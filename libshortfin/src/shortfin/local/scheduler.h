// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_LOCAL_SCHEDULER_H
#define SHORTFIN_LOCAL_SCHEDULER_H

#include <functional>
#include <span>

#include "shortfin/local/device.h"
#include "shortfin/support/iree_helpers.h"

namespace shortfin::local {

class SHORTFIN_API Scope;
class SHORTFIN_API ScopedDevice;

namespace detail {

class SHORTFIN_API Account;

// Transactions are accumulated into a command buffer by type and in
// auto-flush mode, the command buffer is submitted upon a change of type.
enum class TransactionType {
  // Non-aliasing transfer. All transfers submitted in a sequence must not
  // alias each other. If there is a dependency, then they must be flushed
  // before adding an alising transfer.
  TRANSFER = 0,
  // Standalone dispatch that will be enqueued in an individual dispatch.
  SEQUENTIAL_DISPATCH = 1,
  // Parallelizable dispatch that can be enqueued with other parallel
  // dispatches.
  PARALLEL_DISPATCH = 2,
  // Special case: no active transaction type sentinel. Highest number
  // so it doesn't clutter switch-jumps.
  NONE = 3,
};

enum class TransactionMode {
  // All transactions are eagerly flushed in their own command buffers.
  // While potentially not as efficient as possible, this mode is
  // deterministic with respect to typical programming models. The submission
  // logic may apply "as if eager" optimizations in narrow cases where it
  // is obvious that batching is well defined.
  EAGER = 0,
  // Pending command buffers are not flushed until explicitly set to do so.
  EXPLICIT = 1,
};

// Control object for a resource that is tracked on the timeline. Each such
// resource is associated with a single Account. In the case of a resource
// that is shared across device queues, a consistent account will be chosen
// as primary (typically corresponding to the lowest numbered queue). This
// object tracks two things about the resource:
//   1. Mutation Barrier: Idle timepoint of the most recent affecting
//      mutation. Since any such mutation mutation must have been enqueued
//      on a single Account, we only need to track the timepoint here
//      (against the account's sem).
//   2. Use Barrier: Fence joined to all timepoints accessing the resource
//      (read and write). At any given time, this represents the point at
//      which the resource has no further pending uses.
// Since TimelineResources are shared (i.e. across subspan storage, etc),
// they are modeled as reference counted (using non atomics, since this is
// "scoped" same thread access). They must only be held in a context that
// is keeping the containing Scope alive.
//
// Note to the future: in discussing the above, many cases were noted where
// a more advanced programming model would be desirable in order to exercise
// more concurrency. However, the conservative default behavior of
// (effectively) reader/writer locks on resources gives solid, understandable
// behavior for a default programming model. It is not meant to be valid
// for everything over time and should not be considered sacred.
class SHORTFIN_API TimelineResource {
 public:
  class SHORTFIN_API Ref {
   public:
    Ref() : res_(nullptr) {}
    explicit Ref(TimelineResource *res) : res_(res) { res_->Retain(); }
    Ref(const Ref &other) : res_(other.res_) { res_->Retain(); }
    void operator=(const Ref &other) = delete;
    Ref(Ref &&other) : res_(other.res_) { other.res_ = nullptr; }
    ~Ref() {
      if (res_) res_->Release();
    }
    TimelineResource *operator->() { return res_; }

   private:
    TimelineResource *res_;
  };
  TimelineResource(TimelineResource &other) = delete;

  // Sets the mutation barrier.
  // Note that the semaphore set in this way is not retained as it is
  // assumed to be part of the local scheduler.
  void set_mutation_barrier(iree_hal_semaphore_t *sem, uint64_t timepoint) {
    mutation_barrier_sem_ = sem;
    mutation_barrier_timepoint_ = timepoint;
  }
  iree_hal_semaphore_list_t mutation_barrier() {
    if (!mutation_barrier_sem_) {
      return iree_hal_semaphore_list_empty();
    } else {
      return iree_hal_semaphore_list_t{
          .count = 1,
          .semaphores = &mutation_barrier_sem_,
          .payload_values = &mutation_barrier_timepoint_};
    }
  }

  // Use barrier can have new timepoints inserted or converted to a
  // semaphore list.
  void use_barrier_insert(iree_hal_semaphore_t *sem, uint64_t timepoint);
  iree_hal_semaphore_list_t use_barrier() {
    return iree_hal_fence_semaphore_list(use_barrier_fence_);
  }

 private:
  TimelineResource(iree_allocator_t host_allocator, size_t semaphore_capacity);
  void Retain() { refcnt_++; }
  void Release() {
    if (--refcnt_ == 0) delete this;
  }

  int refcnt_ = 0;
  // Non-owning mutation barrier semaphore and timepoint. The fact that this
  // is a single semaphore is an implementation detail that may be generalized
  // in the future should it be necessary to track multiple write sources.
  iree_hal_semaphore_t *mutation_barrier_sem_ = nullptr;
  uint64_t mutation_barrier_timepoint_ = 0;

  // Use barrier fence. The fact that this is a fence object with a fixed
  // capacity is an implementation detail.
  iree_hal_fence_ptr use_barrier_fence_;
  friend class Scheduler;
};

// Accounting structure for a single logical device (Device*), which
// means that each addressable queue gets its own Account.
class SHORTFIN_API Account {
 public:
  Account(Device *device);
  Device *device() const { return device_; }
  iree_hal_device_t *hal_device() { return hal_device_; }
  size_t semaphore_count() const { return 1; }

  // Accesses the active command buffer. This will only be non-null if a
  // pending transaction has been set up (i.e. via AppendCommandBuffer).
  iree_hal_command_buffer_t *active_command_buffer() {
    return active_command_buffer_.get();
  }

  // Extend the current command buffer active deps to join over sem_list.
  void active_deps_extend(iree_hal_semaphore_list_t sem_list);

  // Queue timeline.
  iree_hal_semaphore_t *timeline_sem() { return sem_; }
  uint64_t timeline_idle_timepoint() { return idle_timepoint_; }

 private:
  void Initialize();
  void Reset();
  Device *device_;
  iree_hal_device_t *hal_device_;
  TransactionType active_tx_type_ = TransactionType::NONE;
  iree_hal_fence_ptr active_deps_;
  iree_hal_command_buffer_ptr active_command_buffer_;
  iree_hal_queue_affinity_t active_queue_affinity_bits_;

  // Timepoint at which this device is considered idle, inclusive of any
  // active_command_buffer that has not yet been submitted. This means
  // that at any time, a wait on this timepoint will produce the expected
  // temporal sequencing, but it may represent a point in the future where
  // the work needed to reach it has not been submitted yet.
  // This means that it has two interpretations based on the state of
  // active_command_buffer_:
  //   - nullptr:  All work has been flushed to reach this timepoint.
  //   - !nullptr: When flushed, the active command buffer will be submitted
  //     to signal this timepoint, fulfilling any waiters.
  // Whenever we transition active_command_buffer_ from nullptr,
  // idle_timepoint_ *must* be incremented by at least 1 (or otherwise
  // an eventual submission would submit a duplicate timepoint). This
  // timepoint is only valid for the local sem_.
  uint64_t idle_timepoint_ = 0;
  iree_hal_semaphore_ptr sem_;
  friend class Scheduler;
};

// Handles scheduling state for a scope.
class SHORTFIN_API Scheduler {
 public:
  Scheduler(iree_allocator_t host_allocator)
      : host_allocator_(host_allocator) {}

  TransactionMode transaction_mode() const { return tx_mode_; }

  // Given a ScopedDevice (which may logically bind to multiple queues),
  // returns a deterministic Account associated with the device that can be
  // used for accounting and scheduling.
  Account &GetDefaultAccount(ScopedDevice &device);

  // Sets up |device| for appending commands to a command buffer, invoking
  // callback to complete the mutation. Depending on the current transaction
  // mode and tx_type, this may involve flushing the current command buffer.
  // This will always allocate a new timepoint from the global timeline and
  // increment the elected device account's idle timepoint. It will also
  // set any affinity bits on the pending submission.
  void AppendCommandBuffer(ScopedDevice &device, TransactionType tx_type,
                           std::function<void(Account &)> callback);

  // Flushes any pending accounts that have accumulated commands.
  void Flush();

  // Gets a fresh TimelineResource which can be used for tracking resource
  // read/write and setting barriers. Note that these are all allocated fresh
  // on each call today but may be pooled in the future.
  TimelineResource::Ref NewTimelineResource(iree_allocator_t host_allocator) {
    return TimelineResource::Ref(
        new TimelineResource(host_allocator, semaphore_count_));
  }

 private:
  void Initialize(std::span<Device *const> devices);
  iree_allocator_t host_allocator_;
  // Each distinct hal device gets an account.
  std::vector<Account> accounts_;
  // Accounts indexed by DeviceAddress::device_id().
  std::unordered_map<uint64_t, Account *> accounts_by_device_id_;
  size_t semaphore_count_ = 0;

  // Transaction management.
  TransactionMode tx_mode_ = TransactionMode::EAGER;
  TransactionType current_tx_type_ = TransactionType::NONE;

  friend class local::Scope;
};

}  // namespace detail
}  // namespace shortfin::local

#endif  // SHORTFIN_LOCAL_SCHEDULER_H