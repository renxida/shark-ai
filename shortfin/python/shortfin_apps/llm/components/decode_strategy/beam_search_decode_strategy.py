from .base_decode_strategy import DecodeStrategy, DecodeStrategyConfig


from asyncio import gather
from dataclasses import dataclass
from typing import Dict, List, Tuple
from uuid import uuid4

import numpy as np

from ..messages import InferenceExecRequest, InferencePhase


@dataclass
class ExecRequestSelection:
    """Helper class top make `BeamGroup.evaluate_top_k` return cleaner."""

    log_prob: float
    exec_req: InferenceExecRequest
    token: int
    min_log_prob: float


class BeamGroup:
    def __init__(
        self, beam_group_id: str, n_beams: int, exec_reqs: list[InferenceExecRequest]
    ):
        self.beam_group_id = beam_group_id
        self.n_beams = n_beams
        self.exec_reqs = exec_reqs
        self.completed_reqs: set[InferenceExecRequest] = set()

    async def wait(self):
        done_signals = [
            req.done for req in self.exec_reqs if req not in self.completed_reqs
        ]
        return await gather(*done_signals)

    def topk(
        self, logits: np.array, k: int, axis: int
    ) -> Tuple[List[float], List[int]]:
        # TODO: Move this to sfnp.array
        indices = np.argpartition(logits, -k, axis=axis)
        topk_indices = indices[axis][-k:]
        topk_values = logits[axis][topk_indices]

        sorted_indices = np.argsort(topk_values)[::-1]
        topk_values = topk_values[sorted_indices]
        topk_indices = topk_indices[sorted_indices]

        return topk_values, topk_indices

    def log_softmax(self, logits: np.array) -> np.array:
        # TODO: Move this to sfnp.array
        c = logits.max()
        logsumexp = np.log(np.exp(logits - c).sum())
        return logits - c - logsumexp

    def evaluate_topk(self) -> List[ExecRequestSelection]:
        # TODO: Use temperature when processing logits for better diversity of
        # outputs.
        exec_reqs = self.exec_reqs

        log_prob_map: Dict[float, tuple[InferenceExecRequest, int]] = {}
        min_log_prob = 0.0
        # Find the topk tokens for each req in our beam group
        for exec_req in exec_reqs:
            if exec_req in self.completed_reqs:
                continue
            # NOTE: This copy is slow, and part of why this needs to be moved to
            # `shortfin.array`
            logits = np.array(exec_req.result_logits)
            # Take log_softmax. This is to avoid a req's cumulative probability
            # becoming too small, which can lead precision issues.
            # This allows us to obtain cumulative probability by summing
            # the log_probabilities, instead of multiplying the probabilities.
            log_logits = self.log_softmax(logits)
            log_logits = np.squeeze(log_logits, 1)
            values, tokens = self.topk(log_logits, self.n_beams, -1)
            min_val = values[-1]
            if min_val < min_log_prob:
                min_log_prob = min_val
            for value, token in zip(values, tokens):
                cumulative_log_prob = exec_req.cumulative_log_prob + value
                log_prob_map[cumulative_log_prob] = (exec_req, token)

        # Find the topk tokens across all exec_reqs
        sorted_keys = sorted(log_prob_map.keys(), reverse=True)
        exec_req_selections: List[ExecRequestSelection] = []
        for key in sorted_keys[: self.n_beams - len(self.completed_reqs)]:
            exec_req, token = log_prob_map[key]
            exec_req_selections.append(
                ExecRequestSelection(
                    # Shift log_probs to the right to avoid large
                    # negative numbers
                    log_prob=key - min_log_prob,
                    exec_req=exec_req,
                    token=token,
                    min_log_prob=min_log_prob,
                )
            )

        # TODO: Create a quick helper class so that this tuple doesn't disgust me
        return exec_req_selections

    def process_beams(self, eos_token_id):
        exec_reqs_selections = self.evaluate_topk()
        visited_reqs: Dict[str, InferenceExecRequest] = {}
        new_reqs = set()

        for selection in exec_reqs_selections:
            new_req = selection.exec_req
            token = selection.token
            if new_req.instance_id not in visited_reqs:
                new_req.input_token_ids.append(token)
                new_req.output_token_ids.append(token)
                new_req.start_position += 1
                new_req.accumulated_normalization += abs(selection.min_log_prob)

            else:
                visited_req = visited_reqs[new_req.instance_id]
                new_req = visited_req.replicate_self()
                new_req.input_token_ids.append(token)
                new_req.output_token_ids.append(token)

            new_req.cumulative_log_prob = selection.log_prob
            visited_reqs[new_req.instance_id] = new_req
            new_reqs.add(new_req)
            if token == eos_token_id:
                self.completed_reqs.add(new_req)

        for req in self.exec_reqs:
            if req not in new_reqs:
                req.free_cache_pages()

        self.exec_reqs = list(new_reqs)

    def _final_score(self, exec_req: InferenceExecRequest):
        return (
            exec_req.cumulative_log_prob - exec_req.accumulated_normalization
        ) / len(exec_req.output_token_ids)

    def find_top_beam(self) -> InferenceExecRequest:
        completed_reqs = list(self.completed_reqs)
        if not completed_reqs:
            completed_reqs = self.exec_reqs
        max_score = self._final_score(completed_reqs[0])
        selected_req = completed_reqs[0]
        for req in completed_reqs[1:]:
            score = self._final_score(req)
            if score > max_score:
                selected_req = req
                max_score = score

        return selected_req

    def __del__(self):
        for req in self.exec_reqs:
            req.free_cache_pages()

        for req in self.completed_reqs:
            req.free_cache_pages()


@dataclass
class BeamSearchDecodeStrategyConfig(DecodeStrategyConfig):
    n_beams: int
    return_top_k: bool = False


class BeamSearchDecodeStrategy(DecodeStrategy):
    beam_map: dict[str, BeamGroup] = {}

    def __init__(
        self,
        decode_strategy_config: BeamSearchDecodeStrategyConfig,
    ):
        self._decode_strategy_config = decode_strategy_config

    @property
    def decode_strategy_config(self):
        return self._decode_strategy_config

    def create_beam(self, requests: list[InferenceExecRequest]) -> BeamGroup:
        beam_group_id = str(uuid4())
        for req in requests:
            req.beam_group_id = beam_group_id

        beam_group = BeamGroup(
            beam_group_id,
            self.decode_strategy_config.n_beams,
            requests,
        )
        BeamSearchDecodeStrategy.beam_map[beam_group_id] = beam_group
        return beam_group

    def delete_beam(self, beam_group_id: str):
        beam_group = BeamSearchDecodeStrategy.beam_map[beam_group_id]
        del beam_group

    async def decode(
        self,
        exec_req: InferenceExecRequest,
    ) -> List[int] | List[List[int]]:
        config = self.decode_strategy_config
        decode_reqs = [exec_req]
        for _ in range(config.n_beams - 1):
            decode_req = exec_req.replicate_self()
            decode_reqs.append(decode_req)

        beam_group = self.create_beam(decode_reqs)
        for _ in range(config.max_completion_tokens):
            if len(beam_group.completed_reqs) == config.n_beams:
                break

            for exec in beam_group.exec_reqs:
                if exec in beam_group.completed_reqs:
                    continue
                exec.reset(InferencePhase.DECODE)
                config.batcher_callback(exec)

            await beam_group.wait()
            beam_group.process_beams(config.eos_token_id)

        result = None
        if config.return_top_k:
            reqs = beam_group.completed_reqs
            for req in beam_group.exec_reqs:
                reqs.add(req)
            result = [req.output_token_ids for req in reqs]

        else:
            result = beam_group.find_top_beam().output_token_ids

        self.delete_beam(beam_group.beam_group_id)
        config.streaming_callback(result)
