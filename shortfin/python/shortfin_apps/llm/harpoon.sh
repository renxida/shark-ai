#!/bin/bash

# Default signal is murder the shit out of the process because it is cancerous
signal=9

# Parse command line options
while getopts ":9" opt; do
  case $opt in
    9)
      signal=9
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# Find and kill the processes
pids=$(pgrep -f "python -m shortfin_apps.llm.server")

if [ -z "$pids" ]; then
  echo "No matching processes found."
  exit 0
fi

for pid in $pids; do
  echo "Killing process $pid with signal $signal"
  kill -$signal $pid
done

echo "Done."
