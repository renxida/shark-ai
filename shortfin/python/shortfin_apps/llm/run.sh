#!/bin/bash

# Create or clear the log file
> run.log

# Start the server in the background, redirect output to both file and terminal, and save its PID
python -m shortfin_apps.llm.server   --tokenizer=/tmp/sharktank/llama/tokenizer.json   --model_config=/tmp/sharktank/llama/edited_config.json   --vmfb=/tmp/sharktank/llama/model.vmfb   --parameters=/tmp/sharktank/llama/open-llama-3b-v2-f16.gguf   --device=hip 2>&1 | tee -a run.log &

SERVER_PID=

# Wait a bit for the server to start up
sleep 5

# Run the client
python ~/SHARK-Platform/shortfin/python/shortfin_apps/llm/client.py

# Kill the server
kill

# Wait for the server to shut down
wait
