#!/bin/bash

for i in {1..8}; do
  curl -X POST \
    http://localhost:8000/generate \
    -H 'Content-Type: application/json' \
    -d '{
      "sampling_params": {
        "max_completion_tokens": 15,
        "temperature": 0.7
      },
      "rid": "'$i'",
      "stream": false,
      "text": "1 2 3 4 5 "
    }' &
done
