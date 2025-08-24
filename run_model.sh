#!/bin/bash

# Usage:
#   ./run_model.sh <model> "prompt"
#   ./run_model.sh <model> agent

MODEL=$1
ARG2=$2

if [ -z "$MODEL" ]; then
  echo "❌ You must provide a model, e.g.: ./run_model.sh deepcoder \"Hello world\""
  exit 1
fi

if [ "$ARG2" = "agent" ]; then
  echo "🤖 Starting agent with model $MODEL..."
  source .venv/bin/activate
  python main.py --model $MODEL
  exit 0
fi

if [ -z "$ARG2" ]; then
  docker exec -it ollama ollama run $MODEL
else
  docker exec -it ollama ollama run $MODEL "$ARG2"
fi
