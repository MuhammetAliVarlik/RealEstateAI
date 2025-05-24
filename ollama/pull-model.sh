#!/bin/bash

export OLLAMA_MODELS=/ollama

# Start Ollama
echo "🚀 Starting Ollama server..."
/bin/ollama serve &

pid=$!

# Is Ollama ready? Check using `ollama list`
echo "⏳ Waiting for Ollama to be ready..."
until ollama list >/dev/null 2>&1; do
  sleep 1
done
echo "✅ Ollama is ready."

# Is the model installed?
if ! ollama list | grep -q "qwen2.5"; then
  echo "🔽 Pulling qwen2.5:7b model..."
  ollama pull qwen2.5:7b
else
  echo "📦 Model qwen2.5 already exists. Skipping download."
fi

# Run the model
echo "▶️ Running qwen2.5 model..."
ollama run qwen2.5:latest --keepalive -1m
echo "✅ Model is running."

# Track the lifecycle of the Ollama process
wait $pid