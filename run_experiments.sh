#!/usr/bin/env sh
set -eu

# Common args
EPISODES=1
MAX_STEPS=128
REPLAY_LEN=2
SEED=0

# Model/provider pairs (MODEL PROVIDER per line)
# gpt-5 openai
MODELS_PROVIDERS='
gpt-5-mini openai
gemini-2.5-pro gemini
gemini-2.5-flash gemini
'

# Environments
ENVS="
MiniGrid-SimpleCrossingS11N5-v0
"

# MiniGrid-DoorKey-6x6-v0
# MiniGrid-SimpleCrossingS11N5-v0

# Loops
echo "$MODELS_PROVIDERS" | while read -r MODEL PROVIDER; do
  # skip empty lines
  [ -z "${MODEL:-}" ] && continue
  for ENV_ID in $ENVS; do
    echo "Running model=$MODEL (provider=$PROVIDER) on env=$ENV_ID"
    python bench.py \
      --model="$MODEL" \
      --provider="$PROVIDER" \
      --episodes="$EPISODES" \
      --max_steps="$MAX_STEPS" \
      --replay_len="$REPLAY_LEN" \
      --seed="$SEED" \
      --env_id="$ENV_ID"
  done
done
