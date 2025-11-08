#!/bin/bash

# ====================================================================
# Simple Run Launcher
# Default: Creates isolated git clone
# --local: Runs in current directory
# ====================================================================

PROJECT_DIR="/home/heatz123/workspace/rl/pokeagent-speedrun-heatz"
RUNS_BASE_DIR="/home/heatz123/workspace/rl/pokeagent-runs"

# Parse arguments
LOCAL_MODE=false
if [[ "$1" == "--local" || "$1" == "-l" ]]; then
    LOCAL_MODE=true
    shift
fi

MODEL=${1:-gpt-5}
PORT=${2:-8000}

# === LOCAL MODE ===
if [ "$LOCAL_MODE" = true ]; then
    echo "🏃 Starting local run (model=$MODEL, port=$PORT)"
    cd "$PROJECT_DIR"
    export USE_SUBTASKS=true
    export USE_KNOWLEDGE_BASE=false
    bash start_all.sh $PORT $MODEL
    exit 0
fi

# === ISOLATED MODE ===
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_SAFE=$(echo "$MODEL" | sed 's/[^a-zA-Z0-9]/_/g')
RUN_NAME="run_${TIMESTAMP}_${MODEL_SAFE}_port${PORT}"
RUN_DIR="${RUNS_BASE_DIR}/${RUN_NAME}"

echo "🚀 Starting isolated run: $RUN_NAME"

# Create runs directory
mkdir -p "${RUNS_BASE_DIR}"

# Clone repository
cd "$PROJECT_DIR"
BRANCH=$(git branch --show-current)
git clone --local --branch "$BRANCH" "$PROJECT_DIR" "$RUN_DIR"

# Save metadata
cd "$RUN_DIR"
cat > run_info.json <<EOF
{
  "timestamp": "$TIMESTAMP",
  "model": "$MODEL",
  "port": $PORT,
  "branch": "$BRANCH",
  "commit": "$(git rev-parse --short HEAD)"
}
EOF

# Start processes
export USE_SUBTASKS=true
export USE_KNOWLEDGE_BASE=false
export VLM_MODEL="qwen3-vl:2b"
bash start_all.sh $PORT $MODEL

# Create latest symlink
ln -sfn "$RUN_DIR" "${RUNS_BASE_DIR}/latest"

echo ""
echo "✅ Run started"
echo "📁 $RUN_DIR"
echo "🛑 Stop: ./kill_run.sh $PORT"
