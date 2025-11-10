#!/bin/bash
# Run from current directory (no hardcoded path)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export USE_SUBTASKS=false
export USE_KNOWLEDGE_BASE=true

# Arguments
SERVER_PORT=${1:-8000}  # 기본값 8000
MODEL=${2:-gpt-5}       # 기본값 gpt-5

# 자동 계산
FRAME_PORT=$((SERVER_PORT + 1))
CACHE_DIR=".pokeagent_cache_${SERVER_PORT}"

echo "Starting processes for:"
echo "  Model: $MODEL"
echo "  Server port: $SERVER_PORT"
echo "  Frame port: $FRAME_PORT"
echo "  Cache dir: $CACHE_DIR"
echo ""

# Backup existing cache for this port
if [ -d "$CACHE_DIR" ]; then
    mv "$CACHE_DIR" "${CACHE_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
fi

# Backup main cache if it exists
if [ -d ".pokeagent_cache" ]; then
    mv .pokeagent_cache .pokeagent_cache_backup_$(date +%Y%m%d_%H%M%S)
fi

echo "Starting main server..."
nohup /home/heatz123/anaconda3/envs/pokeagent/bin/python3 -m server.app --port $SERVER_PORT --record > server_${SERVER_PORT}.log 2>&1 &
sleep 2

echo "Starting frame server..."
nohup /home/heatz123/anaconda3/envs/pokeagent/bin/python3 -m server.frame_server --port $FRAME_PORT > frame_server_${FRAME_PORT}.log 2>&1 &
sleep 2

echo "Starting client..."
nohup /home/heatz123/anaconda3/envs/pokeagent/bin/python3 code_client.py --port $SERVER_PORT --model $MODEL --delay 1.0 > client_${SERVER_PORT}.log 2>&1 &

echo "Starting meta-agent daemon..."
nohup /home/heatz123/anaconda3/envs/pokeagent/bin/python3 meta_agent_daemon.py --interval 30 --max-validations 20 > meta_agent_${SERVER_PORT}.log 2>&1 &

echo ""
echo "✅ All processes started!"
echo "Logs: server_${SERVER_PORT}.log, frame_server_${FRAME_PORT}.log, client_${SERVER_PORT}.log, meta_agent_${SERVER_PORT}.log"
echo ""
ps aux | grep -E "(server\.app|frame_server|code_client|meta_agent_daemon)" | grep -v grep

