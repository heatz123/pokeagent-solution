#!/bin/bash

# ====================================================================
# Kill Run by Port
# Stops all processes associated with a specific port
# ====================================================================

PORT=${1}

if [ -z "$PORT" ]; then
    echo "Usage: ./kill_run.sh <port>"
    echo ""
    echo "Example:"
    echo "  ./kill_run.sh 8000"
    exit 1
fi

if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    echo "❌ Error: Port must be a number"
    exit 1
fi

FRAME_PORT=$((PORT + 1))

echo "🛑 Killing processes on port $PORT..."
echo ""

# Kill server
if pkill -f "server.app --port $PORT"; then
    echo "✅ Killed server on port $PORT"
else
    echo "ℹ️  No server process found on port $PORT"
fi

# Kill frame server
if pkill -f "frame_server --port $FRAME_PORT"; then
    echo "✅ Killed frame server on port $FRAME_PORT"
else
    echo "ℹ️  No frame server process found on port $FRAME_PORT"
fi

# Kill client
if pkill -f "code_client.py --port $PORT"; then
    echo "✅ Killed client on port $PORT"
else
    echo "ℹ️  No client process found on port $PORT"
fi

# Kill meta-agent daemon
if pkill -f "meta_agent_daemon.py --port $PORT"; then
    echo "✅ Killed meta-agent daemon on port $PORT"
else
    echo "ℹ️  No meta-agent daemon process found on port $PORT"
fi

echo ""
echo "✅ Done!"
