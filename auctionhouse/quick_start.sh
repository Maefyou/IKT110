#!/bin/bash
# Simple test - Start server and agent, show you how to run the game

echo "🎯 Smart Auction Agent - Simple Test"
echo "===================================="
echo ""

# Kill any existing processes
pkill -f "uvicorn.*server:app" 2>/dev/null
pkill -f "smart_agent_with_dashboard" 2>/dev/null
sleep 2

cd "$(dirname "$0")"

# Start game server in background
echo "1️⃣  Starting game server..."
cd dnd_auction_game
uvicorn dnd_auction_game.server:app > /tmp/game_server.log 2>&1 &
SERVER_PID=$!
sleep 3

if ! lsof -i :8000 >/dev/null 2>&1; then
    echo "❌ Server failed to start. Check /tmp/game_server.log"
    cat /tmp/game_server.log
    exit 1
fi

echo "✅ Game server running (PID: $SERVER_PID)"
echo ""

# Start agent in background
echo "2️⃣  Starting your smart agent..."
cd ../my_agents
python3 smart_agent_with_dashboard.py > /tmp/smart_agent.log 2>&1 &
AGENT_PID=$!
sleep 3

if ! lsof -i :5000 >/dev/null 2>&1; then
    echo "❌ Agent failed to start. Check /tmp/smart_agent.log"
    cat /tmp/smart_agent.log
    kill $SERVER_PID
    exit 1
fi

echo "✅ Agent running (PID: $AGENT_PID)"
echo ""

echo "══════════════════════════════════════════════════════"
echo "✅ READY TO TEST!"
echo "══════════════════════════════════════════════════════"
echo ""
echo "📊 Dashboard: http://localhost:5000"
echo "   Open this in your browser to see live stats!"
echo ""
echo "🎮 To start a game, open a NEW terminal and run:"
echo "   cd $(pwd)/../dnd_auction_game"
echo "   python -m dnd_auction_game.play"
echo ""
echo "📋 Logs:"
echo "   Server: /tmp/game_server.log"
echo "   Agent:  /tmp/smart_agent.log"
echo ""
echo "Press Ctrl+C to stop everything..."
echo "══════════════════════════════════════════════════════"
echo ""

# Handle Ctrl+C
trap "echo ''; echo 'Stopping...'; kill $SERVER_PID $AGENT_PID 2>/dev/null; sleep 1; echo 'Done'; exit 0" INT

# Wait
wait $AGENT_PID
