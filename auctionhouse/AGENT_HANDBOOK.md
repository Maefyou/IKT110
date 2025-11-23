# 📘 Smart Auction Agent - Complete Handbook

## Table of Contents
1. [Quick Start](#quick-start)
2. [Testing Your Agent](#testing-your-agent)
3. [Connecting to Different Servers](#connecting-to-different-servers)
4. [Dashboard Features](#dashboard-features)
5. [Strategy Tuning](#strategy-tuning)
6. [Troubleshooting](#troubleshooting)
7. [Competition Mode](#competition-mode)

---

## Quick Start

### Prerequisites
- Python 3.x installed
- Required packages: `flask`, `flask-cors`, `numpy`

### Starting Your Agent (Localhost)

**Option 1: Automated (Recommended)**
```bash
cd /home/maefyou/UNI/UIA/IKT110/IKT110/auctionhouse
./quick_start.sh
```

**Option 2: Manual**
```bash
# Terminal 1: Start Game Server
cd /home/maefyou/UNI/UIA/IKT110/IKT110/auctionhouse/dnd_auction_game
uvicorn dnd_auction_game.server:app

# Terminal 2: Start Your Agent
cd /home/maefyou/UNI/UIA/IKT110/IKT110/auctionhouse/my_agents
python3 smart_agent_with_dashboard.py
```

### Access Dashboard
Open in browser: **http://localhost:5000**

---

## Testing Your Agent

### Test 1: Solo Test (No Competition)
Tests your agent alone to verify basic functionality.

```bash
# After starting server and agent (see Quick Start above)

# New Terminal: Run Game
cd /home/maefyou/UNI/UIA/IKT110/IKT110/auctionhouse/dnd_auction_game
python -m dnd_auction_game.play
```

**Expected Output:**
- Your agent should connect and make bids
- Dashboard shows live updates
- Game completes successfully

---

### Test 2: Multi-Bot Competition (Recommended)
Tests your agent against multiple opponents to simulate real competition.

**Terminal Setup:**

```bash
# Terminal 1: Start Server & Agent (if not already running)
cd /home/maefyou/UNI/UIA/IKT110/IKT110/auctionhouse
./quick_start.sh

# Terminal 2: Start Random Walk Bot
cd /home/maefyou/UNI/UIA/IKT110/IKT110/auctionhouse/dnd_auction_game/example_agents
python3 agent_random_walk.py

# Terminal 3: Start Random Single Bot
cd /home/maefyou/UNI/UIA/IKT110/IKT110/auctionhouse/dnd_auction_game/example_agents
python3 agent_random_single.py

# Terminal 4: Start Tiny Bid Bot
cd /home/maefyou/UNI/UIA/IKT110/IKT110/auctionhouse/dnd_auction_game/example_agents
python3 agent_tiny_bid.py

# Terminal 5: Start the Game
cd /home/maefyou/UNI/UIA/IKT110/IKT110/auctionhouse/dnd_auction_game
python -m dnd_auction_game.play
```

**What to Look For:**
- ✅ All 4 agents connect successfully
- ✅ Your agent adapts to opponent behavior
- ✅ Dashboard shows opponent statistics
- ✅ Win rate is competitive (aim for >25% against 3 opponents)

---

### Test 3: Quick Multi-Bot Test (One Command Per Terminal)

For faster testing, you can run all bots with one command each:

```bash
# Terminal 1: Server + Your Agent
cd /home/maefyou/UNI/UIA/IKT110/IKT110/auctionhouse && ./quick_start.sh

# Terminal 2: All Example Bots
cd /home/maefyou/UNI/UIA/IKT110/IKT110/auctionhouse/dnd_auction_game/example_agents && \
python3 agent_random_walk.py & \
python3 agent_random_single.py & \
python3 agent_tiny_bid.py & \
wait

# Terminal 3: Run Game
cd /home/maefyou/UNI/UIA/IKT110/IKT110/auctionhouse/dnd_auction_game && \
python -m dnd_auction_game.play
```

**Note:** After the game, you'll need to Ctrl+C in Terminal 2 to stop the example bots.

---

## Connecting to Different Servers

### Localhost (Default)
Your agent is currently configured to connect to `localhost:8000`. This is perfect for local testing.

### Remote Server (Classroom/Competition)

To connect your agent to a different server (e.g., instructor's server or another student's server):

**Step 1: Edit Configuration**

Open the agent file:
```bash
nano /home/maefyou/UNI/UIA/IKT110/IKT110/auctionhouse/my_agents/smart_agent_with_dashboard.py
```

**Step 2: Find Configuration Section (Lines 28-30)**

Look for:
```python
# Server configuration
GAME_HOST = "localhost"
GAME_PORT = 8000
AGENT_NAME = "SmartEV_Agent"
```

**Step 3: Update Host and Port**

Change to remote server details:
```python
# Server configuration
GAME_HOST = "192.168.1.100"  # Replace with actual server IP
GAME_PORT = 8000             # Replace with actual port if different
AGENT_NAME = "YourName_Agent"  # Make it unique in competition
```

**Common Server Scenarios:**

| Scenario | GAME_HOST | GAME_PORT | Notes |
|----------|-----------|-----------|-------|
| Local Testing | `"localhost"` | `8000` | Your own machine |
| Same Network (LAN) | `"192.168.x.x"` | `8000` | Find server's IP with `ip addr` |
| Cloud Server | `"server.example.com"` | `8000` | Use domain or public IP |
| Custom Port | `"localhost"` | `9000` | If server runs on different port |

**Step 4: Save and Restart**

```bash
# Stop current agent (Ctrl+C in its terminal)
# Or kill it:
pkill -f smart_agent_with_dashboard

# Start with new configuration
cd /home/maefyou/UNI/UIA/IKT110/IKT110/auctionhouse/my_agents
python3 smart_agent_with_dashboard.py
```

**Verification:**

Check the agent output:
```
🎯 Smart Auction Agent Starting...
Dashboard running on http://0.0.0.0:5000
Connecting to game server at 192.168.1.100:8000...
✓ Connected to game server
Waiting for game to start...
```

---

### Dashboard Without Server Connection

Your dashboard can run even if the game server isn't available. This is useful for:
- Testing dashboard functionality
- Adjusting strategy parameters
- Reviewing previous game data (if stored)

```bash
# Just start the agent (no server needed for dashboard)
cd /home/maefyou/UNI/UIA/IKT110/IKT110/auctionhouse/my_agents
python3 smart_agent_with_dashboard.py
```

The dashboard will show "Waiting for game..." but you can still access settings.

---

## Dashboard Features

### Real-Time Statistics

| Metric | Description |
|--------|-------------|
| **Current Points** | Your total points in the game |
| **Current Gold** | Gold available for bidding |
| **Rank** | Your position among all players (1 = first place) |
| **Auctions Won/Lost** | Count of won vs. lost auctions you bid on |
| **Round** | Current game round |
| **Opponents** | Number of competing agents |

### Live Charts

1. **Points History**: Your points over time (blue line)
2. **Gold History**: Your gold reserves over time (green line)

### 📜 Complete Auction History

The expandable history section shows **ALL auctions from ALL rounds**.

**How to Use:**
1. Click "📜 Complete Auction History" to expand
2. View results grouped by round (newest first)
3. Each auction shows:
   - 🏆 = You won
   - ❌ = You lost (but participated)
   - ⏭️ = You skipped this auction

**Information Displayed:**
- Auction ID
- Winner and winning bid
- Reward value
- Your bid amount (if you participated)
- Total number of bidders

**Note:** After hard refresh (Ctrl+Shift+R), the history resets. Run a new game to see complete history.

---

### Viewing Past Game Data

**Live Data vs. Historical Data**

- **Live Dashboard**: Shows ONLY the current game session (resets when agent restarts)
- **Historical Logs**: Permanent records stored in `my_agents/logs/*.jsonl`

**To review a previous game:**

1. **Find the log file:**
   ```bash
   cd /home/maefyou/UNI/UIA/IKT110/IKT110/auctionhouse/my_agents/logs
   ls -lt  # Shows most recent files first
   ```

2. **View the log:**
   ```bash
   cat agent_local_rand_id_XXXXX_nY.jsonl | python3 -m json.tool
   ```

3. **Extract specific information:**
   ```python
   # Count total auctions
   cat agent_*.jsonl | grep -o '"a[0-9]*"' | sort -u | wc -l
   
   # See your final stats
   cat agent_*.jsonl | tail -1 | python3 -m json.tool
   ```

**Example - Analyze Win Rate:**
```python
python3 << 'EOF'
import json

# Read the log file (replace with your actual file)
with open('logs/agent_local_rand_id_135252_n8.jsonl') as f:
    lines = [json.loads(line) for line in f]

# Count wins and participations
wins = 0
participated = 0
my_id = "local_rand_id_135252"  # Your agent ID

for line in lines:
    for auction_id, auction_data in line.get('prev_auctions', {}).items():
        bids = auction_data.get('bids', [])
        if any(b['a_id'] == my_id for b in bids):
            participated += 1
            if bids and bids[0]['a_id'] == my_id:
                wins += 1

print(f"Participated: {participated}")
print(f"Won: {wins}")
print(f"Win Rate: {wins/participated*100:.1f}%")
EOF
```

### Strategy Controls (Adjust in Real-Time)

You can adjust these sliders while the game is running:

| Control | Range | Effect |
|---------|-------|--------|
| **Base Aggression** | 0.5 - 1.5 | Higher = bid more aggressively |
| **Risk Tolerance** | 0 - 1 | Higher = prefer risky auctions |
| **Portfolio Size** | 1 - 10 | How many auctions to bid on |
| **Leader Aggression** | 0.5 - 1.5 | Aggression when in first place |
| **Behind Aggression** | 0.5 - 1.5 | Aggression when behind |
| **Max Bid Fraction** | 0.1 - 0.9 | Max % of gold to risk per bid |

**Recommended Settings:**

| Situation | Base Aggr. | Risk Tol. | Portfolio |
|-----------|------------|-----------|-----------|
| **Conservative** | 0.7 | 0.3 | 6-8 |
| **Balanced** | 1.0 | 0.5 | 5-6 |
| **Aggressive** | 1.3 | 0.7 | 3-4 |

---

## Strategy Tuning

### Understanding Your Agent's Strategy

Your agent uses **Expected Value (EV)** calculations:
```
EV = (Probability of Winning) × (Reward - Bid Cost)
```

**Key Features:**
1. **Opponent Learning**: Tracks opponent bid patterns
2. **Outlier Detection**: Identifies unusual opponent behavior
3. **Portfolio Strategy**: Diversifies bids across multiple auctions
4. **Dynamic Aggression**: Adjusts based on position (leading/behind)

### When to Adjust Parameters

**If you're consistently losing:**
- ↑ Increase Base Aggression (to 1.2-1.4)
- ↑ Increase Behind Aggression (to 1.4-1.5)
- ↓ Decrease Portfolio Size (to 3-4, focus bids)

**If you're running out of gold:**
- ↓ Decrease Base Aggression (to 0.7-0.9)
- ↓ Decrease Max Bid Fraction (to 0.3-0.5)
- ↑ Increase Portfolio Size (to 7-9, spread risk)

**If opponents are unpredictable:**
- ↑ Increase Risk Tolerance (to 0.6-0.8)
- Keep Portfolio Size high (7-9)

**If you're leading comfortably:**
- ↓ Decrease Leader Aggression (to 0.6-0.8)
- ↑ Increase Portfolio Size (to 8-10, play it safe)

---

## Troubleshooting

### Important: Agent Lifecycle

**Your agent runs for ONE game session, then stops automatically.**

This is normal behavior! Each time you want to run a new game:
1. Start the agent: `python3 smart_agent_with_dashboard.py`
2. Start other bots (if testing multi-bot)
3. Run the game: `python -m dnd_auction_game.play`
4. After game ends, agent stops
5. Repeat from step 1 for next game

**To keep the agent running between games** (if you plan to run multiple consecutive games):
- The current version exits after each game
- For competition day, this is actually GOOD - it prevents stale data
- For testing, just restart it between games

---

### Dashboard shows "Loading data..."

**Cause:** Browser cache or agent not running

**Solutions:**
1. **Hard refresh browser:** Press `Ctrl + Shift + R` (Linux/Windows) or `Cmd + Shift + R` (Mac)
2. **Check agent is running:**
   ```bash
   ps aux | grep smart_agent
   ```
   If not running, start it:
   ```bash
   cd /home/maefyou/UNI/UIA/IKT110/IKT110/auctionhouse/my_agents
   python3 smart_agent_with_dashboard.py
   ```
3. **Check dashboard port:**
   ```bash
   lsof -i :5000
   ```

### "Connection refused" error

**Cause:** Game server not running or wrong host/port

**Solutions:**
1. **Check game server:**
   ```bash
   lsof -i :8000
   ```
   If not running:
   ```bash
   cd /home/maefyou/UNI/UIA/IKT110/IKT110/auctionhouse/dnd_auction_game
   uvicorn dnd_auction_game.server:app
   ```
2. **Verify configuration** in `smart_agent_with_dashboard.py`:
   - `GAME_HOST` matches server location
   - `GAME_PORT` matches server port

### Agent crashes during game

**Check logs:**
```bash
cat /tmp/smart_agent.log
```

**Common issues:**
- **Division by zero:** Not enough opponent data → Fixed in latest version
- **Invalid bid:** Agent tried to bid more than available gold → Check Max Bid Fraction
- **JSON errors:** Server communication issue → Check server logs

### History only shows recent rounds

**Cause:** You're looking at old cached data, OR the agent stopped after the game ended

**Important:** The agent stops running after each game completes! You need to restart it for the next game.

**Solution:**
1. **Check if agent is still running:**
   ```bash
   lsof -i :5000
   ```
   If nothing shows up, the agent has stopped.

2. **Restart the agent:**
   ```bash
   cd /home/maefyou/UNI/UIA/IKT110/IKT110/auctionhouse/my_agents
   python3 smart_agent_with_dashboard.py
   ```

3. **Hard refresh browser:** `Ctrl + Shift + R`

4. **Run a NEW game** - the dashboard shows data from the CURRENT game session only

**Note:** Each time you start the agent, the history starts fresh. If you want to review previous games, check the log files in `my_agents/logs/`.

### Multiple agents with same name

**Problem:** Your agent uses default name "SmartEV_Agent"

**Solution:** Change `AGENT_NAME` in the configuration:
```python
AGENT_NAME = "YourName_SmartAgent"  # Make it unique
```

---

## Competition Mode

### Preparing for Class Competition

**1. Test Against Multiple Bots**
Run the multi-bot test (see [Testing Your Agent](#testing-your-agent)) and aim for:
- ✅ Consistent top 2 finish
- ✅ Win rate > 30% of auctions participated in
- ✅ Positive point growth throughout game

**2. Fine-Tune Your Strategy**
Use the dashboard to find optimal settings for different scenarios.

**3. Update Agent Name**
```python
AGENT_NAME = "YourName_Agent_v2"  # Make it unique and identifiable
```

**4. Get Server Details**
Ask your instructor for:
- Server IP address (or domain)
- Port number (probably 8000)
- Any special rules or modifications

**5. Connect to Competition Server**

Edit configuration:
```python
GAME_HOST = "instructor-server-ip"  # Get from instructor
GAME_PORT = 8000  # Verify with instructor
AGENT_NAME = "YourName_Agent"  # Unique name
```

**6. Run Your Agent**
```bash
cd /home/maefyou/UNI/UIA/IKT110/IKT110/auctionhouse/my_agents
python3 smart_agent_with_dashboard.py
```

**7. Monitor Performance**
- Dashboard: http://localhost:5000
- Watch live stats during competition
- Note winning strategies from opponents

### During Competition

**Do:**
- ✅ Keep dashboard open to monitor performance
- ✅ Adjust strategy sliders if you're falling behind
- ✅ Watch opponent behavior in auction history
- ✅ Stay calm - the game has multiple rounds

**Don't:**
- ❌ Restart your agent mid-game (you'll lose progress)
- ❌ Change parameters too frequently (let strategy develop)
- ❌ Panic if you're behind early (game is long)

---

## Advanced: Modify Agent Behavior

### Adding New Strategy Rules

Edit `smart_agent_with_dashboard.py` and find the `make_bid()` method (around line 220):

**Example: Always skip auctions with reward < 5**
```python
# In make_bid() method, before portfolio selection:
auctions = {k: v for k, v in auctions.items() 
           if v.get("reward", 0) >= 5}
```

**Example: Bid extra aggressively on high-reward auctions**
```python
# In portfolio selection loop:
if auction["reward"] > 15:
    ev_score *= 1.5  # Boost EV for high-reward auctions
```

### Logging for Analysis

Add custom logging:
```python
# At top of file, add:
import logging
logging.basicConfig(filename='my_strategy.log', level=logging.INFO)

# In make_bid():
logging.info(f"Round {self.rounds_played}: Bid {bid_amount} on {auction_id}")
```

---

## Quick Reference

### Essential Commands

| Task | Command |
|------|---------|
| **Start Everything** | `./quick_start.sh` |
| **Stop Everything** | `Ctrl+C` in quick_start terminal |
| **View Agent Logs** | `tail -f /tmp/smart_agent.log` |
| **View Server Logs** | `tail -f /tmp/game_server.log` |
| **Check Ports** | `lsof -i :5000` (agent) or `lsof -i :8000` (server) |
| **Kill Agent** | `pkill -f smart_agent_with_dashboard` |

### File Locations

| File | Purpose |
|------|---------|
| `my_agents/smart_agent_with_dashboard.py` | Your agent code |
| `my_agents/templates/dashboard.html` | Dashboard interface |
| `quick_start.sh` | Automated startup script |
| `/tmp/smart_agent.log` | Agent runtime logs |
| `/tmp/game_server.log` | Server logs |
| `my_agents/logs/*.jsonl` | Game history |

### URLs

| Service | URL |
|---------|-----|
| **Dashboard** | http://localhost:5000 |
| **Server Leaderboard** | http://localhost:8000 |
| **Debug Test Page** | http://localhost:5000/test |

---

## Getting Help

**Check Logs First:**
```bash
# Agent logs
tail -20 /tmp/smart_agent.log

# Server logs
tail -20 /tmp/game_server.log

# Browser console (press F12, click Console tab)
```

**Common Error Messages:**

| Error | Meaning | Solution |
|-------|---------|----------|
| `Connection refused` | Server not running | Start server with `uvicorn` |
| `Address already in use` | Port 5000/8000 taken | Kill existing process or use different port |
| `Module not found` | Missing dependency | `pip install flask flask-cors numpy` |
| `Invalid bid` | Bid exceeds available gold | Check Max Bid Fraction setting |

---

## Understanding Your Agent's Behavior

### Why Only 2-3 Bids Some Rounds?

**This is your strategy working perfectly!** Your agent uses **portfolio diversification**:

- **portfolio_size** controls how many auctions to bid on
- Instead of all-in on one auction, it spreads risk across multiple
- **It skips bad auctions** (negative or low expected value)

**Example from a real game:**
- Round 11: You had 14,722 gold, 3 auctions available
- Agent chose the 2 best auctions (EV: 13.0 and 11.5)
- **Skipped** the terrible auction (EV: 0.0, negative bonus)
- **Result:** Won 32 points for just 245 gold!
- Opponent wasted 1,025 gold on bad auction, got only 1 point

### Strategy Behavior

**When Leading:**
- Uses `leader_aggression = 0.6` (conservative)
- Protects your lead
- Bids on portfolio_size best auctions

**When Behind:**
- Uses `behind_aggression = 1.2` (aggressive)
- Reduces portfolio to 2-3 auctions
- Goes for high-value wins to catch up

### Agent Features

1. **Expected Value Calculation**: Optimal bidding based on dice probabilities
2. **Opponent Modeling**: Learns from competitors' behavior with outlier detection
3. **Dynamic Strategy**: Adjusts based on game state (leading/behind)
4. **Risk Management**: Adjusts variance tolerance based on position
5. **Portfolio Diversification**: Spreads bids across multiple auctions
6. **Smart Skipping**: Avoids auctions with negative or low expected value

---

## Technical Notes

### Round Numbering

**Dashboard shows "11 rounds" but history shows 10 rounds:**

This is correct! Here's why:
- **Round 0**: Initial state (no auction results yet, nothing to show)
- **Rounds 1-10**: Actual auction rounds with results
- **rounds_played = 11**: Counts all game calls including Round 0

The agent increments `rounds_played` at each call, but Round 0 has no `prev_auctions` (empty), so no history is saved. This is expected behavior.

### Files Structure

```
auctionhouse/
├── my_agents/
│   ├── smart_agent_with_dashboard.py  # Main agent
│   ├── templates/
│   │   └── dashboard.html              # Dashboard UI
│   └── logs/                           # Game history logs
├── dnd_auction_game/                   # Game framework
├── quick_start.sh                      # Auto-start script
└── AGENT_HANDBOOK.md                   # This file
```

---

## Summary

**To Test Solo:**
```bash
./quick_start.sh
# New terminal:
cd dnd_auction_game && python -m dnd_auction_game.play
```

**To Test Multi-Bot:**
```bash
./quick_start.sh
# Start 3 example bots in new terminals
# Then: cd dnd_auction_game && python -m dnd_auction_game.play
```

**To Connect to Remote Server:**
1. Edit `GAME_HOST` in `my_agents/smart_agent_with_dashboard.py` (line 28)
2. Change from `"localhost"` to server IP/domain
3. Restart agent

**To View Dashboard:**
Open http://localhost:5000 (hard refresh with Ctrl+Shift+R if needed)

---

Good luck in the competition! 🎯🏆
