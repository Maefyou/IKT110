"""
Smart Auction Agent with Live Dashboard
Features:
- Expected Value calculation
- Opponent modeling
- Dynamic strategy based on game state
- Live monitoring dashboard
- Adjustable parameters in real-time
- Robust error handling
"""

import random
import os
import time
import numpy as np
from collections import defaultdict, deque
import threading
import json
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

# Import the game client
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dnd_auction_game'))
from dnd_auction_game import AuctionGameClient

# Server configuration
GAME_HOST = "localhost"
GAME_PORT = 8000
AGENT_NAME = "SmartEV_Agent"
PLAYER_ID = "smart_agent_player"
DASHBOARD_PORT = 5000


class AgentConfig:
    """Configuration parameters that can be adjusted via dashboard"""
    def __init__(self):
        # Bidding parameters
        self.base_aggression = 0.9  # Base aggression level (0-2)
        self.risk_tolerance = 0.5   # Higher = prefer high variance (0-1)
        self.max_bid_fraction = 0.4  # Max fraction of gold per auction (0-1)
        self.min_reserve = 50       # Minimum gold to keep in reserve
        self.portfolio_size = 4     # Number of auctions to bid on
        
        # Strategy parameters
        self.leader_aggression = 0.6   # Aggression when leading
        self.behind_aggression = 1.2   # Aggression when behind
        self.desperate_threshold = 10  # Points behind to go desperate
        
        # Opponent modeling
        self.opponent_weight = 1.0  # Weight of opponent bid estimates (0-2)
        self.learning_rate = 0.1    # How fast to adapt to opponents (0-1)
        
        # Robustness
        self.max_reasonable_bid = 500  # Cap on any single bid
        self.outlier_threshold = 3.0   # Std devs to consider outlier
        
        # Dashboard settings
        self.enable_logging = True
        self.log_history_size = 100


class AuctionAnalyzer:
    """Analyzes auctions and opponent behavior with robustness"""
    
    def __init__(self, config):
        self.config = config
        self.opponent_bids = defaultdict(list)
        self.opponent_wins = defaultdict(int)
        self.opponent_losses = defaultdict(int)
        self.auction_history = deque(maxlen=config.log_history_size)
        
    def calculate_expected_value(self, auction):
        """Calculate expected points from an auction"""
        try:
            die = int(auction["die"])
            num = int(auction["num"])
            bonus = int(auction["bonus"])
            
            # Validate reasonable values
            if die > 100 or num > 50 or abs(bonus) > 1000:
                return 0  # Suspicious values
            
            expected_roll = (die + 1) / 2
            expected_points = num * expected_roll + bonus
            
            return max(0, expected_points)
        except (KeyError, ValueError, TypeError):
            return 0
    
    def calculate_variance(self, auction):
        """Calculate variance/risk of an auction"""
        try:
            die = int(auction["die"])
            num = int(auction["num"])
            
            # Variance of a die: (die^2 - 1) / 12
            var_single = (die**2 - 1) / 12
            total_variance = num * var_single
            
            return np.sqrt(total_variance)
        except (KeyError, ValueError, TypeError):
            return 0
    
    def update_opponent_stats(self, prev_auctions, agent_id):
        """Learn from previous round's bids with outlier detection"""
        if not prev_auctions:
            return
        
        try:
            for auction_id, auction_data in prev_auctions.items():
                if "bids" not in auction_data or not auction_data["bids"]:
                    continue
                
                bids = auction_data["bids"]
                
                # Track all bids with outlier filtering
                for bid_info in bids:
                    # Game uses 'a_id' not 'agent_id'
                    opponent_id = bid_info.get("a_id") or bid_info.get("agent_id")
                    if not opponent_id or opponent_id == agent_id:  # Don't track our own bids
                        continue
                    
                    # Game uses 'gold' not 'amount' for bid value
                    amount = float(bid_info.get("gold") or bid_info.get("amount", 0))
                    
                    if amount == 0:
                        continue
                    
                    # Outlier detection: reject extremely high bids
                    if amount > self.config.max_reasonable_bid * 2:
                        continue  # Ignore outliers
                    
                    self.opponent_bids[opponent_id].append(amount)
                    
                    # Keep only recent history
                    if len(self.opponent_bids[opponent_id]) > 50:
                        self.opponent_bids[opponent_id].pop(0)
                
                # Track winner/loser
                if bids:
                    winner_id = bids[0].get("a_id") or bids[0].get("agent_id")
                    if winner_id and winner_id != agent_id:
                        self.opponent_wins[winner_id] += 1
                    
                    for bid_info in bids[1:]:
                        loser_id = bid_info.get("a_id") or bid_info.get("agent_id")
                        if loser_id and loser_id != agent_id:
                            self.opponent_losses[loser_id] += 1
        
        except Exception as e:
            print(f"Error updating opponent stats: {e}")
            import traceback
            traceback.print_exc()
    
    def estimate_opponent_bid(self, auction, num_opponents):
        """Estimate what opponents will bid with robustness"""
        ev = self.calculate_expected_value(auction)
        
        if num_opponents == 0:
            return 10  # Minimum bid if alone
        
        # Collect opponent statistics with outlier filtering
        all_bids = []
        for bids in self.opponent_bids.values():
            # Filter outliers using IQR method
            if len(bids) > 5:
                q1, q3 = np.percentile(bids, [25, 75])
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                filtered = [b for b in bids if lower <= b <= upper]
                all_bids.extend(filtered)
            else:
                all_bids.extend(bids)
        
        if all_bids:
            avg_bid = np.median(all_bids)  # Use median for robustness
            std_bid = np.std(all_bids) if len(all_bids) > 1 else avg_bid * 0.3
            
            # Estimate based on EV and historical behavior
            baseline_ev = 50  # Baseline expectation
            estimated_bid = ev * (avg_bid / baseline_ev) * self.config.opponent_weight
            
            # Add uncertainty
            estimated_bid += std_bid * 0.5
            
            return max(10, min(estimated_bid, self.config.max_reasonable_bid))
        
        # Default: proportion to EV
        return min(ev * 8, self.config.max_reasonable_bid * 0.5)
    
    def get_opponent_aggression(self, agent_id):
        """Get how aggressive an opponent is (0-1 scale)"""
        if agent_id not in self.opponent_bids or not self.opponent_bids[agent_id]:
            return 0.5
        
        bids = self.opponent_bids[agent_id]
        avg_bid = np.median(bids)  # Robust to outliers
        
        # Normalize: 0-100 gold -> 0-1 aggression
        return min(1.0, max(0.0, avg_bid / 100))


class SmartBidder:
    """Intelligent bidding strategy with live monitoring"""
    
    def __init__(self, config):
        self.config = config
        self.analyzer = AuctionAnalyzer(config)
        self.rounds_played = 0
        self.total_points = 0
        self.total_gold_spent = 0
        self.total_gold_won = 0
        self.auctions_won = 0
        self.auctions_lost = 0
        self.agent_id = None  # Will be set on first bid
        
        # Live stats for dashboard
        self.current_bids = {}
        self.current_stats = {}
        self.round_history = deque(maxlen=50)
        self.last_update = datetime.now()
        self.prev_auction_results = []  # Track who won what in previous round
        self.all_auction_history = []  # Track ALL auctions across all rounds
        
    def make_bid(self, agent_id, states, auctions, prev_auctions):
        """Advanced bidding strategy"""
        try:
            # Store agent ID on first call
            if self.agent_id is None:
                self.agent_id = agent_id
                print(f"🆔 My Agent ID: {agent_id}")
            
            self.rounds_played += 1
            self.analyzer.update_opponent_stats(prev_auctions, agent_id)
            
            # Track auction results from previous round
            # Note: prev_auctions contains results from the PREVIOUS round (rounds_played - 1)
            self.prev_auction_results = []
            for auction_id, auction_data in prev_auctions.items():
                bids = auction_data.get("bids", [])
                if bids:
                    winner_id = bids[0].get("a_id")
                    winner_bid = bids[0].get("gold")
                    reward = auction_data.get("reward", 0)
                    
                    # Check if we participated
                    my_bid = next((b for b in bids if b.get("a_id") == agent_id), None)
                    
                    # Simplified result (no all_bids to reduce data size)
                    # Label with correct round number (previous round, not current)
                    result = {
                        "round": self.rounds_played - 1,  # prev_auctions are from previous round
                        "auction_id": auction_id,
                        "winner": winner_id,
                        "winning_bid": float(winner_bid),
                        "reward": int(reward),
                        "i_won": winner_id == agent_id,
                        "i_bid": float(my_bid.get("gold")) if my_bid else None,
                        "total_bidders": len(bids)
                    }
                    
                    self.prev_auction_results.append(result)
                    
                    # Store complete history (no limit)
                    self.all_auction_history.append(result)
                    
                    # Count wins/losses
                    if my_bid:
                        if winner_id == agent_id:
                            self.auctions_won += 1
                        else:
                            self.auctions_lost += 1
            
            my_gold = float(states[agent_id]["gold"])
            my_points = float(states[agent_id]["points"])
            
            # Validate inputs
            if my_gold < 0 or my_gold > 1e6:
                print(f"Warning: Suspicious gold amount: {my_gold}")
                return {}
            
            # Get game state
            num_opponents = len(states) - 1
            if num_opponents == 0:
                return self._bid_alone(auctions, my_gold)
            
            # Analyze position
            opponent_points = [float(s["points"]) for k, s in states.items() if k != agent_id]
            sorted_points = sorted(opponent_points, reverse=True)
            
            if sorted_points:
                leader_points = sorted_points[0]
                my_rank = sum(1 for p in opponent_points if p > my_points) + 1
                points_behind = max(0, leader_points - my_points)
            else:
                leader_points = my_points
                my_rank = 1
                points_behind = 0
            
            # Calculate EV and variance for all auctions
            auction_scores = []
            for auction_id, auction in auctions.items():
                ev = self.analyzer.calculate_expected_value(auction)
                if ev == 0:  # Skip invalid auctions
                    continue
                
                var = self.analyzer.calculate_variance(auction)
                est_opponent_bid = self.analyzer.estimate_opponent_bid(auction, num_opponents)
                
                # Risk-adjusted score
                if my_rank > 1 and points_behind > self.config.desperate_threshold:
                    risk_bonus = var * self.config.risk_tolerance
                else:
                    risk_bonus = -var * (1 - self.config.risk_tolerance)
                
                score = ev + risk_bonus
                
                auction_scores.append({
                    "id": auction_id,
                    "ev": ev,
                    "var": var,
                    "score": score,
                    "est_opponent_bid": est_opponent_bid,
                    "auction": auction
                })
            
            if not auction_scores:
                return {}
            
            # Sort by score
            auction_scores.sort(key=lambda x: x["score"], reverse=True)
            
            # Portfolio strategy
            bids = {}
            remaining_gold = my_gold
            
            # Determine strategy
            if my_rank == 1:
                max_auctions = min(self.config.portfolio_size, len(auction_scores))
                aggression = self.config.leader_aggression
            elif points_behind > self.config.desperate_threshold:
                max_auctions = min(2, len(auction_scores))
                aggression = self.config.behind_aggression
            else:
                max_auctions = min(self.config.portfolio_size, len(auction_scores))
                aggression = self.config.base_aggression
            
            # Allocate gold to top auctions
            for i, auction in enumerate(auction_scores[:max_auctions]):
                if remaining_gold < self.config.min_reserve:
                    break
                
                # Bid above estimated opponent bid
                base_bid = auction["est_opponent_bid"] * aggression
                
                # Portfolio weight
                weight = 1.0 - (i * 0.15)
                bid_amount = base_bid * weight
                
                # Don't overbid on low-EV auctions
                max_reasonable_bid = auction["ev"] * 15
                bid_amount = min(bid_amount, max_reasonable_bid)
                
                # Budget constraint
                bid_amount = min(bid_amount, remaining_gold * self.config.max_bid_fraction)
                
                # Hard cap for robustness
                bid_amount = min(bid_amount, self.config.max_reasonable_bid)
                
                # Minimum bid
                bid_amount = max(10, bid_amount)
                
                if bid_amount <= remaining_gold:
                    bids[auction["id"]] = round(bid_amount, 2)
                    remaining_gold -= bid_amount
            
            # Update stats for dashboard
            self._update_stats(agent_id, states, auctions, bids, my_rank, points_behind)
            
            return bids
        
        except Exception as e:
            print(f"Error in make_bid: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _bid_alone(self, auctions, my_gold):
        """Bid when alone"""
        if not auctions:
            return {}
        
        best_auction = None
        best_ev = -1
        
        for auction_id, auction in auctions.items():
            ev = self.analyzer.calculate_expected_value(auction)
            if ev > best_ev:
                best_ev = ev
                best_auction = auction_id
        
        if best_auction:
            return {best_auction: 10.0}
        return {}
    
    def _update_stats(self, agent_id, states, auctions, bids, rank, points_behind):
        """Update statistics for dashboard"""
        self.current_bids = bids
        self.current_stats = {
            "round": self.rounds_played,
            "gold": states[agent_id]["gold"],
            "points": states[agent_id]["points"],
            "rank": rank,
            "points_behind": points_behind,
            "num_bids": len(bids),
            "total_bid_amount": sum(bids.values()),
            "avg_bid": np.mean(list(bids.values())) if bids else 0,
            "auctions_available": len(auctions),
            "timestamp": datetime.now().isoformat()
        }
        
        self.round_history.append(self.current_stats.copy())
        self.last_update = datetime.now()


# Global instances
config = AgentConfig()
bidder = SmartBidder(config)

# Flask Dashboard
app = Flask(__name__)
CORS(app)


@app.route('/')
def dashboard():
    """Serve dashboard HTML"""
    return render_template('dashboard.html')


@app.route('/test')
def test_simple():
    """Serve simple test page"""
    return render_template('test_simple.html')


@app.route('/api/stats')
def get_stats():
    """Get current statistics"""
    try:
        # Simplify opponent stats - just basics
        opponent_stats_simple = {}
        for k, v in bidder.analyzer.opponent_bids.items():
            if v:
                opponent_stats_simple[k] = {
                    "avg_bid": float(np.mean(v)),
                    "num_bids": len(v)
                }
        
        return jsonify({
            "agent_id": bidder.agent_id or "Waiting...",
            "current": bidder.current_stats,
            "prev_auction_results": bidder.prev_auction_results,
            "all_auction_history": bidder.all_auction_history,  # Send ALL history
            "config": {
                "base_aggression": float(config.base_aggression),
                "risk_tolerance": float(config.risk_tolerance),
                "max_bid_fraction": float(config.max_bid_fraction),
                "min_reserve": int(config.min_reserve),
                "portfolio_size": int(config.portfolio_size),
                "leader_aggression": float(config.leader_aggression),
                "behind_aggression": float(config.behind_aggression),
                "desperate_threshold": int(config.desperate_threshold),
                "opponent_weight": float(config.opponent_weight),
                "max_reasonable_bid": int(config.max_reasonable_bid)
            },
            "history": list(bidder.round_history)[-20:],  # Only last 20 rounds
            "current_bids": bidder.current_bids,
            "opponent_stats": opponent_stats_simple,
            "totals": {
                "rounds_played": int(bidder.rounds_played),
                "auctions_won": int(bidder.auctions_won),
                "auctions_lost": int(bidder.auctions_lost)
            }
        })
    except Exception as e:
        print(f"❌ Error in /api/stats: {e}")
        import traceback
        traceback.print_exc()
        # Return minimal valid response
        return jsonify({
            "agent_id": "Error",
            "current": {},
            "prev_auction_results": [],
            "all_auction_history": [],
            "config": {},
            "history": [],
            "current_bids": {},
            "opponent_stats": {},
            "totals": {"rounds_played": 0, "auctions_won": 0, "auctions_lost": 0},
            "error": str(e)
        })


@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration parameters"""
    data = request.json
    
    try:
        if 'base_aggression' in data:
            config.base_aggression = float(data['base_aggression'])
        if 'risk_tolerance' in data:
            config.risk_tolerance = float(data['risk_tolerance'])
        if 'max_bid_fraction' in data:
            config.max_bid_fraction = float(data['max_bid_fraction'])
        if 'min_reserve' in data:
            config.min_reserve = float(data['min_reserve'])
        if 'portfolio_size' in data:
            config.portfolio_size = int(data['portfolio_size'])
        if 'leader_aggression' in data:
            config.leader_aggression = float(data['leader_aggression'])
        if 'behind_aggression' in data:
            config.behind_aggression = float(data['behind_aggression'])
        if 'desperate_threshold' in data:
            config.desperate_threshold = float(data['desperate_threshold'])
        if 'opponent_weight' in data:
            config.opponent_weight = float(data['opponent_weight'])
        if 'max_reasonable_bid' in data:
            config.max_reasonable_bid = float(data['max_reasonable_bid'])
        
        return jsonify({"status": "success", "message": "Configuration updated"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


def make_bid_callback(agent_id, states, auctions, prev_auctions, pool_gold, prev_pool_buys):
    """Main entry point called by the game client"""
    try:
        # Make bids using our smart bidder
        bids = bidder.make_bid(agent_id, states, auctions, prev_auctions)
        
        # Print round info
        if bidder.current_stats:
            print(f"Round {bidder.rounds_played}: Points={bidder.current_stats.get('points', 0):.1f}, Gold={bidder.current_stats.get('gold', 0):.2f}, Bids={len(bids)}")
        
        # Return in the format expected by the game
        # pool points is set to 1 (minimal - we focus on auctions)
        return {"bids": bids, "pool": 1}
    except Exception as e:
        print(f"Error in make_bid_callback: {e}")
        import traceback
        traceback.print_exc()
        return {"bids": {}, "pool": 1}


def run_agent():
    """Run the agent using the proper game client"""
    print(f"🤖 Starting {AGENT_NAME}")
    print(f"📊 Dashboard will be at http://localhost:{DASHBOARD_PORT}")
    print(f"🎮 Connecting to game server at {GAME_HOST}:{GAME_PORT}")
    print()
    
    try:
        # Create the game client
        game = AuctionGameClient(
            host=GAME_HOST,
            agent_name=AGENT_NAME,
            player_id=PLAYER_ID,
            port=GAME_PORT
        )
        
        # Run the game with our bid callback
        print("✓ Connected to game server")
        print("Waiting for game to start...")
        game.run(make_bid_callback)
        
        print("\n✓ Game completed!")
        if bidder.current_stats:
            print(f"Final Points: {bidder.current_stats.get('points', 0)}")
            print(f"Final Rank: {bidder.current_stats.get('rank', '-')}")
        
    except KeyboardInterrupt:
        print("\n✋ Agent stopped by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


def run_dashboard():
    """Run Flask dashboard"""
    app.run(host='0.0.0.0', port=DASHBOARD_PORT, debug=False, use_reloader=False)


if __name__ == "__main__":
    # Start dashboard in separate thread
    dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
    dashboard_thread.start()
    
    time.sleep(1)  # Let dashboard start
    
    # Run agent
    run_agent()
