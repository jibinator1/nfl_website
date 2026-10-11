"""
Live Pick Forward-Testing Logger.
Logs every breakout pick generated against real market closing lines to data/live_pick_log.csv
for rigorous out-of-sample forward-testing.
"""
import csv
import datetime
import os
from typing import Dict, Any, Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(ROOT_DIR, "data", "live_pick_log.csv")

FIELDNAMES = [
    "logged_at",
    "matchup",
    "game_id",
    "kickoff",
    "player",
    "team",
    "pos",
    "stat",
    "side",
    "market_line",
    "fair_line",
    "win_prob_raw",
    "win_prob_cal",
    "tier",
    "line_source",
    "ev_pct",
    "calibration_status"
]

def log_pick(breakout: Dict[str, Any], game_id: str, kickoff: str, matchup: str) -> bool:
    """
    Logs a single model breakout recommendation.
    Deduplicates by (game_id, player, stat, side) so repeated calls on the same game
    do not pollute the forward-test dataset.
    """
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        player = breakout.get("player", "")
        stat = breakout.get("stat_focus", "")
        side = breakout.get("side", "")

        # Check existing entries to avoid duplicates for the same game
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if (row.get("game_id") == str(game_id) and
                        row.get("player") == player and
                        row.get("stat") == stat and
                        row.get("side") == side):
                        return False  # Already logged

        row = {
            "logged_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "matchup": matchup,
            "game_id": str(game_id),
            "kickoff": kickoff,
            "player": player,
            "team": breakout.get("team", ""),
            "pos": breakout.get("pos", ""),
            "stat": stat,
            "side": side,
            "market_line": breakout.get("market_line", ""),
            "fair_line": breakout.get("fair_line", ""),
            "win_prob_raw": breakout.get("win_prob_raw"),
            "win_prob_cal": breakout.get("win_prob"),
            "tier": breakout.get("confidence_tier", ""),
            "line_source": breakout.get("line_source", ""),
            "ev_pct": breakout.get("ev_pct", ""),
            "calibration_status": breakout.get("calibration_status", "Calibrated" if breakout.get("is_calibrated") else "Uncalibrated (Raw)")
        }

        new_file = not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if new_file:
                w.writeheader()
            w.writerow(row)
        return True
    except Exception as e:
        print(f"[pick_logger] Warning logging pick: {e}")
        return False
