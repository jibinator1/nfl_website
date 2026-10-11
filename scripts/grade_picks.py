"""
Weekly Live Pick Grader & Forward-Testing Evaluation.
Reads data/live_pick_log.csv, joins against completed games in data/weekly_cache.parquet,
grades every pick against real closing lines, and outputs an honest empirical performance report.
"""
import csv
import math
import os
import re
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(ROOT_DIR, "data", "live_pick_log.csv")
WEEKLY_CACHE_PATH = os.path.join(ROOT_DIR, "data", "weekly_cache.parquet")
SCHEDULES_CACHE_PATH = os.path.join(ROOT_DIR, "data", "schedules_cache.parquet")


def wilson_ci(wins: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Calculates Wilson score 95% confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def normalize_name(name: str) -> str:
    """Normalizes player names (e.g. 'T.Kelce' or 'Travis Kelce' -> 'travis kelce')."""
    if not name or not isinstance(name, str):
        return ""
    clean = re.sub(r"[^a-zA-Z\s]", "", name).strip().lower()
    return clean


def extract_line_number(line_str: Any) -> Optional[float]:
    """Extracts numeric line from strings like '64.5 Rec Yds' or raw floats."""
    if isinstance(line_str, (int, float)):
        return float(line_str)
    if not isinstance(line_str, str):
        return None
    match = re.search(r"[-+]?\d*\.\d+|\d+", line_str)
    return float(match.group()) if match else None


def load_actuals() -> pd.DataFrame:
    """Loads player game boxscores from weekly_cache.parquet."""
    if not os.path.exists(WEEKLY_CACHE_PATH):
        print(f"[grade_picks] Missing weekly cache at {WEEKLY_CACHE_PATH}")
        return pd.DataFrame()

    df = pd.read_parquet(WEEKLY_CACHE_PATH)
    # Build clean name keys for matching
    df["norm_display"] = df["player_display_name"].astype(str).apply(normalize_name)
    df["norm_name"] = df["player_name"].astype(str).apply(normalize_name)
    return df


def match_player_actuals(weekly_df: pd.DataFrame, player: str, game_id: str) -> Optional[pd.Series]:
    """Finds player stat row for the game in weekly cache."""
    norm_p = normalize_name(player)
    if not norm_p or weekly_df.empty:
        return None

    # Try matching by game_id first if available
    sub = weekly_df
    if game_id and "game_id" in weekly_df.columns:
        sub_game = weekly_df[weekly_df["game_id"].astype(str) == str(game_id)]
        if not sub_game.empty:
            sub = sub_game

    # Exact or token match on name
    tokens = norm_p.split()
    last_name = tokens[-1] if tokens else norm_p

    # Match display name or player_name
    matched = sub[(sub["norm_display"] == norm_p) | (sub["norm_name"] == norm_p)]
    if matched.empty and len(tokens) > 1:
        # Initial + last name matching (e.g. 't kelce')
        init_last = f"{tokens[0][0]} {last_name}"
        matched = sub[sub["norm_name"].str.endswith(last_name, na=False) & sub["norm_name"].str.startswith(tokens[0][0], na=False)]

    if not matched.empty:
        return matched.iloc[0]
    return None


def grade_live_picks(log_path: str = LOG_PATH) -> Dict[str, Any]:
    """
    Grades all live logged picks in live_pick_log.csv.
    Produces comprehensive breakdown report with Wilson CIs and Brier score.
    """
    if not os.path.exists(log_path):
        print(f"[grade_picks] No live picks log found at {log_path}.")
        return {}

    picks = []
    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            picks.append(row)

    if not picks:
        print("[grade_picks] live_pick_log.csv is empty. No picks to grade yet.")
        return {}

    weekly_df = load_actuals()
    graded_rows = []
    pending_count = 0
    void_count = 0

    for p in picks:
        player = p.get("player", "")
        stat = p.get("stat", "")
        side = p.get("side", "Over")
        mkt_line_num = extract_line_number(p.get("market_line"))
        game_id = p.get("game_id", "")
        p_cal = float(p.get("win_prob_cal") or 0.5)

        if mkt_line_num is None:
            continue

        actual_row = match_player_actuals(weekly_df, player, game_id)
        if actual_row is None:
            pending_count += 1
            continue

        # Extract actual stat
        actual_val = None
        targets = actual_row.get("targets", 0) or 0
        receptions = actual_row.get("receptions", 0) or 0
        rec_yds = actual_row.get("receiving_yards", 0) or 0
        rush_yds = actual_row.get("rushing_yards", 0) or 0
        rush_att = actual_row.get("rushing_attempts", 0) or actual_row.get("attempts", 0) or 0

        # DNP / Inactive detection (sportsbook void rule)
        if "Receiving" in stat or "Receptions" in stat:
            if targets == 0 and receptions == 0 and rec_yds == 0:
                void_count += 1
                continue
            actual_val = rec_yds if "Yards" in stat else receptions
        elif "Rushing" in stat:
            if rush_att == 0 and rush_yds == 0:
                void_count += 1
                continue
            actual_val = rush_yds

        if actual_val is None:
            pending_count += 1
            continue

        # Evaluate outcome
        if side == "Over":
            won = 1 if actual_val > mkt_line_num else (0 if actual_val < mkt_line_num else None)
        else:
            won = 1 if actual_val < mkt_line_num else (0 if actual_val > mkt_line_num else None)

        if won is None:
            # Push
            outcome_str = "PUSH"
        else:
            outcome_str = "WIN" if won == 1 else "LOSS"

        graded_rows.append({
            **p,
            "actual_val": actual_val,
            "won": won,
            "outcome": outcome_str,
            "win_prob_cal": p_cal
        })

    total_graded = [r for r in graded_rows if r["won"] is not None]
    n = len(total_graded)

    print("\n" + "=" * 70)
    print("      NFL PROP LIVE FORWARD-TEST GRADING REPORT")
    print("=" * 70)
    print(f"Total Picks Logged:  {len(picks)}")
    print(f"Pending (Not Final): {pending_count}")
    print(f"Voids (DNP Scratches): {void_count}")
    print(f"Graded Actionable:   {n}")

    if n == 0:
        print("\nNo completed picks found in current weekly boxscores.")
        print("Picks will automatically be graded as official boxscores land in weekly_cache.parquet.")
        print("=" * 70)
        return {"total_picks": len(picks), "pending": pending_count, "voids": void_count, "graded": 0}

    wins = sum(1 for r in total_graded if r["won"] == 1)
    losses = n - wins
    hit_rate = wins / n
    lo, hi = wilson_ci(wins, n)
    roi = ((wins * 100 - losses * 110) / (n * 110)) * 100
    brier = np.mean([(r["win_prob_cal"] - r["won"])**2 for r in total_graded])

    print("\n--- OVERALL PERFORMANCE ---")
    print(f"Record:        {wins}W - {losses}L ({hit_rate:.1%})")
    print(f"95% Wilson CI: [{lo:.1%}, {hi:.1%}]")
    print(f"Flat Unit ROI: {roi:+.2f}% (Standard -110 Juice)")
    print(f"Brier Score:   {brier:.4f}")

    # Stat breakdown
    print("\n--- BY PROP FOCUS ---")
    df_graded = pd.DataFrame(total_graded)
    for stat_name, g in df_graded.groupby("stat"):
        w = g["won"].sum()
        sub_n = len(g)
        hr = w / sub_n if sub_n else 0
        sub_lo, sub_hi = wilson_ci(w, sub_n)
        sub_roi = ((w * 100 - (sub_n - w) * 110) / (sub_n * 110)) * 100
        print(f"{stat_name:<18}: {w}W-{sub_n-w}L ({hr:.1%}) | 95% CI [{sub_lo:.1%}, {sub_hi:.1%}] | ROI: {sub_roi:+.1f}%")

    # Side breakdown
    print("\n--- BY SIDE (OVER VS UNDER) ---")
    for side_name, g in df_graded.groupby("side"):
        w = g["won"].sum()
        sub_n = len(g)
        hr = w / sub_n if sub_n else 0
        sub_lo, sub_hi = wilson_ci(w, sub_n)
        sub_roi = ((w * 100 - (sub_n - w) * 110) / (sub_n * 110)) * 100
        print(f"{side_name:<18}: {w}W-{sub_n-w}L ({hr:.1%}) | 95% CI [{sub_lo:.1%}, {sub_hi:.1%}] | ROI: {sub_roi:+.1f}%")

    # Tier breakdown
    print("\n--- BY MODEL CONVICTION TIER ---")
    for tier_name, g in df_graded.groupby("tier"):
        w = g["won"].sum()
        sub_n = len(g)
        hr = w / sub_n if sub_n else 0
        sub_lo, sub_hi = wilson_ci(w, sub_n)
        sub_roi = ((w * 100 - (sub_n - w) * 110) / (sub_n * 110)) * 100
        print(f"{tier_name:<26}: {w}W-{sub_n-w}L ({hr:.1%}) | 95% CI [{sub_lo:.1%}, {sub_hi:.1%}] | ROI: {sub_roi:+.1f}%")

    print("=" * 70)
    return {
        "n": n, "wins": wins, "losses": losses, "hit_rate": hit_rate,
        "ci": (lo, hi), "roi": roi, "brier": float(brier)
    }


if __name__ == "__main__":
    grade_live_picks()
