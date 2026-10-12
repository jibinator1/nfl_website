"""
backend/tempo_matchup.py

Part 5: Tempo, Game-Flow & Situational Environment Engine.
Replaces asymmetric tempo rules and flawed checkdown metrics:
- Situational pace (seconds/play) and pass rates split by game state:
    Leading (>= +7), Neutral (-6 to +6), Trailing (<= -7)
- No-huddle frequency, two-minute drill pace, and fourth-down aggressiveness
- Implied game-script projection from spread and total (favorite late rush vs. dog pass funnel)
- Symmetric pace verdict (both teams >= 1.0 SD fast, or both >= 1.0 SD slow)
- High-value red zone touch distribution (carries & targets inside the 10 and 5)
"""
from __future__ import annotations

import math
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np

from backend.definitions import game_state, low_sample

# League tempo priors & standard deviations (STARTING values)
PRIOR_NEUTRAL_PACE_SEC = 28.2
SIGMA_NEUTRAL_PACE = 2.1  # ~2.1 seconds per play std dev across NFL offenses

PRIOR_PASS_RATE_LEADING = 0.44
PRIOR_PASS_RATE_NEUTRAL = 0.58
PRIOR_PASS_RATE_TRAILING = 0.68
PRIOR_NO_HUDDLE_RATE = 0.08
PRIOR_4TH_DOWN_GO_RATE = 0.52


def compute_team_tempo_profile(
    arg1: Any,
    arg2: Any = None,
) -> Dict[str, Any]:
    """
    Computes situational tempo, pace by game state, and high-value touch concentration.
    Supports compute_team_tempo_profile("DET") or compute_team_tempo_profile(pbp, "DET").
    """
    if isinstance(arg1, str):
        team = arg1
        team_pbp = arg2
    else:
        team_pbp = arg1
        team = arg2

    if team_pbp is None or not isinstance(team_pbp, pd.DataFrame) or team_pbp.empty or "posteam" not in team_pbp.columns:
        plays = pd.DataFrame()
    else:
        plays = team_pbp[team_pbp["posteam"] == team].copy()
    if plays.empty:
        return {
            "team": team,
            "neutral_pace_sec": PRIOR_NEUTRAL_PACE_SEC,
            "pass_rate_leading": PRIOR_PASS_RATE_LEADING,
            "pass_rate_neutral": PRIOR_PASS_RATE_NEUTRAL,
            "pass_rate_trailing": PRIOR_PASS_RATE_TRAILING,
            "no_huddle_rate": PRIOR_NO_HUDDLE_RATE,
            "go_for_it_4th_rate": PRIOR_4TH_DOWN_GO_RATE,
            "rz_10_touch_leader": f"{team} Primary Back",
            "rz_5_touch_leader": f"{team} Primary Back",
            "proj_plays": 63.5,
            "is_low_sample": True,
        }

    # Game state splits
    if "score_differential" in plays.columns:
        plays["_state"] = game_state(plays["score_differential"])
    else:
        plays["_state"] = "neutral"

    pass_runs = plays[plays["play_type"].isin(["pass", "run"])]
    n_plays = len(pass_runs)

    # Pass rates by state
    def get_pass_rate(sub_df: pd.DataFrame, prior: float) -> float:
        if sub_df.empty:
            return prior
        return float((sub_df["play_type"] == "pass").mean())

    pass_leading = get_pass_rate(pass_runs[pass_runs["_state"] == "leading"], PRIOR_PASS_RATE_LEADING)
    pass_neutral = get_pass_rate(pass_runs[pass_runs["_state"] == "neutral"], PRIOR_PASS_RATE_NEUTRAL)
    pass_trailing = get_pass_rate(pass_runs[pass_runs["_state"] == "trailing"], PRIOR_PASS_RATE_TRAILING)

    # Neutral pace (seconds per play in neutral script)
    neutral_plays = pass_runs[pass_runs["_state"] == "neutral"]
    if len(neutral_plays) >= 10 and "half_seconds_remaining" in neutral_plays.columns:
        hs = neutral_plays["half_seconds_remaining"]
        elapsed = hs.max() - hs.min()
        sec_per_play = float(elapsed / max(len(neutral_plays), 1))
        sec_per_play = max(23.5, min(33.0, sec_per_play))
    else:
        sec_per_play = PRIOR_NEUTRAL_PACE_SEC

    neutral_pace = sec_per_play

    # No huddle rate
    no_huddle = float((plays.get("is_no_huddle", pd.Series(0, index=plays.index)).fillna(0) == 1).mean())
    no_huddle = max(0.02, min(0.30, no_huddle))

    # Fourth down aggressiveness
    d4 = plays[plays["down"] == 4]
    if len(d4) > 3:
        go_4th = float((d4["play_type"].isin(["pass", "run"])).mean())
    else:
        go_4th = PRIOR_4TH_DOWN_GO_RATE

    # High-value touches inside 10 and 5
    inside_10 = pass_runs[pass_runs.get("yardline_100", pd.Series(99, index=pass_runs.index)) <= 10]
    inside_5 = pass_runs[pass_runs.get("yardline_100", pd.Series(99, index=pass_runs.index)) <= 5]

    top_10 = inside_10.get("rusher_player_name", inside_10.get("fantasy_player_name", pd.Series(dtype=object))).dropna()
    top_5 = inside_5.get("rusher_player_name", inside_5.get("fantasy_player_name", pd.Series(dtype=object))).dropna()

    lead_10 = top_10.value_counts().index[0] if len(top_10) > 0 else f"{team} Lead Rusher"
    lead_5 = top_5.value_counts().index[0] if len(top_5) > 0 else lead_10

    return {
        "team": team,
        "neutral_pace_sec": round(neutral_pace, 1),
        "pass_rate_leading": round(pass_leading, 3),
        "pass_rate_neutral": round(pass_neutral, 3),
        "pass_rate_trailing": round(pass_trailing, 3),
        "no_huddle_rate": round(no_huddle, 3),
        "go_for_it_4th_rate": round(go_4th, 3),
        "rz_10_touch_leader": lead_10,
        "rz_5_touch_leader": lead_5,
        "proj_plays": round(1530.0 / neutral_pace, 1),  # ~1530 seconds of game clock in pass/run sequences
        "is_low_sample": low_sample(n_plays=n_plays),
    }


def evaluate_tempo_matchup(
    away_tempo: Dict[str, Any],
    home_tempo: Dict[str, Any],
    spread_line: float = 0.0,
    total_line: float = 45.0,
) -> Dict[str, Any]:
    """
    Evaluates symmetric pace environment and game-script projections.
    """
    a_pace = away_tempo["neutral_pace_sec"]
    h_pace = home_tempo["neutral_pace_sec"]

    z_a = (PRIOR_NEUTRAL_PACE_SEC - a_pace) / SIGMA_NEUTRAL_PACE
    z_h = (PRIOR_NEUTRAL_PACE_SEC - h_pace) / SIGMA_NEUTRAL_PACE

    # Symmetric pace verdict
    if z_a >= 1.0 and z_h >= 1.0:
        verdict = "UP-TEMPO ENVIRONMENT (Both teams >= 1.0 SD fast)"
        pace_tag = "Track Meet Potential"
    elif z_a <= -1.0 and z_h <= -1.0:
        verdict = "SLOW DELIBERATE ENVIRONMENT (Both teams >= 1.0 SD slow)"
        pace_tag = "Clock-Control Slog"
    else:
        verdict = "BALANCED NFL TEMPO ENVIRONMENT"
        pace_tag = "Standard Pace"

    combined_proj_plays = round(away_tempo["proj_plays"] + home_tempo["proj_plays"], 1)

    # Spread script projection:
    # Negative spread means home team favored (e.g. -3.5 means Home is favorite)
    if spread_line <= -4.5:
        script_note = (
            f"{home_tempo['team']} favored by {abs(spread_line):.1f}: game script projects "
            f"elevated late rushing volume for {home_tempo['team']} and negative-script pass funnel for {away_tempo['team']}."
        )
    elif spread_line >= 4.5:
        script_note = (
            f"{away_tempo['team']} favored by {spread_line:.1f}: projects elevated ground script "
            f"for {away_tempo['team']} and trailing pass volume acceleration for {home_tempo['team']}."
        )
    else:
        script_note = f"Competitive one-score spread ({spread_line:+.1f}): projects sustained neutral pace throughout."

    return {
        "away_team": away_tempo["team"],
        "home_team": home_tempo["team"],
        "combined_proj_plays": combined_proj_plays,
        "pace_verdict": verdict,
        "pace_tag": pace_tag,
        "script_note": script_note,
        "away_tempo": away_tempo,
        "home_tempo": home_tempo,
        "total_line": total_line,
        "spread_line": spread_line,
    }
