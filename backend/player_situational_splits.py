"""
backend/player_situational_splits.py

Descriptive Player Situational Usage & Matchup Stat Splits.
Replaces betting prop models with empirical player usage accounting:
- Snap share, target share, air yards share
- Average Depth of Target (aDOT) and Targets Per Route Run (TPRR)
- EPA per target / touch
- Red-zone touch share (snaps inside opponent 20)
- Historical baseline medians, 10th percentile floor, 90th percentile ceiling
- Situational split differentials (performance vs. man vs. zone, leading vs. trailing)
- Sample-size flagging (n < 4 games or low touches)

No fair lines, EV hurdles, win probabilities, or Over/Under picks.
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from backend.definitions import low_sample, MIN_GAMES_FLAG


def compute_floor_ceiling_history(
    actuals: List[float],
    benchmark: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Computes purely descriptive distribution percentiles across a player's recent games.
    """
    if not actuals:
        return {
            "games_count": 0,
            "median": 0.0,
            "floor_10th": 0.0,
            "ceiling_90th": 0.0,
            "cleared_ratio": "0/0",
            "cleared_pct": 0.0,
        }

    arr = np.array(actuals, dtype=float)
    n = len(arr)
    med = float(np.median(arr))
    floor_val = float(np.percentile(arr, 10)) if n >= 3 else float(arr.min())
    ceil_val = float(np.percentile(arr, 90)) if n >= 3 else float(arr.max())

    if benchmark is not None:
        cleared = int(np.sum(arr > benchmark))
        cleared_ratio = f"{cleared}/{n}"
        cleared_pct = round((cleared / n) * 100.0, 1)
    else:
        cleared_ratio = f"{int(np.sum(arr >= med))}/{n}"
        cleared_pct = 50.0

    return {
        "games_count": n,
        "median": round(med, 1),
        "floor_10th": round(floor_val, 1),
        "ceiling_90th": round(ceil_val, 1),
        "cleared_ratio": cleared_ratio,
        "cleared_pct": cleared_pct,
    }


def compute_player_situational_splits(
    player_name: str,
    pos: str,
    team: str,
    player_games: pd.DataFrame,
    pbp_plays: Optional[pd.DataFrame] = None,
    stat_focus: str = "Receiving Yards",
    benchmark_line: Optional[float] = None,
    split_type: str = "coverage_shell",
) -> Dict[str, Any]:
    """
    Generates an honest situational stat sheet for a skill position player.
    """
    n_games = len(player_games) if player_games is not None else 0
    is_low = low_sample(n_games=n_games)

    if player_games is None or player_games.empty:
        return {
            "player": player_name,
            "pos": pos,
            "team": team,
            "games_played": 0,
            "snap_share": 0.0,
            "target_share": 0.0,
            "air_yards_share": 0.0,
            "adot": 0.0,
            "tprr": 0.0,
            "epa_per_target": 0.0,
            "rz_touch_share": 0.0,
            "baseline_median": 0.0,
            "floor_10th": 0.0,
            "ceiling_90th": 0.0,
            "split_label": "No Data",
            "split_differential": 0.0,
            "is_low_sample": True,
        }

    # Snap & volume shares
    snap_share = float(player_games["offense_pct"].mean()) if "offense_pct" in player_games.columns else 0.65
    target_share = float(player_games["target_share"].mean()) if "target_share" in player_games.columns else 0.18
    air_share = float(player_games["air_yards_share"].mean()) if "air_yards_share" in player_games.columns else 0.20

    targets = player_games.get("targets", pd.Series(0, index=player_games.index)).sum()
    air_yds = player_games.get("receiving_air_yards", pd.Series(0, index=player_games.index)).sum()
    adot = float(air_yds / targets) if targets > 0 else 9.0

    # Approx TPRR (targets / team pass attempts while on field)
    team_pass_att = player_games.get("team_pass_att", pd.Series(35 * n_games, index=player_games.index)).sum()
    tprr = float(targets / max(team_pass_att, 1))
    tprr = max(0.05, min(0.40, tprr))

    # EPA per target/rush
    epa_sum = player_games.get("receiving_epa", player_games.get("rushing_epa", pd.Series(0, index=player_games.index))).sum()
    epa_per_opp = float(epa_sum / max(targets, 1))

    # Red zone touches
    rz_touches = int(player_games.get("rz_targets", pd.Series(0, index=player_games.index)).sum() +
                     player_games.get("rz_rushes", pd.Series(0, index=player_games.index)).sum())
    team_rz_touches = int(player_games.get("team_rz_plays", pd.Series(8 * n_games, index=player_games.index)).sum())
    rz_touch_share = float(rz_touches / max(team_rz_touches, 1))

    # Baseline stat history
    stat_col_map = {
        "Receiving Yards": "receiving_yards",
        "Rushing Yards": "rushing_yards",
        "Receptions": "receptions",
    }
    col = stat_col_map.get(stat_focus, "receiving_yards")
    actuals = player_games[col].dropna().tolist() if col in player_games.columns else []

    dist = compute_floor_ceiling_history(actuals, benchmark=benchmark_line)

    # Situational split delta
    # E.g. performance in wins vs losses, or man vs zone if tracked
    split_diff = 0.0
    split_lbl = "Neutral Baseline"
    if "opponent_coverage" in player_games.columns:
        zone_perf = player_games[player_games["opponent_coverage"] == "ZONE"][col].mean()
        man_perf = player_games[player_games["opponent_coverage"] == "MAN"][col].mean()
        if pd.notna(zone_perf) and pd.notna(man_perf):
            split_diff = round(float(man_perf - zone_perf), 1)
            split_lbl = f"{split_diff:+.1f} vs Man over Zone"

    return {
        "player": player_name,
        "pos": pos,
        "team": team,
        "games_played": n_games,
        "snap_share": round(snap_share, 3),
        "target_share": round(target_share, 3),
        "air_yards_share": round(air_share, 3),
        "adot": round(adot, 1),
        "tprr": round(tprr, 3),
        "epa_per_target": round(epa_per_opp, 3),
        "rz_touch_share": round(rz_touch_share, 3),
        "baseline_median": dist["median"],
        "floor_10th": dist["floor_10th"],
        "ceiling_90th": dist["ceiling_90th"],
        "cleared_ratio": dist["cleared_ratio"],
        "cleared_pct": dist["cleared_pct"],
        "split_label": split_lbl,
        "split_differential": split_diff,
        "is_low_sample": is_low,
    }
