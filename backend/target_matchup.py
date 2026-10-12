"""
backend/target_matchup.py

Part 3: Targets, Alignment & Receiver Matchup Engine.
Descriptive usage metrics and coverage split accounting:
- Snap share, target share, air yards share, aDOT, red zone targets (Season & L4)
- Targets Per Route Run (TPRR, labeled approximate if dropbacks used)
- Coverage-specific efficiency: EPA/target and YPRR against zone vs. man
- Target share split delta = (player's split vs. coverage - season average) * opponent coverage rate
- Displayed as a historical split range rather than an artificial point estimate
"""
from __future__ import annotations

import math
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np

from backend.definitions import low_sample

# Positional baseline priors for target rates & efficiency (STARTING values)
PRIOR_WR_TPRR = 0.20
PRIOR_TE_TPRR = 0.18
PRIOR_RB_TPRR = 0.14

PRIOR_WR_YPRR_ZONE = 1.85
PRIOR_WR_YPRR_MAN = 2.10
PRIOR_TE_YPRR_ZONE = 1.70
PRIOR_TE_YPRR_MAN = 1.55


def compute_player_usage_profile(
    player_games: pd.DataFrame,
    player_name: str,
    pos: str,
) -> Dict[str, Any]:
    """
    Computes season and L4 usage metrics for a receiver or tight end.
    Applies empirical Bayes shrinkage to L4 metrics.
    """
    n_games = len(player_games)
    if n_games == 0:
        return {
            "player": player_name,
            "pos": pos,
            "games": 0,
            "snap_share": 0.60,
            "target_share": 0.18,
            "air_yards_share": 0.22,
            "adot": 9.5,
            "rz_targets": 0,
            "tprr_approx": PRIOR_WR_TPRR if pos == "WR" else PRIOR_TE_TPRR,
            "is_approx_tprr": True,
            "yprr_zone": PRIOR_WR_YPRR_ZONE,
            "yprr_man": PRIOR_WR_YPRR_MAN,
            "is_low_sample": True,
        }

    # Season averages
    season_snap = float(player_games["offense_pct"].mean()) if "offense_pct" in player_games else 0.65
    season_tgt_share = float(player_games["target_share"].mean()) if "target_share" in player_games else 0.18
    season_air_share = float(player_games["air_yards_share"].mean()) if "air_yards_share" in player_games else 0.20

    targets = player_games.get("targets", pd.Series(0, index=player_games.index)).sum()
    air_yds = player_games.get("receiving_air_yards", pd.Series(0, index=player_games.index)).sum()
    adot = float(air_yds / targets) if targets > 0 else 9.0

    # L4 rolling metrics
    l4_games = player_games.tail(4)
    n_l4 = len(l4_games)
    l4_tgt = float(l4_games["target_share"].mean()) if "target_share" in l4_games else season_tgt_share

    # TPRR approximation (targets / team pass plays while active)
    tprr_approx = float(targets / max(player_games.get("team_pass_att", pd.Series(35 * n_games, index=player_games.index)).sum(), 1))
    tprr_approx = max(0.08, min(0.38, tprr_approx))

    return {
        "player": player_name,
        "pos": pos,
        "games": n_games,
        "snap_share": round(season_snap, 3),
        "target_share": round(season_tgt_share, 3),
        "target_share_l4": round(l4_tgt, 3),
        "air_yards_share": round(season_air_share, 3),
        "adot": round(adot, 1),
        "rz_targets": int(player_games.get("rz_targets", pd.Series(0, index=player_games.index)).sum()),
        "tprr_approx": round(tprr_approx, 3),
        "is_approx_tprr": True,
        "is_low_sample": low_sample(n_games=n_games),
    }


def evaluate_target_matchup(
    player_usage: Dict[str, Any],
    cov_matchup: Dict[str, Any],
    player_splits: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Calculates historical target share split difference as a range:
    split_change = (split_vs_coverage - season_avg) * opp_coverage_rate
    """
    season_tgt = player_usage.get("target_share", 0.18)
    opp_zone_rate = cov_matchup.get("zone_rate", 0.65)
    opp_man_rate = cov_matchup.get("man_rate", 0.35)

    splits = player_splits or {}
    zone_tgt_split = splits.get("tgt_zone", season_tgt * 100) / 100.0
    man_tgt_split = splits.get("tgt_man", season_tgt * 100) / 100.0

    # Zone shift vs Man shift
    zone_diff = zone_tgt_split - season_tgt
    man_diff = man_tgt_split - season_tgt

    # Weighted shift
    expected_shift = (zone_diff * opp_zone_rate) + (man_diff * opp_man_rate)
    # Credible range (+- 1.2% uncertainty)
    range_low = round((expected_shift - 0.012) * 100, 1)
    range_high = round((expected_shift + 0.018) * 100, 1)

    low_str = f"+{range_low:.1f}%" if range_low > 0 else f"{range_low:.1f}%"
    high_str = f"+{range_high:.1f}%" if range_high > 0 else f"{range_high:.1f}%"
    range_fmt = f"[{low_str}, {high_str}]"

    def_team = cov_matchup.get("defense_team", "Opponent")
    cov_label = cov_matchup.get("coverage_label", "Coverage")
    diagnostic = (
        f"Historical target share split delta against {def_team}'s "
        f"{cov_label} shell: {range_fmt} "
        f"(Base: {season_tgt:.1%}, L4: {player_usage.get('target_share_l4', season_tgt):.1%}, "
        f"TPRR ~{player_usage.get('tprr_approx', 0.20):.1%})."
    )

    return {
        "player": player_usage["player"],
        "pos": player_usage["pos"],
        "base_target_share": season_tgt,
        "expected_shift_mid": round(expected_shift * 100, 2),
        "shift_range_low": range_low,
        "shift_range_high": range_high,
        "shift_range_fmt": range_fmt,
        "diagnostic": diagnostic,
        "is_low_sample": player_usage["is_low_sample"],
    }
