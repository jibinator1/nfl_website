"""
backend/run_matchup.py

Part 2: Run Game & Fronts Matchup Engine.
Descriptive situational run game breakdown:
- Rush EPA/play, success rate, stuff rate (<= 0 yds), explosive rate (>= 10 yds)
- Zone vs. Gap splits derived from run location / gap charting
- Box count distributions (light box <= 6, heavy box >= 8) and situational success
- Designed QB rush volume separated from RB rushes (QB scrambles stripped)
- Matchup differential evaluated in standard deviations with sample-size flags
"""
from __future__ import annotations

import math
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np

from backend.definitions import (
    filter_plays,
    success_col,
    low_sample,
    opponent_adjusted,
    rush_metrics,
    LIGHT_BOX_MAX,
    HEAVY_BOX_MIN,
    EXPLOSIVE_RUN_YDS,
    STUFF_RUN_YDS,
)

# League baselines for rushing metrics (STARTING placeholders)
PRIOR_RUSH_EPA = -0.05
PRIOR_RUSH_SUCCESS = 0.42
PRIOR_STUFF_RATE = 0.18
PRIOR_EXPLOSIVE_RATE = 0.12
PRIOR_LIGHT_BOX_RATE = 0.35
PRIOR_HEAVY_BOX_RATE = 0.22

# Standard deviations across team-seasons for z-score triggers
SIGMA_RUSH_EPA = 0.075
SIGMA_RUSH_SUCCESS = 0.055


def classify_run_concept(row: pd.Series) -> str:
    """Classifies a run as Zone or Gap from location / gap charting."""
    gap = str(row.get("run_gap", "")).lower()
    loc = str(row.get("run_location", "")).lower()
    if gap in ("guard", "tackle") or loc == "middle":
        return "gap"
    elif gap == "end" or loc in ("left", "right"):
        return "zone"
    return "balanced"


def compute_team_rush_profile(
    pbp_df: pd.DataFrame,
    team: str,
    as_defense: bool = False,
) -> Dict[str, Any]:
    """
    Computes a comprehensive rushing profile for a team (offense or defense).
    Strips out QB scrambles from RB rushing stats.
    """
    team_col = "defteam" if as_defense else "posteam"
    runs = pbp_df[pbp_df[team_col] == team].copy()
    runs = filter_plays(runs, kind="run", drop_garbage_time=True)

    if runs.empty:
        return {
            "team": team,
            "plays": 0,
            "epa_per_play": PRIOR_RUSH_EPA,
            "success_rate": PRIOR_RUSH_SUCCESS,
            "stuff_rate": PRIOR_STUFF_RATE,
            "explosive_rate": PRIOR_EXPLOSIVE_RATE,
            "zone_rate": 0.50,
            "gap_rate": 0.50,
            "zone_success": PRIOR_RUSH_SUCCESS,
            "gap_success": PRIOR_RUSH_SUCCESS,
            "light_box_rate": PRIOR_LIGHT_BOX_RATE,
            "light_box_success": PRIOR_RUSH_SUCCESS,
            "heavy_box_rate": PRIOR_HEAVY_BOX_RATE,
            "heavy_box_success": PRIOR_RUSH_SUCCESS,
            "designed_qb_runs": 0,
            "designed_qb_epa": 0.0,
            "is_low_sample": True,
        }

    # Separate designed QB runs vs RB runs
    is_scramble = runs.get("qb_scramble", pd.Series(0, index=runs.index)).fillna(0) == 1
    # Strip scrambles
    rb_runs = runs[~is_scramble].copy()
    n_plays = len(rb_runs)

    if n_plays == 0:
        rb_runs = runs.copy()
        n_plays = len(rb_runs)

    # Calculate core metrics
    raw_epa = float(rb_runs["epa"].mean()) if "epa" in rb_runs else PRIOR_RUSH_EPA
    succ_series = success_col(rb_runs) if "success" in rb_runs else (rb_runs["epa"] > 0).astype(float)
    raw_succ = float(succ_series.mean())

    yards = rb_runs.get("yards_gained", pd.Series(0, index=rb_runs.index)).fillna(0)
    raw_stuff = float((yards <= STUFF_RUN_YDS).mean())
    raw_expl = float((yards >= EXPLOSIVE_RUN_YDS).mean())

    # Empirical rates
    epa_per_play = raw_epa
    success_rate = raw_succ
    stuff_rate = raw_stuff
    explosive_rate = raw_expl

    # Zone vs Gap concept splits
    rb_runs["_concept"] = rb_runs.apply(classify_run_concept, axis=1)
    zone_runs = rb_runs[rb_runs["_concept"] == "zone"]
    gap_runs = rb_runs[rb_runs["_concept"] == "gap"]

    zone_rate = len(zone_runs) / n_plays if n_plays > 0 else 0.50
    gap_rate = len(gap_runs) / n_plays if n_plays > 0 else 0.50
    zone_succ = float(success_col(zone_runs).mean()) if len(zone_runs) > 5 else PRIOR_RUSH_SUCCESS
    gap_succ = float(success_col(gap_runs).mean()) if len(gap_runs) > 5 else PRIOR_RUSH_SUCCESS

    # Box counts (if defenders_in_box column present from participation / PBP)
    if "defenders_in_box" in rb_runs.columns:
        box = rb_runs["defenders_in_box"].dropna()
        light = rb_runs[rb_runs["defenders_in_box"] <= LIGHT_BOX_MAX]
        heavy = rb_runs[rb_runs["defenders_in_box"] >= HEAVY_BOX_MIN]

        light_box_rate = len(light) / len(box) if len(box) > 0 else PRIOR_LIGHT_BOX_RATE
        heavy_box_rate = len(heavy) / len(box) if len(box) > 0 else PRIOR_HEAVY_BOX_RATE
        light_box_succ = float(success_col(light).mean()) if len(light) > 3 else PRIOR_RUSH_SUCCESS
        heavy_box_succ = float(success_col(heavy).mean()) if len(heavy) > 3 else PRIOR_RUSH_SUCCESS
    else:
        light_box_rate = PRIOR_LIGHT_BOX_RATE
        heavy_box_rate = PRIOR_HEAVY_BOX_RATE
        light_box_succ = PRIOR_RUSH_SUCCESS
        heavy_box_succ = PRIOR_RUSH_SUCCESS

    # Designed QB runs
    qb_rushes = runs[~is_scramble & (runs.get("passer_id", pd.Series(index=runs.index)).isna() | (runs.get("play_type") == "run"))]
    designed_qb_cnt = int(is_scramble.sum())
    designed_qb_epa = float(runs[is_scramble]["epa"].mean()) if is_scramble.any() else 0.0

    return {
        "team": team,
        "plays": n_plays,
        "epa_per_play": round(epa_per_play, 3),
        "raw_epa": round(raw_epa, 3),
        "success_rate": round(success_rate, 3),
        "raw_success": round(raw_succ, 3),
        "stuff_rate": round(stuff_rate, 3),
        "explosive_rate": round(explosive_rate, 3),
        "zone_rate": round(zone_rate, 3),
        "gap_rate": round(gap_rate, 3),
        "zone_success": round(zone_succ, 3),
        "gap_success": round(gap_succ, 3),
        "light_box_rate": round(light_box_rate, 3),
        "light_box_success": round(light_box_succ, 3),
        "heavy_box_rate": round(heavy_box_rate, 3),
        "heavy_box_success": round(heavy_box_succ, 3),
        "designed_qb_runs": designed_qb_cnt,
        "designed_qb_epa": round(designed_qb_epa, 3),
        "is_low_sample": low_sample(n_plays=n_plays),
    }


def evaluate_rush_matchup(
    offense_profile: Dict[str, Any],
    defense_profile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluates the matchup differential in standard deviations with sample-size flags.
    Replaces rank-based cutoffs.
    """
    epa_diff = offense_profile["epa_per_play"] - defense_profile["epa_per_play"]
    succ_diff = offense_profile["success_rate"] - defense_profile["success_rate"]

    z_epa = epa_diff / SIGMA_RUSH_EPA
    z_succ = succ_diff / SIGMA_RUSH_SUCCESS
    composite_z = round(0.60 * z_epa + 0.40 * z_succ, 2)

    is_low_sample = offense_profile["is_low_sample"] or defense_profile["is_low_sample"]

    if composite_z >= 1.25:
        verdict = "Favorable Matchup Context"
        tag = "High Positive Leverage"
    elif composite_z >= 0.50:
        verdict = "Moderate Ground Efficiency Differential"
        tag = "Moderate Positive Leverage"
    elif composite_z <= -1.25:
        verdict = "Defensive Front Resistance"
        tag = "Tough Trench Matchup"
    elif composite_z <= -0.50:
        verdict = "Slight Trench Resistance"
        tag = "Moderate Negative Leverage"
    else:
        verdict = "Neutral Line of Scrimmage Context"
        tag = "Neutral / Balanced"

    diagnostic = (
        f"Offense rush success ({offense_profile['success_rate']:.1%}) vs. "
        f"defense allowed ({defense_profile['success_rate']:.1%}): "
        f"{'+' if composite_z > 0 else ''}{composite_z:.2f} SD matchup differential "
        f"(Stuff rate: {defense_profile['stuff_rate']:.1%}, Explosive rate: {offense_profile['explosive_rate']:.1%})."
    )

    return {
        "offense_team": offense_profile["team"],
        "defense_team": defense_profile["team"],
        "composite_z": composite_z,
        "z_epa": round(z_epa, 2),
        "z_success": round(z_succ, 2),
        "verdict": verdict,
        "tag": tag,
        "is_low_sample": is_low_sample,
        "n_plays_offense": offense_profile["plays"],
        "n_plays_defense": defense_profile["plays"],
        "diagnostic": diagnostic,
        "box_context": (
            f"Light box rate {offense_profile.get('light_box_rate', 0.35):.1%} "
            f"(Success {offense_profile.get('light_box_success', 0.42):.1%}) vs. "
            f"Heavy box rate {defense_profile.get('heavy_box_rate', 0.22):.1%}."
        ),
    }


def rank_ground_candidates(
    candidates: List[Dict[str, Any]],
    matchup_eval: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Ranks running back candidates by volume first, then matchup z-score,
    then efficiency. Replaces hard-coded offsets like 94.2, 93.0, 85.0.
    """
    scored = []
    z_matchup = matchup_eval.get("composite_z", 0.0)

    for c in candidates:
        proj_carries = float(c.get("proj_carries", 14.0))
        ypr = float(c.get("yards_per_rush", 4.2))

        # Volume score (centered around 15 carries)
        volume_score = (proj_carries - 15.0) / 4.0
        # Efficiency score
        eff_score = (ypr - 4.2) / 0.50

        # Continuous priority without arbitrary bases
        composite_rank_score = (
            volume_score * 3.0 +
            z_matchup * 2.0 +
            eff_score * 1.5
        )

        c_copy = dict(c)
        c_copy["ground_volume_score"] = round(volume_score, 2)
        c_copy["matchup_z"] = z_matchup
        c_copy["composite_priority_score"] = round(composite_rank_score, 2)
        # Tunable lift based on matchup z-score (clamped to [-0.08, +0.08])
        c_copy["scheme_lift"] = max(-0.08, min(0.08, z_matchup * 0.03))
        scored.append(c_copy)

    scored.sort(key=lambda x: x["composite_priority_score"], reverse=True)
    return scored
