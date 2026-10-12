"""
backend/coverage_matchup.py

Part 1: Defensive Coverage Tendencies & Shell Matchup Engine.
Replaces hard-coded Case A/B/C rules with charted participation data:
- Two orthogonal axes:
    1. Man vs. Zone (defense_man_zone_type)
    2. Middle-Field Open (MFO) vs. Closed (MFC) (defense_coverage_type)
       (Cover 2-Man counts as Open and Man)
- Pressure generation: Blitz rate (pass_rushers >= 5), 4-man pressure rate, time to throw
- Production allowed by shell: Dropback EPA and success rate allowed split by alignment / shell
- Descriptive coverage rates from charted snaps with sample-size flags
- Trend detection: Last 3 games vs. season rate shifts
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
    dropback_metrics,
    BLITZ_MIN_RUSHERS,
)

# League priors for coverage tendencies (STARTING placeholders)
PRIOR_ZONE_RATE = 0.65
PRIOR_MAN_RATE = 0.35
PRIOR_MFO_RATE = 0.52
PRIOR_MFC_RATE = 0.48
PRIOR_BLITZ_RATE = 0.28
PRIOR_PRESSURE_NO_BLITZ = 0.24
PRIOR_TIME_TO_THROW = 2.75

PRIOR_PASS_EPA_ZONE = -0.02
PRIOR_PASS_EPA_MAN = 0.04
PRIOR_PASS_SUCCESS_ZONE = 0.44
PRIOR_PASS_SUCCESS_MAN = 0.46


def map_coverage_shell(cov_type_str: str) -> Tuple[bool, bool]:
    """
    Maps defense_coverage_type string to (is_mfo, is_man).
    Cover 2-Man counts as Open and Man.
    Cover 1 / Cover 3 / Cover 0 -> Closed (MFC).
    Cover 2 / Cover 4 / Cover 6 / Quarters -> Open (MFO).
    """
    s = str(cov_type_str).upper().strip()
    is_mfo = False
    is_man = False

    if "2-MAN" in s or "2 MAN" in s or "COVER 2 MAN" in s:
        return True, True
    elif any(c in s for c in ["COVER 2", "COVER 4", "COVER 6", "QUARTERS", "2 DEEP", "TWO HIGH"]):
        is_mfo = True
    elif any(c in s for c in ["COVER 1", "COVER 3", "COVER 0", "SINGLE HIGH"]):
        is_mfo = False
    else:
        is_mfo = True  # Default modern two-high lean

    if "MAN" in s or "COVER 1" in s or "COVER 0" in s:
        is_man = True

    return is_mfo, is_man


def compute_team_coverage_profile(
    pbp_part_df: pd.DataFrame,
    def_team: str,
    prior_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Computes charted coverage profile for def_team across both axes with sample size flags.
    """
    if pbp_part_df is None or pbp_part_df.empty or "defteam" not in pbp_part_df.columns:
        passes = pd.DataFrame()
    else:
        plays = pbp_part_df[pbp_part_df["defteam"] == def_team].copy()
        passes = filter_plays(plays, kind="pass", drop_garbage_time=True)
    n_dropbacks = len(passes)

    # Use prior profile defaults if provided
    prior_zone = prior_profile.get("zone_pct", PRIOR_ZONE_RATE * 100) / 100.0 if prior_profile else PRIOR_ZONE_RATE
    prior_man = prior_profile.get("man_pct", PRIOR_MAN_RATE * 100) / 100.0 if prior_profile else PRIOR_MAN_RATE
    prior_mfo = prior_profile.get("mfo_pct", PRIOR_MFO_RATE * 100) / 100.0 if prior_profile else PRIOR_MFO_RATE
    prior_blitz = prior_profile.get("blitz_pct", PRIOR_BLITZ_RATE * 100) / 100.0 if prior_profile else PRIOR_BLITZ_RATE

    if passes.empty:
        return {
            "team": def_team,
            "dropbacks": 0,
            "zone_rate": round(prior_zone, 3),
            "man_rate": round(prior_man, 3),
            "mfo_rate": round(prior_mfo, 3),
            "mfc_rate": round(1.0 - prior_mfo, 3),
            "blitz_rate": round(prior_blitz, 3),
            "pressure_no_blitz": round(PRIOR_PRESSURE_NO_BLITZ, 3),
            "time_to_throw": PRIOR_TIME_TO_THROW,
            "pass_epa_zone": PRIOR_PASS_EPA_ZONE,
            "pass_epa_man": PRIOR_PASS_EPA_MAN,
            "pass_success_zone": PRIOR_PASS_SUCCESS_ZONE,
            "pass_success_man": PRIOR_PASS_SUCCESS_MAN,
            "recent_trend_man": 0.0,
            "recent_trend_blitz": 0.0,
            "is_low_sample": True,
        }

    # 1. Man vs Zone axis
    if "defense_man_zone_type" in passes.columns and passes["defense_man_zone_type"].notna().any():
        mz = passes["defense_man_zone_type"].str.upper()
        raw_man_rate = float((mz == "MAN").mean())
        raw_zone_rate = float((mz == "ZONE").mean())
    else:
        raw_man_rate = prior_man
        raw_zone_rate = prior_zone

    # 2. MFO vs MFC axis (Middle-Field Open vs Closed)
    if "defense_coverage_type" in passes.columns and passes["defense_coverage_type"].notna().any():
        mapped = passes["defense_coverage_type"].apply(map_coverage_shell)
        raw_mfo_rate = float(mapped.apply(lambda x: x[0]).mean())
    else:
        raw_mfo_rate = prior_mfo

    # 3. Blitz & Pressure rates
    if "number_of_pass_rushers" in passes.columns:
        rushers = passes["number_of_pass_rushers"].fillna(4)
        is_blitz = rushers >= BLITZ_MIN_RUSHERS
        raw_blitz_rate = float(is_blitz.mean())

        if "was_pressure" in passes.columns:
            press = passes["was_pressure"].fillna(0) == 1
            raw_press_no_blitz = float((press & ~is_blitz).mean())
        else:
            raw_press_no_blitz = PRIOR_PRESSURE_NO_BLITZ
    else:
        raw_blitz_rate = prior_blitz
        raw_press_no_blitz = PRIOR_PRESSURE_NO_BLITZ

    time_to_throw = float(passes["time_to_throw"].mean()) if "time_to_throw" in passes and passes["time_to_throw"].notna().any() else PRIOR_TIME_TO_THROW

    # Empirical rates
    man_rate = raw_man_rate
    zone_rate = 1.0 - man_rate
    mfo_rate = raw_mfo_rate
    mfc_rate = 1.0 - mfo_rate
    blitz_rate = raw_blitz_rate
    press_no_blitz = raw_press_no_blitz

    # Production allowed by shell
    succ = success_col(passes) if "success" in passes else (passes["epa"] > 0).astype(float)
    passes["_succ"] = succ

    if "defense_man_zone_type" in passes.columns:
        zone_dropbacks = passes[passes["defense_man_zone_type"].str.upper() == "ZONE"]
        man_dropbacks = passes[passes["defense_man_zone_type"].str.upper() == "MAN"]
        epa_zone = float(zone_dropbacks["epa"].mean()) if len(zone_dropbacks) > 5 else PRIOR_PASS_EPA_ZONE
        epa_man = float(man_dropbacks["epa"].mean()) if len(man_dropbacks) > 5 else PRIOR_PASS_EPA_MAN
        succ_zone = float(zone_dropbacks["_succ"].mean()) if len(zone_dropbacks) > 5 else PRIOR_PASS_SUCCESS_ZONE
        succ_man = float(man_dropbacks["_succ"].mean()) if len(man_dropbacks) > 5 else PRIOR_PASS_SUCCESS_MAN
    else:
        epa_zone, epa_man = PRIOR_PASS_EPA_ZONE, PRIOR_PASS_EPA_MAN
        succ_zone, succ_man = PRIOR_PASS_SUCCESS_ZONE, PRIOR_PASS_SUCCESS_MAN

    # 4. Recent trend (Last 3 games vs. Season)
    recent_trend_man = 0.0
    recent_trend_blitz = 0.0
    if "week" in passes.columns and passes["week"].nunique() >= 4:
        max_wk = passes["week"].max()
        last3 = passes[passes["week"] >= max_wk - 2]
        if not last3.empty and "defense_man_zone_type" in last3.columns:
            l3_man = (last3["defense_man_zone_type"].str.upper() == "MAN").mean()
            recent_trend_man = round(float(l3_man - raw_man_rate), 3)
        if not last3.empty and "number_of_pass_rushers" in last3.columns:
            l3_blitz = (last3["number_of_pass_rushers"].fillna(4) >= BLITZ_MIN_RUSHERS).mean()
            recent_trend_blitz = round(float(l3_blitz - raw_blitz_rate), 3)

    return {
        "team": def_team,
        "dropbacks": n_dropbacks,
        "zone_rate": round(zone_rate, 3),
        "man_rate": round(man_rate, 3),
        "mfo_rate": round(mfo_rate, 3),
        "mfc_rate": round(mfc_rate, 3),
        "blitz_rate": round(blitz_rate, 3),
        "pressure_no_blitz": round(press_no_blitz, 3),
        "time_to_throw": round(time_to_throw, 2),
        "pass_epa_zone": round(epa_zone, 3),
        "pass_epa_man": round(epa_man, 3),
        "pass_success_zone": round(succ_zone, 3),
        "pass_success_man": round(succ_man, 3),
        "recent_trend_man": recent_trend_man,
        "recent_trend_blitz": recent_trend_blitz,
        "is_low_sample": low_sample(n_plays=n_dropbacks),
    }


def evaluate_coverage_matchup(
    offense_team: str,
    def_profile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Produces honest scheme-based matchup context (replacing static Case A/B/C sentences).
    """
    is_mfo = def_profile["mfo_rate"] >= 0.50
    is_man_heavy = def_profile["man_rate"] >= 0.35

    shell_label = "Middle-Field Open (Two-High)" if is_mfo else "Middle-Field Closed (Single-High)"
    coverage_label = "Man-Heavy" if is_man_heavy else "Zone-Heavy"

    # Contextual tendencies without claims of certainty
    if is_mfo and not is_man_heavy:
        matchup_context = (
            f"{def_profile['team']} secondary operates primarily in {shell_label} Zone "
            f"({def_profile['zone_rate']:.1%} zone, {def_profile['mfo_rate']:.1%} MFO). "
            f"Production allowed: {def_profile['pass_epa_zone']:+.2f} EPA/play vs Zone ({def_profile['pass_success_zone']:.1%} success rate). "
            f"Tends to surrender intermediate middle cushions and seam targets while protecting against deep boundary explosives."
        )
    elif not is_mfo and not is_man_heavy:
        matchup_context = (
            f"{def_profile['team']} relies on {shell_label} Zone "
            f"({def_profile['zone_rate']:.1%} zone, {def_profile['mfc_rate']:.1%} MFC). "
            f"Single-high safety rotations funnel intermediate looks toward outside flats and boundary crossing routes."
        )
    else:
        matchup_context = (
            f"{def_profile['team']} plays {coverage_label} coverage ({def_profile['man_rate']:.1%} Man). "
            f"Production allowed against man looks: {def_profile['pass_epa_man']:+.2f} EPA/play ({def_profile['pass_success_man']:.1%} success). "
            f"Frequently challenges perimeter receivers to win isolated 1-on-1 boundary releases."
        )

    # Trend commentary if rate shift >= 8%
    trend_notes = []
    if abs(def_profile["recent_trend_man"]) >= 0.08:
        shift_str = f"+{def_profile['recent_trend_man']:.1%}" if def_profile["recent_trend_man"] > 0 else f"{def_profile['recent_trend_man']:.1%}"
        trend_notes.append(f"Man coverage frequency shifted {shift_str} in recent 3-game sample.")
    if abs(def_profile["recent_trend_blitz"]) >= 0.08:
        shift_str = f"+{def_profile['recent_trend_blitz']:.1%}" if def_profile["recent_trend_blitz"] > 0 else f"{def_profile['recent_trend_blitz']:.1%}"
        trend_notes.append(f"Blitz rate shifted {shift_str} over the last 3 games.")

    return {
        "offense_team": offense_team,
        "defense_team": def_profile["team"],
        "shell_label": shell_label,
        "coverage_label": coverage_label,
        "man_rate": def_profile["man_rate"],
        "zone_rate": def_profile["zone_rate"],
        "mfo_rate": def_profile["mfo_rate"],
        "mfc_rate": def_profile["mfc_rate"],
        "blitz_rate": def_profile["blitz_rate"],
        "pressure_no_blitz": def_profile["pressure_no_blitz"],
        "time_to_throw": def_profile["time_to_throw"],
        "matchup_context": matchup_context,
        "trend_notes": trend_notes,
        "is_low_sample": def_profile["is_low_sample"],
        "n_dropbacks": def_profile["dropbacks"],
    }
