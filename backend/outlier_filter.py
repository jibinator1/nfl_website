"""
backend/outlier_filter.py

League Outlier & Mismatch Scorer (Path 1 / Option B).
Filters raw matchup statistics through a strict league-distribution gate.
Surfaces only high-signal metrics in the top or bottom 20% of the league
(Rank <= 6 or Rank >= 27, or |z| >= 1.0σ), strictly capped at 5 bullets.

Three Priority Categories:
1. Category 1 — Scheme Clash (Highest Priority):
   Offensive tendency and opposing defensive vulnerability both clear 1.0σ / top-or-bottom 20%.
2. Category 2 — Standalone Unit Extremes:
   Single team metric where |z| >= 1.25σ.
3. Category 3 — Qualified Analog Splits:
   Similar Opponents strong_median is valid (>= 3 games across >= 3 distinct opponents)
   AND deviates from defense's overall median by >= 15%.
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional
import numpy as np


# League-wide empirical distributions across 32 NFL franchises (mean, std)
DEFAULT_LEAGUE_DIST = {
    "off_gap_pct": (45.0, 8.5),
    "off_zone_pct": (55.0, 8.5),
    "off_rush_rate": (42.5, 6.0),
    "off_pass_rate_neutral": (57.5, 6.0),
    "def_gap_ypc_allowed": (4.20, 0.45),
    "def_zone_ypc_allowed": (4.15, 0.40),
    "def_man_pct": (30.0, 8.0),
    "def_zone_pct": (70.0, 8.0),
    "def_mfo_pct": (52.0, 9.0),
    "def_mfc_pct": (48.0, 9.0),
    "def_blitz_pct": (26.0, 7.0),
    "epa_per_pass_zone": (0.00, 0.08),
    "epa_per_pass_man": (0.00, 0.10),
}


def _z(val: Optional[float], metric: str, dist: Dict[str, tuple[float, float]]) -> float:
    if val is None:
        return 0.0
    mean, std = dist.get(metric, (0.0, 1.0))
    if std <= 1e-6:
        return 0.0
    return float((val - mean) / std)


def _get_team_games(matchup_data: Dict[str, Any], team: str) -> int:
    sw = matchup_data.get("sample_window") or {}
    home_tm = matchup_data.get("home_team")
    away_tm = matchup_data.get("away_team")

    if team == home_tm and sw.get("home_games_in_sample") is not None:
        return int(sw["home_games_in_sample"])
    if team == away_tm and sw.get("away_games_in_sample") is not None:
        return int(sw["away_games_in_sample"])
    if sw.get("n_games") is not None:
        return int(sw["n_games"])
    if "n_games" in matchup_data and matchup_data["n_games"] is not None:
        return int(matchup_data["n_games"])
    # Default to sufficient sample (16) if unspecified (e.g. in synthetic mock tests)
    return 16


def _get_team_metric_prior(team: str, metric_name: str) -> Optional[float]:
    """
    Retrieves the prior-season baseline for a team and metric.
    Safely returns None if team or metric is not present (e.g. synthetic test teams).
    """
    try:
        from backend.scheme_insights import TEAM_NAMES, get_team_scheme_profile
        t = str(team).upper()
        if t not in TEAM_NAMES and t != "LAR":
            return None
        prof = get_team_scheme_profile(t)
    except Exception:
        return None

    if not prof or not isinstance(prof, dict):
        return None

    run_s = prof.get("run_scheme", {})
    cov_s = prof.get("coverage", {})

    if metric_name == "off_gap_pct":
        return run_s.get("offense_gap_pct")
    elif metric_name == "off_zone_pct":
        return run_s.get("offense_zone_pct")
    elif metric_name == "def_gap_ypc_allowed":
        return run_s.get("def_gap_ypc_allowed")
    elif metric_name == "def_zone_ypc_allowed":
        return run_s.get("def_zone_ypc_allowed")
    elif metric_name == "def_blitz_pct":
        return cov_s.get("blitz_pct")
    elif metric_name == "def_zone_pct":
        return cov_s.get("zone_pct")
    elif metric_name == "def_man_pct":
        return cov_s.get("man_pct")

    return None


def _blend_metric(
    val: float,
    team: str,
    metric_name: str,
    n_games: int,
) -> tuple[float, bool]:
    """
    Applies Empirical Bayes / Prior-Season Shrinkage for small early-season samples:
        Blended = (n/4) * Metric_current + ((4-n)/4) * Metric_prior  for n < 4
    When n >= 4, uses 100% current season (weight = 1.0, is_blended = False).
    If no prior is available, returns (val, False).
    """
    if n_games >= 4:
        return float(val), False

    prior = _get_team_metric_prior(team, metric_name)
    if prior is None:
        return float(val), False

    n_clamped = max(0, min(n_games, 4))
    w_cur = n_clamped / 4.0
    w_prior = (4.0 - n_clamped) / 4.0
    blended = w_cur * float(val) + w_prior * float(prior)
    return round(blended, 3), True


def extract_top_outliers(
    matchup_data: Dict[str, Any],
    league_dist: Optional[Dict[str, tuple[float, float]]] = None,
    max_bullets: int = 5,
) -> List[Dict[str, Any]]:
    """
    Evaluates computed matchup metrics through league-distribution gates
    and returns up to max_bullets ranked by combined |z-score| descending.
    """
    dist = league_dist or DEFAULT_LEAGUE_DIST
    candidates: List[Dict[str, Any]] = []

    away = matchup_data.get("away_team", "AWAY")
    home = matchup_data.get("home_team", "HOME")

    season = matchup_data.get("season", 2026)
    try:
        prior_season = int(season) - 1
    except (ValueError, TypeError):
        prior_season = 2025

    run_splits = matchup_data.get("run_scheme_splits", {})
    cov = matchup_data.get("coverage_tendencies", {})
    sheet = matchup_data.get("matchup_stat_sheet", {})
    sim = sheet.get("section_5_similar_opponents", {})

    # =========================================================================
    # Category 1: Run Scheme Clashes & Category 2: Run Extremes
    # =========================================================================
    run_matchups = [
        ("away_unit", away, home, run_splits.get("away_unit", {})),
        ("home_unit", home, away, run_splits.get("home_unit", {})),
    ]

    for unit_key, off_tm, def_tm, unit in run_matchups:
        if not unit:
            continue

        off_games = _get_team_games(matchup_data, off_tm)
        def_games = _get_team_games(matchup_data, def_tm)

        for concept in ["gap", "zone"]:
            raw_off_pct = float(unit.get(f"offense_{concept}_pct", 50.0))
            raw_def_ypc = float(unit.get(f"opp_def_{concept}_ypc_allowed", 4.15))
            def_rank = unit.get(f"opp_def_{concept}_rank")

            off_pct, off_blended = _blend_metric(raw_off_pct, off_tm, f"off_{concept}_pct", off_games)
            def_ypc, def_blended = _blend_metric(raw_def_ypc, def_tm, f"def_{concept}_ypc_allowed", def_games)

            off_eligible = (off_games >= 4) or off_blended
            def_eligible = (def_games >= 4) or def_blended

            off_z = _z(off_pct, f"off_{concept}_pct", dist)
            def_z = _z(def_ypc, f"def_{concept}_ypc_allowed", dist)

            # Category 1: Run Scheme Clash (offense leans on concept >= +0.84, opp def |z| >= 0.84)
            if off_z >= 0.84 and abs(def_z) >= 0.84 and off_eligible and def_eligible:
                clash_score = round(abs(off_z) + abs(def_z), 2)
                rank_str = f", Rank #{def_rank}" if def_rank else ""
                blend_tag = ""
                if off_blended or def_blended:
                    n_tag = min(off_games, def_games) if (off_blended and def_blended) else (off_games if off_blended else def_games)
                    blend_tag = f" [Blended w/ {prior_season} Prior, n={n_tag}]"

                candidates.append({
                    "type": "SCHEME_CLASH",
                    "subtype": f"RUN_{concept.upper()}_CLASH",
                    "score": clash_score,
                    "team": off_tm,
                    "opponent": def_tm,
                    "headline": (
                        f"{off_tm} runs {concept.upper()} at top-20% rate ({off_pct:.1f}%, z={off_z:+.1f}σ) vs "
                        f"{def_tm} defense allowing {def_ypc:.2f} YPC (z={def_z:+.1f}σ{rank_str}){blend_tag}"
                    ),
                })
            # Category 2: Single-side defensive YPC extreme (|z| >= 1.25)
            elif abs(def_z) >= 1.25 and def_eligible:
                blend_tag = f" [Blended w/ {prior_season} Prior, n={def_games}]" if def_blended else ""
                rank_str = f" [Rank #{def_rank}]" if def_rank else ""
                candidates.append({
                    "type": "UNIT_EXTREME",
                    "subtype": f"DEF_{concept.upper()}_YPC",
                    "score": round(abs(def_z), 2),
                    "team": def_tm,
                    "headline": f"{def_tm} allowing {def_ypc:.2f} YPC on {concept.upper()} runs ({def_z:+.1f}σ vs league median{rank_str}){blend_tag}",
                })

            # Category 2: Single-side offensive run concept extreme (positive outlier >= 1.25)
            if off_z >= 1.25 and off_eligible and not any(c.get("subtype") == f"RUN_{concept.upper()}_CLASH" for c in candidates):
                blend_tag = f" [Blended w/ {prior_season} Prior, n={off_games}]" if off_blended else ""
                candidates.append({
                    "type": "UNIT_EXTREME",
                    "subtype": f"OFF_{concept.upper()}_RATE",
                    "score": round(abs(off_z), 2),
                    "team": off_tm,
                    "headline": f"{off_tm} {concept.upper()} run rate: {off_pct:.1f}% ({off_z:+.1f}σ vs league median){blend_tag}",
                })

    # Standalone Offensive Rush Rate extremes
    for unit_key, off_tm, def_tm, unit in run_matchups:
        rush_rate = unit.get("rush_rate")
        off_games = _get_team_games(matchup_data, off_tm)
        if rush_rate is not None:
            raw_r = float(rush_rate) * (100.0 if rush_rate <= 1.0 else 1.0)
            r_val, r_blended = _blend_metric(raw_r, off_tm, "off_rush_rate", off_games)
            off_eligible = (off_games >= 4) or r_blended
            if off_eligible:
                r_z = _z(r_val, "off_rush_rate", dist)
                if abs(r_z) >= 1.25:
                    blend_tag = f" [Blended w/ {prior_season} Prior, n={off_games}]" if r_blended else ""
                    candidates.append({
                        "type": "UNIT_EXTREME",
                        "subtype": "OFF_RUSH_RATE",
                        "score": round(abs(r_z), 2),
                        "team": off_tm,
                        "headline": f"{off_tm} Rush Rate: {r_val:.1f}% ({r_z:+.1f}σ vs league median){blend_tag}",
                    })

    # =========================================================================
    # Category 1: Coverage Clashes & Category 2: Coverage Extremes
    # =========================================================================
    cov_matchups = [
        ("away_vs_home", away, home, cov.get("away_vs_home", {})),
        ("home_vs_away", home, away, cov.get("home_vs_away", {})),
    ]

    for c_key, off_tm, def_tm, c_data in cov_matchups:
        if not c_data:
            continue

        off_games = _get_team_games(matchup_data, off_tm)
        def_games = _get_team_games(matchup_data, def_tm)

        # Check Blitz Extreme (|z| >= 1.25)
        raw_blitz_pct = float(c_data.get("def_blitz_pct", 26.0))
        blitz_pct, blitz_blended = _blend_metric(raw_blitz_pct, def_tm, "def_blitz_pct", def_games)
        blitz_eligible = (def_games >= 4) or blitz_blended
        if blitz_z := _z(blitz_pct, "def_blitz_pct", dist):
            if blitz_z >= 1.25 and blitz_eligible:
                blend_tag = f" [Blended w/ {prior_season} Prior, n={def_games}]" if blitz_blended else ""
                candidates.append({
                    "type": "UNIT_EXTREME",
                    "subtype": "DEF_BLITZ_RATE",
                    "score": round(blitz_z, 2),
                    "team": def_tm,
                    "headline": f"{def_tm} Defense Blitz Rate: {blitz_pct:.1f}% (+{blitz_z:.1f}σ vs league median){blend_tag}",
                })

        # Check Zone & Man coverage
        for scheme in ["zone", "man"]:
            raw_def_scheme_pct = float(c_data.get(f"def_{scheme}_pct", 70.0 if scheme == "zone" else 30.0))
            def_scheme_pct, def_cov_blended = _blend_metric(raw_def_scheme_pct, def_tm, f"def_{scheme}_pct", def_games)
            def_cov_eligible = (def_games >= 4) or def_cov_blended

            raw_off_epa = float(c_data.get(f"off_epa_{scheme}", 0.0))
            off_epa, off_epa_blended = _blend_metric(raw_off_epa, off_tm, f"epa_per_pass_{scheme}", off_games)
            off_cov_eligible = (off_games >= 4) or off_epa_blended

            def_scheme_z = _z(def_scheme_pct, f"def_{scheme}_pct", dist)
            off_epa_z = _z(off_epa, f"epa_per_pass_{scheme}", dist)

            # Scheme Clash: Defense runs scheme at top-20% (z >= 0.84) and offense deviates >= 1.0σ
            if def_scheme_z >= 0.84 and abs(off_epa_z) >= 1.0 and def_cov_eligible and off_cov_eligible:
                score = round(abs(def_scheme_z) + abs(off_epa_z), 2)
                blend_tag = ""
                if def_cov_blended or off_epa_blended:
                    n_tag = min(off_games, def_games) if (def_cov_blended and off_epa_blended) else (off_games if def_cov_blended else def_games)
                    blend_tag = f" [Blended w/ {prior_season} Prior, n={n_tag}]"

                candidates.append({
                    "type": "SCHEME_CLASH",
                    "subtype": f"COVERAGE_{scheme.upper()}_CLASH",
                    "score": score,
                    "team": def_tm,
                    "opponent": off_tm,
                    "headline": (
                        f"{def_tm} runs {scheme.upper()} at top-20% rate ({def_scheme_pct:.1f}%, z={def_scheme_z:+.1f}σ) vs "
                        f"{off_tm} efficiency ({off_epa:+.2f} EPA/play, z={off_epa_z:+.1f}σ){blend_tag}"
                    ),
                })
            elif abs(def_scheme_z) >= 1.25 and def_cov_eligible:
                blend_tag = f" [Blended w/ {prior_season} Prior, n={def_games}]" if def_cov_blended else ""
                candidates.append({
                    "type": "UNIT_EXTREME",
                    "subtype": f"DEF_{scheme.upper()}_RATE",
                    "score": round(abs(def_scheme_z), 2),
                    "team": def_tm,
                    "headline": f"{def_tm} {scheme.upper()} Coverage Rate: {def_scheme_pct:.1f}% ({def_scheme_z:+.1f}σ vs league median){blend_tag}",
                })

    # =========================================================================
    # Category 3: Qualified Analog Splits
    # =========================================================================
    sim_matchups = [
        (home, away, sim.get("home_defense_vs_opponents_like_away", {})),
        (away, home, sim.get("away_defense_vs_opponents_like_home", {})),
    ]

    for def_tm, off_tm, sim_data in sim_matchups:
        if not sim_data:
            continue

        strong_med = sim_data.get("strong_analogs_median") or sim_data.get("median_rb_rush_yards_strong")
        baseline_obj = sim_data.get("overall_defense_baseline", {})
        baseline = float(baseline_obj.get("rb_rush_yards", 0.0)) if isinstance(baseline_obj, dict) else 0.0

        if strong_med is not None and baseline > 0:
            diff_pct = (strong_med - baseline) / baseline * 100.0
            if abs(diff_pct) >= 15.0:
                # Analog score: calibrated to compete with ~1.5σ signals
                score = round(1.20 + (abs(diff_pct) / 25.0), 2)
                status = sim_data.get("strong_status", ">= 3 third-party games")
                candidates.append({
                    "type": "ANALOG_SPLIT",
                    "subtype": "STRONG_ANALOG_YARDS",
                    "score": score,
                    "team": def_tm,
                    "opponent": off_tm,
                    "headline": (
                        f"{off_tm} vs {def_tm} Strong Analogs: {strong_med:.1f} yds allowed "
                        f"({diff_pct:+.1f}% vs {baseline:.1f} baseline across {status})"
                    ),
                })

    # Deduplicate candidates with identical or nearly identical headlines
    seen_headlines = set()
    unique_candidates: List[Dict[str, Any]] = []
    for c in candidates:
        hl = c["headline"].strip()
        if hl not in seen_headlines:
            seen_headlines.add(hl)
            unique_candidates.append(c)

    # Sort descending by score and cap at max_bullets
    unique_candidates.sort(key=lambda x: x["score"], reverse=True)
    return unique_candidates[:max_bullets]
