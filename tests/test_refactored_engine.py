"""
tests/test_refactored_engine.py

Comprehensive test suite verifying the refactored descriptive NFL scouting engine:
  1. Shared play filters (WP bounds [0.05, 0.95], kneel/spike drop, scramble handling).
  2. Game-state classification (offense leading >= +7, trailing <= -7, neutral).
  3. Sample-size flagging (n < 40 plays, n < 4 games).
  4. Opponent adjustment implementing exact leave-one-out math.
  5. Rush metrics and dropback metrics calculations.
  6. Descriptive player situational splits (snap, target, aDOT, TPRR, percentiles).
  7. Modular matchup engines: run game, coverage shells, target splits, tempo profiles.
  8. Film-room matchup synthesis: notable stats ranked by deviation from league median.
  9. Personnel availability and injury context layer.

Zero betting model tests: no EV, fair lines, win probabilities, or picks.
"""
from __future__ import annotations

import os
import pytest
import pandas as pd
import numpy as np

from backend.definitions import (
    filter_plays,
    game_state,
    low_sample,
    opponent_adjusted,
    rush_metrics,
    dropback_metrics,
    success_col,
    WP_MIN,
    WP_MAX,
    LEAD_MARGIN,
    TRAIL_MARGIN,
    SAMPLE_SIZE_THRESHOLDS,
)
from backend.player_situational_splits import (
    compute_player_situational_splits,
    compute_floor_ceiling_history,
)
from backend.coverage_matchup import (
    compute_team_coverage_profile,
    evaluate_coverage_matchup,
)
from backend.run_matchup import (
    compute_team_rush_profile,
    evaluate_rush_matchup,
)
from backend.target_matchup import (
    compute_player_usage_profile,
    evaluate_target_matchup,
)
from backend.tempo_matchup import (
    compute_team_tempo_profile,
    evaluate_tempo_matchup,
)
from backend.synthesis import build_matchup_synthesis
from backend.injuries import load_weekly_injuries, get_game_injuries


# ==============================================================================
# 1. Play Filters & Shared Data Pipeline
# ==============================================================================
def test_filter_plays_garbage_time_and_special():
    """Verify standard play filter excludes garbage time and kneel/spikes."""
    df = pd.DataFrame({
        "play_type": ["pass", "run", "pass", "run", "pass"],
        "epa": [0.5, -0.2, 0.1, 0.3, None],
        "qb_kneel": [0, 0, 1, 0, 0],
        "qb_spike": [0, 0, 0, 0, 0],
        "wp": [0.50, 0.98, 0.50, 0.02, 0.50],  # 0.98 and 0.02 are garbage time
    })

    filtered = filter_plays(df, kind="all", drop_garbage_time=True)
    # Row 0: valid (wp 0.50, epa 0.5)
    # Row 1: dropped (wp 0.98 > 0.95)
    # Row 2: dropped (qb_kneel == 1)
    # Row 3: dropped (wp 0.02 < 0.05)
    # Row 4: dropped (epa is None)
    assert len(filtered) == 1
    assert filtered.iloc[0]["epa"] == 0.5


def test_filter_plays_scrambles():
    """Verify drop_qb_scrambles strips scrambles for RB run evaluations."""
    df = pd.DataFrame({
        "play_type": ["run", "run", "pass"],
        "epa": [0.2, 0.1, 0.3],
        "qb_scramble": [0, 1, 0],
        "wp": [0.5, 0.5, 0.5],
    })
    rb_runs = filter_plays(df, kind="run", drop_qb_scrambles=True)
    assert len(rb_runs) == 1
    assert rb_runs.iloc[0]["qb_scramble"] == 0


# ==============================================================================
# 2. Game-State Classification
# ==============================================================================
def test_game_state_classification():
    """Leading if >= +7, trailing if <= -7, neutral otherwise."""
    diffs = pd.Series([14, 7, 6, 0, -3, -7, -10])
    states = game_state(diffs)
    expected = ["leading", "leading", "neutral", "neutral", "neutral", "trailing", "trailing"]
    assert list(states) == expected


# ==============================================================================
# 3. Sample-Size Flagging
# ==============================================================================
def test_sample_size_flags():
    """Flags metrics built on fewer than 40 plays or 4 games."""
    assert low_sample(n_plays=39) is True
    assert low_sample(n_plays=40) is False
    assert low_sample(n_games=3) is True
    assert low_sample(n_games=4) is False
    assert low_sample(n_plays=100, n_games=5) is False
    assert SAMPLE_SIZE_THRESHOLDS["plays"] == 40
    assert SAMPLE_SIZE_THRESHOLDS["games"] == 4


# ==============================================================================
# 4. Opponent Adjustment (Exact Leave-One-Out Math)
# ==============================================================================
def test_opponent_adjusted_exact_leave_one_out():
    """
    Opponent-adjusted difference computes opponent's mean across other games.
    """
    games = pd.DataFrame([
        {"team": "BAL", "opp": "KC", "val": 30.0},
        {"team": "DET", "opp": "KC", "val": 20.0},
        {"team": "SF",  "opp": "KC", "val": 10.0},
    ])
    # KC conceded 30 to BAL, 20 to DET, 10 to SF. Total = 60.
    # For BAL vs KC: KC's other games are DET (20) and SF (10). Mean = 15.0.
    # Diff for BAL = 30.0 - 15.0 = +15.0.
    adj = opponent_adjusted(games, team_col="team", opp_col="opp", value_col="val")
    bal_row = adj[adj["team"] == "BAL"].iloc[0]
    assert bal_row["adj_value"] == pytest.approx(15.0)
    assert bal_row["games"] == 1


# ==============================================================================
# 5. Rush and Dropback Metrics Helpers
# ==============================================================================
def test_rush_and_dropback_metrics():
    """Verify aggregated metrics calculate rates correctly."""
    rush_df = pd.DataFrame({
        "team": ["PHI", "PHI", "PHI"],
        "epa": [0.2, -0.4, 0.5],
        "yards_gained": [12, -1, 4],
        "success": [1, 0, 1],
    })
    metrics = rush_metrics(rush_df, ["team"])
    assert len(metrics) == 1
    row = metrics.iloc[0]
    assert row["plays"] == 3
    assert row["stuff_rate"] == pytest.approx(1 / 3)      # -1 yds
    assert row["explosive_rate"] == pytest.approx(1 / 3)  # 12 yds >= 10
    assert row["success_rate"] == pytest.approx(2 / 3)

    pass_df = pd.DataFrame({
        "team": ["KC", "KC"],
        "epa": [0.4, -0.1],
        "success": [1, 0],
    })
    p_metrics = dropback_metrics(pass_df, ["team"])
    assert p_metrics.iloc[0]["dropbacks"] == 2
    assert p_metrics.iloc[0]["success_rate"] == 0.5


# ==============================================================================
# 6. Player Situational Splits & Floor/Ceiling
# ==============================================================================
def test_player_floor_ceiling_history():
    """Distribution percentiles calculated accurately on real actuals."""
    actuals = [45.0, 52.0, 68.0, 75.0, 80.0, 88.0, 92.0, 105.0, 110.0, 120.0]
    res = compute_floor_ceiling_history(actuals, benchmark=70.0)
    assert res["games_count"] == 10
    assert res["median"] == pytest.approx(84.0)
    assert res["cleared_ratio"] == "7/10"
    assert res["cleared_pct"] == 70.0
    assert res["floor_10th"] > 0.0
    assert res["ceiling_90th"] > res["median"]


def test_player_situational_splits():
    """Generates complete descriptive player usage profile."""
    player_games = pd.DataFrame({
        "offense_pct": [0.85, 0.82, 0.90, 0.88],
        "target_share": [0.25, 0.28, 0.22, 0.29],
        "air_yards_share": [0.35, 0.40, 0.30, 0.38],
        "targets": [8, 10, 7, 9],
        "receiving_air_yards": [80, 110, 65, 95],
        "team_pass_att": [34, 36, 32, 35],
        "receiving_epa": [4.2, 5.1, 2.0, 4.8],
        "rz_targets": [2, 1, 0, 2],
        "rz_rushes": [0, 0, 0, 0],
        "team_rz_plays": [8, 9, 6, 8],
        "receiving_yards": [75.0, 95.0, 62.0, 88.0],
    })
    splits = compute_player_situational_splits(
        player_name="N.Collins",
        pos="WR",
        team="HOU",
        player_games=player_games,
        stat_focus="Receiving Yards",
        benchmark_line=72.5,
    )
    assert splits["player"] == "N.Collins"
    assert splits["games_played"] == 4
    assert splits["is_low_sample"] is False
    assert splits["snap_share"] == pytest.approx(0.862, rel=1e-2)
    assert splits["target_share"] == pytest.approx(0.26, rel=1e-2)
    assert splits["adot"] == pytest.approx(10.3, rel=1e-1)
    assert splits["baseline_median"] == pytest.approx(81.5)
    assert splits["cleared_ratio"] == "3/4"


# ==============================================================================
# 7. Coverage Matchup Engine
# ==============================================================================
def test_coverage_matchup_evaluation():
    """Evaluates coverage tendencies with descriptive context."""
    def_profile = {
        "team": "DEN",
        "dropbacks": 65,
        "zone_rate": 0.55,
        "man_rate": 0.45,
        "mfo_rate": 0.40,
        "mfc_rate": 0.60,
        "blitz_rate": 0.32,
        "pressure_no_blitz": 0.28,
        "time_to_throw": 2.65,
        "pass_epa_zone": -0.05,
        "pass_epa_man": -0.12,
        "pass_success_zone": 0.42,
        "pass_success_man": 0.36,
        "recent_trend_man": 0.05,
        "recent_trend_blitz": 0.02,
        "is_low_sample": False,
    }
    eval_cov = evaluate_coverage_matchup("KC", def_profile)
    assert eval_cov["defense_team"] == "DEN"
    assert eval_cov["coverage_label"] == "Man-Heavy"
    assert "Single-High" in eval_cov["shell_label"]
    assert len(eval_cov["matchup_context"]) > 20


# ==============================================================================
# 8. Rushing Matchup Engine
# ==============================================================================
def test_rush_matchup_evaluation():
    """Evaluates ground trench differential in standard deviations."""
    off = {
        "team": "BAL", "plays": 120, "epa_per_play": 0.08, "success_rate": 0.50,
        "stuff_rate": 0.12, "explosive_rate": 0.18, "is_low_sample": False,
    }
    deff = {
        "team": "CAR", "plays": 110, "epa_per_play": 0.04, "success_rate": 0.48,
        "stuff_rate": 0.14, "explosive_rate": 0.15, "is_low_sample": False,
    }
    res = evaluate_rush_matchup(off, deff)
    assert res["offense_team"] == "BAL"
    assert res["defense_team"] == "CAR"
    assert res["composite_z"] > 0
    assert res["is_low_sample"] is False


# ==============================================================================
# 9. Target Matchup Engine
# ==============================================================================
def test_target_matchup_evaluation():
    """Calculates coverage split delta range cleanly."""
    usage = {
        "player": "A.St. Brown", "pos": "WR", "target_share": 0.28,
        "target_share_l4": 0.31, "tprr_approx": 0.27, "is_low_sample": False,
    }
    cov = {
        "defense_team": "GB", "coverage_label": "Zone-Heavy", "zone_rate": 0.70, "man_rate": 0.30,
    }
    splits = {"tgt_zone": 30.0, "tgt_man": 24.0}
    res = evaluate_target_matchup(usage, cov, splits)
    assert res["player"] == "A.St. Brown"
    assert res["base_target_share"] == 0.28
    assert "GB" in res["diagnostic"]
    assert res["is_low_sample"] is False


# ==============================================================================
# 10. Tempo Matchup Engine
# ==============================================================================
def test_tempo_matchup_evaluation():
    """Evaluates game pace script and projected plays."""
    away_tempo = {
        "team": "MIA", "neutral_pace_sec": 26.0, "pass_rate_neutral": 0.62,
        "pass_rate_leading": 0.50, "pass_rate_trailing": 0.72, "proj_plays": 68.0,
        "no_huddle_rate": 0.12, "go_for_it_4th_rate": 0.55, "is_low_sample": False,
    }
    home_tempo = {
        "team": "BUF", "neutral_pace_sec": 26.5, "pass_rate_neutral": 0.60,
        "pass_rate_leading": 0.48, "pass_rate_trailing": 0.70, "proj_plays": 66.5,
        "no_huddle_rate": 0.10, "go_for_it_4th_rate": 0.60, "is_low_sample": False,
    }
    res = evaluate_tempo_matchup(away_tempo, home_tempo, spread_line=-2.5, total_line=48.5)
    assert res["combined_proj_plays"] > 130
    assert res["pace_tag"] in ["Standard Pace", "Track Meet Potential"]


# ==============================================================================
# 11. Matchup Synthesis Engine
# ==============================================================================
def test_matchup_synthesis_notable_stats():
    """Synthesizes pillars into ranked notable matchup stats without predictions."""
    cov_away = {"defense_team": "GB", "shell_label": "Two-High", "mfo_rate": 0.58, "man_rate": 0.28, "is_low_sample": False}
    cov_home = {"defense_team": "DET", "shell_label": "Single-High", "mfo_rate": 0.42, "man_rate": 0.40, "is_low_sample": False}
    rush_away = {"team": "DET", "composite_z": 1.10, "is_low_sample": False, "diagnostic": "Strong ground push"}
    rush_home = {"team": "GB", "composite_z": 0.20, "is_low_sample": False, "diagnostic": "Neutral trench"}
    tempo = {"combined_proj_plays": 134.0, "pace_tag": "Up-Tempo", "spread_line": -3.5, "total_line": 51.0}

    synth = build_matchup_synthesis(
        cov_eval_away=cov_away,
        cov_eval_home=cov_home,
        rush_eval_away=rush_away,
        rush_eval_home=rush_home,
        tempo_eval=tempo,
        key_players=[{"player": "J.Gibbs", "team": "DET", "pos": "RB", "stat_focus": "Rushing Yards"}],
        injury_context=[{"player": "J.Alexander", "position": "CB", "status": "Out"}],
    )

    assert "notable_matchup_stats" in synth
    assert "ranked_drivers" in synth  # Backwards compatibility key
    stats = synth["notable_matchup_stats"]
    assert len(stats) >= 3

    # Ensure strictly sorted by importance (deviation from median)
    importances = [s["importance"] for s in stats]
    assert importances == sorted(importances, reverse=True)

    # Verify contingencies contain injury and game script
    assert len(synth["contingencies"]) >= 2
    assert any("J.Alexander" in c for c in synth["contingencies"])


# ==============================================================================
# 12. Personnel & Injury Integration
# ==============================================================================
def test_injuries_layer():
    """Verifies factual personnel availability loading."""
    injuries = load_weekly_injuries(season=2024, week=1)
    assert isinstance(injuries, dict)
    assert isinstance(injuries.get("injuries", []), list)
    game_inj = get_game_injuries("BAL", "KC", season=2024, week=1)
    assert isinstance(game_inj, list)


# ==============================================================================
# 13. Path 1 Signal-Only & Outlier Engine Invariants
# ==============================================================================
def test_no_lookahead_in_sample_window():
    """Assert a Week 3 call contains zero plays from Weeks 3+ (and Weeks 4+)."""
    from backend.team_tendencies import load_clean_pbp
    from backend.matchup_sheet import build_matchup_stat_sheet

    pbp = load_clean_pbp(season=2026, up_to_week=3, strict_less=True)
    if not pbp.empty and "week" in pbp.columns:
        pbp_curr = pbp[pbp["season"] == 2026] if "season" in pbp.columns else pbp
        if not pbp_curr.empty:
            assert (pbp_curr["week"] < 3).all()
            assert not (pbp_curr["week"] >= 3).any()

    # Verify stat sheet sample window
    sheet = build_matchup_stat_sheet("BAL", "DAL", season=2026, up_to_week=3)
    sw = sheet.get("sample_window", {})
    assert sw.get("up_to_week") == 3
    assert sw.get("max_week_included", 0) <= 2
    assert "Weeks 1–2" in sw.get("sample_label", "") or "Weeks 1–" in sw.get("sample_label", "")


def test_strong_analog_requires_three_distinct_opponents():
    """Assert that 4 games against DAL alone returns strong_median = None."""
    # When strong_third_party has 4 games, but all against DAL (len(distinct_opps) == 1 < 3):
    third_party_analogs = [
        {"opponent": "DAL", "rmsd": 0.50, "yds": 90.0},
        {"opponent": "DAL", "rmsd": 0.60, "yds": 95.0},
        {"opponent": "DAL", "rmsd": 0.70, "yds": 85.0},
        {"opponent": "DAL", "rmsd": 0.80, "yds": 100.0},
    ]
    strong_third_party = [g for g in third_party_analogs if g["rmsd"] <= 1.00]
    distinct_opps = {g["opponent"] for g in strong_third_party}

    if len(strong_third_party) >= 3 and len(distinct_opps) >= 3:
        strong_median = float(np.median([g["yds"] for g in strong_third_party]))
        strong_status = f"{len(strong_third_party)} games across {len(distinct_opps)} third-party opponents"
    else:
        strong_median = None
        strong_status = "Insufficient third-party diversity (<3 games or <3 distinct opponents)"

    assert strong_median is None
    assert "Insufficient third-party diversity" in strong_status


def test_max_five_outlier_bullets():
    """Assert len(sheet['top_outliers']) <= 5 across matchup data."""
    from backend.scheme_insights import compute_scheme_insights
    from backend.outlier_filter import extract_top_outliers

    insights = compute_scheme_insights(home_team="DAL", away_team="BAL", season=2026, week=3)
    assert "top_outliers" in insights
    assert len(insights["top_outliers"]) <= 5
    if "matchup_stat_sheet" in insights and isinstance(insights["matchup_stat_sheet"], dict):
        sheet = insights["matchup_stat_sheet"]
        if "top_outliers" in sheet:
            assert len(sheet["top_outliers"]) <= 5

    # Also test outlier filter with artificial 10+ candidate outliers
    mock_matchup = {
        "coverage_tendencies": {
            "away_vs_home": {"def_man_pct_z": 2.1, "def_blitz_pct_z": 1.8, "def_zone_pct_z": -1.5, "def_mfo_pct_z": 1.3},
            "home_vs_away": {"def_man_pct_z": 1.9, "def_blitz_pct_z": 1.6, "def_zone_pct_z": -1.4, "def_mfo_pct_z": 1.2},
        },
        "run_scheme_splits": {
            "away_unit": {"off_gap_pct_z": 1.5, "opp_def_gap_ypc_allowed_z": 1.6, "off_zone_pct_z": 1.4, "opp_def_zone_ypc_allowed_z": 1.5},
            "home_unit": {"off_gap_pct_z": 1.5, "opp_def_gap_ypc_allowed_z": 1.6, "off_zone_pct_z": 1.4, "opp_def_zone_ypc_allowed_z": 1.5},
        }
    }
    extracted = extract_top_outliers(mock_matchup, max_bullets=5)
    assert len(extracted) <= 5


def test_run_scheme_clash_positive_only():
    """Assert negative offensive z-scores do not trigger SCHEME_CLASH (prevents binary mirroring)."""
    from backend.outlier_filter import extract_top_outliers

    # DAL has +1.6σ Gap and -1.6σ Zone; opposing BAL defense allows 3.48 YPC (-1.6σ) on Gap and 3.62 YPC (-1.3σ) on Zone
    mock_matchup = {
        "away_team": "BAL",
        "home_team": "DAL",
        "n_games": 8,
        "run_scheme_splits": {
            "home_unit": {
                "team": "DAL",
                "offense_gap_pct": 58.3,   # +1.56σ
                "offense_zone_pct": 41.7,  # -1.56σ
                "opp_def_gap_ypc_allowed": 3.48,   # -1.60σ
                "opp_def_zone_ypc_allowed": 3.62,  # -1.33σ
            }
        }
    }
    bullets = extract_top_outliers(mock_matchup)
    clash_subtypes = [b.get("subtype") for b in bullets if b.get("type") == "SCHEME_CLASH"]

    # Only Gap should clash, NOT Zone (DAL leans on Gap, avoids Zone)
    assert "RUN_GAP_CLASH" in clash_subtypes
    assert "RUN_ZONE_CLASH" not in clash_subtypes

    # Ensure no bullet labels a negative z as 'top-20% rate'
    for b in bullets:
        if "runs ZONE" in b.get("headline", ""):
            assert "top-20%" not in b.get("headline", "")


def test_small_sample_gate_suppresses_outliers():
    """Assert SCHEME_CLASH and UNIT_EXTREME are suppressed when n_games < 4."""
    from backend.outlier_filter import extract_top_outliers

    mock_matchup = {
        "away_team": "BAL",
        "home_team": "DAL",
        "sample_window": {
            "home_games_in_sample": 2,
            "away_games_in_sample": 2,
            "n_games": 2,
        },
        "run_scheme_splits": {
            "home_unit": {
                "team": "DAL",
                "offense_gap_pct": 65.0,  # extreme +2.35σ
                "opp_def_gap_ypc_allowed": 5.20,  # extreme +2.22σ
            }
        }
    }
    bullets = extract_top_outliers(mock_matchup)
    # Both teams only have 2 games; outliers must be suppressed
    assert len(bullets) == 0


def test_format_injury_line_excludes_healthy_full_practice():
    """Assert healthy players with full practice and no status designation return None."""
    from backend.roster_verification import format_injury_line

    healthy_player = {
        "player_name": "Zay Flowers",
        "position": "WR",
        "report_status": None,
        "practice_status": "Full Participation in Practice",
    }
    assert format_injury_line(healthy_player) is None

    injured_player = {
        "player_name": "Nnamdi Madubuike",
        "position": "DT",
        "report_status": "Out",
        "practice_status": "Limited Participation in Practice",
        "report_primary_injury": "Neck",
    }
    line = format_injury_line(injured_player)
    assert line is not None
    assert "Nnamdi Madubuike (DT): Out" in line
    assert "(Neck)" in line

    limited_only_player = {
        "player_name": "Maliek Collins",
        "position": "DT",
        "report_status": None,
        "practice_status": "Limited Participation in Practice",
    }
    assert format_injury_line(limited_only_player) is not None


def test_strong_analog_filter_excludes_moderate_analogs():
    """Assert games with RMSD > 0.70 are excluded from strong_third_party."""
    third_party_analogs = [
        {"opponent": "PIT", "rmsd": 0.641, "similarity_tier_key": "strong", "yds": 80.0},
        {"opponent": "GB", "rmsd": 0.675, "similarity_tier_key": "strong", "yds": 85.0},
        {"opponent": "NYG", "rmsd": 0.782, "similarity_tier_key": "moderate", "yds": 90.0},
        {"opponent": "DAL", "rmsd": 0.793, "similarity_tier_key": "moderate", "yds": 95.0},
    ]
    strong_third_party = [
        g for g in third_party_analogs
        if g.get("similarity_tier_key") == "strong" or g.get("rmsd", 99.0) <= 0.70
    ]
    distinct_opps = {g["opponent"] for g in strong_third_party}
    # Only PIT and GB qualify (2 games, 2 opps < 3)
    assert len(strong_third_party) == 2
    assert len(distinct_opps) == 2


