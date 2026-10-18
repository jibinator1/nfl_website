"""
tests/test_team_tendencies.py

Comprehensive test suite verifying the descriptive team tendency,
opponent vulnerability, similar-opponent lookup, and matchup stat sheet engine:
  1. Every output metric contains raw value, league median, p25, p75, rank, rank direction, diff, and sample size.
  2. Every matchup section is clearly labeled descriptive, not predictive.
  3. Similar opponent historical lookup returns a table of real past games with medians and ranges.
  4. Dual-split lookup presents ground-heavy and aerial-heavy matches side by side.
  5. Head-to-head recent meeting returns historical box score stats when available.
  6. Zero predictive or model language: no 'projected', 'expected', 'fair', 'edge', 'advantage', 'exploit', 'pick', 'probability', 'ev'.
"""
from __future__ import annotations

import re
import pytest
import pandas as pd

from backend.team_tendencies import (
    compute_team_tendency_profile,
    compute_opponent_vulnerability_profile,
    format_stat_comparison,
)
from backend.similar_opponents import (
    find_similar_opponent_games,
    run_dual_archetype_lookup,
    extract_team_offensive_profile,
)
from backend.matchup_sheet import (
    build_matchup_stat_sheet,
    get_head_to_head_meeting,
)


PROHIBITED_WORDS = [
    "projected",
    "expected",
    "fair",
    "edge",
    "advantage",
    "exploit",
    "pick",
    "probability",
    "ev",
]


# ==============================================================================
# 1. Format Stat Comparison Helper
# ==============================================================================
def test_format_stat_comparison():
    """format_stat_comparison computes median, percentiles, rank, diffs, and sample size flag."""
    league_series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    res = format_stat_comparison(
        team_val=35.0,
        league_series=league_series,
        sample_size=45,
        higher_is_better=True,
        unit="yards",
        min_sample=40,
    )
    assert res["raw_value"] == 35.0
    assert res["league_median"] == 30.0
    assert res["league_p25"] == 20.0
    assert res["league_p75"] == 40.0
    assert res["rank"] == 3  # 50, 40 are higher -> rank 3
    assert res["rank_direction"] == "higher is better"
    assert res["diff_from_median"] == 5.0
    assert res["diff_pct"] == pytest.approx(16.7, rel=1e-1)
    assert res["sample_size"] == 45
    assert res["is_small_sample"] is False

    # Small sample flag test
    res_small = format_stat_comparison(team_val=35.0, league_series=league_series, sample_size=12, min_sample=40)
    assert res_small["is_small_sample"] is True


# ==============================================================================
# 2. Team Tendency Profile (What this team tends to do)
# ==============================================================================
def test_team_tendency_profile():
    """compute_team_tendency_profile calculates situational splits with medians and sample sizes."""
    profile = compute_team_tendency_profile("BAL", season=2026)
    assert profile["team"] == "BAL"
    assert profile["total_plays_evaluated"] > 0

    # Game State Splits
    gs = profile["game_state_splits"]
    for state in ["leading", "neutral", "trailing"]:
        assert state in gs
        s_data = gs[state]
        assert "pass_rate" in s_data
        assert "rush_rate" in s_data
        assert "epa_per_play" in s_data
        assert "success_rate" in s_data
        assert "plays_per_game" in s_data
        assert s_data["pass_rate"]["sample_size"] >= 0
        assert "league_median" in s_data["pass_rate"]
        assert "rank" in s_data["pass_rate"]

    # Down Splits
    downs = profile["down_splits"]
    for d_num in [1, 2, 3, 4]:
        d_key = f"down_{d_num}"
        assert d_key in downs
        assert "pass_rate" in downs[d_key]
        assert "league_median" in downs[d_key]["pass_rate"]

    # Field Position Splits
    fp = profile["field_position_splits"]
    assert "red_zone_20_to_1" in fp
    assert "own_1_to_20" in fp
    assert "pass_rate" in fp["red_zone_20_to_1"]
    assert "td_rate" in fp["red_zone_20_to_1"]

    # Red zone shares
    rz = profile["red_zone_shares"]["inside_20"]
    assert "wr_target_share" in rz
    assert "te_target_share" in rz
    assert "rb_carry_share" in rz


# ==============================================================================
# 3. Opponent Vulnerability Profile (What they allow, and to whom)
# ==============================================================================
def test_opponent_vulnerability_profile():
    """compute_opponent_vulnerability_profile calculates rush, pass, depth, and archetype metrics."""
    vuln = compute_opponent_vulnerability_profile("KC", season=2026)
    assert vuln["defense_team"] == "KC"
    assert vuln["total_plays_defended"] > 0

    # Rush Defense
    rd = vuln["rush_defense"]["overall_rb"]
    assert "epa_per_carry" in rd
    assert "epa_per_carry_opp_adj" in rd
    assert "yards_per_carry" in rd
    assert "stuff_rate_under_zero" in rd
    assert rd["epa_per_carry"]["sample_size"] > 0
    assert "league_median" in rd["epa_per_carry"]

    # Directional Gaps
    gaps = vuln["rush_defense"]["directional_gaps"]
    assert "inside_tackles" in gaps
    assert "outside_tackles" in gaps
    assert "yards_per_carry" in gaps["inside_tackles"]

    # Pass Defense & Depth Bins
    pd_depth = vuln["pass_defense"]["target_depth_allowed"]
    assert "deep_20_plus" in pd_depth
    assert "short_0_to_9" in pd_depth
    assert "yards_per_attempt" in pd_depth["deep_20_plus"]

    # Positional Breakdown
    pos_all = vuln["positions_allowed"]
    assert "wr" in pos_all
    assert "te" in pos_all
    assert "rb" in pos_all
    assert "yards_per_game" in pos_all["te"]
    assert "sample_targets" in pos_all["te"]

    # Archetype Breakdown
    arch = vuln["player_archetypes_allowed"]
    assert "vs_workhorse_rbs_15_plus_carries" in arch
    assert "vs_wr1_top_target" in arch
    assert "games_faced" in arch["vs_workhorse_rbs_15_plus_carries"]


# ==============================================================================
# 4. Similar Opponent Historical Lookup
# ==============================================================================
def test_similar_opponent_historical_lookup():
    """find_similar_opponent_games returns table of real historical games with medians and ranges."""
    res = find_similar_opponent_games("BAL", "KC", season=2026, n_similar=5)
    assert res["offense_team"] == "BAL"
    assert res["defense_team"] == "KC"
    assert len(res["similar_games"]) == 5

    # Check game row fields
    first_game = res["similar_games"][0]
    assert "season" in first_game
    assert "week" in first_game
    assert "opponent" in first_game
    assert "rb_rush_yards_allowed" in first_game
    assert "wr_rec_yards_allowed" in first_game
    assert "te_rec_yards_allowed" in first_game
    assert "pass_yards_allowed" in first_game
    assert "sacks" in first_game
    assert "epa_per_play_allowed" in first_game
    assert "success_rate_allowed" in first_game

    # Summary medians and ranges
    summ = res["similar_games_summary"]
    assert "median" in summ["rb_rush_yards"]
    assert "min" in summ["rb_rush_yards"]
    assert "max" in summ["rb_rush_yards"]
    assert summ["rb_rush_yards"]["min"] <= summ["rb_rush_yards"]["median"] <= summ["rb_rush_yards"]["max"]

    # Baseline comparison across all games
    base = res["overall_defense_baseline"]
    assert "rb_rush_yards" in base
    assert "total_games_evaluated" in base
    assert base["total_games_evaluated"] >= 5

    # Narrative text
    narrative = res["descriptive_narrative"]
    assert "KC faced 5 opponents" in narrative
    assert "rushing yards" in narrative
    assert "disclaimer" in res


# ==============================================================================
# 5. Dual-Split Lookup (Run-Heavy vs Pass-Heavy side-by-side)
# ==============================================================================
def test_dual_archetype_lookup():
    """run_dual_archetype_lookup presents run-heavy and pass-heavy historical samples side-by-side with distance ceiling <= 1.50sigma."""
    dual = run_dual_archetype_lookup("BAL", "KC", season=2026, n_similar=5)
    assert "ground_match" in dual
    assert "aerial_match" in dual
    # Distance ceiling drops any match > 1.50 sigma
    ground_games = dual["ground_match"]["similar_games"]
    assert len(ground_games) <= 5
    for g in ground_games:
        assert g["rmsd"] <= 1.50, f"Game RMSD exceeded 1.50 ceiling: {g}"



# ==============================================================================
# 6. Head-to-Head Recent Meeting Box Score
# ==============================================================================
def test_head_to_head_meeting():
    """get_head_to_head_meeting retrieves recent box score stats for past matchups."""
    h2h = get_head_to_head_meeting("BAL", "KC")
    if h2h.get("played_recently"):
        assert "season" in h2h
        assert "week" in h2h
        assert "away_team" in h2h
        assert "home_team" in h2h
        assert "rush_yards" in h2h["away_team"]
        assert "pass_yards" in h2h["away_team"]
        assert "leading_rusher" in h2h["away_team"]
        assert "leading_receiver" in h2h["away_team"]


# ==============================================================================
# 7. Complete Matchup Stat Sheet Assembly & Acceptance Checks
# ==============================================================================
def test_matchup_stat_sheet_assembly():
    """build_matchup_stat_sheet compiles complete stat sheet without model/predictive wording."""
    sheet = build_matchup_stat_sheet("BAL", "KC", season=2026)
    assert sheet["matchup"] == "BAL @ KC"
    assert "section_1_away_offense_tendencies" in sheet
    assert "section_2_home_defense_vulnerabilities" in sheet
    assert "section_3_home_offense_tendencies" in sheet
    assert "section_4_away_defense_vulnerabilities" in sheet
    assert "section_5_similar_opponents" in sheet
    assert "section_6_head_to_head" in sheet
    assert "section_7_key_personnel" in sheet

    # Acceptance check: verify no prohibited model words in text descriptions
    def _check_no_prohibited(obj, path="root"):
        if isinstance(obj, str):
            lower_str = obj.lower()
            for bad in PROHIBITED_WORDS:
                # Disallow words when standalone or in predictions
                pattern = r"\b" + re.escape(bad) + r"\b"
                assert not re.search(pattern, lower_str), f"Found prohibited model word '{bad}' at {path}: '{obj}'"
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _check_no_prohibited(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, elem in enumerate(obj):
                _check_no_prohibited(elem, f"{path}[{i}]")

    # Run check across descriptive texts
    _check_no_prohibited(sheet["disclaimer"], "disclaimer")
    _check_no_prohibited(sheet["section_5_similar_opponents"]["home_defense_vs_opponents_like_away"]["descriptive_narrative"], "narrative")
    _check_no_prohibited(sheet["section_7_key_personnel"]["note"], "personnel_note")


def test_player_historical_red_zone_touches():
    """Verify compute_player_historical_box_stats computes non-zero red zone touches for known active players."""
    from backend.roster_verification import compute_player_historical_box_stats
    st = compute_player_historical_box_stats(
        player_name="Bijan Robinson",
        team="ATL",
        stat_focus="Rushing Yards",
        season=2026,
        up_to_week=5,
    )
    assert st is not None
    assert st["rz_20"] > 0
    assert st["rz_10"] > 0


def test_dynamic_scouting_data_date_range():
    """Verify dynamic scouting features recalculate when date range changes and properly suppress ranks for n < 6 games."""
    from backend.scheme_insights import get_dynamic_scouting_data, compute_scheme_insights

    # 1. Custom 6-week window in 2025
    dyn_early = get_dynamic_scouting_data(2025, start_date="2025-09-04", end_date="2025-10-15")
    assert "HOU" in dyn_early
    assert "JAX" in dyn_early
    assert len(dyn_early) == 32

    hou_early = dyn_early["HOU"]
    jax_early = dyn_early["JAX"]

    # HOU played 5 games by Oct 15 -> rank suppressed
    assert hou_early["n_games"] == 5
    assert hou_early["offense"]["passing_ypa"]["rank"] is None
    assert hou_early["offense"]["game_state_splits"]["leading"]["pass_rate"]["rank"] is None

    # JAX played 6 games by Oct 15 -> rank computed
    assert jax_early["n_games"] == 6
    assert jax_early["defense"]["passing_ypa_allowed"]["rank"] is not None
    assert 1 <= jax_early["defense"]["passing_ypa_allowed"]["rank"] <= 32

    # 2. Full 2025 season (17 games) -> all ranks active
    dyn_full = get_dynamic_scouting_data(2025, start_date="2025-09-01", end_date="2026-01-15")
    hou_full = dyn_full["HOU"]
    jax_full = dyn_full["JAX"]

    assert hou_full["n_games"] == 17
    assert hou_full["offense"]["passing_ypa"]["rank"] == 17
    assert jax_full["n_games"] == 17
    assert jax_full["defense"]["passing_ypa_allowed"]["rank"] == 7

    # 3. Scheme insights output incorporates date range dynamically
    res = compute_scheme_insights("JAX", "HOU", season=2025, start_date="2025-09-04", end_date="2025-10-15")
    assert res["sample_window"]["home_games_in_sample"] == 6
    assert res["sample_window"]["away_games_in_sample"] == 5
    sheet = res["matchup_stat_sheet"]
    assert sheet["section_1_away_offense_tendencies"]["n_games"] == 5
    assert sheet["section_2_home_defense_vulnerabilities"]["n_games"] == 6


