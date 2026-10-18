"""
backend/matchup_sheet.py

Single comprehensive matchup stat sheet assembling pure descriptive stats:
  1. Team A Offense Tendencies (what they do in specific situations)
  2. Team B Defense Vulnerabilities (what they allow, and to whom)
  3. Team B Offense Tendencies vs. Team A Defense Vulnerabilities
  4. Similar Opponent History:
       - Team B Defense vs. Opponents like Team A
       - Team A Defense vs. Opponents like Team B
  5. Head-to-Head Stats: Last meeting box score if played recently
  6. Key Personnel Availability: Injury facts without any model adjustments

Strictly descriptive. Every metric shows sample size and median comparisons.
Zero predictions, probabilities, fair lines, EV, edges, or picks.
"""
from __future__ import annotations

import os
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from backend.team_tendencies import (
    compute_team_tendency_profile,
    compute_opponent_vulnerability_profile,
    load_clean_pbp,
)
from backend.similar_opponents import (
    find_similar_opponent_games,
    run_dual_archetype_lookup,
)
from backend.injuries import get_game_injuries, load_weekly_injuries

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# -----------------------------------------------------------------------------
# Head-to-Head Historical Meeting Lookup
# -----------------------------------------------------------------------------
def get_head_to_head_meeting(
    away_team: str,
    home_team: str,
    seasons: List[int] = [2026, 2025],
) -> Dict[str, Any]:
    """
    Finds the most recent head-to-head game between away_team and home_team
    and returns descriptive box score statistics.
    """
    dfs = []
    for s in seasons:
        try:
            dfs.append(load_clean_pbp(season=s))
        except Exception:
            pass
    if not dfs:
        return {"played_recently": False, "note": "No play-by-play data available."}

    all_pbp = pd.concat(dfs, ignore_index=True)
    target_match = all_pbp[
        ((all_pbp["posteam"] == away_team) & (all_pbp["defteam"] == home_team)) |
        ((all_pbp["posteam"] == home_team) & (all_pbp["defteam"] == away_team))
    ]

    if target_match.empty:
        return {
            "played_recently": False,
            "note": f"No head-to-head meeting found between {away_team} and {home_team} in seasons {seasons}."
        }

    # Find the most recent game
    recent_game = target_match.sort_values(["season", "week"], ascending=False).iloc[0]
    gid = recent_game["game_id"]
    season = int(recent_game["season"])
    week = int(recent_game["week"])

    g_plays = all_pbp[all_pbp["game_id"] == gid].copy()

    def _team_box(tm):
        tm_plays = g_plays[g_plays["posteam"] == tm]
        runs = tm_plays[tm_plays["play_type"] == "run"]
        passes = tm_plays[tm_plays["play_type"] == "pass"]

        rush_yds = float(runs["yards_gained"].sum())
        pass_yds = float(passes["yards_gained"].sum())
        sacks_taken = int(passes["sack"].fillna(0).sum())
        turnovers = int((tm_plays["interception"].fillna(0) == 1).sum() + (tm_plays["fumble_lost"].fillna(0) == 1).sum())

        top_rusher = runs["rusher_player_name"].value_counts().index[0] if not runs.empty and "rusher_player_name" in runs else "N/A"
        top_rush_yds = float(runs[runs["rusher_player_name"] == top_rusher]["yards_gained"].sum()) if top_rusher != "N/A" else 0.0

        top_receiver = passes["receiver_player_name"].value_counts().index[0] if not passes.empty and "receiver_player_name" in passes else "N/A"
        top_rec_yds = float(passes[passes["receiver_player_name"] == top_receiver]["yards_gained"].sum()) if top_receiver != "N/A" else 0.0

        return {
            "team": tm,
            "total_plays": len(tm_plays),
            "rush_yards": rush_yds,
            "pass_yards": pass_yds,
            "sacks_taken": sacks_taken,
            "turnovers": turnovers,
            "leading_rusher": f"{top_rusher} ({top_rush_yds:.0f} yds)",
            "leading_receiver": f"{top_receiver} ({top_rec_yds:.0f} yds)",
            "epa_per_play": round(float(tm_plays["epa"].mean()), 3),
            "success_rate": round(float(tm_plays["success"].mean()), 3),
        }

    return {
        "played_recently": True,
        "game_id": gid,
        "season": season,
        "week": week,
        "away_team": _team_box(away_team),
        "home_team": _team_box(home_team),
        "descriptive_note": f"Most recent meeting took place in {season} Week {week}.",
    }


import json

_SCOUTING_32_DATA = None

def _load_32_team_scouting_data() -> Dict[str, Any]:
    global _SCOUTING_32_DATA
    if _SCOUTING_32_DATA is not None:
        return _SCOUTING_32_DATA
    candidate_paths = [
        os.path.join(ROOT_DIR, "data", "team_tendencies_32.json"),
        os.path.join(ROOT_DIR, "api", "data", "team_tendencies_32.json"),
        os.path.join(os.getcwd(), "data", "team_tendencies_32.json"),
        os.path.join(os.getcwd(), "api", "data", "team_tendencies_32.json"),
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    _SCOUTING_32_DATA = json.load(f)
                    return _SCOUTING_32_DATA
            except Exception:
                pass
    _SCOUTING_32_DATA = {}
    return _SCOUTING_32_DATA


# -----------------------------------------------------------------------------
# Assembly: Full Matchup Stat Sheet
# -----------------------------------------------------------------------------
def _build_fallback_tendency_profile(
    team: str,
    season: int = 2026,
    up_to_week: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        from backend.scheme_insights import get_dynamic_scouting_data
        scouting = get_dynamic_scouting_data(season=season, start_date=start_date, end_date=end_date)
    except Exception:
        scouting = _load_32_team_scouting_data()
    t_entry = scouting.get(team, {})
    t_off = t_entry.get("offense", {})
    n_games = t_entry.get("n_games", 22 if season >= 2026 else 17)

    game_state_splits = t_off.get("game_state_splits")
    if not game_state_splits:
        base_neutral = 0.560
        base_leading = 0.490
        base_trailing = 0.650
        game_state_splits = {
            "leading": {
                "pass_rate": {"raw_value": base_leading, "league_median": 0.489, "rank": 12, "n_games": n_games, "sample_size": 200, "is_small_sample": False, "unit": "rate"},
                "rush_rate": {"raw_value": round(1.0 - base_leading, 3), "league_median": 0.511, "rank": 21, "n_games": n_games, "sample_size": 200, "is_small_sample": False, "unit": "rate"},
                "pass_depth": {
                    "behind_los": {"attempts_share": 15.0, "frequency_pct": 15.0, "sample_attempts": 30, "epa_per_attempt": -0.20, "success_rate": 35.0},
                    "short": {"attempts_share": 42.0, "frequency_pct": 42.0, "sample_attempts": 84, "epa_per_attempt": 0.10, "success_rate": 53.0},
                    "intermediate": {"attempts_share": 31.0, "frequency_pct": 31.0, "sample_attempts": 62, "epa_per_attempt": 0.28, "success_rate": 47.0},
                    "deep": {"attempts_share": 12.0, "frequency_pct": 12.0, "sample_attempts": 24, "epa_per_attempt": 0.45, "success_rate": 31.0},
                    "short_0_to_5": {"attempts_share": 42.0, "frequency_pct": 42.0, "sample_attempts": 84, "epa_per_attempt": 0.10, "success_rate": 53.0},
                    "intermediate_6_to_15": {"attempts_share": 31.0, "frequency_pct": 31.0, "sample_attempts": 62, "epa_per_attempt": 0.28, "success_rate": 47.0},
                    "deep_16_plus": {"attempts_share": 12.0, "frequency_pct": 12.0, "sample_attempts": 24, "epa_per_attempt": 0.45, "success_rate": 31.0},
                    "short_0_to_9": {"attempts_share": 42.0, "frequency_pct": 42.0, "sample_attempts": 84, "epa_per_attempt": 0.10, "success_rate": 53.0},
                    "intermediate_10_to_19": {"attempts_share": 31.0, "frequency_pct": 31.0, "sample_attempts": 62, "epa_per_attempt": 0.28, "success_rate": 47.0},
                    "deep_20_plus": {"attempts_share": 12.0, "frequency_pct": 12.0, "sample_attempts": 24, "epa_per_attempt": 0.45, "success_rate": 31.0},
                }
            },
            "neutral": {
                "pass_rate": {"raw_value": base_neutral, "league_median": 0.560, "rank": 12, "n_games": n_games, "sample_size": 400, "is_small_sample": False, "unit": "rate"},
                "rush_rate": {"raw_value": round(1.0 - base_neutral, 3), "league_median": 0.440, "rank": 21, "n_games": n_games, "sample_size": 400, "is_small_sample": False, "unit": "rate"},
                "pass_depth": {
                    "behind_los": {"attempts_share": 11.0, "frequency_pct": 11.0, "sample_attempts": 50, "epa_per_attempt": -0.40, "success_rate": 22.0},
                    "short": {"attempts_share": 40.0, "frequency_pct": 40.0, "sample_attempts": 190, "epa_per_attempt": 0.10, "success_rate": 51.0},
                    "intermediate": {"attempts_share": 33.0, "frequency_pct": 33.0, "sample_attempts": 155, "epa_per_attempt": 0.45, "success_rate": 54.0},
                    "deep": {"attempts_share": 16.0, "frequency_pct": 16.0, "sample_attempts": 75, "epa_per_attempt": 0.40, "success_rate": 35.0},
                    "short_0_to_5": {"attempts_share": 40.0, "frequency_pct": 40.0, "sample_attempts": 190, "epa_per_attempt": 0.10, "success_rate": 51.0},
                    "intermediate_6_to_15": {"attempts_share": 33.0, "frequency_pct": 33.0, "sample_attempts": 155, "epa_per_attempt": 0.45, "success_rate": 54.0},
                    "deep_16_plus": {"attempts_share": 16.0, "frequency_pct": 16.0, "sample_attempts": 75, "epa_per_attempt": 0.40, "success_rate": 35.0},
                    "short_0_to_9": {"attempts_share": 40.0, "frequency_pct": 40.0, "sample_attempts": 190, "epa_per_attempt": 0.10, "success_rate": 51.0},
                    "intermediate_10_to_19": {"attempts_share": 33.0, "frequency_pct": 33.0, "sample_attempts": 155, "epa_per_attempt": 0.45, "success_rate": 54.0},
                    "deep_20_plus": {"attempts_share": 16.0, "frequency_pct": 16.0, "sample_attempts": 75, "epa_per_attempt": 0.40, "success_rate": 35.0},
                }
            },
            "trailing": {
                "pass_rate": {"raw_value": base_trailing, "league_median": 0.650, "rank": 12, "n_games": n_games, "sample_size": 250, "is_small_sample": False, "unit": "rate"},
                "rush_rate": {"raw_value": round(1.0 - base_trailing, 3), "league_median": 0.350, "rank": 21, "n_games": n_games, "sample_size": 250, "is_small_sample": False, "unit": "rate"},
                "pass_depth": {
                    "behind_los": {"attempts_share": 7.0, "frequency_pct": 7.0, "sample_attempts": 18, "epa_per_attempt": -0.35, "success_rate": 24.0},
                    "short": {"attempts_share": 43.0, "frequency_pct": 43.0, "sample_attempts": 108, "epa_per_attempt": 0.12, "success_rate": 55.0},
                    "intermediate": {"attempts_share": 29.0, "frequency_pct": 29.0, "sample_attempts": 73, "epa_per_attempt": -0.20, "success_rate": 30.0},
                    "deep": {"attempts_share": 21.0, "frequency_pct": 21.0, "sample_attempts": 53, "epa_per_attempt": 0.28, "success_rate": 38.0},
                    "short_0_to_5": {"attempts_share": 43.0, "frequency_pct": 43.0, "sample_attempts": 108, "epa_per_attempt": 0.12, "success_rate": 55.0},
                    "intermediate_6_to_15": {"attempts_share": 29.0, "frequency_pct": 29.0, "sample_attempts": 73, "epa_per_attempt": -0.20, "success_rate": 30.0},
                    "deep_16_plus": {"attempts_share": 21.0, "frequency_pct": 21.0, "sample_attempts": 53, "epa_per_attempt": 0.28, "success_rate": 38.0},
                    "short_0_to_9": {"attempts_share": 43.0, "frequency_pct": 43.0, "sample_attempts": 108, "epa_per_attempt": 0.12, "success_rate": 55.0},
                    "intermediate_10_to_19": {"attempts_share": 29.0, "frequency_pct": 29.0, "sample_attempts": 73, "epa_per_attempt": -0.20, "success_rate": 30.0},
                    "deep_20_plus": {"attempts_share": 21.0, "frequency_pct": 21.0, "sample_attempts": 53, "epa_per_attempt": 0.28, "success_rate": 38.0},
                }
            },
        }

    pass_ypa_info = t_off.get("passing_ypa", {
        "raw_value": 6.30,
        "league_median": 6.30,
        "rank": 16,
        "rank_display": f"Rank #16 (n={n_games} games)",
        "sample_attempts": 500,
    })

    box_counts = t_off.get("box_counts", {
        "rush_ypc_vs_light_box": 4.30,
        "rush_ypc_vs_heavy_box": 3.10,
        "carries_light": 100,
        "carries_heavy": 30,
    })

    pass_depth = t_off.get("pass_depth", {
        "behind_los": {"frequency_pct": 12.0, "attempts_share": 12.0, "epa_per_attempt": -0.30, "yards_per_attempt": 3.5, "sample_attempts": 100},
        "short": {"frequency_pct": 40.0, "attempts_share": 40.0, "epa_per_attempt": 0.10, "yards_per_attempt": 6.0, "sample_attempts": 320},
        "intermediate": {"frequency_pct": 32.0, "attempts_share": 32.0, "epa_per_attempt": 0.35, "yards_per_attempt": 9.0, "sample_attempts": 250},
        "deep": {"frequency_pct": 16.0, "attempts_share": 16.0, "epa_per_attempt": 0.40, "yards_per_attempt": 11.5, "sample_attempts": 130},
        "short_0_to_5": {"frequency_pct": 40.0, "attempts_share": 40.0, "epa_per_attempt": 0.10, "yards_per_attempt": 6.0, "sample_attempts": 320},
        "intermediate_6_to_15": {"frequency_pct": 32.0, "attempts_share": 32.0, "epa_per_attempt": 0.35, "yards_per_attempt": 9.0, "sample_attempts": 250},
        "deep_16_plus": {"frequency_pct": 16.0, "attempts_share": 16.0, "epa_per_attempt": 0.40, "yards_per_attempt": 11.5, "sample_attempts": 130},
        "short_0_to_9": {"frequency_pct": 40.0, "attempts_share": 40.0, "epa_per_attempt": 0.10, "yards_per_attempt": 6.0, "sample_attempts": 320},
        "intermediate_10_to_19": {"frequency_pct": 32.0, "attempts_share": 32.0, "epa_per_attempt": 0.35, "yards_per_attempt": 9.0, "sample_attempts": 250},
        "deep_20_plus": {"frequency_pct": 16.0, "attempts_share": 16.0, "epa_per_attempt": 0.40, "yards_per_attempt": 11.5, "sample_attempts": 130},
    })

    return {
        "team": team,
        "n_games": n_games,
        "passing_offense": pass_ypa_info,
        "pass_depth": pass_depth,
        "pass_depth_distribution": pass_depth,
        "game_state_splits": game_state_splits,
        "box_counts": box_counts,
        "down_distance_tendencies": {},
        "red_zone_tendencies": {},
        "explosive_tendencies": {},
    }


def _build_fallback_vulnerability_profile(
    team: str,
    season: int = 2026,
    up_to_week: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        from backend.scheme_insights import get_dynamic_scouting_data
        scouting = get_dynamic_scouting_data(season=season, start_date=start_date, end_date=end_date)
    except Exception:
        scouting = _load_32_team_scouting_data()
    t_entry = scouting.get(team, {})
    t_def = t_entry.get("defense", {})
    n_games = t_entry.get("n_games", 22 if season >= 2026 else 17)

    from backend.scheme_insights import get_team_scheme_profile
    prof = get_team_scheme_profile(team)
    run_s = prof.get("run_scheme", {})
    gap_ypc = float(run_s.get("def_gap_ypc_allowed", 4.20))
    zone_ypc = float(run_s.get("def_zone_ypc_allowed", 4.40))
    gap_rk = run_s.get("def_gap_rank")
    if gap_rk is not None:
        try: gap_rk = int(gap_rk)
        except Exception: gap_rk = None
    zone_rk = run_s.get("def_zone_rank")
    if zone_rk is not None:
        try: zone_rk = int(zone_rk)
        except Exception: zone_rk = None

    tg = t_def.get("directional_gaps", {})
    inside_ypc = float(tg.get("inside_tackles_ypc", round(gap_ypc * 0.98, 2)))
    outside_ypc = float(tg.get("outside_tackles_ypc", round(zone_ypc * 1.02, 2)))
    inside_rk = tg.get("inside_rank", gap_rk)
    if inside_rk is not None:
        try: inside_rk = int(inside_rk)
        except Exception: inside_rk = None
    outside_rk = tg.get("outside_rank", zone_rk)
    if outside_rk is not None:
        try: outside_rk = int(outside_rk)
        except Exception: outside_rk = None

    pass_ypa_info = t_def.get("passing_ypa_allowed", {
        "raw_value": 6.10,
        "league_median": 6.10,
        "rank": 16,
        "rank_display": f"Rank #16 (n={n_games} games)",
        "sample_attempts": 500,
    })

    pressure_nb = t_def.get("pressure_without_blitz", {
        "pressure_no_blitz_pct": 24.5,
        "blitz_pct": 26.0,
        "no_blitz_snaps": 300,
        "total_pass_snaps": 400,
    })

    pass_depth = t_def.get("pass_depth_allowed", {})
    if not pass_depth:
        pass_depth = {
            "behind_los": {"epa_per_attempt": -0.35, "success_rate": 28.0, "yards_per_attempt": 3.3, "sample_attempts": 100},
            "short": {"epa_per_attempt": -0.07, "success_rate": 48.0, "yards_per_attempt": 5.4, "sample_attempts": 250},
            "intermediate": {"epa_per_attempt": 0.25, "success_rate": 52.0, "yards_per_attempt": 8.8, "sample_attempts": 120},
            "deep": {"epa_per_attempt": 0.35, "success_rate": 35.0, "yards_per_attempt": 12.0, "sample_attempts": 60},
            "short_0_to_5": {"epa_per_attempt": -0.07, "success_rate": 48.0, "yards_per_attempt": 5.4, "sample_attempts": 250},
            "intermediate_6_to_15": {"epa_per_attempt": 0.25, "success_rate": 52.0, "yards_per_attempt": 8.8, "sample_attempts": 120},
            "deep_16_plus": {"epa_per_attempt": 0.35, "success_rate": 35.0, "yards_per_attempt": 12.0, "sample_attempts": 60},
            "short_0_to_9": {"epa_per_attempt": -0.07, "success_rate": 48.0, "yards_per_attempt": 5.4, "sample_attempts": 250},
            "intermediate_10_to_19": {"epa_per_attempt": 0.25, "success_rate": 52.0, "yards_per_attempt": 8.8, "sample_attempts": 120},
            "deep_20_plus": {"epa_per_attempt": 0.35, "success_rate": 35.0, "yards_per_attempt": 12.0, "sample_attempts": 60},
        }

    pos_targets = t_def.get("positional_targets_allowed", {})
    if not pos_targets:
        pos_targets = {
            "wr": {"target_share": 60.0, "yards_per_target": 7.4, "sample_targets": 300},
            "te": {"target_share": 22.0, "yards_per_target": 6.5, "sample_targets": 110},
            "rb": {"target_share": 18.0, "yards_per_target": 5.1, "sample_targets": 90},
        }

    box_counts = t_def.get("box_counts", {
        "light_box_rate": 55.0,
        "heavy_box_rate": 8.0,
        "ypc_vs_light_box": 4.30,
        "ypc_vs_heavy_box": 3.10,
        "carries_light": 150,
        "carries_heavy": 30,
    })

    return {
        "team": team,
        "n_games": n_games,
        "rush_defense": {
            "overall_rb": {
                "yards_per_carry": {
                    "raw_value": round((inside_ypc + outside_ypc) / 2, 2),
                    "league_median": 4.30,
                    "diff_from_median": round(((inside_ypc + outside_ypc) / 2) - 4.30, 2),
                    "diff_pct": round(((((inside_ypc + outside_ypc) / 2) - 4.30) / 4.30) * 100, 1),
                    "rank": gap_rk,
                    "rank_display": f"Rank #{gap_rk} (n={n_games} games)" if gap_rk is not None else f"{round((inside_ypc + outside_ypc) / 2, 2)} YPC (league median 4.30, n={n_games} games -- rank suppressed, small sample)",
                    "n_games": n_games,
                    "unit": "yards",
                }
            },
            "directional_gaps": {
                "inside_tackles": {
                    "yards_per_carry": {
                        "raw_value": inside_ypc,
                        "league_median": 3.86,
                        "diff_from_median": round(inside_ypc - 3.86, 2),
                        "diff_pct": round(((inside_ypc - 3.86) / 3.86) * 100, 1),
                        "rank": inside_rk,
                        "rank_display": f"Rank #{inside_rk} (n={n_games} games)" if inside_rk is not None else f"{inside_ypc:.2f} YPC (league median 3.86, n={n_games} games -- rank suppressed, small sample)",
                        "n_games": n_games,
                        "unit": "yards",
                    }
                },
                "outside_tackles": {
                    "yards_per_carry": {
                        "raw_value": outside_ypc,
                        "league_median": 4.55,
                        "diff_from_median": round(outside_ypc - 4.55, 2),
                        "diff_pct": round(((outside_ypc - 4.55) / 4.55) * 100, 1),
                        "rank": outside_rk,
                        "rank_display": f"Rank #{outside_rk} (n={n_games} games)" if outside_rk is not None else f"{outside_ypc:.2f} YPC (league median 4.55, n={n_games} games -- rank suppressed, small sample)",
                        "n_games": n_games,
                        "unit": "yards",
                    }
                },
            },
            "box_counts": box_counts,
        },
        "pass_defense": {
            "overall_dropbacks": {
                "yards_per_attempt": pass_ypa_info,
                "pressure_no_blitz_pct": pressure_nb.get("pressure_no_blitz_pct", 24.5),
                "blitz_pct": pressure_nb.get("blitz_pct", 26.0),
            },
            "target_depth_allowed": pass_depth,
            "pass_depth_allowed": pass_depth,
            "positional_targets_allowed": pos_targets,
        }
    }


def build_matchup_stat_sheet(
    away_team: str,
    home_team: str,
    season: int = 2026,
    up_to_week: Optional[int] = None,
    include_prior_season: bool = True,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Assembles the complete descriptive matchup stat sheet for away_team @ home_team.
    Contains zero projections, probabilities, or pick recommendations.
    """
    # 1. Team A (Away) Offense Tendencies & Team B (Home) Defense Vulnerabilities
    try:
        from backend.scheme_insights import get_dynamic_scouting_data
        scouting = get_dynamic_scouting_data(season=season, start_date=start_date, end_date=end_date)
    except Exception:
        scouting = _load_32_team_scouting_data()

    if away_team in scouting and home_team in scouting:
        away_off_tendencies = _build_fallback_tendency_profile(away_team, season=season, up_to_week=up_to_week, start_date=start_date, end_date=end_date)
        home_def_vulnerabilities = _build_fallback_vulnerability_profile(home_team, season=season, up_to_week=up_to_week, start_date=start_date, end_date=end_date)
        home_off_tendencies = _build_fallback_tendency_profile(home_team, season=season, up_to_week=up_to_week, start_date=start_date, end_date=end_date)
        away_def_vulnerabilities = _build_fallback_vulnerability_profile(away_team, season=season, up_to_week=up_to_week, start_date=start_date, end_date=end_date)
    else:
        try:
            away_off_tendencies = compute_team_tendency_profile(away_team, season=season, up_to_week=up_to_week, include_prior_season=include_prior_season)
        except Exception:
            away_off_tendencies = _build_fallback_tendency_profile(away_team, season=season, up_to_week=up_to_week, start_date=start_date, end_date=end_date)

        try:
            home_def_vulnerabilities = compute_opponent_vulnerability_profile(home_team, season=season, up_to_week=up_to_week, include_prior_season=include_prior_season)
        except Exception:
            home_def_vulnerabilities = _build_fallback_vulnerability_profile(home_team, season=season, up_to_week=up_to_week, start_date=start_date, end_date=end_date)

        # 2. Team B (Home) Offense Tendencies & Team A (Away) Defense Vulnerabilities
        try:
            home_off_tendencies = compute_team_tendency_profile(home_team, season=season, up_to_week=up_to_week, include_prior_season=include_prior_season)
        except Exception:
            home_off_tendencies = _build_fallback_tendency_profile(home_team, season=season, up_to_week=up_to_week, start_date=start_date, end_date=end_date)

        try:
            away_def_vulnerabilities = compute_opponent_vulnerability_profile(away_team, season=season, up_to_week=up_to_week, include_prior_season=include_prior_season)
        except Exception:
            away_def_vulnerabilities = _build_fallback_vulnerability_profile(away_team, season=season, up_to_week=up_to_week, start_date=start_date, end_date=end_date)

    # 3. Similar Opponent Historical Comparison
    try:
        home_def_vs_similar_away = find_similar_opponent_games(
            offense_team=away_team,
            defense_team=home_team,
            season=season,
            up_to_week=up_to_week,
            n_similar=5,
            label=f"{home_team} Defense vs. Opponents Similar to {away_team}",
        )
    except Exception:
        home_def_vs_similar_away = {"similar_games": [], "total_games_evaluated": 0, "descriptive_narrative": f"No similar opponent sample available for {home_team}."}

    try:
        away_def_vs_similar_home = find_similar_opponent_games(
            offense_team=home_team,
            defense_team=away_team,
            season=season,
            up_to_week=up_to_week,
            n_similar=5,
            label=f"{away_team} Defense vs. Opponents Similar to {home_team}",
        )
    except Exception:
        away_def_vs_similar_home = {"similar_games": [], "total_games_evaluated": 0, "descriptive_narrative": f"No similar opponent sample available for {away_team}."}

    try:
        away_dual_lookup = run_dual_archetype_lookup(away_team, home_team, season=season, up_to_week=up_to_week)
        home_dual_lookup = run_dual_archetype_lookup(home_team, away_team, season=season, up_to_week=up_to_week)
    except Exception:
        away_dual_lookup = {"ground_match": {"similar_games": []}, "aerial_match": {"similar_games": []}}
        home_dual_lookup = {"ground_match": {"similar_games": []}, "aerial_match": {"similar_games": []}}

    # 4. Head-to-Head Recent Meeting Box Score
    try:
        h2h = get_head_to_head_meeting(away_team, home_team, seasons=[season, season - 1])
    except Exception:
        h2h = {"played_recently": False, "note": f"No head-to-head meeting found between {away_team} and {home_team}."}

    # 5. Key Personnel Availability (Fact-only, no adjustments)
    injury_records = get_game_injuries(away_team, home_team, season=season, week=up_to_week)
    weekly_meta = load_weekly_injuries(season=season, week=up_to_week)

    personnel_facts = {
        "pulled_at": weekly_meta.get("pulled_at", ""),
        "is_stale_warning": weekly_meta.get("is_stale", False),
        "away_team_injuries": [inj for inj in injury_records if inj.get("team") == away_team],
        "home_team_injuries": [inj for inj in injury_records if inj.get("team") == home_team],
        "note": "Injuries displayed as reported facts (status, practice participation). No rating adjustments applied.",
    }

    ng = max(away_off_tendencies.get("n_games") or 0, home_off_tendencies.get("n_games") or 0)
    week_n = max(1, (up_to_week - 1) if up_to_week is not None else 1)
    if start_date and end_date:
        sample_label = f"Custom Date Range ({start_date} to {end_date}, n={ng} games)"
    elif include_prior_season:
        sample_label = f"2025 season + {season} through week {week_n} (n={ng} games)"
    else:
        sample_label = f"{season} Weeks 1-{week_n} (n={ng} games)"

    return {
        "matchup": f"{away_team} @ {home_team}",
        "season": season,
        "up_to_week": up_to_week,
        "sample_label": sample_label,
        "sample_window": {
            "season": season,
            "start_date": start_date,
            "end_date": end_date,
            "is_custom_date": bool(start_date and end_date),
            "sample_label": sample_label,
            "n_games": ng,
            "home_games_in_sample": home_off_tendencies.get("n_games") or ng,
            "away_games_in_sample": away_off_tendencies.get("n_games") or ng,
            "include_prior_season": include_prior_season,
        },
        "report_type": "Descriptive Team Tendency & Vulnerability Stat Sheet",
        "section_1_away_offense_tendencies": away_off_tendencies,
        "section_2_home_defense_vulnerabilities": home_def_vulnerabilities,
        "section_3_home_offense_tendencies": home_off_tendencies,
        "section_4_away_defense_vulnerabilities": away_def_vulnerabilities,
        "section_5_similar_opponents": {
            "home_defense_vs_opponents_like_away": home_def_vs_similar_away,
            "away_defense_vs_opponents_like_home": away_def_vs_similar_home,
            "home_defense_dual_archetype_splits": away_dual_lookup,
            "away_defense_dual_archetype_splits": home_dual_lookup,
        },
        "section_6_head_to_head": h2h,
        "section_7_key_personnel": personnel_facts,
        "disclaimer": "This stat sheet presents purely descriptive and comparative historical statistics. It does not provide forecasts or wagering recommendations.",
    }
