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


# -----------------------------------------------------------------------------
# Assembly: Full Matchup Stat Sheet
# -----------------------------------------------------------------------------
def build_matchup_stat_sheet(
    away_team: str,
    home_team: str,
    season: int = 2026,
    up_to_week: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Assembles the complete descriptive matchup stat sheet for away_team @ home_team.
    Contains zero projections, probabilities, or pick recommendations.
    """
    # 1. Team A (Away) Offense Tendencies & Team B (Home) Defense Vulnerabilities
    away_off_tendencies = compute_team_tendency_profile(away_team, season=season, up_to_week=up_to_week)
    home_def_vulnerabilities = compute_opponent_vulnerability_profile(home_team, season=season, up_to_week=up_to_week)

    # 2. Team B (Home) Offense Tendencies & Team A (Away) Defense Vulnerabilities
    home_off_tendencies = compute_team_tendency_profile(home_team, season=season, up_to_week=up_to_week)
    away_def_vulnerabilities = compute_opponent_vulnerability_profile(away_team, season=season, up_to_week=up_to_week)

    # 3. Similar Opponent Historical Comparison
    # Home defense vs. opponents like Away offense
    home_def_vs_similar_away = find_similar_opponent_games(
        offense_team=away_team,
        defense_team=home_team,
        season=season,
        up_to_week=up_to_week,
        n_similar=5,
        label=f"{home_team} Defense vs. Opponents Similar to {away_team}",
    )

    # Away defense vs. opponents like Home offense
    away_def_vs_similar_home = find_similar_opponent_games(
        offense_team=home_team,
        defense_team=away_team,
        season=season,
        up_to_week=up_to_week,
        n_similar=5,
        label=f"{away_team} Defense vs. Opponents Similar to {home_team}",
    )

    # Dual-split archetype matches (ground-heavy vs aerial-heavy)
    away_dual_lookup = run_dual_archetype_lookup(away_team, home_team, season=season, up_to_week=up_to_week)
    home_dual_lookup = run_dual_archetype_lookup(home_team, away_team, season=season, up_to_week=up_to_week)

    # 4. Head-to-Head Recent Meeting Box Score
    h2h = get_head_to_head_meeting(away_team, home_team, seasons=[season, season - 1])

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

    pbp_cur = load_clean_pbp(season=season, up_to_week=up_to_week)
    if not pbp_cur.empty and "week" in pbp_cur.columns and pbp_cur["week"].notna().any():
        max_wk = int(pbp_cur["week"].max())
        n_gm = int(pbp_cur[pbp_cur["posteam"].isin([away_team, home_team])]["game_id"].nunique()) // 2
        sample_label = f"Sample Window: {season} Weeks 1–{max_wk} (n={n_gm} games)"
    else:
        max_wk = 0
        n_gm = 0
        sample_label = f"Sample Window: {season} (n=0 games)"

    sample_window_data = {
        "sample_label": sample_label,
        "up_to_week": up_to_week,
        "max_week_included": max_wk,
        "n_games": n_gm,
        "label": sample_label,
    }

    return {
        "matchup": f"{away_team} @ {home_team}",
        "season": season,
        "up_to_week": up_to_week,
        "sample_window": sample_window_data,
        "sample_label": sample_label,
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


def build_matchup_sheet(
    away_team: str,
    home_team: str,
    season: int = 2026,
    up_to_week: Optional[int] = None,
) -> Dict[str, Any]:
    """Alias for build_matchup_stat_sheet."""
    return build_matchup_stat_sheet(
        away_team=away_team,
        home_team=home_team,
        season=season,
        up_to_week=up_to_week,
    )

