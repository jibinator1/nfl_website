"""
backend/roster_verification.py

Roster Verification & Player Historical Box Stats Engine
========================================================
1. Verifies every player name against the current season's weekly_cache.
   Ensures that any player displayed has at least one recorded snap or touch
   for the team being analyzed. Stale or traded player names are dropped.
2. Extracts pure historical box stats from player game logs (last 10 games:
   median, 25th percentile, 75th percentile, n games, red zone touches).
   No market lines, no split diff, no EV, no win probabilities.
"""
from __future__ import annotations

import os
import re
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd
import numpy as np

from backend.team_tendencies import load_weekly_stats

_ROSTER_CACHE: Dict[Tuple[int, str, Optional[int]], pd.DataFrame] = {}


def get_active_team_roster(
    team: str,
    season: int = 2026,
    up_to_week: Optional[int] = None,
) -> pd.DataFrame:
    """Returns players with at least 1 touch/snap for team in season up to up_to_week."""
    w = load_weekly_stats(season=season, up_to_week=up_to_week)
    if w.empty or "team" not in w.columns:
        return pd.DataFrame()

    team_norm = "LA" if team == "LAR" else team
    sub = w[w["team"].replace({"LAR": "LA"}) == team_norm].copy()
    if sub.empty:
        return pd.DataFrame()

    # Identify players with >= 1 touch
    carries = sub["carries"].fillna(0) if "carries" in sub.columns else pd.Series(0, index=sub.index)
    targets = sub["targets"].fillna(0) if "targets" in sub.columns else pd.Series(0, index=sub.index)
    pass_yds = sub["passing_yards"].fillna(0) if "passing_yards" in sub.columns else pd.Series(0, index=sub.index)

    active_mask = (carries > 0) | (targets > 0) | (pass_yds > 0)
    active_df = sub[active_mask]
    return active_df


def verify_player_on_roster(
    player_name: str,
    team: str,
    season: int = 2026,
    up_to_week: Optional[int] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Verifies that player_name appears in the specified season's weekly_cache
    for team with at least one touch/snap.
    Returns (is_verified, verified_display_name).
    """
    if not player_name or player_name.strip() in ("", "None", "Unknown"):
        return False, None

    active_df = get_active_team_roster(team=team, season=season, up_to_week=up_to_week)
    if active_df.empty:
        # Fallback to prior season active check if current season has 0 games recorded
        active_df = get_active_team_roster(team=team, season=season - 1)
        if active_df.empty:
            return False, None

    clean_target = re.sub(r"[^a-zA-Z]", "", player_name).lower()

    # 1. Exact normalized match on player_display_name or player_name
    for col in ["player_display_name", "player_name"]:
        if col in active_df.columns:
            matches = active_df[
                active_df[col].astype(str).str.replace(r"[^a-zA-Z]", "", regex=True).str.lower() == clean_target
            ]
            if not matches.empty:
                disp_col = "player_display_name" if "player_display_name" in matches.columns else col
                return True, str(matches[disp_col].iloc[0])

    # 2. Match by last name + first initial
    parts = player_name.replace(".", " ").split()
    if len(parts) >= 2:
        first_init = parts[0][0].lower()
        last_name = re.sub(r"[^a-zA-Z]", "", parts[-1]).lower()

        if "player_name" in active_df.columns:
            for _, row in active_df.iterrows():
                p_str = str(row["player_name"]).lower()
                clean_p = re.sub(r"[^a-zA-Z]", "", p_str)
                if clean_p.endswith(last_name) and clean_p.startswith(first_init):
                    disp = row.get("player_display_name", row["player_name"])
                    return True, str(disp)

    return False, None


def clean_or_drop_player(
    player_name: str,
    team: str,
    season: int = 2026,
    up_to_week: Optional[int] = None,
    fallback_placeholder: str = "No qualifying player identified",
) -> str:
    """Verifies player and returns display name, or fallback if unverified."""
    is_ok, disp_name = verify_player_on_roster(player_name, team, season=season, up_to_week=up_to_week)
    if is_ok and disp_name:
        return disp_name
    return fallback_placeholder


def compute_player_historical_box_stats(
    player_name: str,
    team: str,
    stat_focus: str = "Rushing Yards",
    season: int = 2026,
    up_to_week: Optional[int] = None,
    max_games: int = 10,
) -> Optional[Dict[str, Any]]:
    """
    Extracts pure historical box stats from player's game log:
    - Last 10 games: median, 25th percentile, 75th percentile, n games, range.
    - Coverage splits note: 'n=X games, sub-split not available.'
    - Red zone: touches inside 20, touches inside 10.
    Returns None if player is unverified or has 0 games.
    """
    is_ok, disp_name = verify_player_on_roster(player_name, team, season=season, up_to_week=up_to_week)
    if not is_ok or not disp_name:
        return None

    # Load player's games across current and prior season
    frames = []
    df_curr = load_weekly_stats(season=season, up_to_week=up_to_week)
    if not df_curr.empty:
        frames.append(df_curr)
    df_prev = load_weekly_stats(season=season - 1)
    if not df_prev.empty:
        frames.append(df_prev)

    if not frames:
        return None

    all_w = pd.concat(frames, ignore_index=True)
    team_norm = "LA" if team == "LAR" else team
    sub = all_w[all_w["team"].replace({"LAR": "LA"}) == team_norm].copy()
    if sub.empty:
        return None

    clean_target = re.sub(r"[^a-zA-Z]", "", disp_name).lower()
    matches = sub[
        sub["player_display_name"].astype(str).str.replace(r"[^a-zA-Z]", "", regex=True).str.lower() == clean_target
    ]
    if matches.empty:
        matches = sub[
            sub["player_name"].astype(str).str.replace(r"[^a-zA-Z]", "", regex=True).str.lower() == clean_target
        ]

    if matches.empty:
        return None

    # Sort descending by season and week and take up to max_games
    matches = matches.sort_values(by=["season", "week"], ascending=[False, False]).head(max_games)
    n_games = len(matches)
    if n_games == 0:
        return None

    # Determine metric series
    col_map = {
        "Rushing Yards": "rushing_yards",
        "Receiving Yards": "receiving_yards",
        "Receptions": "receptions",
        "Passing Yards": "passing_yards",
    }
    col = col_map.get(stat_focus, "rushing_yards")
    unit = "rec" if stat_focus == "Receptions" else "yds"

    series = matches[col].dropna() if col in matches.columns else pd.Series(dtype=float)
    if series.empty:
        med = 0.0
        p25 = 0.0
        p75 = 0.0
        min_val = 0.0
        max_val = 0.0
    else:
        med = round(float(series.median()), 1)
        p25 = round(float(series.quantile(0.25)), 1)
        p75 = round(float(series.quantile(0.75)), 1)
        min_val = round(float(series.min()), 1)
        max_val = round(float(series.max()), 1)

    # Red zone counts
    rz20_col = matches.get("rushing_20", pd.Series(0, index=matches.index)).fillna(0) + \
               matches.get("receiving_20", pd.Series(0, index=matches.index)).fillna(0)
    rz10_col = matches.get("rushing_10", pd.Series(0, index=matches.index)).fillna(0) + \
               matches.get("receiving_10", pd.Series(0, index=matches.index)).fillna(0)
    rz20_sum = int(rz20_col.sum())
    rz10_sum = int(rz10_col.sum())

    summary_str = f"Median: {med:.1f} {unit} (25th: {p25:.1f}, 75th: {p75:.1f}, range: {min_val:.1f}-{max_val:.1f}, n={n_games} games)"
    rz_summary_str = f"{rz20_sum} touches inside 20, {rz10_sum} inside 10 (n={n_games} games)"
    split_str = f"n={n_games} games, sub-split not available."

    return {
        "verified_name": disp_name,
        "n_games": n_games,
        "median": med,
        "p25": p25,
        "p75": p75,
        "min": min_val,
        "max": max_val,
        "unit": unit,
        "stat_focus": stat_focus,
        "summary_str": summary_str,
        "rz_20": rz20_sum,
        "rz_10": rz10_sum,
        "rz_summary": rz_summary_str,
        "splits_note": split_str,
    }
