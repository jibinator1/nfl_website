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

    # 1. Real Red Zone touches from play-by-play (yardline_100 <= 20 and <= 10)
    rz20_sum = 0
    rz10_sum = 0
    try:
        from backend.team_tendencies import load_clean_pbp
        pbp_dfs = []
        for s in matches["season"].unique():
            try:
                pbp_dfs.append(load_clean_pbp(season=int(s), up_to_week=up_to_week if int(s) == season else None))
            except Exception:
                pass
        if pbp_dfs:
            pbp_comb = pd.concat(pbp_dfs, ignore_index=True)
            game_keys = set(zip(matches["season"].astype(int), matches["week"].astype(int)))
            pbp_sub = pbp_comb[pbp_comb.apply(lambda r: (int(r["season"]), int(r["week"])) in game_keys, axis=1)]

            parts = disp_name.replace(".", " ").split()
            first_init = parts[0][0].upper()
            last_name = parts[-1].lower()

            rush_plays = pbp_sub[
                (pbp_sub["play_type"] == "run") &
                (pbp_sub["rusher_player_name"].fillna("").str.lower().str.endswith(last_name)) &
                (pbp_sub["rusher_player_name"].fillna("").str.startswith(first_init))
            ]
            rec_plays = pbp_sub[
                (pbp_sub["complete_pass"] == 1) &
                (pbp_sub["receiver_player_name"].fillna("").str.lower().str.endswith(last_name)) &
                (pbp_sub["receiver_player_name"].fillna("").str.startswith(first_init))
            ]

            rz20_sum = int((rush_plays["yardline_100"] <= 20).sum() + (rec_plays["yardline_100"] <= 20).sum())
            rz10_sum = int((rush_plays["yardline_100"] <= 10).sum() + (rec_plays["yardline_100"] <= 10).sum())
    except Exception:
        rz20_sum = 0
        rz10_sum = 0

    # 2. Explosive plays (10+ yards and 20+ yards) from weekly box stats
    exp_10 = int(matches.get("rushing_10", pd.Series(0, index=matches.index)).fillna(0).sum() +
                 matches.get("receiving_10", pd.Series(0, index=matches.index)).fillna(0).sum())
    exp_20 = int(matches.get("rushing_20", pd.Series(0, index=matches.index)).fillna(0).sum() +
                 matches.get("receiving_20", pd.Series(0, index=matches.index)).fillna(0).sum())

    summary_str = f"Median: {med:.1f} {unit} (25th: {p25:.1f}, 75th: {p75:.1f}, range: {min_val:.1f}-{max_val:.1f}, n={n_games} games)"
    rz_summary_str = f"Red zone touches: {rz20_sum} inside 20, {rz10_sum} inside 10 (n={n_games} games)"
    exp_summary_str = f"Explosive plays: {exp_10} gains of 10+ yds, {exp_20} of 20+ yds (n={n_games} games)"

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
        "exp_10": exp_10,
        "exp_20": exp_20,
        "exp_summary": exp_summary_str,
        "splits_note": None,  # Omit filler when sub-split is not available
    }


def format_injury_line(player_row: dict) -> str | None:
    """
    Formats a clean non-null injury string for display with team tags.
    Ensures 'Status = None' and 'Injury = None' never render.
    Suppresses redundant practice status when game status is Out, Doubtful, or IR.
    Excludes fully cleared players who have no game status designation and full practice participation.
    """
    status = str(player_row.get("report_status") or player_row.get("status") or "").strip()
    practice = str(player_row.get("practice_status") or player_row.get("practice") or "").strip()
    injury = str(player_row.get("report_primary_injury") or player_row.get("primary_injury") or player_row.get("injury") or "").strip()

    status = "" if status in ("None", "null", "nan") else status
    practice = "" if practice in ("None", "null", "nan") else practice
    injury = "" if injury in ("None", "null", "nan", "Undisclosed") else injury

    # Exclude healthy full-practice players
    if not status and (practice in ("Full Participation in Practice", "Full", "") or "full participation" in practice.lower()):
        return None

    # Only show practice status if game status is Questionable or unassigned
    details = []
    if status:
        details.append(status)
    if practice and status not in ("Out", "Doubtful", "IR"):
        # Shorten verbose NFL strings
        short_prac = practice.replace(" Participation in Practice", "").replace(" In Practice", "").replace("in Practice", "").strip()
        details.append(short_prac)

    detail_str = ", ".join(details)
    if injury:
        detail_str = f"{detail_str} ({injury})" if detail_str else f"({injury})"

    if not detail_str:
        return None

    name = player_row.get("player_name") or player_row.get("player") or "Unknown Player"
    pos = player_row.get("position") or player_row.get("pos") or ""
    team = player_row.get("team") or player_row.get("team_abbr") or ""
    team_prefix = f"{team} - " if team else ""
    return f"{name} ({team_prefix}{pos}): {detail_str}"


def get_roster_profiles(
    team: str,
    season: int = 2026,
    up_to_week: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Returns player usage profiles for team up to up_to_week,
    calculating target share, rush share, and red zone touch share.
    Alpha threshold: target_share >= 22%, rush_share >= 45%, or rz_touch_share >= 35%.
    """
    roster_df = get_active_team_roster(team=team, season=season, up_to_week=up_to_week)
    if roster_df.empty:
        return []

    tot_carries = float(roster_df["carries"].sum()) if "carries" in roster_df.columns else 0.0
    tot_targets = float(roster_df["targets"].sum()) if "targets" in roster_df.columns else 0.0

    # Red zone totals from PBP
    rz_touches_by_player: Dict[str, int] = {}
    rz_tot = 0
    try:
        from backend.team_tendencies import load_clean_pbp
        pbp = load_clean_pbp(season=season, up_to_week=up_to_week)
        if not pbp.empty:
            tm_rz = pbp[(pbp["posteam"] == team) & (pbp["yardline_100"] <= 20)]
            rz_tot = len(tm_rz)
            for _, r_play in tm_rz.iterrows():
                if r_play["play_type"] == "run" and pd.notna(r_play.get("rusher_player_name")):
                    p_n = str(r_play["rusher_player_name"])
                    rz_touches_by_player[p_n] = rz_touches_by_player.get(p_n, 0) + 1
                elif r_play["play_type"] == "pass" and r_play.get("complete_pass") == 1 and pd.notna(r_play.get("receiver_player_name")):
                    p_n = str(r_play["receiver_player_name"])
                    rz_touches_by_player[p_n] = rz_touches_by_player.get(p_n, 0) + 1
    except Exception:
        pass

    profiles = []
    name_col = "player_display_name" if "player_display_name" in roster_df.columns else "player_name"
    for p_name, p_group in roster_df.groupby(name_col):
        carries = float(p_group["carries"].sum()) if "carries" in p_group.columns else 0.0
        targets = float(p_group["targets"].sum()) if "targets" in p_group.columns else 0.0
        pos = p_group["position"].iloc[-1] if "position" in p_group.columns else "Unknown"

        rush_share = carries / tot_carries if tot_carries > 0 else 0.0
        target_share = targets / tot_targets if tot_targets > 0 else 0.0

        # Match RZ touches by name
        rz_touches = 0
        p_name_str = str(p_name)
        parts = p_name_str.split()
        last_name = parts[-1].lower() if parts else ""
        first_init = parts[0][0].upper() if parts else ""
        for rz_k, rz_v in rz_touches_by_player.items():
            if rz_k.lower().endswith(last_name) and rz_k.startswith(first_init):
                rz_touches += rz_v

        rz_touch_share = rz_touches / rz_tot if rz_tot > 0 else 0.0
        is_alpha = bool(target_share >= 0.22 or rush_share >= 0.45 or rz_touch_share >= 0.35)

        profiles.append({
            "player": p_name_str,
            "player_display_name": p_name_str,
            "team": team,
            "pos": pos,
            "carries": carries,
            "targets": targets,
            "rush_share": round(rush_share, 3),
            "target_share": round(target_share, 3),
            "rz_touches": rz_touches,
            "rz_touch_share": round(rz_touch_share, 3),
            "is_alpha": is_alpha,
        })

    profiles.sort(key=lambda x: (x["is_alpha"], x["target_share"] + x["rush_share"]), reverse=True)
    return profiles

