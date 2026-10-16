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
import logging
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

from backend.team_tendencies import load_weekly_stats

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RZ_CACHE_PATH = os.path.join(ROOT_DIR, "data", "player_red_zone_cache.parquet")
_CACHED_RZ_DF: Optional[pd.DataFrame] = None
USAGE_CACHE_PATH = os.path.join(ROOT_DIR, "data", "player_usage_cache.parquet")
_CACHED_USAGE_DF: Optional[pd.DataFrame] = None
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

    # Load player's games: evaluate current season first so n matches the current season window
    df_curr = load_weekly_stats(season=season, up_to_week=up_to_week)
    all_w = df_curr if not df_curr.empty else load_weekly_stats(season=season - 1)
    if all_w.empty:
        return None

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

    # If no games found in current season, check prior season as fallback
    if matches.empty and not df_curr.empty:
        df_prev = load_weekly_stats(season=season - 1)
        if not df_prev.empty:
            sub_prev = df_prev[df_prev["team"].replace({"LAR": "LA"}) == team_norm].copy()
            matches = sub_prev[
                sub_prev["player_display_name"].astype(str).str.replace(r"[^a-zA-Z]", "", regex=True).str.lower() == clean_target
            ]
            if matches.empty:
                matches = sub_prev[
                    sub_prev["player_name"].astype(str).str.replace(r"[^a-zA-Z]", "", regex=True).str.lower() == clean_target
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

    # 1. Real Red Zone touches from precomputed cache or play-by-play (yardline_100 <= 20 and <= 10)
    rz20_sum = 0
    rz10_sum = 0
    try:
        global _CACHED_RZ_DF
        if _CACHED_RZ_DF is None and os.path.exists(RZ_CACHE_PATH):
            _CACHED_RZ_DF = pd.read_parquet(RZ_CACHE_PATH)

        if _CACHED_RZ_DF is not None and not _CACHED_RZ_DF.empty:
            # Fast vectorized key matching on season and week (replaces row-by-row apply)
            rz_keys = _CACHED_RZ_DF["season"].astype(int).astype(str) + "_" + _CACHED_RZ_DF["week"].astype(int).astype(str)
            target_keys = set(matches["season"].astype(int).astype(str) + "_" + matches["week"].astype(int).astype(str))
            sub_rz = _CACHED_RZ_DF[rz_keys.isin(target_keys)]
            player_ids = set(matches["player_id"].dropna().astype(str).unique().tolist()) if "player_id" in matches.columns else set()
            disp_names = set(matches["player_display_name"].dropna().astype(str).unique().tolist()) if "player_display_name" in matches.columns else set()
            disp_names.add(disp_name)

            clean_name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", disp_name, flags=re.IGNORECASE).strip()
            parts = clean_name.replace(".", " ").split()
            first_init = parts[0][0].upper() if parts else ""
            last_name = parts[-1].lower() if parts else ""

            p_mask = (
                sub_rz["player_id"].astype(str).isin(player_ids) |
                sub_rz["player_display_name"].isin(disp_names) |
                sub_rz["player_name"].isin(disp_names)
            )
            if not p_mask.any() and last_name and first_init:
                name_col = sub_rz["player_name"].fillna("").astype(str)
                p_mask = (name_col.str.lower().str.endswith(last_name)) & (name_col.str.startswith(first_init))

            matched_rz = sub_rz[p_mask]
            if not matched_rz.empty:
                rz20_sum = int(matched_rz["rz20"].sum())
                rz10_sum = int(matched_rz["rz10"].sum())
        else:
            from backend.team_tendencies import load_clean_pbp
            pbp_dfs = []
            for s in matches["season"].unique():
                try:
                    pbp_dfs.append(load_clean_pbp(season=int(s)))
                except Exception:
                    pass
            if pbp_dfs:
                pbp_comb = pd.concat(pbp_dfs, ignore_index=True)
                game_keys = set(zip(matches["season"].astype(int), matches["week"].astype(int)))
                pbp_sub = pbp_comb[pbp_comb.apply(lambda r: (int(r["season"]), int(r["week"])) in game_keys, axis=1)]

                player_ids = set(matches["player_id"].dropna().astype(str).unique().tolist()) if "player_id" in matches.columns else set()
                player_names = set(matches["player_name"].dropna().astype(str).unique().tolist()) if "player_name" in matches.columns else set()
                disp_names = set(matches["player_display_name"].dropna().astype(str).unique().tolist()) if "player_display_name" in matches.columns else set()
                disp_names.add(disp_name)

                clean_name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", disp_name, flags=re.IGNORECASE).strip()
                parts = clean_name.replace(".", " ").split()
                first_init = parts[0][0].upper() if parts else ""
                last_name = parts[-1].lower() if parts else ""

                # Rush plays: must be run play, matching either player_id, full/abbrev name, or first_init + clean last name
                rush_mask = (pbp_sub["play_type"] == "run")
                rush_id_mask = pbp_sub["rusher_player_id"].astype(str).isin(player_ids) if (player_ids and "rusher_player_id" in pbp_sub.columns) else pd.Series(False, index=pbp_sub.index)
                rush_name_col = pbp_sub["rusher_player_name"].fillna("")
                rush_name_mask = rush_name_col.isin(player_names | disp_names)
                rush_part_mask = (rush_name_col.str.lower().str.endswith(last_name)) & (rush_name_col.str.startswith(first_init)) if (last_name and first_init) else pd.Series(False, index=pbp_sub.index)

                rush_plays = pbp_sub[rush_mask & (rush_id_mask | rush_name_mask | rush_part_mask)]

                # Rec plays: completed pass, matching player_id, full/abbrev name, or first_init + clean last name
                rec_mask = (pbp_sub["complete_pass"] == 1)
                rec_id_mask = pbp_sub["receiver_player_id"].astype(str).isin(player_ids) if (player_ids and "receiver_player_id" in pbp_sub.columns) else pd.Series(False, index=pbp_sub.index)
                rec_name_col = pbp_sub["receiver_player_name"].fillna("")
                rec_name_mask = rec_name_col.isin(player_names | disp_names)
                rec_part_mask = (rec_name_col.str.lower().str.endswith(last_name)) & (rec_name_col.str.startswith(first_init)) if (last_name and first_init) else pd.Series(False, index=pbp_sub.index)

                rec_plays = pbp_sub[rec_mask & (rec_id_mask | rec_name_mask | rec_part_mask)]

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

    if series.empty:
        summary_str = f"No recorded stats (n={n_games} games)"
    else:
        summary_str = f"Median: {med:.1f} {unit} (25th: {p25:.1f}, 75th: {p75:.1f}, range: {min_val:.1f}-{max_val:.1f}, n={n_games} games)"
    rz_summary_str = f"Red zone touches: {rz20_sum} inside 20, {rz10_sum} inside 10 (n={n_games} games)"
    exp_summary_str = f"Explosive plays: {exp_10} gains of 10+ yds, {exp_20} of 20+ yds (n={n_games} games)"
    is_small_sample = n_games < 4

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
        "is_small_sample": is_small_sample,
        "small_sample_warning": "Small sample (n < 4 games)" if is_small_sample else None,
    }


def compute_player_usage_profile(
    player_name: str,
    team: str,
    season: int = 2026,
    up_to_week: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Computes per-game median usage metrics for a skill player from active games (snaps > 0).
    Descriptive only: snap share, carries, target share, WOPR.
    Returns None if player has no active games in the window.
    """
    global _CACHED_USAGE_DF
    if _CACHED_USAGE_DF is None:
        if os.path.exists(USAGE_CACHE_PATH):
            try:
                _CACHED_USAGE_DF = pd.read_parquet(USAGE_CACHE_PATH)
            except Exception as e:
                logger.warning(f"[Usage Profile] Failed reading usage cache from {USAGE_CACHE_PATH}: {e}")
                _CACHED_USAGE_DF = None
        else:
            logger.warning(
                f"[Usage Profile] Usage cache not found at {USAGE_CACHE_PATH}. "
                "Season usage profiles will be omitted from player cards."
            )
            return None

    if _CACHED_USAGE_DF is None or _CACHED_USAGE_DF.empty:
        return None

    df = _CACHED_USAGE_DF
    team_norm = "LA" if team == "LAR" else team

    # Filter season (with prior-season fallback if current has 0 records)
    sub = df[df["season"] == season].copy()
    if sub.empty:
        sub = df[df["season"] == season - 1].copy()
        if sub.empty:
            return None

    # Filter team
    sub = sub[sub["team"].replace({"LAR": "LA"}) == team_norm]
    if sub.empty:
        return None

    # Filter week
    if up_to_week is not None:
        sub = sub[sub["week"] <= up_to_week]
        if sub.empty:
            return None

    # Name matching (display name or short name)
    clean_target = re.sub(r"[^a-zA-Z]", "", player_name).lower()
    matches = sub[
        sub["player_display_name"].astype(str).str.replace(r"[^a-zA-Z]", "", regex=True).str.lower() == clean_target
    ]
    if matches.empty and "player_name" in sub.columns:
        matches = sub[
            sub["player_name"].astype(str).str.replace(r"[^a-zA-Z]", "", regex=True).str.lower() == clean_target
        ]
    if matches.empty:
        return None

    # Exclude inactive weeks (must have offense_snaps > 0)
    active_games = matches[matches["offense_snaps"] > 0].copy()
    n_games = len(active_games)
    if n_games == 0:
        return None

    pos = str(active_games["position"].iloc[0]).upper() if "position" in active_games.columns else "WR"

    # Per-game medians
    med_snap = round(float(active_games["offense_pct"].median() * 100))
    med_tgt_share = round(float(active_games["target_share"].median() * 100), 1)

    if pos == "RB":
        med_carries = round(float(active_games["carries"].median()), 1)
        carries_val = int(med_carries) if med_carries == int(med_carries) else med_carries
        lines = [
            {"label": "Snap share", "value": f"{med_snap}%"},
            {"label": "Carries", "value": str(carries_val)},
            {"label": "Target share", "value": f"{med_tgt_share:.1f}%"},
        ]
        raw_medians = {
            "snap_share": float(med_snap),
            "carries": float(carries_val),
            "target_share": float(med_tgt_share),
        }
    else:
        med_wopr = round(float(active_games["wopr"].median()), 2)
        lines = [
            {"label": "Snap share", "value": f"{med_snap}%"},
            {"label": "Target share", "value": f"{med_tgt_share:.1f}%"},
            {"label": "WOPR", "value": f"{med_wopr:.2f}"},
        ]
        raw_medians = {
            "snap_share": float(med_snap),
            "target_share": float(med_tgt_share),
            "wopr": float(med_wopr),
        }

    return {
        "n_games": n_games,
        "small_sample": n_games < 4,
        "position": pos,
        "lines": lines,
        "raw_medians": raw_medians,
    }

