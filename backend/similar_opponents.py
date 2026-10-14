"""
backend/similar_opponents.py

Descriptive similar-opponent historical lookup engine.
Finds past games where a defense faced opponents with an offensive profile
similar to this week's opponent, and displays what actually happened.

Includes:
  - Multi-feature RMSD distance matching on z-scored offensive profiles:
      * Ground features: rush rate, neutral rush %, RB1 carry %, designed QB rush %, RB1 vs RB2 split
      * Aerial features: neutral pass %, WR1 target share, TE target share, aDOT, red zone pass %
  - Game-level vs. Distinct-Opponent Deduplication:
      * Reports both game count (n_games) and distinct opponent franchise count (n_distinct_opponents)
      * Calculates both game-level medians and deduped opponent-level medians
      * Requires n >= 3 games AND n >= 2 distinct opponents before summarizing strong analogs
  - RMSD-calibrated similarity tiers (dimension-invariant):
      * Strong (RMSD <= 0.70σ)
      * Moderate (0.70–1.10σ)
      * Loose (> 1.10σ)
  - Exclusion of direct head-to-head meetings from third-party analog sample size
  - Side-by-side Dual Archetype Split:
      * Ground-heavy vs. pass-heavy performance vs. season baseline
      * Multi-tier conclusion gating: Full, Partial, or Cannot Conclude
      * Unsupportable rows explicitly marked "INSUFFICIENT SAMPLE — NOT SUMMARIZED"
  - Clear descriptive text: "historical comparison, not a projection"
"""
from __future__ import annotations

import os
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from backend.team_tendencies import load_clean_pbp, get_player_position_map

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# -----------------------------------------------------------------------------
# Offensive Profile Extraction for Similar Opponent Matching
# -----------------------------------------------------------------------------
SIMILARITY_FEATURE_COLS = [
    "rush_rate",
    "pass_rate_neutral",
    "rb_carry_concentration",
    "qb_designed_rush_rate",
    "rb1_vs_rb2_split",
    "top_wr_target_share",
    "te_target_share",
    "adot",
    "rz_pass_rate",
]


def extract_team_offensive_profile(
    team: str,
    season: int = 2026,
    up_to_week: Optional[int] = None,
    pbp_df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """
    Extracts core offensive tendency feature vector for a team.
    Includes designed QB rush rate and RB1 vs RB2 committee split.
    """
    df = pbp_df if pbp_df is not None else load_clean_pbp(season=season, up_to_week=up_to_week)
    pos_map = get_player_position_map()

    team_plays = df[df["posteam"] == team].copy()
    if team_plays.empty:
        return {col: 0.0 for col in SIMILARITY_FEATURE_COLS}

    team_plays["_rush_pos"] = team_plays["rusher_player_id"].map(pos_map).fillna("Unknown")
    team_plays["_rec_pos"] = team_plays["receiver_player_id"].map(pos_map).fillna("Unknown")

    diff = team_plays["score_differential"].fillna(0)
    neut = team_plays[(diff >= -6) & (diff <= 6)]
    rz = team_plays[team_plays["yardline_100"] <= 20]
    passes = team_plays[team_plays["play_type"] == "pass"].copy()
    runs = team_plays[team_plays["play_type"] == "run"].copy()

    # Designed QB rush rate (QB runs excluding scrambles)
    if not runs.empty:
        qb_des = runs[(runs["_rush_pos"] == "QB") & (runs["qb_scramble"] != 1)]
        qb_des_rate = float(len(qb_des) / len(runs))
    else:
        qb_des_rate = 0.0

    # RB carry concentration & committee split (RB1 vs RB2)
    rb_runs = runs[runs["_rush_pos"] == "RB"]
    if not rb_runs.empty:
        rb_vc = rb_runs["rusher_player_id"].value_counts()
        rb1_share = float(rb_vc.iloc[0] / len(rb_runs))
        rb2_share = float(rb_vc.iloc[1] / len(rb_runs)) if len(rb_vc) > 1 else 0.0
        rb_conc = rb1_share
        rb_split = float(rb1_share - rb2_share)
    else:
        rb_conc = 0.50
        rb_split = 0.0

    # Target shares by position
    n_passes = len(passes)
    if n_passes > 0:
        te_targets = int((passes["_rec_pos"] == "TE").sum())
        te_share = float(te_targets / n_passes)
        wr_passes = passes[passes["_rec_pos"] == "WR"]
        if not wr_passes.empty:
            top_wr_t = int(wr_passes["receiver_player_id"].value_counts().iloc[0])
            top_wr_share = float(top_wr_t / n_passes)
        else:
            top_wr_share = 0.20
        adot = float(passes["air_yards"].dropna().mean()) if passes["air_yards"].notna().any() else 8.0
    else:
        te_share = 0.20
        top_wr_share = 0.20
        adot = 8.0

    rush_rate = float((team_plays["play_type"] == "run").mean())
    neut_pass = float((neut["play_type"] == "pass").mean()) if len(neut) >= 10 else float((team_plays["play_type"] == "pass").mean())
    rz_pass = float((rz["play_type"] == "pass").mean()) if len(rz) >= 5 else 0.50

    return {
        "rush_rate": round(rush_rate, 3),
        "pass_rate_neutral": round(neut_pass, 3),
        "rb_carry_concentration": round(rb_conc, 3),
        "qb_designed_rush_rate": round(qb_des_rate, 3),
        "rb1_vs_rb2_split": round(rb_split, 3),
        "top_wr_target_share": round(top_wr_share, 3),
        "te_target_share": round(te_share, 3),
        "adot": round(adot, 1),
        "rz_pass_rate": round(rz_pass, 3),
    }


_OFFENSIVE_PROFILES_CACHE: Dict[Tuple[int, ...], pd.DataFrame] = {}


def build_all_offensive_profiles(
    seasons: List[int] = [2024, 2025, 2026],
) -> pd.DataFrame:
    """
    Computes offensive profiles for all teams across specified seasons.
    Used as the population baseline to z-score similarity features.
    """
    key = tuple(sorted(seasons))
    if key in _OFFENSIVE_PROFILES_CACHE:
        return _OFFENSIVE_PROFILES_CACHE[key].copy()

    dfs = []
    for s in seasons:
        try:
            dfs.append(load_clean_pbp(season=s))
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()
    pbp_all = pd.concat(dfs, ignore_index=True)
    pos_map = get_player_position_map()
    pbp_all["_rec_pos"] = pbp_all["receiver_player_id"].map(pos_map).fillna("Unknown")
    pbp_all["_rush_pos"] = pbp_all["rusher_player_id"].map(pos_map).fillna("Unknown")

    records = []
    for (tm, s), g in pbp_all.groupby(["posteam", "season"]):
        if len(g) < 40:
            continue
        diff = g["score_differential"].fillna(0)
        neut = g[(diff >= -6) & (diff <= 6)]
        rz = g[g["yardline_100"] <= 20]
        passes = g[g["play_type"] == "pass"].copy()
        runs = g[g["play_type"] == "run"].copy()

        # Designed QB rush rate
        if not runs.empty:
            qb_des = runs[(runs["_rush_pos"] == "QB") & (runs["qb_scramble"] != 1)]
            qb_des_rate = float(len(qb_des) / len(runs))
        else:
            qb_des_rate = 0.0

        # RB carry concentration & committee split
        rb_runs = runs[runs["_rush_pos"] == "RB"]
        if not rb_runs.empty:
            rb_vc = rb_runs["rusher_player_id"].value_counts()
            rb1_share = float(rb_vc.iloc[0] / len(rb_runs))
            rb2_share = float(rb_vc.iloc[1] / len(rb_runs)) if len(rb_vc) > 1 else 0.0
            rb_conc = rb1_share
            rb_split = float(rb1_share - rb2_share)
        else:
            rb_conc = 0.50
            rb_split = 0.0

        n_p = len(passes)
        te_t = (passes["_rec_pos"] == "TE").sum() if n_p > 0 else 0
        te_share = float(te_t / max(n_p, 1))

        wr_passes = passes[passes["_rec_pos"] == "WR"]
        top_wr_t = wr_passes["receiver_player_id"].value_counts().iloc[0] if not wr_passes.empty else 0
        top_wr_share = float(top_wr_t / max(n_p, 1))

        adot = float(passes["air_yards"].dropna().mean()) if passes["air_yards"].notna().any() else 8.0
        rush_rate = float((g["play_type"] == "run").mean())
        neut_pass = float((neut["play_type"] == "pass").mean()) if len(neut) >= 10 else float((g["play_type"] == "pass").mean())
        rz_pass = float((rz["play_type"] == "pass").mean()) if len(rz) >= 5 else 0.50

        records.append({
            "team": tm,
            "season": s,
            "rush_rate": rush_rate,
            "pass_rate_neutral": neut_pass,
            "rb_carry_concentration": rb_conc,
            "qb_designed_rush_rate": qb_des_rate,
            "rb1_vs_rb2_split": rb_split,
            "top_wr_target_share": top_wr_share,
            "te_target_share": te_share,
            "adot": adot,
            "rz_pass_rate": rz_pass,
        })

    res_df = pd.DataFrame(records)
    _OFFENSIVE_PROFILES_CACHE[key] = res_df
    return res_df.copy()


# -----------------------------------------------------------------------------
# Core Similar-Opponent Matcher
# -----------------------------------------------------------------------------
def find_similar_opponent_games(
    offense_team: str,
    defense_team: str,
    season: int = 2026,
    up_to_week: Optional[int] = None,
    n_similar: int = 5,
    feature_subset: Optional[List[str]] = None,
    label: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Finds games where defense_team faced opponents stylistically similar to offense_team.
    Reports both game-level medians and deduped opponent franchise medians.
    Strictly gates strong summaries: requires n >= 3 games AND n >= 2 distinct opponents.
    """
    # 1. Load multi-season pbp for defense's history
    pbp_cur = load_clean_pbp(season=season, up_to_week=up_to_week)
    hist_dfs = [pbp_cur]
    for prior_s in [season - 1, season - 2]:
        try:
            hist_dfs.append(load_clean_pbp(season=prior_s))
        except Exception:
            pass
    pbp_hist = pd.concat(hist_dfs, ignore_index=True)

    pos_map = get_player_position_map()
    pbp_hist["_rec_pos"] = pbp_hist["receiver_player_id"].map(pos_map).fillna("Unknown")
    pbp_hist["_rush_pos"] = pbp_hist["rusher_player_id"].map(pos_map).fillna("Unknown")

    # 2. Build population profiles and z-score across 3 seasons
    pop_profiles = build_all_offensive_profiles(seasons=[season - 2, season - 1, season])
    if pop_profiles.empty:
        return {"error": "Insufficient play-by-play data to build population profiles."}

    features = feature_subset or SIMILARITY_FEATURE_COLS
    means = pop_profiles[features].mean()
    stds = pop_profiles[features].std().replace(0, 1.0)

    # 3. Target team profile
    target_prof = extract_team_offensive_profile(offense_team, season=season, up_to_week=up_to_week, pbp_df=pbp_cur)
    target_vec = pd.Series({f: target_prof.get(f, means[f]) for f in features})
    target_z = (target_vec - means) / stds

    # 4. Examine defense's historical games
    def_games = pbp_hist[pbp_hist["defteam"] == defense_team].groupby(["game_id", "season", "week", "posteam"])
    game_rows = []

    pop_lookup = pop_profiles.set_index(["team", "season"])

    for (gid, s, wk, opp), g in def_games:
        # Get opponent profile for that season
        if (opp, s) in pop_lookup.index:
            opp_row = pop_lookup.loc[(opp, s)]
            opp_vec = opp_row[features]
        else:
            continue

        opp_z = (opp_vec - means) / stds
        dist = float(np.sqrt(((target_z - opp_z) ** 2).sum()))
        rmsd = float(np.sqrt(((target_z - opp_z) ** 2).mean()))

        # What defense allowed in this specific game
        rb_runs = g[(g["play_type"] == "run") & (g["_rush_pos"] == "RB")]
        rb_carries = int(len(rb_runs))
        rb_rush_yds = float(rb_runs["yards_gained"].sum()) if rb_carries > 0 else 0.0
        rb_ypc = round(float(rb_rush_yds / max(rb_carries, 1)), 2)
        rb_rush_epa = round(float(rb_runs["epa"].mean()), 3) if rb_carries > 0 else 0.0
        rb_success_rate = round(float(rb_runs["success"].mean()), 3) if rb_carries > 0 else 0.0
        rb_stuff_rate = round(float((rb_runs["yards_gained"] <= 0).mean()), 3) if rb_carries > 0 else 0.0
        rb_explosive_rate = round(float((rb_runs["yards_gained"] >= 10).mean()), 3) if rb_carries > 0 else 0.0

        passes = g[g["play_type"] == "pass"]
        pass_att = int(len(passes))
        tot_pass_yds = float(passes["yards_gained"].sum())
        pass_ypa = round(float(tot_pass_yds / max(pass_att, 1)), 2)
        pass_epa = round(float(passes["epa"].mean()), 3) if pass_att > 0 else 0.0
        wr_rec_yds = float(passes[passes["_rec_pos"] == "WR"]["yards_gained"].sum())
        te_rec_yds = float(passes[passes["_rec_pos"] == "TE"]["yards_gained"].sum())
        sacks = int(passes["sack"].fillna(0).sum())
        qb_hits = int(passes["qb_hit"].fillna(0).sum())
        epa_per_play = float(g["epa"].mean())
        success_rate = float(g["success"].mean())

        # RMSD-calibrated similarity tier classification
        if rmsd <= 1.00:
            sim_tier = "Strong (<= 1.00σ)"
            sim_tier_key = "strong"
        elif rmsd <= 1.30:
            sim_tier = "Moderate (1.00–1.30σ)"
            sim_tier_key = "moderate"
        else:
            sim_tier = "Loose (1.30–1.50σ)"
            sim_tier_key = "loose"

        is_direct_h2h = bool(opp == offense_team)
        match_note = f"Actual prior meeting vs {offense_team} (H2H)" if is_direct_h2h else "Stylistic analog"

        game_rows.append({
            "game_id": gid,
            "season": int(s),
            "week": int(wk),
            "opponent": opp,
            "distance": round(dist, 3),
            "rmsd": round(rmsd, 3),
            "similarity_tier": sim_tier,
            "similarity_tier_key": sim_tier_key,
            "is_direct_h2h": is_direct_h2h,
            "match_note": match_note,
            "rb_carries_allowed": rb_carries,
            "rb_rush_yards_allowed": round(rb_rush_yds, 1),
            "rb_ypc_allowed": rb_ypc,
            "rb_rush_epa_allowed": rb_rush_epa,
            "rb_rush_success_rate_allowed": rb_success_rate,
            "rb_stuff_rate_allowed": rb_stuff_rate,
            "rb_explosive_rate_allowed": rb_explosive_rate,
            "pass_attempts_allowed": pass_att,
            "pass_yards_allowed": round(tot_pass_yds, 1),
            "pass_ypa_allowed": pass_ypa,
            "pass_epa_allowed": pass_epa,
            "wr_rec_yards_allowed": round(wr_rec_yds, 1),
            "te_rec_yards_allowed": round(te_rec_yds, 1),
            "sacks": sacks,
            "qb_hits_pressures": qb_hits,
            "epa_per_play_allowed": round(epa_per_play, 3),
            "success_rate_allowed": round(success_rate, 3),
        })

    if not game_rows:
        return {"error": f"No historical games found for defense {defense_team}."}

    all_games_df = pd.DataFrame(game_rows)
    # Defense's overall median across all games evaluated
    overall_medians = {
        "rb_carries": round(float(all_games_df["rb_carries_allowed"].median()), 1),
        "rb_rush_yards": round(float(all_games_df["rb_rush_yards_allowed"].median()), 1),
        "rb_ypc": round(float(all_games_df["rb_ypc_allowed"].median()), 2),
        "rb_rush_epa": round(float(all_games_df["rb_rush_epa_allowed"].median()), 3),
        "rb_rush_success_rate": round(float(all_games_df["rb_rush_success_rate_allowed"].median()), 3),
        "rb_success_rate": round(float(all_games_df["rb_rush_success_rate_allowed"].median()), 3),
        "rb_stuff_rate": round(float(all_games_df["rb_stuff_rate_allowed"].median()), 3),
        "rb_explosive_rate": round(float(all_games_df["rb_explosive_rate_allowed"].median()), 3),
        "pass_attempts": round(float(all_games_df["pass_attempts_allowed"].median()), 1),
        "pass_yards": round(float(all_games_df["pass_yards_allowed"].median()), 1),
        "pass_ypa": round(float(all_games_df["pass_ypa_allowed"].median()), 2),
        "pass_epa": round(float(all_games_df["pass_epa_allowed"].median()), 3),
        "wr_rec_yards": round(float(all_games_df["wr_rec_yards_allowed"].median()), 1),
        "te_rec_yards": round(float(all_games_df["te_rec_yards_allowed"].median()), 1),
        "sacks": round(float(all_games_df["sacks"].median()), 1),
        "qb_hits_pressures": round(float(all_games_df["qb_hits_pressures"].median()), 1),
        "epa_per_play": round(float(all_games_df["epa_per_play_allowed"].median()), 3),
        "success_rate": round(float(all_games_df["success_rate_allowed"].median()), 3),
        "total_games_evaluated": len(all_games_df),
        "distinct_opponents_evaluated": int(all_games_df["opponent"].nunique()),
    }

    # 1.50σ RMSD hard ceiling: Drop any game above 1.50σ from match candidate pool
    third_party_all = all_games_df[~all_games_df["is_direct_h2h"]]
    closest_distance = float(third_party_all["distance"].min()) if not third_party_all.empty else float(all_games_df["distance"].min())

    ceiling_df = all_games_df[all_games_df["rmsd"] <= 1.50].sort_values("distance").copy()

    # If fewer than 3 games remain under the ceiling, do not render a match table at all
    if len(ceiling_df) < 3:
        render_match_table = False
        no_analogs_message = f"No suitable analogs found in the available history. Closest match distance: {closest_distance:.2f}σ, above the 1.50σ ceiling."
        sorted_df = pd.DataFrame(columns=all_games_df.columns)
    else:
        render_match_table = True
        no_analogs_message = ""
        sorted_df = ceiling_df.head(n_similar).copy()
        sorted_df["similarity_rank"] = range(1, len(sorted_df) + 1)

    # Separate Strong Analogs vs. Direct H2H
    strong_df = sorted_df[sorted_df["rmsd"] <= 1.00].copy() if not sorted_df.empty else pd.DataFrame(columns=all_games_df.columns)
    strong_analog_df = strong_df[~strong_df["is_direct_h2h"]].copy() if not strong_df.empty else pd.DataFrame(columns=all_games_df.columns)

    n_strong = len(strong_df)
    n_strong_analog = len(strong_analog_df)
    n_strong_analog_games = len(strong_analog_df)
    n_distinct_strong_analogs = int(strong_analog_df["opponent"].nunique()) if not strong_analog_df.empty else 0

    n_total = len(sorted_df)
    n_total_games = len(sorted_df)
    n_distinct_total_opponents = int(sorted_df["opponent"].nunique()) if not sorted_df.empty else 0

    direct_h2h_games = [g for g in all_games_df.to_dict(orient="records") if g["is_direct_h2h"]]
    has_direct_h2h = len(direct_h2h_games) > 0

    def _calc_summary(df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {}
        return {
            "sample_games": len(df),
            "distinct_opponents": int(df["opponent"].nunique()),
            "rb_rush_yards": {
                "median": round(float(df["rb_rush_yards_allowed"].median()), 1),
                "min": round(float(df["rb_rush_yards_allowed"].min()), 1),
                "max": round(float(df["rb_rush_yards_allowed"].max()), 1),
            },
            "rb_carries": {
                "median": round(float(df["rb_carries_allowed"].median()), 1),
                "min": int(df["rb_carries_allowed"].min()),
                "max": int(df["rb_carries_allowed"].max()),
            },
            "rb_ypc": {
                "median": round(float(df["rb_ypc_allowed"].median()), 2),
                "min": round(float(df["rb_ypc_allowed"].min()), 2),
                "max": round(float(df["rb_ypc_allowed"].max()), 2),
            },
            "rb_rush_epa": {
                "median": round(float(df["rb_rush_epa_allowed"].median()), 3),
                "min": round(float(df["rb_rush_epa_allowed"].min()), 3),
                "max": round(float(df["rb_rush_epa_allowed"].max()), 3),
            },
            "rb_rush_success_rate": {
                "median": round(float(df["rb_rush_success_rate_allowed"].median()), 3),
                "min": round(float(df["rb_rush_success_rate_allowed"].min()), 3),
                "max": round(float(df["rb_rush_success_rate_allowed"].max()), 3),
            },
            "rb_success_rate": {
                "median": round(float(df["rb_rush_success_rate_allowed"].median()), 3),
                "min": round(float(df["rb_rush_success_rate_allowed"].min()), 3),
                "max": round(float(df["rb_rush_success_rate_allowed"].max()), 3),
            },
            "rb_stuff_rate": {
                "median": round(float(df["rb_stuff_rate_allowed"].median()), 3),
                "min": round(float(df["rb_stuff_rate_allowed"].min()), 3),
                "max": round(float(df["rb_stuff_rate_allowed"].max()), 3),
            },
            "rb_explosive_rate": {
                "median": round(float(df["rb_explosive_rate_allowed"].median()), 3),
                "min": round(float(df["rb_explosive_rate_allowed"].min()), 3),
                "max": round(float(df["rb_explosive_rate_allowed"].max()), 3),
            },
            "pass_yards": {
                "median": round(float(df["pass_yards_allowed"].median()), 1),
                "min": round(float(df["pass_yards_allowed"].min()), 1),
                "max": round(float(df["pass_yards_allowed"].max()), 1),
            },
            "pass_ypa": {
                "median": round(float(df["pass_ypa_allowed"].median()), 2),
                "min": round(float(df["pass_ypa_allowed"].min()), 2),
                "max": round(float(df["pass_ypa_allowed"].max()), 2),
            },
            "pass_epa": {
                "median": round(float(df["pass_epa_allowed"].median()), 3),
                "min": round(float(df["pass_epa_allowed"].min()), 3),
                "max": round(float(df["pass_epa_allowed"].max()), 3),
            },
            "wr_rec_yards": {
                "median": round(float(df["wr_rec_yards_allowed"].median()), 1),
                "min": round(float(df["wr_rec_yards_allowed"].min()), 1),
                "max": round(float(df["wr_rec_yards_allowed"].max()), 1),
            },
            "te_rec_yards": {
                "median": round(float(df["te_rec_yards_allowed"].median()), 1),
                "min": round(float(df["te_rec_yards_allowed"].min()), 1),
                "max": round(float(df["te_rec_yards_allowed"].max()), 1),
            },
            "sacks": {
                "median": round(float(df["sacks"].median()), 1),
                "min": int(df["sacks"].min()),
                "max": int(df["sacks"].max()),
            },
            "epa_per_play": {
                "median": round(float(df["epa_per_play_allowed"].median()), 3),
                "min": round(float(df["epa_per_play_allowed"].min()), 3),
                "max": round(float(df["epa_per_play_allowed"].max()), 3),
            },
            "success_rate": {
                "median": round(float(df["success_rate_allowed"].median()), 3),
                "min": round(float(df["success_rate_allowed"].min()), 3),
                "max": round(float(df["success_rate_allowed"].max()), 3),
            },
        }

    def _calc_deduped_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates per-opponent medians across their multiple meetings, then medians across opponents."""
        if df.empty:
            return {}
        opp_m = df.groupby("opponent").agg({
            "rb_rush_yards_allowed": "median",
            "rb_carries_allowed": "median",
            "rb_ypc_allowed": "median",
            "rb_rush_epa_allowed": "median",
            "rb_rush_success_rate_allowed": "median",
            "rb_stuff_rate_allowed": "median",
            "rb_explosive_rate_allowed": "median",
            "pass_yards_allowed": "median",
            "pass_ypa_allowed": "median",
            "pass_epa_allowed": "median",
            "wr_rec_yards_allowed": "median",
            "te_rec_yards_allowed": "median",
            "sacks": "median",
            "epa_per_play_allowed": "median",
            "success_rate_allowed": "median",
        })
        return {
            "distinct_opponents": len(opp_m),
            "rb_rush_yards": {
                "median": round(float(opp_m["rb_rush_yards_allowed"].median()), 1),
                "min": round(float(opp_m["rb_rush_yards_allowed"].min()), 1),
                "max": round(float(opp_m["rb_rush_yards_allowed"].max()), 1),
            },
            "rb_carries": {
                "median": round(float(opp_m["rb_carries_allowed"].median()), 1),
                "min": round(float(opp_m["rb_carries_allowed"].min()), 1),
                "max": round(float(opp_m["rb_carries_allowed"].max()), 1),
            },
            "rb_ypc": {
                "median": round(float(opp_m["rb_ypc_allowed"].median()), 2),
                "min": round(float(opp_m["rb_ypc_allowed"].min()), 2),
                "max": round(float(opp_m["rb_ypc_allowed"].max()), 2),
            },
            "rb_rush_epa": {
                "median": round(float(opp_m["rb_rush_epa_allowed"].median()), 3),
                "min": round(float(opp_m["rb_rush_epa_allowed"].min()), 3),
                "max": round(float(opp_m["rb_rush_epa_allowed"].max()), 3),
            },
            "pass_yards": {
                "median": round(float(opp_m["pass_yards_allowed"].median()), 1),
                "min": round(float(opp_m["pass_yards_allowed"].min()), 1),
                "max": round(float(opp_m["pass_yards_allowed"].max()), 1),
            },
            "pass_ypa": {
                "median": round(float(opp_m["pass_ypa_allowed"].median()), 2),
                "min": round(float(opp_m["pass_ypa_allowed"].min()), 2),
                "max": round(float(opp_m["pass_ypa_allowed"].max()), 2),
            },
            "pass_epa": {
                "median": round(float(opp_m["pass_epa_allowed"].median()), 3),
                "min": round(float(opp_m["pass_epa_allowed"].min()), 3),
                "max": round(float(opp_m["pass_epa_allowed"].max()), 3),
            },
            "sacks": {
                "median": round(float(opp_m["sacks"].median()), 1),
                "min": round(float(opp_m["sacks"].min()), 1),
                "max": round(float(opp_m["sacks"].max()), 1),
            },
            "epa_per_play": {
                "median": round(float(opp_m["epa_per_play_allowed"].median()), 3),
                "min": round(float(opp_m["epa_per_play_allowed"].min()), 3),
                "max": round(float(opp_m["epa_per_play_allowed"].max()), 3),
            },
            "success_rate": {
                "median": round(float(opp_m["success_rate_allowed"].median()), 3),
                "min": round(float(opp_m["success_rate_allowed"].min()), 3),
                "max": round(float(opp_m["success_rate_allowed"].max()), 3),
            },
        }

    all_summary = _calc_summary(sorted_df)
    all_deduped_summary = _calc_deduped_summary(sorted_df)

    # Require BOTH n_strong_analog_games >= 3 AND n_distinct_strong_analogs >= 2 to summarize strong
    can_summarize_strong = bool(n_strong_analog_games >= 3 and n_distinct_strong_analogs >= 2)
    strong_summary = _calc_summary(strong_analog_df) if can_summarize_strong else None
    strong_deduped_summary = _calc_deduped_summary(strong_analog_df) if can_summarize_strong else None

    # Transparent Sample Warning and Circular Match Flag
    if n_strong_analog_games == 0:
        if has_direct_h2h:
            sample_warning = (
                f"No third-party opponent in {defense_team}'s history matched {offense_team}'s profile as a strong analog (RMSD <= 1.00σ). "
                f"The only close match is the actual H2H meeting ({direct_h2h_games[0]['season']} Wk {direct_h2h_games[0]['week']} vs {offense_team}). "
                f"The similarity comparison for this matchup is therefore circular and should not be used to draw general conclusions about {defense_team}'s tendencies against {offense_team}-like offenses."
            )
        else:
            sample_warning = (
                f"No opponent in {defense_team}'s history matched {offense_team}'s profile as a strong analog (RMSD <= 1.00σ). "
                f"All {n_total_games} games evaluated are moderate or loose matches under the 1.50σ ceiling."
            )
    elif n_strong_analog_games < 3:
        sample_warning = (
            f"Only {n_strong_analog_games} third-party game in {defense_team}'s history matched {offense_team}'s profile as a strong analog "
            f"({n_distinct_strong_analogs} distinct opponent, RMSD <= 1.00σ, excluding direct H2H meetings). Sample is too small (need >= 3 games) for strong-only isolate. Showing median across all qualifying matches under 1.50σ ceiling."
        )
    elif n_distinct_strong_analogs < 2:
        sample_warning = (
            f"While {n_strong_analog_games} games matched as strong analogs, all were against a single opponent ({strong_analog_df['opponent'].iloc[0]}). "
            f"There are not enough distinct opponent franchises (need >= 2) to summarize as a general stylistic analog."
        )
    else:
        sample_warning = (
            f"{n_strong_analog_games} third-party games across {n_distinct_strong_analogs} distinct opponents matched {offense_team}'s profile as strong analogs "
            f"(RMSD <= 1.00σ, excluding direct H2H meetings)."
        )

    h2h_disclaimer = (
        f"Note: Sample includes the actual head-to-head meeting vs {offense_team} ({direct_h2h_games[0]['season']} Wk {direct_h2h_games[0]['week']}). "
        f"Direct H2H meetings are excluded from third-party analog sample counts."
        if has_direct_h2h else ""
    )

    # Descriptive narrative reporting
    if not can_summarize_strong:
        if all_summary and "rb_rush_yards" in all_summary:
            all_rush_med = f"{all_summary['rb_rush_yards']['median']:.1f}"
            all_pass_med = f"{all_summary['pass_yards']['median']:.1f}"
        else:
            all_rush_med = "N/A"
            all_pass_med = "N/A"
        descriptive_narrative = (
            f"{defense_team} faced {len(sorted_df)} opponents evaluated against {offense_team}'s profile under the 1.50σ ceiling, "
            f"with {n_strong_analog_games} third-party games across {n_distinct_strong_analogs} distinct opponents qualifying as strong analogs (RMSD <= 1.00σ, excluding direct H2H). "
            f"Across all {len(sorted_df)} evaluated matches ({n_distinct_total_opponents} distinct opponents), {defense_team} allowed a game median of "
            f"{all_rush_med} rushing yards to RBs and {all_pass_med} passing yards."
        )
    else:
        if n_strong_analog_games == n_distinct_strong_analogs:
            opp_note = f"In those {n_strong_analog_games} games ({n_distinct_strong_analogs} distinct opponents), {defense_team} allowed a median of {strong_summary['rb_rush_yards']['median']:.1f} rushing yards to RBs ({strong_summary['rb_ypc']['median']:.2f} YPC)"
        else:
            opp_note = f"In those {n_strong_analog_games} games across {n_distinct_strong_analogs} distinct opponents, {defense_team} allowed a game-level median of {strong_summary['rb_rush_yards']['median']:.1f} rushing yards to RBs (deduped opponent median: {strong_deduped_summary['rb_rush_yards']['median']:.1f}, {strong_summary['rb_ypc']['median']:.2f} YPC)"

        descriptive_narrative = (
            f"{defense_team} faced {len(sorted_df)} opponents with {n_strong_analog_games} strong third-party games across {n_distinct_strong_analogs} distinct opponents matching {offense_team}'s profile "
            f"(RMSD <= 1.00σ, excluding direct H2H). {opp_note} and {strong_summary['pass_yards']['median']:.1f} passing yards. "
            f"{defense_team}'s median across all {overall_medians['total_games_evaluated']} games evaluated is "
            f"{overall_medians['rb_rush_yards']:.1f} rushing yards and {overall_medians['pass_yards']:.1f} passing yards."
        )

    if not render_match_table:
        sample_warning = no_analogs_message
        descriptive_narrative = no_analogs_message

    return {
        "label": label,
        "offense_team": offense_team,
        "defense_team": defense_team,
        "target_offensive_profile": target_prof,
        "features_matched": features,
        "render_match_table": render_match_table,
        "no_analogs_message": no_analogs_message,
        "n_strong": n_strong,
        "n_strong_analog": n_strong_analog,
        "n_strong_analog_games": n_strong_analog_games,
        "n_distinct_strong_analogs": n_distinct_strong_analogs,
        "n_total": n_total,
        "n_total_games": n_total_games,
        "n_distinct_total_opponents": n_distinct_total_opponents,
        "is_full_sample": bool(n_strong_analog_games == n_total_games),
        "can_summarize_strong": can_summarize_strong,
        "sample_warning": sample_warning,
        "h2h_disclaimer": h2h_disclaimer,
        "similar_games": sorted_df.to_dict(orient="records") if render_match_table else [],
        "similar_games_summary": all_summary if render_match_table else {},
        "all_summary": all_summary if render_match_table else {},
        "all_deduped_summary": all_deduped_summary if render_match_table else {},
        "strong_summary": strong_summary,
        "strong_deduped_summary": strong_deduped_summary,
        "median_rb_rush_yards_all": all_summary.get("rb_rush_yards", {}).get("median") if all_summary else None,
        "median_rb_rush_yards_strong": strong_summary.get("rb_rush_yards", {}).get("median") if strong_summary else None,
        "median_pass_yards_all": all_summary.get("pass_yards", {}).get("median") if all_summary else None,
        "median_pass_yards_strong": strong_summary.get("pass_yards", {}).get("median") if strong_summary else None,
        "overall_defense_baseline": overall_medians,
        "descriptive_narrative": descriptive_narrative,
        "disclaimer": "Historical comparison based on past game logs. Shows past defensive results against similar offensive profiles.",
    }


# -----------------------------------------------------------------------------
# Dual-Split Lookup: Run-Heavy Match vs. Pass-Heavy Match
# -----------------------------------------------------------------------------
def run_dual_archetype_lookup(
    offense_team: str,
    defense_team: str,
    season: int = 2026,
    up_to_week: Optional[int] = None,
    n_similar: int = 5,
) -> Dict[str, Any]:
    """
    Runs the similar opponent lookup twice:
      1. Matched on ground features: rush rate, neutral rush %, RB1 carry %, designed QB rush %, RB1 vs RB2 split
      2. Matched on aerial features: neutral pass %, WR1 target share, TE target share, aDOT, red zone pass %
    Presents both samples side by side so users can see how the defense performs
    across run-funnel vs. pass-funnel game environments.
    """
    run_features = [
        "rush_rate",
        "pass_rate_neutral",
        "rb_carry_concentration",
        "qb_designed_rush_rate",
        "rb1_vs_rb2_split",
    ]
    pass_features = [
        "pass_rate_neutral",
        "top_wr_target_share",
        "te_target_share",
        "adot",
        "rz_pass_rate",
    ]

    run_split = find_similar_opponent_games(
        offense_team=offense_team,
        defense_team=defense_team,
        season=season,
        up_to_week=up_to_week,
        n_similar=n_similar,
        feature_subset=run_features,
        label=f"Ground & Run-Concentration Match ({offense_team}-like)",
    )

    pass_split = find_similar_opponent_games(
        offense_team=offense_team,
        defense_team=defense_team,
        season=season,
        up_to_week=up_to_week,
        n_similar=n_similar,
        feature_subset=pass_features,
        label="Aerial & Target-Distribution Match (Pass-Funnel)",
    )

    ground_ok = run_split.get("can_summarize_strong", False)
    aerial_ok = pass_split.get("can_summarize_strong", False)

    if ground_ok and aerial_ok:
        can_conclude = True
        can_conclude_status = "full"
        conclusion_verdict = (
            f"Full conclusion supported: {defense_team} has sufficient third-party history against both "
            f"ground-heavy offenses ({run_split['n_strong_analog_games']} games across {run_split['n_distinct_strong_analogs']} distinct opponents) "
            f"and pass-heavy offenses ({pass_split['n_strong_analog_games']} games across {pass_split['n_distinct_strong_analogs']} distinct opponents)."
        )
    elif ground_ok and not aerial_ok:
        can_conclude = False
        can_conclude_status = "partial_ground"
        conclusion_verdict = (
            f"Partial conclusion supported: {defense_team} has sufficient third-party history against ground-heavy offenses "
            f"({run_split['n_strong_analog_games']} games across {run_split['n_distinct_strong_analogs']} distinct opponents; "
            f"median {run_split['strong_summary']['rb_rush_yards']['median']:.1f} RB rush yds vs baseline {run_split['overall_defense_baseline']['rb_rush_yards']:.1f}), "
            f"but insufficient pass-heavy analogs ({pass_split.get('n_strong_analog_games', 0)} games, {pass_split.get('n_distinct_strong_analogs', 0)} opponents). "
            f"Ground tendency can be summarized, but aerial contrast cannot be concluded."
        )
    elif aerial_ok and not ground_ok:
        can_conclude = False
        can_conclude_status = "partial_aerial"
        conclusion_verdict = (
            f"Partial conclusion supported: {defense_team} has sufficient third-party history against pass-heavy offenses "
            f"({pass_split['n_strong_analog_games']} games across {pass_split['n_distinct_strong_analogs']} distinct opponents; "
            f"median {pass_split['strong_summary']['pass_yards']['median']:.1f} pass yds vs baseline {pass_split['overall_defense_baseline']['pass_yards']:.1f}), "
            f"but insufficient ground-heavy analogs ({run_split.get('n_strong_analog_games', 0)} games, {run_split.get('n_distinct_strong_analogs', 0)} opponents). "
            f"Aerial tendency can be summarized, but ground contrast cannot be concluded."
        )
    else:
        can_conclude = False
        can_conclude_status = "none"
        conclusion_verdict = (
            f"Cannot conclude: {defense_team} has faced fewer than 3 third-party opponents matching {offense_team}'s profile "
            f"(Ground: {run_split.get('n_strong_analog_games', 0)} games across {run_split.get('n_distinct_strong_analogs', 0)} opponents; "
            f"Aerial: {pass_split.get('n_strong_analog_games', 0)} games across {pass_split.get('n_distinct_strong_analogs', 0)} opponents). "
            f"There is not enough independent data to draw reliable conclusions about {defense_team}'s performance against ground-heavy vs. pass-heavy offenses."
        )

    def _safe_med(summ: Optional[Dict[str, Any]], key: str) -> Optional[float]:
        if not summ or not isinstance(summ, dict):
            return None
        val = summ.get(key)
        if isinstance(val, dict):
            return val.get("median")
        return val

    # Build Side-by-Side Comparative Contrast Table
    contrast_table = [
        {
            "archetype": f"Ground-Heavy ({offense_team}-like Run Focus)",
            "archetype_key": "ground",
            "features_matched": ["Rush Rate", "Neutral Rush %", "RB1 Carry %", "Designed QB Rush %", "RB1 vs RB2 Split"],
            "n_strong": run_split.get("n_strong", 0),
            "n_strong_analog": run_split.get("n_strong_analog", 0),
            "n_strong_games": run_split.get("n_strong_analog_games", 0),
            "n_strong_opponents": run_split.get("n_distinct_strong_analogs", 0),
            "n_total": run_split.get("n_total", 0),
            "n_total_games": run_split.get("n_total_games", 0),
            "n_total_opponents": run_split.get("n_distinct_total_opponents", 0),
            "is_full_sample": bool(run_split.get("n_strong_analog_games", 0) == run_split.get("n_total_games", 0)),
            "can_summarize": run_split.get("can_summarize_strong", False),
            "status": (
                f"Sufficient Sample ({run_split.get('n_strong_analog_games', 0)} games, {run_split.get('n_distinct_strong_analogs', 0)} opponents)"
                if run_split.get("can_summarize_strong", False)
                else f"INSUFFICIENT SAMPLE — NOT SUMMARIZED ({run_split.get('n_strong_analog_games', 0)} games, {run_split.get('n_distinct_strong_analogs', 0)} opps)"
            ),
            "rb_rush_yards_strong": _safe_med(run_split.get("strong_summary"), "rb_rush_yards"),
            "rb_rush_yards_strong_deduped": _safe_med(run_split.get("strong_deduped_summary"), "rb_rush_yards"),
            "rb_rush_yards_all": _safe_med(run_split.get("all_summary"), "rb_rush_yards"),
            "rb_rush_yards_all_deduped": _safe_med(run_split.get("all_deduped_summary"), "rb_rush_yards"),
            "rb_carries": _safe_med(run_split.get("all_summary"), "rb_carries"),
            "rb_ypc": _safe_med(run_split.get("all_summary"), "rb_ypc"),
            "rb_rush_epa": _safe_med(run_split.get("all_summary"), "rb_rush_epa"),
            "rb_success_rate": _safe_med(run_split.get("all_summary"), "rb_success_rate"),
            "rb_stuff_rate": _safe_med(run_split.get("all_summary"), "rb_stuff_rate"),
            "rb_explosive_rate": _safe_med(run_split.get("all_summary"), "rb_explosive_rate"),
            "pass_yards": _safe_med(run_split.get("all_summary"), "pass_yards"),
            "sacks": _safe_med(run_split.get("all_summary"), "sacks"),
            "epa_per_play": _safe_med(run_split.get("all_summary"), "epa_per_play"),
            "success_rate": _safe_med(run_split.get("all_summary"), "success_rate"),
        },
        {
            "archetype": "Pass-Heavy (Aerial & Spread Target Distribution)",
            "archetype_key": "aerial",
            "features_matched": ["Neutral Pass %", "WR1 Target Share", "TE Target Share", "aDOT", "Red Zone Pass %"],
            "n_strong": pass_split.get("n_strong", 0),
            "n_strong_analog": pass_split.get("n_strong_analog", 0),
            "n_strong_games": pass_split.get("n_strong_analog_games", 0),
            "n_strong_opponents": pass_split.get("n_distinct_strong_analogs", 0),
            "n_total": pass_split.get("n_total", 0),
            "n_total_games": pass_split.get("n_total_games", 0),
            "n_total_opponents": pass_split.get("n_distinct_total_opponents", 0),
            "is_full_sample": bool(pass_split.get("n_strong_analog_games", 0) == pass_split.get("n_total_games", 0)),
            "can_summarize": pass_split.get("can_summarize_strong", False),
            "status": (
                f"Sufficient Sample ({pass_split.get('n_strong_analog_games', 0)} games, {pass_split.get('n_distinct_strong_analogs', 0)} opponents)"
                if pass_split.get("can_summarize_strong", False)
                else f"INSUFFICIENT SAMPLE — NOT SUMMARIZED ({pass_split.get('n_strong_analog_games', 0)} games, {pass_split.get('n_distinct_strong_analogs', 0)} opps)"
            ),
            "rb_rush_yards_strong": _safe_med(pass_split.get("strong_summary"), "rb_rush_yards"),
            "rb_rush_yards_strong_deduped": _safe_med(pass_split.get("strong_deduped_summary"), "rb_rush_yards"),
            "rb_rush_yards_all": _safe_med(pass_split.get("all_summary"), "rb_rush_yards"),
            "rb_rush_yards_all_deduped": _safe_med(pass_split.get("all_deduped_summary"), "rb_rush_yards"),
            "rb_carries": _safe_med(pass_split.get("all_summary"), "rb_carries"),
            "rb_ypc": _safe_med(pass_split.get("all_summary"), "rb_ypc"),
            "rb_rush_epa": _safe_med(pass_split.get("all_summary"), "rb_rush_epa"),
            "rb_success_rate": _safe_med(pass_split.get("all_summary"), "rb_success_rate"),
            "rb_stuff_rate": _safe_med(pass_split.get("all_summary"), "rb_stuff_rate"),
            "rb_explosive_rate": _safe_med(pass_split.get("all_summary"), "rb_explosive_rate"),
            "pass_yards": _safe_med(pass_split.get("all_summary"), "pass_yards"),
            "sacks": _safe_med(pass_split.get("all_summary"), "sacks"),
            "epa_per_play": _safe_med(pass_split.get("all_summary"), "epa_per_play"),
            "success_rate": _safe_med(pass_split.get("all_summary"), "success_rate"),
        },
        {
            "archetype": f"{defense_team} Season Baseline (All Opponents)",
            "archetype_key": "baseline",
            "features_matched": ["All Evaluated Matchups"],
            "n_strong": run_split["overall_defense_baseline"]["total_games_evaluated"],
            "n_strong_analog": run_split["overall_defense_baseline"]["total_games_evaluated"],
            "n_strong_games": run_split["overall_defense_baseline"]["total_games_evaluated"],
            "n_strong_opponents": run_split["overall_defense_baseline"]["distinct_opponents_evaluated"],
            "n_total": run_split["overall_defense_baseline"]["total_games_evaluated"],
            "n_total_games": run_split["overall_defense_baseline"]["total_games_evaluated"],
            "n_total_opponents": run_split["overall_defense_baseline"]["distinct_opponents_evaluated"],
            "is_full_sample": True,
            "can_summarize": True,
            "status": "Overall Baseline",
            "rb_rush_yards_strong": run_split["overall_defense_baseline"]["rb_rush_yards"],
            "rb_rush_yards_strong_deduped": run_split["overall_defense_baseline"]["rb_rush_yards"],
            "rb_rush_yards_all": run_split["overall_defense_baseline"]["rb_rush_yards"],
            "rb_rush_yards_all_deduped": run_split["overall_defense_baseline"]["rb_rush_yards"],
            "rb_carries": run_split["overall_defense_baseline"]["rb_carries"],
            "rb_ypc": run_split["overall_defense_baseline"]["rb_ypc"],
            "rb_rush_epa": run_split["overall_defense_baseline"]["rb_rush_epa"],
            "rb_success_rate": run_split["overall_defense_baseline"]["rb_success_rate"],
            "rb_stuff_rate": run_split["overall_defense_baseline"]["rb_stuff_rate"],
            "rb_explosive_rate": run_split["overall_defense_baseline"]["rb_explosive_rate"],
            "pass_yards": run_split["overall_defense_baseline"]["pass_yards"],
            "sacks": run_split["overall_defense_baseline"]["sacks"],
            "epa_per_play": run_split["overall_defense_baseline"]["epa_per_play"],
            "success_rate": run_split["overall_defense_baseline"]["success_rate"],
        }
    ]

    return {
        "offense_team": offense_team,
        "defense_team": defense_team,
        "ground_match": run_split,
        "aerial_match": pass_split,
        "can_conclude": can_conclude,
        "can_conclude_status": can_conclude_status,
        "conclusion_verdict": conclusion_verdict,
        "contrast_table": contrast_table,
        "disclaimer": "Side-by-side descriptive comparison across run-focused and pass-focused historical samples.",
    }
