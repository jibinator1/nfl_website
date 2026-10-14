"""
backend/team_tendencies.py

Descriptive team tendency and opponent vulnerability stat engine.
Answers three core questions without models, predictions, or betting picks:
  1. What does this team tend to do in specific situations? (Tendency profile)
  2. What does this team allow, and to whom? (Vulnerability profile)
  3. How does every metric compare to the league median, percentiles, and rank?

Every metric includes:
  - raw_value
  - league_median
  - league_p25
  - league_p75
  - rank (1 = best, 32 = worst)
  - rank_direction ("higher is better" or "lower is better")
  - diff_from_median (raw difference and percentage difference)
  - sample_size (plays or games)
  - is_small_sample (flag if plays < 40 or games < 4)

No projections. No probabilities. No fair lines. No edge or advantage claims.
"""
from __future__ import annotations

import os
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PBP_DIR = os.path.join(ROOT_DIR, "data", "pbp_cache")
WEEKLY_CACHE_PATH = os.path.join(ROOT_DIR, "data", "weekly_cache.parquet")

_CACHED_PBP: Dict[int, pd.DataFrame] = {}
_CACHED_WEEKLY: Optional[pd.DataFrame] = None
_CACHED_POS_MAP: Optional[Dict[str, str]] = None


# -----------------------------------------------------------------------------
# Data Loader Helpers
# -----------------------------------------------------------------------------
def load_clean_pbp(
    season: int = 2026,
    up_to_week: Optional[int] = None,
    strict_less: bool = True,
) -> pd.DataFrame:
    """Loads and caches clean regular season pass/run plays from local parquet or nflreadpy."""
    global _CACHED_PBP
    if season not in _CACHED_PBP:
        local_path = os.path.join(PBP_DIR, f"pbp_{season}.parquet")
        if os.path.exists(local_path):
            df = pd.read_parquet(local_path)
        else:
            import nflreadpy as nfl
            df = nfl.load_pbp(seasons=[season]).to_pandas()
            os.makedirs(PBP_DIR, exist_ok=True)
            try:
                df.to_parquet(local_path, index=False)
            except Exception:
                pass
        # Keep pass and run plays with valid epa
        df = df[df["play_type"].isin(["pass", "run"]) & df["epa"].notna()].copy()
        if "qb_kneel" in df.columns:
            df = df[df["qb_kneel"] == 0]
        if "qb_spike" in df.columns:
            df = df[df["qb_spike"] == 0]
        _CACHED_PBP[season] = df

    df = _CACHED_PBP[season]
    if up_to_week is not None and "week" in df.columns:
        if strict_less:
            df = df[df["week"] < up_to_week].copy()
        else:
            df = df[df["week"] <= up_to_week].copy()
    return df


def load_weekly_stats(
    season: Optional[int] = None,
    up_to_week: Optional[int] = None,
    strict_less: bool = True,
) -> pd.DataFrame:
    """Loads player weekly box stats for archetype analysis."""
    global _CACHED_WEEKLY
    if _CACHED_WEEKLY is None:
        if os.path.exists(WEEKLY_CACHE_PATH):
            _CACHED_WEEKLY = pd.read_parquet(WEEKLY_CACHE_PATH)
        else:
            import nflreadpy as nfl
            _CACHED_WEEKLY = nfl.load_player_stats(seasons=[2025, 2026]).to_pandas()
    df = _CACHED_WEEKLY.copy()
    if season is not None and "season" in df.columns:
        df = df[df["season"] == season]
    if up_to_week is not None and "week" in df.columns:
        if strict_less:
            df = df[df["week"] < up_to_week]
        else:
            df = df[df["week"] <= up_to_week]
    return df


def get_player_position_map() -> Dict[str, str]:
    """Returns mapping from player_id to primary offensive/defensive position."""
    global _CACHED_POS_MAP
    if _CACHED_POS_MAP is None:
        w = load_weekly_stats()
        if not w.empty and "player_id" in w.columns and "position" in w.columns:
            _CACHED_POS_MAP = w.groupby("player_id")["position"].last().dropna().to_dict()
        else:
            _CACHED_POS_MAP = {}
    return _CACHED_POS_MAP


# -----------------------------------------------------------------------------
# Metric Comparison Formatter
# -----------------------------------------------------------------------------
def format_stat_comparison(
    team_val: float,
    league_series: pd.Series | Dict[str, float] | List[float],
    sample_size: int,
    higher_is_better: bool = True,
    unit: str = "",
    min_sample: int = 40,
    n_games: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Computes league median, percentiles, rank, differences, and sample size flag.
    If n_games < 6, rank is suppressed to prevent misleading small-sample claims.
    """
    if isinstance(league_series, dict):
        vals = pd.Series(list(league_series.values())).dropna()
    elif isinstance(league_series, list):
        vals = pd.Series(league_series).dropna()
    else:
        vals = pd.Series(league_series).dropna()

    if vals.empty:
        median = team_val
        p25 = team_val
        p75 = team_val
        rank = 16
    else:
        median = float(vals.median())
        p25 = float(vals.quantile(0.25))
        p75 = float(vals.quantile(0.75))
        if higher_is_better:
            rank = int((vals > team_val).sum() + 1)
        else:
            rank = int((vals < team_val).sum() + 1)

    diff_raw = team_val - median
    diff_pct = (diff_raw / abs(median) * 100.0) if abs(median) > 1e-5 else 0.0
    is_small = sample_size < min_sample

    rank_suppressed = False
    if n_games is not None:
        if n_games < 6:
            rank_suppressed = True
            rank_out = None
            rank_display = f"{team_val:.2f} {unit} (league median {median:.2f}, n={n_games} games -- rank suppressed, small sample)"
        else:
            rank_out = rank
            rank_display = f"Rank #{rank} (n={n_games} games)"
    else:
        rank_out = rank
        rank_display = f"Rank #{rank} (n={sample_size} plays)"

    return {
        "raw_value": round(float(team_val), 3),
        "league_median": round(float(median), 3),
        "league_p25": round(float(p25), 3),
        "league_p75": round(float(p75), 3),
        "rank": rank_out,
        "rank_display": rank_display,
        "rank_suppressed": rank_suppressed,
        "n_games": n_games,
        "total_teams": len(vals) if not vals.empty else 32,
        "rank_direction": "higher is better" if higher_is_better else "lower is better",
        "diff_from_median": round(float(diff_raw), 3),
        "diff_pct": round(float(diff_pct), 1),
        "sample_size": int(sample_size),
        "is_small_sample": bool(is_small),
        "unit": unit,
    }


# -----------------------------------------------------------------------------
# 1. Team Tendency Profile (What this team tends to do)
# -----------------------------------------------------------------------------
def compute_team_tendency_profile(
    team: str,
    season: int = 2026,
    up_to_week: Optional[int] = None,
    pbp_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Computes situational tendency splits for a team's offense compared to league medians:
      - By game state (leading >= +7, neutral -6 to +6, trailing <= -7)
      - By down (1st, 2nd, 3rd, 4th)
      - By field position (own 1-20, own 21-50, opp 49-21, red zone 20-1)
      - By 4th quarter margin / score state
      - Personnel and formation (shotgun, no-huddle)
      - Red zone positional shares (inside 20, inside 10, inside 5)
    """
    df = pbp_df if pbp_df is not None else load_clean_pbp(season=season, up_to_week=up_to_week)
    all_teams = sorted(df["posteam"].dropna().unique().tolist())
    pos_map = get_player_position_map()

    # Pre-calculate fields
    df = df.copy()
    diff = df["score_differential"].fillna(0)
    df["_state"] = np.where(diff >= 7, "leading", np.where(diff <= -7, "trailing", "neutral"))
    df["_is_pass"] = (df["play_type"] == "pass").astype(int)
    df["_is_run"] = (df["play_type"] == "run").astype(int)
    df["_rec_pos"] = df["receiver_player_id"].map(pos_map).fillna("Unknown")
    df["_rush_pos"] = df["rusher_player_id"].map(pos_map).fillna("Unknown")

    n_games = int(df[df["posteam"] == team]["game_id"].nunique()) if "game_id" in df.columns else None

    # Helper: team vs league series
    def _agg_metric(
        filter_fn,
        val_fn,
        higher_is_better: bool = True,
        unit: str = "",
        min_sample: int = 40,
    ) -> Dict[str, Any]:
        league_vals = {}
        target_val = 0.0
        target_n = 0
        for tm in all_teams:
            tm_df = df[df["posteam"] == tm]
            sub = filter_fn(tm_df)
            n_sub = len(sub)
            val = val_fn(sub) if n_sub > 0 else 0.0
            league_vals[tm] = val
            if tm == team:
                target_val = val
                target_n = n_sub
        return format_stat_comparison(
            target_val, league_vals, target_n, higher_is_better, unit=unit, min_sample=min_sample, n_games=n_games
        )

    # 1. Game State Splits
    game_state_splits = {}
    for st in ["leading", "neutral", "trailing"]:
        f_st = lambda d, s=st: d[d["_state"] == s]
        pass_rate = _agg_metric(f_st, lambda d: float(d["_is_pass"].mean()), higher_is_better=True, unit="rate", min_sample=20)
        rush_rate = _agg_metric(f_st, lambda d: float(d["_is_run"].mean()), higher_is_better=True, unit="rate", min_sample=20)
        epa_per_play = _agg_metric(f_st, lambda d: float(d["epa"].mean()), higher_is_better=True, unit="epa", min_sample=20)
        succ_rate = _agg_metric(f_st, lambda d: float(d["success"].mean()), higher_is_better=True, unit="rate", min_sample=20)
        yds_per_play = _agg_metric(f_st, lambda d: float(d["yards_gained"].mean()), higher_is_better=True, unit="yards", min_sample=20)

        # Plays per game in this state
        def _plays_per_game(d):
            n_games = d["game_id"].nunique() if "game_id" in d else 1
            return float(len(d) / max(n_games, 1))

        ppg = _agg_metric(f_st, _plays_per_game, higher_is_better=True, unit="plays", min_sample=4)
        no_hud = _agg_metric(f_st, lambda d: float((d["no_huddle"] == 1).mean()), higher_is_better=True, unit="rate", min_sample=20)

        game_state_splits[st] = {
            "pass_rate": pass_rate,
            "rush_rate": rush_rate,
            "epa_per_play": epa_per_play,
            "success_rate": succ_rate,
            "yards_per_play": yds_per_play,
            "plays_per_game": ppg,
            "no_huddle_rate": no_hud,
            "sample_plays": pass_rate["sample_size"],
        }

    # 2. Down Splits (1st, 2nd, 3rd, 4th)
    down_splits = {}
    for d_num in [1, 2, 3, 4]:
        f_dn = lambda d, dn=d_num: d[d["down"] == dn]
        p_rate = _agg_metric(f_dn, lambda d: float(d["_is_pass"].mean()), higher_is_better=True, unit="rate", min_sample=15)
        r_rate = _agg_metric(f_dn, lambda d: float(d["_is_run"].mean()), higher_is_better=True, unit="rate", min_sample=15)
        epa = _agg_metric(f_dn, lambda d: float(d["epa"].mean()), higher_is_better=True, unit="epa", min_sample=15)
        down_splits[f"down_{d_num}"] = {
            "pass_rate": p_rate,
            "rush_rate": r_rate,
            "epa_per_play": epa,
            "sample_plays": p_rate["sample_size"],
        }

    # 3. Field Position Splits
    # Own 1-20 (yardline_100 >= 80), Own 21-50 (50 <= yardline_100 < 80)
    # Opp 49-21 (21 <= yardline_100 < 50), Red Zone (yardline_100 <= 20)
    zones = {
        "own_1_to_20": lambda d: d[d["yardline_100"] >= 80],
        "own_21_to_50": lambda d: d[(d["yardline_100"] >= 50) & (d["yardline_100"] < 80)],
        "opp_49_to_21": lambda d: d[(d["yardline_100"] > 20) & (d["yardline_100"] < 50)],
        "red_zone_20_to_1": lambda d: d[d["yardline_100"] <= 20],
    }
    field_pos_splits = {}
    for z_name, z_fn in zones.items():
        p_rate = _agg_metric(z_fn, lambda d: float(d["_is_pass"].mean()), higher_is_better=True, unit="rate", min_sample=15)
        r_rate = _agg_metric(z_fn, lambda d: float(d["_is_run"].mean()), higher_is_better=True, unit="rate", min_sample=15)
        td_rate = _agg_metric(z_fn, lambda d: float(d["touchdown"].fillna(0).mean()), higher_is_better=True, unit="rate", min_sample=15)
        epa = _agg_metric(z_fn, lambda d: float(d["epa"].mean()), higher_is_better=True, unit="epa", min_sample=15)
        field_pos_splits[z_name] = {
            "pass_rate": p_rate,
            "rush_rate": r_rate,
            "td_rate": td_rate,
            "epa_per_play": epa,
            "sample_plays": p_rate["sample_size"],
        }

    # 4. 4th Quarter Margin Splits
    q4_all = lambda d: d[d["qtr"] == 4]
    q4_close = lambda d: d[(d["qtr"] == 4) & (d["score_differential"].abs() <= 8)]
    q4_trailing = lambda d: d[(d["qtr"] == 4) & (d["score_differential"] <= -9)]
    q4_leading = lambda d: d[(d["qtr"] == 4) & (d["score_differential"] >= 9)]

    fourth_quarter = {
        "one_score_game": {
            "pass_rate": _agg_metric(q4_close, lambda d: float(d["_is_pass"].mean()), unit="rate", min_sample=10),
            "epa_per_play": _agg_metric(q4_close, lambda d: float(d["epa"].mean()), unit="epa", min_sample=10),
            "sample_plays": len(q4_close(df[df["posteam"] == team])),
        },
        "trailing_two_scores": {
            "pass_rate": _agg_metric(q4_trailing, lambda d: float(d["_is_pass"].mean()), unit="rate", min_sample=10),
            "sample_plays": len(q4_trailing(df[df["posteam"] == team])),
        },
        "leading_two_scores": {
            "rush_rate": _agg_metric(q4_leading, lambda d: float(d["_is_run"].mean()), unit="rate", min_sample=10),
            "sample_plays": len(q4_leading(df[df["posteam"] == team])),
        },
    }

    # 5. Formation & Pace
    shotgun = _agg_metric(lambda d: d, lambda d: float(d["shotgun"].fillna(0).mean()), unit="rate")
    no_huddle_overall = _agg_metric(lambda d: d, lambda d: float(d["no_huddle"].fillna(0).mean()), unit="rate")

    # 6. Red Zone Positional Shares
    rz_df_fn = lambda d: d[d["yardline_100"] <= 20]
    rz_10_fn = lambda d: d[d["yardline_100"] <= 10]
    rz_5_fn = lambda d: d[d["yardline_100"] <= 5]

    rz_wr_tgt = _agg_metric(
        lambda d: d[(d["yardline_100"] <= 20) & (d["_is_pass"] == 1)],
        lambda d: float((d["_rec_pos"] == "WR").mean()),
        unit="share", min_sample=10
    )
    rz_te_tgt = _agg_metric(
        lambda d: d[(d["yardline_100"] <= 20) & (d["_is_pass"] == 1)],
        lambda d: float((d["_rec_pos"] == "TE").mean()),
        unit="share", min_sample=10
    )
    rz_rb_tgt = _agg_metric(
        lambda d: d[(d["yardline_100"] <= 20) & (d["_is_pass"] == 1)],
        lambda d: float((d["_rec_pos"] == "RB").mean()),
        unit="share", min_sample=10
    )
    rz_rb_carries = _agg_metric(
        lambda d: d[(d["yardline_100"] <= 20) & (d["_is_run"] == 1)],
        lambda d: float((d["_rush_pos"] == "RB").mean()),
        unit="share", min_sample=10
    )
    rz_qb_carries = _agg_metric(
        lambda d: d[(d["yardline_100"] <= 20) & (d["_is_run"] == 1)],
        lambda d: float((d["_rush_pos"] == "QB").mean()),
        unit="share", min_sample=10
    )

    rz_10_pass_rate = _agg_metric(rz_10_fn, lambda d: float(d["_is_pass"].mean()), unit="rate", min_sample=10)
    rz_5_pass_rate = _agg_metric(rz_5_fn, lambda d: float(d["_is_pass"].mean()), unit="rate", min_sample=5)

    return {
        "team": team,
        "season": season,
        "up_to_week": up_to_week,
        "total_plays_evaluated": len(df[df["posteam"] == team]),
        "game_state_splits": game_state_splits,
        "down_splits": down_splits,
        "field_position_splits": field_pos_splits,
        "fourth_quarter_margin": fourth_quarter,
        "formation_and_pace": {
            "shotgun_rate": shotgun,
            "no_huddle_rate": no_huddle_overall,
        },
        "red_zone_shares": {
            "inside_20": {
                "wr_target_share": rz_wr_tgt,
                "te_target_share": rz_te_tgt,
                "rb_target_share": rz_rb_tgt,
                "rb_carry_share": rz_rb_carries,
                "qb_carry_share": rz_qb_carries,
            },
            "inside_10": {
                "pass_rate": rz_10_pass_rate,
                "sample_plays": rz_10_pass_rate["sample_size"],
            },
            "inside_5": {
                "pass_rate": rz_5_pass_rate,
                "sample_plays": rz_5_pass_rate["sample_size"],
            },
        },
    }


# -----------------------------------------------------------------------------
# 2. Opponent Vulnerability Profile (What this defense allows)
# -----------------------------------------------------------------------------
def compute_opponent_vulnerability_profile(
    defense_team: str,
    season: int = 2026,
    up_to_week: Optional[int] = None,
    pbp_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Computes what a defense allows across ground, aerial, depth, and positional axes.
    Includes opponent-adjusted versions alongside raw descriptive values.
    """
    df = pbp_df if pbp_df is not None else load_clean_pbp(season=season, up_to_week=up_to_week)
    all_teams = sorted(df["defteam"].dropna().unique().tolist())
    pos_map = get_player_position_map()
    weekly_stats = load_weekly_stats(season=season, up_to_week=up_to_week)

    df = df.copy()
    df["_is_pass"] = (df["play_type"] == "pass").astype(int)
    df["_is_run"] = (df["play_type"] == "run").astype(int)
    df["_rec_pos"] = df["receiver_player_id"].map(pos_map).fillna("Unknown")
    df["_rush_pos"] = df["rusher_player_id"].map(pos_map).fillna("Unknown")

    # Leave-one-out opponent adjustment helper
    def _calc_opp_adj_series(play_subset_fn, val_col):
        g = df[play_subset_fn(df)].copy()
        if g.empty:
            return {tm: 0.0 for tm in all_teams}
        per_game = g.groupby(["defteam", "posteam", "game_id"])[val_col].mean().reset_index()
        # opponent baseline across all its other games
        opp_totals = per_game.groupby("posteam")[val_col].agg(["sum", "count"]).rename(
            columns={"sum": "opp_sum", "count": "opp_n"}
        )
        merged = per_game.join(opp_totals, on="posteam")
        n_other = merged["opp_n"] - 1
        opp_other_mean = (merged["opp_sum"] - merged[val_col]) / n_other.where(n_other > 0)
        merged["diff"] = merged[val_col] - opp_other_mean
        adj_per_team = merged.groupby("defteam")["diff"].mean().to_dict()
        return {tm: float(adj_per_team.get(tm, 0.0)) for tm in all_teams}

    n_games = int(df[df["defteam"] == defense_team]["game_id"].nunique()) if "game_id" in df.columns else None

    def _agg_def_metric(
        filter_fn,
        val_fn,
        higher_is_better: bool = False,  # For defenses, lower EPA / yards allowed is usually better
        unit: str = "",
        min_sample: int = 40,
    ) -> Dict[str, Any]:
        league_vals = {}
        target_val = 0.0
        target_n = 0
        for tm in all_teams:
            tm_df = df[df["defteam"] == tm]
            sub = filter_fn(tm_df)
            n_sub = len(sub)
            val = val_fn(sub) if n_sub > 0 else 0.0
            league_vals[tm] = val
            if tm == defense_team:
                target_val = val
                target_n = n_sub
        return format_stat_comparison(
            target_val, league_vals, target_n, higher_is_better=higher_is_better, unit=unit, min_sample=min_sample, n_games=n_games
        )

    # A. Rush Defense (RB Carries Only)
    f_rb_runs = lambda d: d[(d["_is_run"] == 1) & (d["qb_scramble"] == 0) & (d["_rush_pos"] != "QB")]
    f_rb_inside = lambda d: d[(d["_is_run"] == 1) & (d["qb_scramble"] == 0) & (d["_rush_pos"] != "QB") & ((d["run_location"] == "middle") | (d["run_gap"] == "guard"))]
    f_rb_outside = lambda d: d[(d["_is_run"] == 1) & (d["qb_scramble"] == 0) & (d["_rush_pos"] != "QB") & (d["run_location"].isin(["left", "right"])) & (d["run_gap"].isin(["tackle", "end"]))]

    rb_epa = _agg_def_metric(f_rb_runs, lambda d: float(d["epa"].mean()), unit="epa")
    rb_ypc = _agg_def_metric(f_rb_runs, lambda d: float(d["yards_gained"].mean()), unit="yards")
    rb_succ = _agg_def_metric(f_rb_runs, lambda d: float(d["success"].mean()), unit="rate")
    rb_stuff = _agg_def_metric(f_rb_runs, lambda d: float((d["yards_gained"] <= 0).mean()), higher_is_better=True, unit="rate")
    rb_expl = _agg_def_metric(f_rb_runs, lambda d: float((d["yards_gained"] >= 10).mean()), higher_is_better=False, unit="rate")

    # Opponent-adjusted RB rush EPA
    adj_rush_epa_series = _calc_opp_adj_series(lambda d: (d["_is_run"] == 1) & (d["_rush_pos"] != "QB"), "epa")
    rb_epa_opp_adj = format_stat_comparison(
        adj_rush_epa_series.get(defense_team, 0.0),
        adj_rush_epa_series,
        sample_size=rb_epa["sample_size"],
        higher_is_better=False,
        unit="epa_diff",
    )

    # Inside vs Outside
    inside_ypc = _agg_def_metric(f_rb_inside, lambda d: float(d["yards_gained"].mean()), unit="yards", min_sample=20)
    inside_epa = _agg_def_metric(f_rb_inside, lambda d: float(d["epa"].mean()), unit="epa", min_sample=20)
    outside_ypc = _agg_def_metric(f_rb_outside, lambda d: float(d["yards_gained"].mean()), unit="yards", min_sample=20)
    outside_epa = _agg_def_metric(f_rb_outside, lambda d: float(d["epa"].mean()), unit="epa", min_sample=20)

    rush_defense = {
        "overall_rb": {
            "epa_per_carry": rb_epa,
            "epa_per_carry_opp_adj": rb_epa_opp_adj,
            "yards_per_carry": rb_ypc,
            "success_rate": rb_succ,
            "stuff_rate_under_zero": rb_stuff,
            "explosive_rate_10_plus": rb_expl,
            "sample_carries": rb_epa["sample_size"],
        },
        "directional_gaps": {
            "inside_tackles": {
                "yards_per_carry": inside_ypc,
                "epa_per_carry": inside_epa,
                "sample_carries": inside_ypc["sample_size"],
            },
            "outside_tackles": {
                "yards_per_carry": outside_ypc,
                "epa_per_carry": outside_epa,
                "sample_carries": outside_ypc["sample_size"],
            },
        },
    }

    # B. Pass Defense
    f_passes = lambda d: d[d["_is_pass"] == 1]
    pass_epa = _agg_def_metric(f_passes, lambda d: float(d["epa"].mean()), unit="epa")
    pass_succ = _agg_def_metric(f_passes, lambda d: float(d["success"].mean()), unit="rate")
    pass_ypa = _agg_def_metric(f_passes, lambda d: float(d["yards_gained"].mean()), unit="yards")
    pass_sack_rate = _agg_def_metric(f_passes, lambda d: float(d["sack"].fillna(0).mean()), higher_is_better=True, unit="rate")
    pass_pressure_rate = _agg_def_metric(f_passes, lambda d: float(((d["sack"].fillna(0) == 1) | (d["qb_hit"].fillna(0) == 1)).mean()), higher_is_better=True, unit="rate")

    # Opponent-adjusted Pass EPA
    adj_pass_epa_series = _calc_opp_adj_series(lambda d: d["_is_pass"] == 1, "epa")
    pass_epa_opp_adj = format_stat_comparison(
        adj_pass_epa_series.get(defense_team, 0.0),
        adj_pass_epa_series,
        sample_size=pass_epa["sample_size"],
        higher_is_better=False,
        unit="epa_diff",
    )

    # Pass Depth Bins
    depth_bins = {
        "behind_los": lambda d: d[(d["_is_pass"] == 1) & (d["air_yards"] < 0)],
        "short_0_to_9": lambda d: d[(d["_is_pass"] == 1) & (d["air_yards"] >= 0) & (d["air_yards"] < 10)],
        "intermediate_10_to_19": lambda d: d[(d["_is_pass"] == 1) & (d["air_yards"] >= 10) & (d["air_yards"] < 20)],
        "deep_20_plus": lambda d: d[(d["_is_pass"] == 1) & (d["air_yards"] >= 20)],
    }
    pass_depth_allowed = {}
    for d_name, d_fn in depth_bins.items():
        ypa = _agg_def_metric(d_fn, lambda d: float(d["yards_gained"].mean()), unit="yards", min_sample=10)
        depa = _agg_def_metric(d_fn, lambda d: float(d["epa"].mean()), unit="epa", min_sample=10)
        pass_depth_allowed[d_name] = {
            "yards_per_attempt": ypa,
            "epa_per_attempt": depa,
            "sample_attempts": ypa["sample_size"],
        }

    pass_defense = {
        "overall_dropbacks": {
            "epa_per_dropback": pass_epa,
            "epa_per_dropback_opp_adj": pass_epa_opp_adj,
            "success_rate": pass_succ,
            "yards_per_attempt": pass_ypa,
            "sack_rate": pass_sack_rate,
            "pressure_rate": pass_pressure_rate,
            "sample_dropbacks": pass_epa["sample_size"],
        },
        "target_depth_allowed": pass_depth_allowed,
    }

    # C. By Position Allowed (RB, WR, TE)
    f_pos_targets = lambda pos: (lambda d: d[(d["_is_pass"] == 1) & (d["_rec_pos"] == pos)])
    pos_allowed = {}
    for pos_label in ["WR", "TE", "RB"]:
        pos_fn = f_pos_targets(pos_label)
        pos_ypa = _agg_def_metric(pos_fn, lambda d: float(d["yards_gained"].mean()), unit="yards", min_sample=15)
        pos_epa = _agg_def_metric(pos_fn, lambda d: float(d["epa"].mean()), unit="epa", min_sample=15)
        pos_succ = _agg_def_metric(pos_fn, lambda d: float(d["success"].mean()), unit="rate", min_sample=15)

        # Total yards allowed per game to this position
        def _pos_yds_per_game(d):
            n_g = d["game_id"].nunique() if "game_id" in d else 1
            return float(d["yards_gained"].sum() / max(n_g, 1))

        pos_ypg = _agg_def_metric(pos_fn, _pos_yds_per_game, unit="yards_per_game", min_sample=4)

        pos_allowed[pos_label.lower()] = {
            "yards_per_game": pos_ypg,
            "yards_per_target": pos_ypa,
            "epa_per_target": pos_epa,
            "success_rate": pos_succ,
            "sample_targets": pos_ypa["sample_size"],
        }

    # D. Player Archetypes Allowed (from weekly box score logs)
    archetype_stats = {}
    if not weekly_stats.empty and "opponent_team" in weekly_stats.columns:
        w_df = weekly_stats.copy()

        # 1. Workhorse RBs (>= 15 carries)
        workhorses = w_df[(w_df["position"] == "RB") & (w_df["carries"] >= 15)]
        def_wh = workhorses[workhorses["opponent_team"] == defense_team]
        n_wh_games = len(def_wh)
        med_wh_yds = float(def_wh["rushing_yards"].median()) if n_wh_games > 0 else 0.0
        med_wh_ypc = float((def_wh["rushing_yards"] / def_wh["carries"].replace(0, 1)).median()) if n_wh_games > 0 else 0.0

        wh_all_meds = workhorses.groupby("opponent_team")["rushing_yards"].median().to_dict()
        wh_yds_comp = format_stat_comparison(med_wh_yds, wh_all_meds, n_wh_games, higher_is_better=False, unit="yards", min_sample=3)

        # 2. Committee RBs (< 10 carries)
        committee = w_df[(w_df["position"] == "RB") & (w_df["carries"] < 10) & (w_df["carries"] > 0)]
        def_comm = committee[committee["opponent_team"] == defense_team]
        n_comm_games = len(def_comm)
        med_comm_yds = float(def_comm["rushing_yards"].median()) if n_comm_games > 0 else 0.0
        comm_all_meds = committee.groupby("opponent_team")["rushing_yards"].median().to_dict()
        comm_yds_comp = format_stat_comparison(med_comm_yds, comm_all_meds, n_comm_games, higher_is_better=False, unit="yards", min_sample=4)

        # 3. WR1s (highest targets on their team for the game)
        wr_only = w_df[w_df["position"] == "WR"].copy()
        if not wr_only.empty:
            wr1_idx = wr_only.groupby(["team", "week"])["targets"].idxmax()
            wr1s = wr_only.loc[wr1_idx]
            def_wr1 = wr1s[wr1s["opponent_team"] == defense_team]
            n_wr1 = len(def_wr1)
            med_wr1_yds = float(def_wr1["receiving_yards"].median()) if n_wr1 > 0 else 0.0
            wr1_all_meds = wr1s.groupby("opponent_team")["receiving_yards"].median().to_dict()
            wr1_comp = format_stat_comparison(med_wr1_yds, wr1_all_meds, n_wr1, higher_is_better=False, unit="yards", min_sample=3)
        else:
            wr1_comp = format_stat_comparison(0.0, {}, 0)

        # 4. High-Volume TEs (>= 15% target share or >= 5 targets)
        tes = w_df[(w_df["position"] == "TE") & (w_df["targets"] >= 5)]
        def_te = tes[tes["opponent_team"] == defense_team]
        n_te = len(def_te)
        med_te_yds = float(def_te["receiving_yards"].median()) if n_te > 0 else 0.0
        te_all_meds = tes.groupby("opponent_team")["receiving_yards"].median().to_dict()
        te_comp = format_stat_comparison(med_te_yds, te_all_meds, n_te, higher_is_better=False, unit="yards", min_sample=3)

        archetype_stats = {
            "vs_workhorse_rbs_15_plus_carries": {
                "median_rush_yards_allowed": wh_yds_comp,
                "median_ypc_allowed": round(med_wh_ypc, 2),
                "games_faced": n_wh_games,
            },
            "vs_committee_rbs_under_10_carries": {
                "median_rush_yards_allowed": comm_yds_comp,
                "games_faced": n_comm_games,
            },
            "vs_wr1_top_target": {
                "median_receiving_yards_allowed": wr1_comp,
                "games_faced": wr1_comp["sample_size"],
            },
            "vs_high_volume_tes_5_plus_targets": {
                "median_receiving_yards_allowed": te_comp,
                "games_faced": te_comp["sample_size"],
            },
        }

    return {
        "defense_team": defense_team,
        "season": season,
        "up_to_week": up_to_week,
        "total_plays_defended": len(df[df["defteam"] == defense_team]),
        "rush_defense": rush_defense,
        "pass_defense": pass_defense,
        "positions_allowed": pos_allowed,
        "player_archetypes_allowed": archetype_stats,
    }


def get_team_tendencies(
    team: str,
    season: int = 2026,
    up_to_week: Optional[int] = None,
    pbp_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Descriptive team tendency profile lookup enforcing point-in-time up_to_week cutoff.
    """
    return compute_team_tendency_profile(
        team=team,
        season=season,
        up_to_week=up_to_week,
        pbp_df=pbp_df,
    )
