"""
backend/definitions.py

Single source of truth for play filters, game-state buckets, sample-size flags,
opponent adjustment, and situational metrics. Every part of the scouting engine
imports from here so definitions remain consistent across all matchup sheets.

All metrics are strictly descriptive: what happened, sample size, and comparison
to league median. No betting models, win probabilities, or fair lines.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Situational & Context Thresholds
# --------------------------------------------------------------------------
WP_MIN, WP_MAX = 0.05, 0.95          # drop garbage time outside this range (win prob 5% - 95%)
LEAD_MARGIN = 7                      # leading if score_differential >= +7
TRAIL_MARGIN = -7                    # trailing if score_differential <= -7

MIN_PLAYS_FLAG = 40                  # flag any metric built on fewer plays than this
MIN_GAMES_FLAG = 4                   # flag any metric built on fewer games than this
SAMPLE_SIZE_THRESHOLDS = {
    "plays": MIN_PLAYS_FLAG,
    "games": MIN_GAMES_FLAG,
}

LIGHT_BOX_MAX = 6                    # defenders_in_box <= 6  -> light box
HEAVY_BOX_MIN = 8                    # defenders_in_box >= 8  -> heavy box
BLITZ_MIN_RUSHERS = 5                # number_of_pass_rushers >= 5 -> blitz
EXPLOSIVE_RUN_YDS = 10               # explosive run = 10+ yards
STUFF_RUN_YDS = 0                    # stuffed = 0 or fewer yards


# --------------------------------------------------------------------------
# Play filters
# --------------------------------------------------------------------------
def filter_plays(
    pbp: pd.DataFrame,
    kind: str = "all",
    drop_garbage_time: bool = True,
    drop_qb_scrambles: bool = False,
) -> pd.DataFrame:
    """
    Standard play filter for descriptive scouting.

    kind: 'all' (pass + run), 'pass', or 'run'.
    drop_qb_scrambles: set True for RB rushing numbers (scrambles are
    recorded as run plays with qb_scramble == 1).
    """
    df = pbp
    if "play_type" in df.columns:
        types = {"all": ["pass", "run"], "pass": ["pass"], "run": ["run"]}[kind]
        df = df[df["play_type"].isin(types)]
    if "epa" in df.columns:
        df = df[df["epa"].notna()]
    for col in ("qb_kneel", "qb_spike"):
        if col in df.columns:
            df = df[df[col].fillna(0) != 1]
    if drop_garbage_time and "wp" in df.columns:
        df = df[(df["wp"] >= WP_MIN) & (df["wp"] <= WP_MAX)]
    if drop_qb_scrambles and "qb_scramble" in df.columns:
        df = df[df["qb_scramble"].fillna(0) != 1]
    return df


def success_col(df: pd.DataFrame) -> pd.Series:
    """Use the nflverse `success` column everywhere (EPA > 0)."""
    return df["success"].astype(float)


# --------------------------------------------------------------------------
# Game state
# --------------------------------------------------------------------------
def game_state(score_differential: pd.Series) -> pd.Series:
    """
    Offense-perspective game state. score_differential is the possession
    team's lead (positive = leading), as in nflverse.
    Returns 'leading' / 'neutral' / 'trailing'.
    """
    s = score_differential
    out = pd.Series("neutral", index=s.index, dtype="object")
    out[s >= LEAD_MARGIN] = "leading"
    out[s <= TRAIL_MARGIN] = "trailing"
    out[s.isna()] = np.nan
    return out


# --------------------------------------------------------------------------
# Sample-size flags
# --------------------------------------------------------------------------
def low_sample(n_plays: Optional[float] = None, n_games: Optional[float] = None) -> bool:
    """True if the metric rests on too little data (plays < 40 or games < 4)."""
    if n_plays is not None and n_plays < MIN_PLAYS_FLAG:
        return True
    if n_games is not None and n_games < MIN_GAMES_FLAG:
        return True
    return False


# --------------------------------------------------------------------------
# Opponent adjustment
# --------------------------------------------------------------------------
def opponent_adjusted(
    games: pd.DataFrame,
    team_col: str,
    opp_col: str,
    value_col: str,
) -> pd.DataFrame:
    """
    Opponent-adjust a per-game metric (one row per team-game) using leave-one-out.

    For each team-game, compares the opponent's value that game to the
    opponent's average in its OTHER games, then averages those differences
    per team. Positive adj = the team's opponents did better than usual
    against it (matters when value_col is 'offense produced' and the
    opponent is a defense; sign flips depending on what you feed in).

    Returns one row per team: team, adj_value, games.
    """
    g = games[[team_col, opp_col, value_col]].dropna().copy()
    totals = g.groupby(opp_col)[value_col].agg(["sum", "count"]).rename(
        columns={"sum": "opp_sum", "count": "opp_n"}
    )

    # Exact leave-out: opponent's mean excluding its game(s) vs. this team.
    merged = g.join(totals, on=opp_col)
    n_other = merged["opp_n"] - 1
    opp_other_mean = (merged["opp_sum"] - merged[value_col]) / n_other.where(n_other > 0)
    merged["diff"] = merged[value_col] - opp_other_mean
    out = merged.groupby(team_col).agg(adj_value=("diff", "mean"), games=("diff", "count"))
    return out.reset_index()


# --------------------------------------------------------------------------
# Situational rate helpers
# --------------------------------------------------------------------------
def rush_metrics(rush_plays: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Rush EPA/play, success rate, stuff rate, explosive rate, plays."""
    r = rush_plays.copy()
    r["_stuff"] = (r["yards_gained"] <= STUFF_RUN_YDS).astype(float)
    r["_expl"] = (r["yards_gained"] >= EXPLOSIVE_RUN_YDS).astype(float)
    r["_succ"] = success_col(r)
    return (
        r.groupby(group_cols)
        .agg(
            plays=("epa", "size"),
            epa_per_play=("epa", "mean"),
            success_rate=("_succ", "mean"),
            stuff_rate=("_stuff", "mean"),
            explosive_rate=("_expl", "mean"),
        )
        .reset_index()
    )


def dropback_metrics(pass_plays: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Dropback EPA/play and success rate (sacks count as dropbacks)."""
    p = pass_plays.copy()
    p["_succ"] = success_col(p)
    return (
        p.groupby(group_cols)
        .agg(
            dropbacks=("epa", "size"),
            epa_per_dropback=("epa", "mean"),
            success_rate=("_succ", "mean"),
        )
        .reset_index()
    )
