"""
PBP Advanced Feature Extraction & Caching Engine
=================================================
Extracts leak-free, shift(1) rolling features from play-by-play data:
1. Pass-depth & aDOT splits: Short (<=10), Intermediate (11-19), Deep (20+) air yards & aDOT trajectory.
2. Neutral game-script metrics: Win probability between 20% and 80%, filtering out blowout / garbage-time noise.
3. Pressure & Trench Protection: Dropbacks, sacks, QB hits, offensive pressure rate allowed vs defense generated.
4. Explosive Play Rates: 20+ yard passes & 10+ yard rushes created vs allowed.
5. Persistent disk caching to parquet for sub-second reloads.
"""

import os
import time
import numpy as np
import pandas as pd
import nflreadpy as nfl

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
PBP_CACHE_DIR = os.path.join(DATA_DIR, 'pbp_cache')
os.makedirs(PBP_CACHE_DIR, exist_ok=True)

PLAYER_CACHE_FILE = os.path.join(PBP_CACHE_DIR, 'player_pbp_features.parquet')
DEF_CACHE_FILE = os.path.join(PBP_CACHE_DIR, 'def_depth_features.parquet')
NEUTRAL_CACHE_FILE = os.path.join(PBP_CACHE_DIR, 'team_neutral_features.parquet')
TRENCH_CACHE_FILE = os.path.join(PBP_CACHE_DIR, 'trench_features.parquet')
EXPLOSIVE_CACHE_FILE = os.path.join(PBP_CACHE_DIR, 'explosive_features.parquet')
SUMMARY_CACHE_FILE = os.path.join(PBP_CACHE_DIR, 'team_pbp_summary.parquet')

# Authoritative completed seasons for macro baseline (2-year quality benchmark)
# Must only be updated manually after a full season settles, NEVER dynamically.
MACRO_BASELINE_SEASONS = [2024, 2025]


def extract_pbp_features(pbp: pd.DataFrame, rolling_window=5):
    """
    Computes all advanced player & team rolling metrics with strict .shift(1) chronological lag.
    Zero future leakage guaranteed.
    """
    print("Engineering advanced PBP features (Pass Depth, Neutral Script, Pressure, Explosiveness)...")
    t0 = time.time()

    # Normalize basic columns
    pbp['air_yards'] = pd.to_numeric(pbp.get('air_yards', 0), errors='coerce')
    pbp['yards_gained'] = pd.to_numeric(pbp.get('yards_gained', 0), errors='coerce').fillna(0)
    pbp['epa'] = pd.to_numeric(pbp.get('epa', 0), errors='coerce').fillna(0)
    pbp['cpoe'] = pd.to_numeric(pbp.get('cpoe', 0), errors='coerce').fillna(0)
    pbp['wp'] = pd.to_numeric(pbp.get('wp', 0.5), errors='coerce').fillna(0.5)
    pbp['half_seconds_remaining'] = pd.to_numeric(pbp.get('half_seconds_remaining', 900), errors='coerce').fillna(900)

    # Pressure indicator: sacks + qb_hits (and was_pressure if present)
    has_wp = 'was_pressure' in pbp.columns
    press_mask = (pbp.get('qb_hit', 0) == 1) | (pbp.get('sack', 0) == 1)
    if has_wp:
        press_mask = press_mask | (pbp['was_pressure'] == 1)
    pbp['is_pressured'] = press_mask.astype(int)

    # Pass depth buckets
    passes = pbp[pbp['play_type'] == 'pass'].copy()
    passes['is_short'] = (passes['air_yards'] <= 10).astype(int)
    passes['is_inter'] = ((passes['air_yards'] > 10) & (passes['air_yards'] < 20)).astype(int)
    passes['is_deep'] = (passes['air_yards'] >= 20).astype(int)
    passes['is_explosive_pass'] = (passes['yards_gained'] >= 20).astype(int)

    runs = pbp[pbp['play_type'] == 'run'].copy()
    runs['is_explosive_rush'] = (runs['yards_gained'] >= 10).astype(int)

    # -------------------------------------------------------------------------
    # 1. Receiver Pass Depth & aDOT Trajectory
    # -------------------------------------------------------------------------
    rec_plays = passes.dropna(subset=['receiver_player_id']).copy()
    rec_game = rec_plays.groupby(['season', 'week', 'receiver_player_id']).agg(
        rec_targets=('play_id', 'count'),
        rec_short_targets=('is_short', 'sum'),
        rec_deep_targets=('is_deep', 'sum'),
        rec_adot=('air_yards', 'mean'),
        rec_short_yards=('yards_gained', lambda x: x[rec_plays.loc[x.index, 'is_short'] == 1].sum()),
        rec_deep_yards=('yards_gained', lambda x: x[rec_plays.loc[x.index, 'is_deep'] == 1].sum()),
    ).reset_index().rename(columns={'receiver_player_id': 'player_id'})

    rec_game['short_target_share'] = rec_game['rec_short_targets'] / rec_game['rec_targets'].clip(lower=1)
    rec_game['deep_target_share'] = rec_game['rec_deep_targets'] / rec_game['rec_targets'].clip(lower=1)

    rec_game = rec_game.sort_values(['player_id', 'season', 'week'])
    for col in ['short_target_share', 'deep_target_share', 'rec_adot', 'rec_short_yards', 'rec_deep_yards']:
        rec_game[f'roll_{col}'] = rec_game.groupby('player_id')[col].transform(
            lambda x: x.rolling(rolling_window, min_periods=2).mean().shift(1)
        )
    # aDOT trend: 3-week recent vs 5-week baseline
    rec_game['roll_rec_adot_recent'] = rec_game.groupby('player_id')['rec_adot'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
    )
    rec_game['adot_trend'] = rec_game['roll_rec_adot_recent'] - rec_game['roll_rec_adot']

    rec_depth_df = rec_game[['player_id', 'season', 'week', 'roll_short_target_share',
                             'roll_deep_target_share', 'roll_rec_adot', 'adot_trend']].copy()

    # -------------------------------------------------------------------------
    # 2. Passer Pass Depth & Deep Accuracy (QB)
    # -------------------------------------------------------------------------
    passer_plays = passes.dropna(subset=['passer_player_id']).copy()
    qb_game = passer_plays.groupby(['season', 'week', 'passer_player_id']).agg(
        qb_pass_att=('play_id', 'count'),
        qb_deep_att=('is_deep', 'sum'),
        qb_adot=('air_yards', 'mean'),
        qb_deep_epa=('epa', lambda x: x[passer_plays.loc[x.index, 'is_deep'] == 1].mean()),
    ).reset_index().rename(columns={'passer_player_id': 'player_id'})

    qb_game['qb_deep_att_rate'] = qb_game['qb_deep_att'] / qb_game['qb_pass_att'].clip(lower=1)
    qb_game = qb_game.sort_values(['player_id', 'season', 'week'])
    for col in ['qb_deep_att_rate', 'qb_adot', 'qb_deep_epa']:
        qb_game[f'roll_{col}'] = qb_game.groupby('player_id')[col].transform(
            lambda x: x.rolling(rolling_window, min_periods=2).mean().shift(1)
        )

    qb_depth_df = qb_game[['player_id', 'season', 'week', 'roll_qb_deep_att_rate',
                           'roll_qb_adot', 'roll_qb_deep_epa']].copy()

    # -------------------------------------------------------------------------
    # 3. Defensive Pass Depth Allowed (Opponent Matchup)
    # -------------------------------------------------------------------------
    def_pass = passes.dropna(subset=['defteam']).groupby(['season', 'week', 'defteam']).agg(
        opp_pass_att=('play_id', 'count'),
        opp_deep_att=('is_deep', 'sum'),
        opp_deep_epa=('epa', lambda x: x[passes.loc[x.index, 'is_deep'] == 1].mean()),
        opp_short_epa=('epa', lambda x: x[passes.loc[x.index, 'is_short'] == 1].mean()),
    ).reset_index().rename(columns={'defteam': 'opponent_team'}).sort_values(['opponent_team', 'season', 'week'])

    def_pass['opp_deep_rate'] = def_pass['opp_deep_att'] / def_pass['opp_pass_att'].clip(lower=1)
    for col in ['opp_deep_rate', 'opp_deep_epa', 'opp_short_epa']:
        def_pass[f'roll_{col}'] = def_pass.groupby('opponent_team')[col].transform(
            lambda x: x.rolling(rolling_window, min_periods=2).mean().shift(1)
        )

    def_depth_df = def_pass[['opponent_team', 'season', 'week', 'roll_opp_deep_rate',
                             'roll_opp_deep_epa', 'roll_opp_short_epa']].copy()

    # -------------------------------------------------------------------------
    # 4. Neutral Game-Script (WP 20-80%, Excl. Final 2 Mins of Half)
    # -------------------------------------------------------------------------
    neutral_mask = (
        (pbp['wp'] >= 0.20) & 
        (pbp['wp'] <= 0.80) & 
        (pbp['half_seconds_remaining'] > 120) & 
        (pbp['play_type'].isin(['pass', 'run']))
    )
    pbp_neutral = pbp[neutral_mask].copy()
    pbp_neutral['is_pass'] = (pbp_neutral['play_type'] == 'pass').astype(int)

    team_neutral = pbp_neutral.dropna(subset=['posteam']).groupby(['season', 'week', 'posteam']).agg(
        neutral_plays=('play_id', 'count'),
        neutral_pass_plays=('is_pass', 'sum'),
        neutral_epa=('epa', 'mean'),
    ).reset_index().rename(columns={'posteam': 'team'}).sort_values(['team', 'season', 'week'])

    team_neutral['neutral_pass_rate'] = team_neutral['neutral_pass_plays'] / team_neutral['neutral_plays'].clip(lower=1)
    for col in ['neutral_pass_rate', 'neutral_epa']:
        team_neutral[f'roll_{col}'] = team_neutral.groupby('team')[col].transform(
            lambda x: x.rolling(rolling_window, min_periods=2).mean().shift(1)
        )
    team_neutral['roll_neutral_plays'] = team_neutral.groupby('team')['neutral_plays'].transform(
        lambda x: x.rolling(rolling_window, min_periods=2).sum().shift(1)
    )

    team_neutral_df = team_neutral[['team', 'season', 'week', 'roll_neutral_pass_rate', 'roll_neutral_epa', 'roll_neutral_plays']].copy()

    # RB Neutral Rush Share
    neutral_runs = pbp_neutral[pbp_neutral['play_type'] == 'run'].dropna(subset=['rusher_player_id', 'posteam']).copy()
    rb_neutral_counts = neutral_runs.groupby(['season', 'week', 'rusher_player_id', 'posteam']).size().reset_index(name='rb_neutral_rushes')
    team_neutral_runs = neutral_runs.groupby(['season', 'week', 'posteam']).size().reset_index(name='team_neutral_rushes')

    rb_neutral = rb_neutral_counts.merge(team_neutral_runs, on=['season', 'week', 'posteam'], how='left')
    rb_neutral['neutral_rush_share'] = rb_neutral['rb_neutral_rushes'] / rb_neutral['team_neutral_rushes'].clip(lower=1)
    rb_neutral = rb_neutral.rename(columns={'rusher_player_id': 'player_id'}).sort_values(['player_id', 'season', 'week'])

    rb_neutral['roll_neutral_rush_share'] = rb_neutral.groupby('player_id')['neutral_rush_share'].transform(
        lambda x: x.rolling(rolling_window, min_periods=2).mean().shift(1)
    )
    rb_neutral_df = rb_neutral[['player_id', 'season', 'week', 'roll_neutral_rush_share']].copy()

    # -------------------------------------------------------------------------
    # 5. Trench Pressure Metrics (Allowed by Offense vs Generated by Defense)
    # -------------------------------------------------------------------------
    off_press = passes.dropna(subset=['posteam']).groupby(['season', 'week', 'posteam']).agg(
        dropbacks=('play_id', 'count'),
        pressures_allowed=('is_pressured', 'sum')
    ).reset_index().rename(columns={'posteam': 'team'}).sort_values(['team', 'season', 'week'])
    off_press['pressure_rate_allowed'] = off_press['pressures_allowed'] / off_press['dropbacks'].clip(lower=1)
    off_press['roll_pressure_rate_allowed'] = off_press.groupby('team')['pressure_rate_allowed'].transform(
        lambda x: x.rolling(rolling_window, min_periods=2).mean().shift(1)
    )

    def_press = passes.dropna(subset=['defteam']).groupby(['season', 'week', 'defteam']).agg(
        def_dropbacks=('play_id', 'count'),
        pressures_gen=('is_pressured', 'sum')
    ).reset_index().rename(columns={'defteam': 'team'}).sort_values(['team', 'season', 'week'])
    def_press['pressure_rate_gen'] = def_press['pressures_gen'] / def_press['def_dropbacks'].clip(lower=1)
    def_press['roll_pressure_rate_gen'] = def_press.groupby('team')['pressure_rate_gen'].transform(
        lambda x: x.rolling(rolling_window, min_periods=2).mean().shift(1)
    )

    trench_df = off_press[['team', 'season', 'week', 'roll_pressure_rate_allowed']].merge(
        def_press[['team', 'season', 'week', 'roll_pressure_rate_gen']],
        on=['team', 'season', 'week'], how='outer'
    )

    # -------------------------------------------------------------------------
    # 6. Explosive Play Volatility (20+ Pass, 10+ Rush)
    # -------------------------------------------------------------------------
    off_exp_p = passes.dropna(subset=['posteam']).groupby(['season', 'week', 'posteam']).agg(
        p_att=('play_id', 'count'), exp_p=('is_explosive_pass', 'sum')
    ).reset_index().rename(columns={'posteam': 'team'})
    off_exp_p['exp_pass_rate'] = off_exp_p['exp_p'] / off_exp_p['p_att'].clip(lower=1)

    off_exp_r = runs.dropna(subset=['posteam']).groupby(['season', 'week', 'posteam']).agg(
        r_att=('play_id', 'count'), exp_r=('is_explosive_rush', 'sum')
    ).reset_index().rename(columns={'posteam': 'team'})
    off_exp_r['exp_rush_rate'] = off_exp_r['exp_r'] / off_exp_r['r_att'].clip(lower=1)

    def_exp_p = passes.dropna(subset=['defteam']).groupby(['season', 'week', 'defteam']).agg(
        opp_p_att=('play_id', 'count'), opp_exp_p=('is_explosive_pass', 'sum')
    ).reset_index().rename(columns={'defteam': 'team'})
    def_exp_p['opp_exp_pass_rate'] = def_exp_p['opp_exp_p'] / def_exp_p['opp_p_att'].clip(lower=1)

    explosive_raw = off_exp_p[['team', 'season', 'week', 'exp_pass_rate']].merge(
        off_exp_r[['team', 'season', 'week', 'exp_rush_rate']], on=['team', 'season', 'week'], how='outer'
    ).merge(
        def_exp_p[['team', 'season', 'week', 'opp_exp_pass_rate']], on=['team', 'season', 'week'], how='outer'
    ).sort_values(['team', 'season', 'week'])

    for col in ['exp_pass_rate', 'exp_rush_rate', 'opp_exp_pass_rate']:
        explosive_raw[f'roll_{col}'] = explosive_raw.groupby('team')[col].transform(
            lambda x: x.rolling(rolling_window, min_periods=2).mean().shift(1)
        )
    explosive_df = explosive_raw[['team', 'season', 'week', 'roll_exp_pass_rate', 'roll_exp_rush_rate', 'roll_opp_exp_pass_rate']].copy()

    # Consolidate player metrics
    player_df = rec_depth_df.merge(rb_neutral_df, on=['player_id', 'season', 'week'], how='outer')
    player_df = player_df.merge(qb_depth_df, on=['player_id', 'season', 'week'], how='outer')

    # -------------------------------------------------------------------------
    # 7. Team Season Summary (for Analytics Deep Dive Cards)
    # -------------------------------------------------------------------------
    # Strictly isolate macro baseline to explicit completed historical regular seasons (MACRO_BASELINE_SEASONS)
    # This guarantees zero in-season leakage and prevents partial/future 2026 game weeks from contaminating the 2-year macro baseline.
    recent_pbp = pbp[pbp['season'].isin(MACRO_BASELINE_SEASONS)].copy()
    rec_passes = recent_pbp[recent_pbp['play_type'] == 'pass'].copy()
    rec_passes['is_deep'] = (rec_passes['air_yards'] >= 20).astype(int)
    rec_passes['is_exp'] = (rec_passes['yards_gained'] >= 20).astype(int)

    team_deep_off = rec_passes.groupby('posteam').agg(
        deep_pass_attempts=('is_deep', 'sum'),
        total_pass_att=('play_id', 'count'),
        deep_pass_epa=('epa', lambda x: x[rec_passes.loc[x.index, 'is_deep'] == 1].mean()),
        exp_pass_rate=('is_exp', 'mean'),
        avg_air_yards=('air_yards', 'mean')
    ).reset_index().rename(columns={'posteam': 'team'})
    team_deep_off['deep_pass_rate'] = team_deep_off['deep_pass_attempts'] / team_deep_off['total_pass_att'].clip(lower=1)

    team_deep_def = rec_passes.groupby('defteam').agg(
        opp_deep_pass_attempts=('is_deep', 'sum'),
        opp_total_pass_att=('play_id', 'count'),
        opp_deep_pass_epa=('epa', lambda x: x[rec_passes.loc[x.index, 'is_deep'] == 1].mean()),
        opp_exp_pass_rate=('is_exp', 'mean'),
    ).reset_index().rename(columns={'defteam': 'team'})
    team_deep_def['opp_deep_pass_rate'] = team_deep_def['opp_deep_pass_attempts'] / team_deep_def['opp_total_pass_att'].clip(lower=1)

    # Combine into team summary
    team_summary = team_deep_off.merge(team_deep_def, on='team', how='outer')

    # Add pressure summary
    team_press_off = rec_passes.groupby('posteam')['is_pressured'].mean().reset_index().rename(
        columns={'posteam': 'team', 'is_pressured': 'pressure_rate_allowed'}
    )
    team_press_def = rec_passes.groupby('defteam')['is_pressured'].mean().reset_index().rename(
        columns={'defteam': 'team', 'is_pressured': 'pressure_rate_generated'}
    )
    team_summary = team_summary.merge(team_press_off, on='team', how='outer').merge(team_press_def, on='team', how='outer')

    # Add league ranks for team summary
    for c, asc in [
        ('deep_pass_epa', False), ('opp_deep_pass_epa', True),
        ('pressure_rate_allowed', True), ('pressure_rate_generated', False),
        ('exp_pass_rate', False), ('opp_exp_pass_rate', True)
    ]:
        if c in team_summary.columns:
            team_summary[f'rank_{c}'] = team_summary[c].rank(ascending=asc, method='min').fillna(16).astype(int)

    print(f"Engineered PBP features in {time.time()-t0:.2f}s.")
    return {
        'player_feats': player_df,
        'def_depth': def_depth_df,
        'team_neutral': team_neutral_df,
        'trench': trench_df,
        'explosive': explosive_df,
        'team_summary': team_summary
    }


def _pbp_cache_is_fresh(max_age_days=3):
    """
    Returns True if all cache files exist AND none is older than max_age_days.
    If any file is stale, the cache must be rebuilt to incorporate newly played weeks.
    """
    import time
    cache_files = [PLAYER_CACHE_FILE, DEF_CACHE_FILE, NEUTRAL_CACHE_FILE,
                   TRENCH_CACHE_FILE, EXPLOSIVE_CACHE_FILE, SUMMARY_CACHE_FILE]
    max_age_seconds = max_age_days * 86400
    now = time.time()
    for f in cache_files:
        if not os.path.exists(f):
            return False
        if now - os.path.getmtime(f) > max_age_seconds:
            print(f"[PBP Cache] Stale ({f.split(os.sep)[-1]} > {max_age_days}d old). Rebuilding for new game data...")
            return False
    return True


def load_pbp_features(seasons=[2023, 2024, 2025, 2026], force_reload=False):
    """
    Loads advanced PBP features from parquet cache if present and fresh; otherwise fetches PBP and compiles.
    Cache is automatically invalidated after 3 days to incorporate newly played game weeks.
    Always returns the uniform dict of feature dataframes.
    """
    all_cached = not force_reload and _pbp_cache_is_fresh(max_age_days=3)

    if all_cached:
        try:
            return {
                'player_feats': pd.read_parquet(PLAYER_CACHE_FILE),
                'def_depth': pd.read_parquet(DEF_CACHE_FILE),
                'team_neutral': pd.read_parquet(NEUTRAL_CACHE_FILE),
                'trench': pd.read_parquet(TRENCH_CACHE_FILE),
                'explosive': pd.read_parquet(EXPLOSIVE_CACHE_FILE),
                'team_summary': pd.read_parquet(SUMMARY_CACHE_FILE)
            }
        except Exception as e:
            print(f"[PBP Cache] Rebuilding due to cache read error: {e}")

    # Fetch PBP
    print(f"Loading PBP data for seasons {seasons} via nflreadpy...")
    pbp = nfl.load_pbp(seasons=seasons).to_pandas()
    feats = extract_pbp_features(pbp)

    # Save cache files
    try:
        feats['player_feats'].to_parquet(PLAYER_CACHE_FILE, index=False)
        feats['def_depth'].to_parquet(DEF_CACHE_FILE, index=False)
        feats['team_neutral'].to_parquet(NEUTRAL_CACHE_FILE, index=False)
        feats['trench'].to_parquet(TRENCH_CACHE_FILE, index=False)
        feats['explosive'].to_parquet(EXPLOSIVE_CACHE_FILE, index=False)
        feats['team_summary'].to_parquet(SUMMARY_CACHE_FILE, index=False)
        print(f"[PBP Cache] Saved all feature cache tables to {PBP_CACHE_DIR}.")
    except Exception as e:
        print(f"[PBP Cache] Note saving cache: {e}")

    return feats
