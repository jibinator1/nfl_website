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

# Determine cache directory dynamically (supporting Vercel api/data and local data)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDIDATE_CACHE_DIRS = [
    os.path.join(ROOT_DIR, 'api', 'data', 'pbp_cache'),
    os.path.join(ROOT_DIR, 'data', 'pbp_cache'),
    os.path.join(os.getcwd(), 'api', 'data', 'pbp_cache'),
    os.path.join(os.getcwd(), 'data', 'pbp_cache'),
]

PBP_CACHE_DIR = os.path.join(ROOT_DIR, 'data', 'pbp_cache')
for c in CANDIDATE_CACHE_DIRS:
    if os.path.exists(os.path.join(c, 'player_pbp_features.parquet')):
        PBP_CACHE_DIR = c
        break

os.makedirs(PBP_CACHE_DIR, exist_ok=True)

PLAYER_CACHE_FILE = os.path.join(PBP_CACHE_DIR, 'player_pbp_features.parquet')
DEF_CACHE_FILE = os.path.join(PBP_CACHE_DIR, 'def_depth_features.parquet')
NEUTRAL_CACHE_FILE = os.path.join(PBP_CACHE_DIR, 'team_neutral_features.parquet')
TRENCH_CACHE_FILE = os.path.join(PBP_CACHE_DIR, 'trench_features.parquet')
EXPLOSIVE_CACHE_FILE = os.path.join(PBP_CACHE_DIR, 'explosive_features.parquet')
SUMMARY_CACHE_FILE = os.path.join(PBP_CACHE_DIR, 'team_pbp_summary.parquet')

# Authoritative completed seasons for macro baseline (2-year quality benchmark)
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

    # 1. Receiver Pass-Depth Targets & Target Shares
    rec_depth = passes.dropna(subset=['receiver_player_id']).groupby(
        ['season', 'week', 'receiver_player_id', 'posteam']
    ).agg(
        targets=('play_id', 'count'),
        short_targets=('is_short', 'sum'),
        inter_targets=('is_inter', 'sum'),
        deep_targets=('is_deep', 'sum'),
        total_air_yards=('air_yards', 'sum')
    ).reset_index().rename(columns={'receiver_player_id': 'player_id', 'posteam': 'team'})

    rec_depth['short_target_share'] = rec_depth['short_targets'] / rec_depth['targets'].clip(lower=1)
    rec_depth['inter_target_share'] = rec_depth['inter_targets'] / rec_depth['targets'].clip(lower=1)
    rec_depth['deep_target_share'] = rec_depth['deep_targets'] / rec_depth['targets'].clip(lower=1)
    rec_depth['adot'] = rec_depth['total_air_yards'] / rec_depth['targets'].clip(lower=1)

    rec_depth = rec_depth.sort_values(['player_id', 'season', 'week'])
    for col in ['short_target_share', 'inter_target_share', 'deep_target_share', 'adot']:
        rec_depth[f'roll_{col}'] = rec_depth.groupby('player_id')[col].transform(
            lambda x: x.rolling(rolling_window, min_periods=2).mean().shift(1)
        )
    rec_depth['adot_trajectory'] = rec_depth['roll_adot'] - rec_depth.groupby('player_id')['roll_adot'].shift(1)
    rec_depth_df = rec_depth[['player_id', 'season', 'week', 'roll_short_target_share', 'roll_inter_target_share', 'roll_deep_target_share', 'roll_adot', 'adot_trajectory']].copy()

    # 2. QB Pass-Depth Distribution & EPA/Att
    qb_depth = passes.dropna(subset=['passer_player_id']).groupby(
        ['season', 'week', 'passer_player_id', 'posteam']
    ).agg(
        attempts=('play_id', 'count'),
        deep_att=('is_deep', 'sum'),
        deep_epa=('epa', lambda x: x[passes.loc[x.index, 'is_deep'] == 1].sum() if (passes.loc[x.index, 'is_deep'] == 1).any() else 0.0),
        cpoe=('cpoe', 'mean'),
        air_yards=('air_yards', 'mean')
    ).reset_index().rename(columns={'passer_player_id': 'player_id', 'posteam': 'team'})

    qb_depth['deep_pass_rate'] = qb_depth['deep_att'] / qb_depth['attempts'].clip(lower=1)
    qb_depth['deep_pass_epa_rate'] = qb_depth['deep_epa'] / qb_depth['attempts'].clip(lower=1)
    qb_depth = qb_depth.sort_values(['player_id', 'season', 'week'])
    for col in ['deep_pass_rate', 'deep_pass_epa_rate', 'cpoe', 'air_yards']:
        qb_depth[f'roll_{col}'] = qb_depth.groupby('player_id')[col].transform(
            lambda x: x.rolling(rolling_window, min_periods=2).mean().shift(1)
        )
    qb_depth_df = qb_depth[['player_id', 'season', 'week', 'roll_deep_pass_rate', 'roll_deep_pass_epa_rate', 'roll_cpoe', 'roll_air_yards']].copy()

    # 3. Defensive Pass-Depth Vulnerability
    def_depth = passes.dropna(subset=['defteam']).groupby(['season', 'week', 'defteam']).agg(
        opp_att=('play_id', 'count'),
        opp_deep_att=('is_deep', 'sum'),
        opp_deep_epa=('epa', lambda x: x[passes.loc[x.index, 'is_deep'] == 1].sum() if (passes.loc[x.index, 'is_deep'] == 1).any() else 0.0),
        opp_short_att=('is_short', 'sum'),
        opp_inter_att=('is_inter', 'sum'),
    ).reset_index().rename(columns={'defteam': 'team'})

    def_depth['opp_deep_pass_rate'] = def_depth['opp_deep_att'] / def_depth['opp_att'].clip(lower=1)
    def_depth['opp_deep_epa_rate'] = def_depth['opp_deep_epa'] / def_depth['opp_att'].clip(lower=1)
    def_depth = def_depth.sort_values(['team', 'season', 'week'])
    for col in ['opp_deep_pass_rate', 'opp_deep_epa_rate']:
        def_depth[f'roll_{col}'] = def_depth.groupby('team')[col].transform(
            lambda x: x.rolling(rolling_window, min_periods=2).mean().shift(1)
        )
    def_depth_df = def_depth[['team', 'season', 'week', 'roll_opp_deep_pass_rate', 'roll_opp_deep_epa_rate']].copy()

    # 4. Neutral Game-Script Metrics (20% <= WP <= 80%)
    neutral_pbp = pbp[(pbp['wp'] >= 0.20) & (pbp['wp'] <= 0.80)].copy()
    team_neutral = neutral_pbp.dropna(subset=['posteam']).groupby(['season', 'week', 'posteam']).agg(
        neutral_plays=('play_id', 'count'),
        neutral_passes=('play_type', lambda x: (x == 'pass').sum()),
        neutral_epa=('epa', 'mean'),
        neutral_sec_per_play=('half_seconds_remaining', lambda x: (x.max() - x.min()) / max(len(x), 1))
    ).reset_index().rename(columns={'posteam': 'team'})

    team_neutral['neutral_pass_rate'] = team_neutral['neutral_passes'] / team_neutral['neutral_plays'].clip(lower=1)
    team_neutral = team_neutral.sort_values(['team', 'season', 'week'])
    for col in ['neutral_pass_rate', 'neutral_epa', 'neutral_sec_per_play']:
        team_neutral[f'roll_{col}'] = team_neutral.groupby('team')[col].transform(
            lambda x: x.rolling(rolling_window, min_periods=2).mean().shift(1)
        )
    team_neutral_df = team_neutral[['team', 'season', 'week', 'roll_neutral_pass_rate', 'roll_neutral_epa', 'roll_neutral_sec_per_play']].copy()

    # RB Neutral Efficiency
    rb_neutral = neutral_pbp[neutral_pbp['play_type'] == 'run'].dropna(subset=['rusher_player_id']).groupby(
        ['season', 'week', 'rusher_player_id']
    ).agg(
        neutral_carries=('play_id', 'count'),
        neutral_rush_yards=('yards_gained', 'sum'),
        neutral_rush_epa=('epa', 'mean')
    ).reset_index().rename(columns={'rusher_player_id': 'player_id'})

    rb_neutral['neutral_ypc'] = rb_neutral['neutral_rush_yards'] / rb_neutral['neutral_carries'].clip(lower=1)
    rb_neutral = rb_neutral.sort_values(['player_id', 'season', 'week'])
    for col in ['neutral_ypc', 'neutral_rush_epa']:
        rb_neutral[f'roll_{col}'] = rb_neutral.groupby('player_id')[col].transform(
            lambda x: x.rolling(rolling_window, min_periods=2).mean().shift(1)
        )
    rb_neutral_df = rb_neutral[['player_id', 'season', 'week', 'roll_neutral_ypc', 'roll_neutral_rush_epa']].copy()

    # 5. Pressure & Trench Metrics
    off_press = passes.dropna(subset=['posteam']).groupby(['season', 'week', 'posteam']).agg(
        dropbacks=('play_id', 'count'),
        pressures=('is_pressured', 'sum')
    ).reset_index().rename(columns={'posteam': 'team'})
    off_press['pressure_rate_allowed'] = off_press['pressures'] / off_press['dropbacks'].clip(lower=1)
    off_press = off_press.sort_values(['team', 'season', 'week'])
    off_press['roll_pressure_rate_allowed'] = off_press.groupby('team')['pressure_rate_allowed'].transform(
        lambda x: x.rolling(rolling_window, min_periods=2).mean().shift(1)
    )

    def_press = passes.dropna(subset=['defteam']).groupby(['season', 'week', 'defteam']).agg(
        opp_dropbacks=('play_id', 'count'),
        pressures_gen=('is_pressured', 'sum')
    ).reset_index().rename(columns={'defteam': 'team'})
    def_press['pressure_rate_gen'] = def_press['pressures_gen'] / def_press['opp_dropbacks'].clip(lower=1)
    def_press = def_press.sort_values(['team', 'season', 'week'])
    def_press['roll_pressure_rate_gen'] = def_press.groupby('team')['pressure_rate_gen'].transform(
        lambda x: x.rolling(rolling_window, min_periods=2).mean().shift(1)
    )

    trench_df = off_press[['team', 'season', 'week', 'roll_pressure_rate_allowed']].merge(
        def_press[['team', 'season', 'week', 'roll_pressure_rate_gen']],
        on=['team', 'season', 'week'], how='outer'
    )

    # 6. Explosive Play Volatility
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
    def_exp_p['opp_exp_pass_rate'] = def_exp_p['opp_exp_p'] / def_exp_p['opp_att'].clip(lower=1)

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

    # 7. Team Season Summary
    recent_pbp = pbp[pbp['season'].isin(MACRO_BASELINE_SEASONS)].copy()
    rec_passes = recent_pbp[recent_pbp['play_type'] == 'pass'].copy()
    rec_passes['is_deep'] = (rec_passes['air_yards'] >= 20).astype(int)
    rec_passes['is_exp'] = (rec_passes['yards_gained'] >= 20).astype(int)

    team_deep_off = rec_passes.groupby('posteam').agg(
        deep_pass_attempts=('is_deep', 'sum'),
        total_pass_att=('play_id', 'count'),
        deep_pass_epa=('epa', lambda x: x[rec_passes.loc[x.index, 'is_deep'] == 1].mean() if (rec_passes.loc[x.index, 'is_deep'] == 1).any() else 0.0),
        exp_pass_rate=('is_exp', 'mean'),
        avg_air_yards=('air_yards', 'mean')
    ).reset_index().rename(columns={'posteam': 'team'})
    team_deep_off['deep_pass_rate'] = team_deep_off['deep_pass_attempts'] / team_deep_off['total_pass_att'].clip(lower=1)

    team_deep_def = rec_passes.groupby('defteam').agg(
        opp_deep_pass_attempts=('is_deep', 'sum'),
        opp_total_pass_att=('play_id', 'count'),
        opp_deep_pass_epa=('epa', lambda x: x[rec_passes.loc[x.index, 'is_deep'] == 1].mean() if (rec_passes.loc[x.index, 'is_deep'] == 1).any() else 0.0),
        opp_exp_pass_rate=('is_exp', 'mean'),
    ).reset_index().rename(columns={'defteam': 'team'})
    team_deep_def['opp_deep_pass_rate'] = team_deep_def['opp_deep_pass_attempts'] / team_deep_def['opp_total_pass_att'].clip(lower=1)

    team_summary = team_deep_off.merge(team_deep_def, on='team', how='outer')

    team_press_off = rec_passes.groupby('posteam')['is_pressured'].mean().reset_index().rename(
        columns={'posteam': 'team', 'is_pressured': 'pressure_rate_allowed'}
    )
    team_press_def = rec_passes.groupby('defteam')['is_pressured'].mean().reset_index().rename(
        columns={'defteam': 'team', 'is_pressured': 'pressure_rate_generated'}
    )
    team_summary = team_summary.merge(team_press_off, on='team', how='outer').merge(team_press_def, on='team', how='outer')

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


def _pbp_cache_files_exist():
    """Returns True if all required parquet cache files exist on disk."""
    cache_files = [PLAYER_CACHE_FILE, DEF_CACHE_FILE, NEUTRAL_CACHE_FILE,
                   TRENCH_CACHE_FILE, EXPLOSIVE_CACHE_FILE, SUMMARY_CACHE_FILE]
    return all(os.path.exists(f) for f in cache_files)


def load_pbp_features(seasons=[2023, 2024, 2025, 2026], force_reload=False):
    """
    Loads advanced PBP features from parquet cache.
    On Vercel (serverless), ALWAYS reads pre-baked cache to stay within memory limits.
    Only downloads and compiles raw PBP when running offline/locally with force_reload.
    """
    is_serverless = os.getenv('VERCEL') is not None or os.getenv('AWS_LAMBDA_FUNCTION_NAME') is not None
    
    # 1. Fast Cache Read (0.01s, < 50MB RAM)
    if _pbp_cache_files_exist() and (is_serverless or not force_reload):
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
            print(f"[PBP Cache] Error reading cache file: {e}")
            if is_serverless:
                # Return empty safe fallbacks on serverless to prevent out-of-memory crash
                return {
                    'player_feats': pd.DataFrame(), 'def_depth': pd.DataFrame(),
                    'team_neutral': pd.DataFrame(), 'trench': pd.DataFrame(),
                    'explosive': pd.DataFrame(), 'team_summary': pd.DataFrame()
                }

    if is_serverless:
        # Prevent any multi-GB PBP download on Vercel
        print("[PBP Cache] Serverless environment: Skipping raw PBP extraction.")
        return {
            'player_feats': pd.DataFrame(), 'def_depth': pd.DataFrame(),
            'team_neutral': pd.DataFrame(), 'trench': pd.DataFrame(),
            'explosive': pd.DataFrame(), 'team_summary': pd.DataFrame()
        }

    # 2. Local / Offline Ingestion (Only when run locally via daily_update.py)
    try:
        import nflreadpy as nfl
        print(f"Loading PBP data for seasons {seasons} via nflreadpy...")
        pbp = nfl.load_pbp(seasons=seasons).to_pandas()
        feats = extract_pbp_features(pbp)

        for target_pbp_dir in [PBP_CACHE_DIR, os.path.join(ROOT_DIR, 'api', 'data', 'pbp_cache')]:
            os.makedirs(target_pbp_dir, exist_ok=True)
            feats['player_feats'].to_parquet(os.path.join(target_pbp_dir, 'player_pbp_features.parquet'), index=False)
            feats['def_depth'].to_parquet(os.path.join(target_pbp_dir, 'def_depth_features.parquet'), index=False)
            feats['team_neutral'].to_parquet(os.path.join(target_pbp_dir, 'team_neutral_features.parquet'), index=False)
            feats['trench'].to_parquet(os.path.join(target_pbp_dir, 'trench_features.parquet'), index=False)
            feats['explosive'].to_parquet(os.path.join(target_pbp_dir, 'explosive_features.parquet'), index=False)
            feats['team_summary'].to_parquet(os.path.join(target_pbp_dir, 'team_pbp_summary.parquet'), index=False)
        print(f"[PBP Cache] Saved all feature cache tables.")
        return feats
    except Exception as e:
        print(f"[PBP Cache] Local PBP extraction error: {e}")
        return {
            'player_feats': pd.DataFrame(), 'def_depth': pd.DataFrame(),
            'team_neutral': pd.DataFrame(), 'trench': pd.DataFrame(),
            'explosive': pd.DataFrame(), 'team_summary': pd.DataFrame()
        }
