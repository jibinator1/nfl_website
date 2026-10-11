"""
Generates game-by-game situational and tempo features parquet table
for all 32 NFL teams across seasons 2023-2026.
Saved to data/pbp_cache/team_situational_game.parquet
and api/data/pbp_cache/team_situational_game.parquet.
"""

import os
import nflreadpy as nfl
import pandas as pd
import numpy as np

def build_situational_features():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_paths = [
        os.path.join(root_dir, 'data', 'pbp_cache', 'team_situational_game.parquet'),
        os.path.join(root_dir, 'api', 'data', 'pbp_cache', 'team_situational_game.parquet')
    ]

    seasons = [2023, 2024, 2025, 2026]
    print(f"Loading PBP data for seasons {seasons}...")
    pbp = nfl.load_pbp(seasons=seasons).to_pandas()
    reg = pbp[pbp['season_type'] == 'REG'].copy()

    # Normalize team codes: LAR -> LA
    reg['team'] = reg['posteam'].replace({'LAR': 'LA'})

    rows = []
    print("Extracting game-by-game situational metrics...")
    for (gid, tm), g_df in reg.groupby(['game_id', 'team']):
        if not tm or pd.isna(tm):
            continue
        season = int(g_df['season'].iloc[0])
        week = int(g_df['week'].iloc[0])
        gameday = str(g_df['game_date'].iloc[0]) if 'game_date' in g_df.columns else ''

        # Offensive plays
        off_plays = g_df[g_df['play_type'].isin(['pass', 'run'])]
        plays_count = len(off_plays)

        # Neutral pace (20% <= win prob <= 80%)
        neutral_pbp = g_df[(g_df['wp'] >= 0.20) & (g_df['wp'] <= 0.80)]
        if len(neutral_pbp) >= 5 and 'half_seconds_remaining' in neutral_pbp.columns:
            hs = neutral_pbp['half_seconds_remaining']
            sec_per_play = float((hs.max() - hs.min()) / max(len(neutral_pbp), 1))
            sec_per_play = max(24.0, min(33.0, sec_per_play))
        else:
            sec_per_play = 28.0

        # Red Zone (inside 20)
        rz = off_plays[off_plays['yardline_100'] <= 20]
        rz_pass = int((rz['play_type'] == 'pass').sum())
        rz_run = int((rz['play_type'] == 'run').sum())
        rz_tot = rz_pass + rz_run

        # Top RZ rusher
        rz_rushes = rz[rz['play_type'] == 'run']
        if not rz_rushes.empty:
            r_counts = rz_rushes['rusher_player_name'].dropna().value_counts()
            top_rusher = r_counts.index[0] if len(r_counts) > 0 else ''
            top_rushes = int(r_counts.iloc[0]) if len(r_counts) > 0 else 0
        else:
            top_rusher = ''
            top_rushes = 0

        # 3rd Down
        d3 = off_plays[off_plays['down'] == 3]
        d3_pass = int((d3['play_type'] == 'pass').sum())
        d3_run = int((d3['play_type'] == 'run').sum())
        d3_tot = d3_pass + d3_run

        # Passes & Checkdowns (air_yards <= 0)
        passes = off_plays[off_plays['play_type'] == 'pass']
        pass_cnt = len(passes)
        checkdowns = int((passes['air_yards'] <= 0).sum())

        rows.append({
            'game_id': str(gid),
            'team': str(tm),
            'season': season,
            'week': week,
            'gameday': gameday,
            'plays': plays_count,
            'neutral_sec_per_play': round(sec_per_play, 1),
            'rz_pass': rz_pass,
            'rz_run': rz_run,
            'rz_total': rz_tot,
            'top_rz_rusher': top_rusher,
            'top_rz_rushes': top_rushes,
            'd3_pass': d3_pass,
            'd3_run': d3_run,
            'd3_total': d3_tot,
            'passes': pass_cnt,
            'checkdowns': checkdowns
        })

    df_sit = pd.DataFrame(rows)
    print(f"Total processed team-games: {len(df_sit)}")

    for p in target_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        df_sit.to_parquet(p, index=False)
        print(f"Saved: {p} ({os.path.getsize(p)} bytes)")

if __name__ == '__main__':
    build_situational_features()
