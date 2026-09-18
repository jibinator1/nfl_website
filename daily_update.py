#!/usr/bin/env python3
"""
NFL Analytics Hub - Daily Data Ingestion & Auto-Deploy Pipeline
==============================================================
Designed to run on Mac / Linux / Windows terminal or via cron.
1. Pulls the latest regular season player stats & game schedules via nflreadpy.
2. Recomputes implied team totals, rest days, and weather factors.
3. Updates pre-baked parquet caches in both api/data/ (Vercel) and data/ (local).
4. Synchronizes PBP feature caches.
5. Automatically commits and pushes to GitHub to trigger an instant Vercel redeploy.
"""

import os
import sys
import subprocess
import time
from datetime import datetime
import pandas as pd
import numpy as np

# Ensure project root in sys.path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import nflreadpy as nfl
from backend.pbp_features import load_pbp_features

SEASONS = [2023, 2024, 2025, 2026]
AUTO_PUSH = True


def fetch_and_update_data():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetching fresh NFL data for seasons {SEASONS}...")
    
    # 1. Fetch schedules
    sched_dfs = []
    for s in SEASONS:
        try:
            sc = nfl.load_schedules(seasons=[s]).to_pandas()
            sc = sc[sc['game_type'] == 'REG'].copy()
            sched_dfs.append(sc)
            print(f"  [OK] Schedules loaded for {s} ({len(sc)} games)")
        except Exception as e:
            print(f"  [Notice] Loading {s} schedules: {e}")

    # 2. Fetch weekly player stats
    weekly_dfs = []
    for s in SEASONS:
        try:
            w = nfl.load_player_stats(seasons=[s]).to_pandas()
            w = w[w['season_type'] == 'REG'].copy()
            weekly_dfs.append(w)
            print(f"  [OK] Player stats loaded for {s} ({len(w)} rows)")
        except Exception as e:
            print(f"  [Notice] Loading {s} player stats: {e}")

    if not sched_dfs or not weekly_dfs:
        print("[Error] Failed to fetch essential schedule or player stats from nflreadpy.")
        return False

    sched_df = pd.concat(sched_dfs, ignore_index=True)
    weekly_df = pd.concat(weekly_dfs, ignore_index=True).dropna(subset=['player_id', 'team', 'position'])

    # 3. Merge schedule context with weekly player data
    print("Computing implied team totals, spreads, and rest factors...")
    home_lookup = sched_df[['season', 'week', 'home_team', 'away_team', 'home_rest', 'roof', 'temp', 'wind', 'gameday', 'spread_line', 'total_line']].copy()
    home_lookup = home_lookup.rename(columns={'home_team': 'team_sched', 'away_team': 'opp_sched', 'home_rest': 'days_rest'})
    home_lookup['is_home'] = 1
    home_lookup['team_spread'] = pd.to_numeric(home_lookup['spread_line'], errors='coerce').fillna(0)
    home_lookup['total_line'] = pd.to_numeric(home_lookup['total_line'], errors='coerce').fillna(44.0)
    home_lookup['implied_team_total'] = (home_lookup['total_line'] + home_lookup['team_spread']) / 2.0

    away_lookup = sched_df[['season', 'week', 'home_team', 'away_team', 'away_rest', 'roof', 'temp', 'wind', 'gameday', 'spread_line', 'total_line']].copy()
    away_lookup = away_lookup.rename(columns={'away_team': 'team_sched', 'home_team': 'opp_sched', 'away_rest': 'days_rest'})
    away_lookup['is_home'] = 0
    away_lookup['team_spread'] = -pd.to_numeric(away_lookup['spread_line'], errors='coerce').fillna(0)
    away_lookup['total_line'] = pd.to_numeric(away_lookup['total_line'], errors='coerce').fillna(44.0)
    away_lookup['implied_team_total'] = (away_lookup['total_line'] + away_lookup['team_spread']) / 2.0

    team_sched = pd.concat([home_lookup, away_lookup], ignore_index=True)
    weekly_df = weekly_df.merge(
        team_sched,
        left_on=['season', 'week', 'team'],
        right_on=['season', 'week', 'team_sched'],
        how='left'
    )

    weekly_df['is_dome'] = weekly_df['roof'].isin(['dome', 'closed']).astype(int)
    weekly_df.loc[weekly_df['is_dome'] == 1, 'temp'] = 72.0
    weekly_df.loc[weekly_df['is_dome'] == 1, 'wind'] = 0.0
    weekly_df['temp'] = pd.to_numeric(weekly_df['temp'], errors='coerce').fillna(68.0)
    weekly_df['wind'] = pd.to_numeric(weekly_df['wind'], errors='coerce').fillna(4.0)
    weekly_df['days_rest'] = pd.to_numeric(weekly_df['days_rest'], errors='coerce').fillna(7).clip(upper=14)
    weekly_df['team_spread'] = pd.to_numeric(weekly_df['team_spread'], errors='coerce').fillna(0)
    weekly_df['implied_team_total'] = pd.to_numeric(weekly_df['implied_team_total'], errors='coerce').fillna(22.0)

    # 4. Save to both api/data/ and data/
    target_dirs = [
        os.path.join(PROJECT_DIR, 'data'),
        os.path.join(PROJECT_DIR, 'api', 'data'),
    ]

    for d in target_dirs:
        os.makedirs(d, exist_ok=True)
        w_path = os.path.join(d, 'weekly_cache.parquet')
        s_path = os.path.join(d, 'schedules_cache.parquet')
        weekly_df.to_parquet(w_path, index=False)
        sched_df.to_parquet(s_path, index=False)
        print(f"  [Saved] Updated cache in {os.path.relpath(d, PROJECT_DIR)}: weekly ({len(weekly_df)} rows), schedules ({len(sched_df)} games)")

    # 5. Refresh / verify PBP features
    try:
        load_pbp_features(SEASONS, force_reload=False)
        print("  [OK] Advanced PBP feature tables verified.")
    except Exception as e:
        print(f"  [Notice] PBP features notice: {e}")

    return True


def push_to_github():
    print("\n--- Syncing updated datasets to GitHub ---")
    try:
        os.chdir(PROJECT_DIR)
        
        # Check if there are modified files
        status_res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        if not status_res.stdout.strip():
            print("No data changes detected. Everything is up to date!")
            return True

        print("Staging data updates...")
        subprocess.run(["git", "add", "data/", "api/data/"], check=True)
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        commit_msg = f"Auto-update NFL stats & schedules: {timestamp}"
        print(f"Committing: {commit_msg}")
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        
        print("Pushing to GitHub (origin main)...")
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("Successfully pushed to GitHub! Vercel will automatically redeploy the latest stats.")
        return True
    except Exception as e:
        print(f"[Error] Git sync failed: {e}")
        return False


def main():
    t_start = time.time()
    print("=" * 65)
    print("NFL ANALYTICS HUB - DAILY DATA SYNC")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    success = fetch_and_update_data()
    if success:
        if AUTO_PUSH:
            push_to_github()
        elapsed = time.time() - t_start
        print(f"\nProcess completed successfully in {elapsed:.1f} seconds.")
    else:
        print("\nData sync encountered an issue. Exiting.")


if __name__ == "__main__":
    main()
