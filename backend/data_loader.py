import os
import json
import pandas as pd
import numpy as np

from backend.pbp_features import load_pbp_features

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDIDATE_DATA_DIRS = [
    os.path.join(ROOT_DIR, 'api', 'data'),
    os.path.join(ROOT_DIR, 'data'),
    os.path.join(os.getcwd(), 'api', 'data'),
    os.path.join(os.getcwd(), 'data'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'api', 'data'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'),
]


def _find_data_file(filename):
    for d in CANDIDATE_DATA_DIRS:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return os.path.join(ROOT_DIR, 'data', filename)


class NFLDataLoader:
    """Canonical data manager for NFL stats, schedules, and matchup analytics."""

    def __init__(self, seasons=[2023, 2024, 2025, 2026]):
        self.seasons = sorted(list(set(seasons)))
        self.weekly = None
        self.schedules = None
        self.rosters = None
        self.pbp_feats = None
        self.is_loaded = False

    def _load_manual_overrides(self):
        """Loads trade and injury overrides from manual_overrides.json."""
        overrides_file = _find_data_file('manual_overrides.json')
        if not os.path.exists(overrides_file):
            return {'trade_overrides': {}, 'injury_overrides': {}, 'player_blacklist': []}
        try:
            with open(overrides_file, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            t_map = {}
            for item in raw.get('trade_overrides', []):
                t_map[item.get('player_id', '')] = item.get('team', '')
                t_map[item.get('player', '')] = item.get('team', '')
            return {
                'trade_overrides': t_map,
                'injury_overrides': raw.get('injury_overrides', []),
                'player_blacklist': raw.get('player_blacklist', [])
            }
        except Exception as e:
            print(f"Notice loading manual overrides: {e}")
            return {'trade_overrides': {}, 'injury_overrides': {}, 'player_blacklist': []}

    def load_data(self):
        """Loads player stats, schedules, and advanced metrics (cache-first for instant Vercel startup)."""
        weekly_path = _find_data_file('weekly_cache.parquet')
        schedules_path = _find_data_file('schedules_cache.parquet')
        is_serverless = os.getenv('VERCEL') is not None or os.getenv('AWS_LAMBDA_FUNCTION_NAME') is not None

        # 1. Instant Cache Path (0.05s cold start on Vercel, < 50MB RAM)
        if os.path.exists(weekly_path) and os.path.exists(schedules_path):
            try:
                print(f"Loading NFL data from cache ({weekly_path})...")
                self.weekly = pd.read_parquet(weekly_path)
                self.schedules = pd.read_parquet(schedules_path)
                try:
                    self.pbp_feats = load_pbp_features(self.seasons)
                except Exception as pe:
                    print(f"Note loading PBP features: {pe}")
                self.is_loaded = True
                print(f"NFL Data Engine ready (cached)! Loaded {len(self.weekly)} player games and {len(self.schedules)} matchups.")
                return
            except Exception as e:
                print(f"Note loading parquet cache: {e}")

        if is_serverless:
            print("[Data Engine] Serverless mode: cache unreadable, initializing minimal safe state.")
            self.weekly = pd.DataFrame()
            self.schedules = pd.DataFrame()
            self.pbp_feats = load_pbp_features(self.seasons)
            self.is_loaded = True
            return

        # 2. Local Ingestion Fallback (Only run offline/locally)
        import nflreadpy as nfl
        print(f"Loading NFL data for seasons {self.seasons} via nflreadpy...")
        weekly_dfs = []
        sched_dfs = []
        for s in self.seasons:
            try:
                w = nfl.load_player_stats(seasons=[s]).to_pandas()
                w = w[w['season_type'] == 'REG'].copy()
                weekly_dfs.append(w)
            except Exception as e:
                print(f"Note loading {s} player stats: {e}")

            try:
                sc = nfl.load_schedules(seasons=[s]).to_pandas()
                sc = sc[sc['game_type'] == 'REG'].copy()
                sched_dfs.append(sc)
            except Exception as e:
                print(f"Note loading {s} schedules: {e}")

        try:
            rost_seasons = [s for s in [2025, 2026] if s in self.seasons]
            if rost_seasons:
                self.rosters = nfl.load_rosters(seasons=rost_seasons).to_pandas()
        except Exception as e:
            print(f"Note loading rosters: {e}")

        if not weekly_dfs or not sched_dfs:
            raise RuntimeError("Failed to load core NFL data from nflreadpy.")

        self.weekly = pd.concat(weekly_dfs, ignore_index=True).dropna(subset=['player_id', 'team', 'position'])
        self.schedules = pd.concat(sched_dfs, ignore_index=True)

        # Merge schedule context with weekly player data
        home_lookup = self.schedules[['season', 'week', 'home_team', 'away_team', 'home_rest', 'roof', 'temp', 'wind', 'gameday', 'spread_line', 'total_line']].copy()
        home_lookup = home_lookup.rename(columns={'home_team': 'team_sched', 'away_team': 'opp_sched', 'home_rest': 'days_rest'})
        home_lookup['is_home'] = 1
        home_lookup['team_spread'] = pd.to_numeric(home_lookup['spread_line'], errors='coerce').fillna(0)
        home_lookup['total_line'] = pd.to_numeric(home_lookup['total_line'], errors='coerce').fillna(44.0)
        home_lookup['implied_team_total'] = (home_lookup['total_line'] + home_lookup['team_spread']) / 2.0

        away_lookup = self.schedules[['season', 'week', 'home_team', 'away_team', 'away_rest', 'roof', 'temp', 'wind', 'gameday', 'spread_line', 'total_line']].copy()
        away_lookup = away_lookup.rename(columns={'away_team': 'team_sched', 'home_team': 'opp_sched', 'away_rest': 'days_rest'})
        away_lookup['is_home'] = 0
        away_lookup['team_spread'] = -pd.to_numeric(away_lookup['spread_line'], errors='coerce').fillna(0)
        away_lookup['total_line'] = pd.to_numeric(away_lookup['total_line'], errors='coerce').fillna(44.0)
        away_lookup['implied_team_total'] = (away_lookup['total_line'] + away_lookup['team_spread']) / 2.0

        team_sched = pd.concat([home_lookup, away_lookup], ignore_index=True)
        self.weekly = self.weekly.merge(
            team_sched,
            left_on=['season', 'week', 'team'],
            right_on=['season', 'week', 'team_sched'],
            how='left'
        )

        # Weather and rest normalizations
        self.weekly['is_dome'] = self.weekly['roof'].isin(['dome', 'closed']).astype(int)
        self.weekly.loc[self.weekly['is_dome'] == 1, 'temp'] = 72.0
        self.weekly.loc[self.weekly['is_dome'] == 1, 'wind'] = 0.0
        self.weekly['temp'] = pd.to_numeric(self.weekly['temp'], errors='coerce').fillna(68.0)
        self.weekly['wind'] = pd.to_numeric(self.weekly['wind'], errors='coerce').fillna(4.0)
        self.weekly['days_rest'] = pd.to_numeric(self.weekly['days_rest'], errors='coerce').fillna(7).clip(upper=14)
        self.weekly['team_spread'] = pd.to_numeric(self.weekly['team_spread'], errors='coerce').fillna(0)
        self.weekly['implied_team_total'] = pd.to_numeric(self.weekly['implied_team_total'], errors='coerce').fillna(22.0)

        # Cache freshly fetched datasets
        try:
            for save_dir in [os.path.join(ROOT_DIR, 'data'), os.path.join(ROOT_DIR, 'api', 'data')]:
                os.makedirs(save_dir, exist_ok=True)
                self.weekly.to_parquet(os.path.join(save_dir, 'weekly_cache.parquet'), index=False)
                self.schedules.to_parquet(os.path.join(save_dir, 'schedules_cache.parquet'), index=False)
            print("Saved updated data to local parquet caches.")
        except Exception as se:
            print(f"Note caching datasets: {se}")

        try:
            self.pbp_feats = load_pbp_features(self.seasons)
            print("Advanced PBP features loaded successfully.")
        except Exception as e:
            print(f"Note loading PBP features: {e}")

        self.is_loaded = True
        print(f"NFL Data Engine ready! Loaded {len(self.weekly)} player games and {len(self.schedules)} matchups.")
