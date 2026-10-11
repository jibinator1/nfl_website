"""
The Odds API Multi-Key Ingestion & Caching Engine
=================================================
Fetches consensus spreads, totals, and player props from The Odds API.
Automatically rotates across the 3 keys configured in .env:
  - THE_ODDS_API_KEY_1
  - THE_ODDS_API_KEY_2
  - THE_ODDS_API_KEY_3
  - THE_ODDS_API_KEYS (comma-separated fallback)

Supported prop markets:
  - player_reception_yds  ('Receiving Yards')
  - player_receptions     ('Receptions')
  - player_rush_yds       ('Rushing Yards')
  - player_pass_yds       ('Passing Yards')
  - spreads, totals

Persists output to:
  - data/live_odds_cache.json & api/data/live_odds_cache.json
  - data/historical_closing_odds_archive.csv & api/data/historical_closing_odds_archive.csv
"""

import os
import sys
import json
import time
import argparse
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()


def get_api_keys():
    """Extracts all available Odds API keys from .env pool."""
    pool_str = os.getenv('THE_ODDS_API_KEYS', '')
    if pool_str:
        keys = [k.strip() for k in pool_str.split(',') if k.strip()]
        if keys:
            return keys
    keys = []
    for i in range(1, 10):
        val = os.getenv(f'THE_ODDS_API_KEY_{i}')
        if not val and i == 1:
            val = os.getenv('THE_ODDS_API_KEY')
        if val and val.strip():
            keys.append(val.strip())
    return keys


API_KEYS = get_api_keys()
SPORT = 'americanfootball_nfl'
BOOKMAKERS_PARAM = 'draftkings,fanduel,betmgm,bovada,betrivers'
REGIONS = 'us,us2'
PRIORITY_ORDER = ['draftkings', 'fanduel', 'betmgm', 'bovada', 'betrivers']

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIRS = [
    os.path.join(ROOT_DIR, 'data'),
    os.path.join(ROOT_DIR, 'api', 'data')
]
for d in DATA_DIRS:
    os.makedirs(d, exist_ok=True)

TARGET_PROP_MARKETS = [
    'player_reception_yds',
    'player_receptions',
    'player_rush_yds',
    'player_pass_yds'
]

MARKET_DISPLAY_NAMES = {
    'player_reception_yds': 'Receiving Yards',
    'player_receptions': 'Receptions',
    'player_rush_yds': 'Rushing Yards',
    'player_pass_yds': 'Passing Yards'
}


class OddsAPIKeyManager:
    """Manages multi-key rotation and monitors remaining quota."""

    def __init__(self, keys=None):
        self.keys = keys or API_KEYS
        self.current_idx = 0
        self.exhausted_keys = set()
        self.key_stats = {k: {'remaining': 'unknown', 'used': 'unknown', 'status': 'active'} for k in self.keys}

    @property
    def current_key(self):
        return self.keys[self.current_idx] if self.keys else ""

    def rotate_to_next(self, reason="quota depleted"):
        if not self.keys:
            return False
        prev_key = self.current_key
        self.exhausted_keys.add(prev_key)
        self.key_stats[prev_key]['status'] = f'exhausted ({reason})'
        print(f"  [API Key Pool] Key {self.current_idx + 1} ({prev_key[:6]}...{prev_key[-4:]}) {reason}. Rotating...")

        for offset in range(1, len(self.keys)):
            next_idx = (self.current_idx + offset) % len(self.keys)
            next_k = self.keys[next_idx]
            if next_k not in self.exhausted_keys:
                self.current_idx = next_idx
                print(f"  [API Key Pool] Switched to Key {self.current_idx + 1} ({next_k[:6]}...{next_k[-4:]})")
                return True
        print("  [API Key Pool] WARNING: All API keys in pool are exhausted!")
        return False

    def request(self, url, params=None, timeout=15):
        if not self.keys:
            print("  [API Key Pool] No Odds API keys found in environment.")
            return None
        params = dict(params or {})
        attempts = 0
        max_attempts = len(self.keys)

        while attempts < max_attempts:
            active_key = self.current_key
            params['apiKey'] = active_key
            try:
                res = requests.get(url, params=params, timeout=timeout)
                rem = res.headers.get('x-requests-remaining')
                used = res.headers.get('x-requests-used')
                if rem is not None:
                    self.key_stats[active_key]['remaining'] = rem
                    self.key_stats[active_key]['used'] = used

                if res.status_code == 200:
                    if rem is not None and str(rem).isdigit() and int(rem) <= 0:
                        self.rotate_to_next("remaining reached 0")
                    return res
                elif res.status_code in [401, 429]:
                    rotated = self.rotate_to_next(f"HTTP {res.status_code}")
                    if not rotated:
                        return res
                    attempts += 1
                else:
                    return res
            except Exception as e:
                print(f"  [API Key Pool] Network error on key {self.current_idx + 1}: {e}")
                rotated = self.rotate_to_next(f"network error: {e}")
                if not rotated:
                    raise e
                attempts += 1

        return None


def normalize_odds_name(name):
    if not name:
        return ""
    clean = str(name).lower().strip()
    clean = clean.replace('.', ' ').replace("'", "").replace('-', ' ')
    for suffix in [' jr', ' sr', ' ii', ' iii', ' iv', ' v']:
        if clean.endswith(suffix):
            clean = clean[:-len(suffix)].strip()
    return " ".join(clean.split())


def fetch_and_save_odds(dry_run=False, max_games=None):
    key_manager = OddsAPIKeyManager()
    print(f"[*] Initialized Odds API Key Pool with {len(key_manager.keys)} keys from .env.")

    if not key_manager.keys:
        print("[!] Error: No API keys configured.")
        return

    # 1. Fetch upcoming events
    events_url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/events'
    print("[*] Fetching upcoming NFL games from The Odds API...")
    res = key_manager.request(events_url, params={}, timeout=15)
    if not res or res.status_code != 200:
        print(f"[!] Failed to fetch games (Status {res.status_code if res else 'None'}): {res.text if res else ''}")
        return

    events = res.json()
    print(f"[OK] Found {len(events)} upcoming events.")

    if dry_run:
        print("[Dry Run] Exiting without consuming further request quota.")
        return

    now_iso = datetime.now(timezone.utc).isoformat()
    odds_cache = {
        'cached_at': now_iso,
        'spread': {},
        'total': {},
        'player_reception_yds': {},
        'player_receptions': {},
        'player_rush_yds': {},
        'player_pass_yds': {},
        'props_detailed': {},
        'events_meta': [],
        'key_stats': key_manager.key_stats
    }

    target_events = events[:max_games] if max_games else events
    archive_rows = []
    market_str = ','.join(TARGET_PROP_MARKETS + ['spreads', 'totals'])

    for idx, ev in enumerate(target_events):
        eid = ev['id']
        h_team = ev.get('home_team')
        a_team = ev.get('away_team')
        commence = ev.get('commence_time')
        print(f"[{idx+1}/{len(target_events)}] Fetching lines & props: {a_team} @ {h_team}...")

        p_url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/events/{eid}/odds'
        p_res = key_manager.request(p_url, params={
            'regions': REGIONS,
            'markets': market_str,
            'bookmakers': BOOKMAKERS_PARAM
        }, timeout=15)

        if not p_res or p_res.status_code != 200:
            print(f"    Notice: Failed for {eid} ({p_res.status_code if p_res else 'None'})")
            continue

        p_data = p_res.json()
        odds_cache['events_meta'].append({
            'id': eid, 'home': h_team, 'away': a_team, 'commence': commence
        })

        bms = {b['key']: b for b in p_data.get('bookmakers', [])}

        # Extract consensus game spreads and totals
        for bk in PRIORITY_ORDER:
            if bk in bms:
                for m in bms[bk].get('markets', []):
                    if m['key'] == 'spreads':
                        for out in m.get('outcomes', []):
                            if out.get('name') == h_team and out.get('point') is not None:
                                odds_cache['spread'][f"{h_team}_vs_{a_team}"] = float(out['point'])
                                break
                    if m['key'] == 'totals':
                        for out in m.get('outcomes', []):
                            if out.get('point') is not None:
                                odds_cache['total'][f"{h_team}_vs_{a_team}"] = float(out['point'])
                                break
                if f"{h_team}_vs_{a_team}" in odds_cache['spread'] and f"{h_team}_vs_{a_team}" in odds_cache['total']:
                    break

        # Extract Player Props
        for m_key in TARGET_PROP_MARKETS:
            stat_name = MARKET_DISPLAY_NAMES.get(m_key, m_key)
            for bk in PRIORITY_ORDER:
                if bk not in bms:
                    continue
                for m_obj in bms[bk].get('markets', []):
                    if m_obj.get('key') == m_key:
                        for out in m_obj.get('outcomes', []):
                            p_name = out.get('description', '')
                            pt = out.get('point')
                            if not p_name or pt is None:
                                continue
                            norm_p = normalize_odds_name(p_name)
                            pt_val = float(pt)

                            # Save top consensus / priority line into cache dictionary
                            if norm_p not in odds_cache[m_key]:
                                odds_cache[m_key][norm_p] = pt_val

                            if norm_p not in odds_cache['props_detailed']:
                                odds_cache['props_detailed'][norm_p] = {}

                            if stat_name not in odds_cache['props_detailed'][norm_p]:
                                odds_cache['props_detailed'][norm_p][stat_name] = {
                                    'line': pt_val,
                                    'sportsbook': bk,
                                    'price': out.get('price'),
                                    'side': out.get('name')
                                }

                            archive_rows.append({
                                'Archived_At': now_iso,
                                'Season': 2026,
                                'Week': 6,
                                'Sportsbook': bk,
                                'Player': p_name,
                                'Stat': stat_name,
                                'Closing_Line': pt_val,
                                'Juice_Over': out.get('price') if out.get('name') == 'Over' else None,
                                'Juice_Under': out.get('price') if out.get('name') == 'Under' else None,
                                'Source': 'The Odds API'
                            })

        time.sleep(0.10)

    # Persist live cache JSON to data/ and api/data/
    for d in DATA_DIRS:
        cache_file = os.path.join(d, 'live_odds_cache.json')
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(odds_cache, f, indent=2)
            print(f"[OK] Saved live odds cache to {cache_file}")
        except Exception as e:
            print(f"[!] Warning saving {cache_file}: {e}")

    # Append to historical closing odds archive CSV
    if archive_rows:
        new_df = pd.DataFrame(archive_rows)
        for d in DATA_DIRS:
            csv_file = os.path.join(d, 'historical_closing_odds_archive.csv')
            try:
                if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
                    old_df = pd.read_csv(csv_file)
                    combined = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates(
                        subset=['Season', 'Week', 'Player', 'Stat', 'Sportsbook'], keep='last'
                    )
                else:
                    combined = new_df.drop_duplicates(
                        subset=['Season', 'Week', 'Player', 'Stat', 'Sportsbook'], keep='last'
                    )
                combined.to_csv(csv_file, index=False)
                print(f"[OK] Synced {len(combined):,} total odds archive rows to {csv_file}")
            except Exception as e:
                print(f"[!] Warning syncing {csv_file}: {e}")

    rec_count = len(odds_cache['player_reception_yds'])
    rush_count = len(odds_cache['player_rush_yds'])
    recept_count = len(odds_cache['player_receptions'])
    print(f"\n[DONE] Live Odds Ingestion Complete: {rec_count} Rec Yds, {recept_count} Receptions, {rush_count} Rush Yds props cached.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Live Odds Ingestion Engine")
    parser.add_argument('--dry-run', action='store_true', help="Check key status without consuming quota")
    parser.add_argument('--max-games', type=int, default=None, help="Limit number of games to fetch")
    args = parser.parse_args()

    fetch_and_save_odds(dry_run=args.dry_run, max_games=args.max_games)
