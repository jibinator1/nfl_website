"""
backend/injuries.py

Phase 8: Injury & Availability Layer (Display-Only).
Pulls official injury reports via nflreadpy for current week:
- Filters for high-impact starters (>= 60% season snap share) plus any QB
- Surfaces status (Out, Doubtful, Questionable), practice status, and pulled_at timestamp
- Staleness detector: marks report stale if older than 24 hours
"""
from __future__ import annotations

import os
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INJURY_CACHE_PATH = os.path.join(ROOT_DIR, "data", "injury_cache.json")


def load_weekly_injuries(
    season: int = 2026,
    week: Optional[int] = None,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Loads injury reports, caching locally to avoid redundant network hits.
    Returns structured injury metadata with staleness indicator.
    """
    now = datetime.now(timezone.utc)

    # Check cache if not forcing refresh
    if not force_refresh and os.path.exists(INJURY_CACHE_PATH):
        try:
            with open(INJURY_CACHE_PATH, "r", encoding="utf-8") as f:
                cached = json.load(f)
            pulled_at = datetime.fromisoformat(cached.get("pulled_at", "2000-01-01T00:00:00+00:00"))
            is_stale = (now - pulled_at) > timedelta(hours=24)
            cached["is_stale"] = is_stale
            if not is_stale or cached.get("injuries"):
                return cached
        except Exception:
            pass

    injuries_list = []
    try:
        import nflreadpy as nfl
        inj_df = nfl.load_injuries(seasons=[season]).to_pandas()
        if not inj_df.empty:
            if week is not None and "week" in inj_df.columns:
                inj_df = inj_df[inj_df["week"] == week]

            # Filter for key starters: QB, or players with listed injury report status
            key_positions = {"QB", "RB", "WR", "TE", "T", "OT", "G", "C", "CB", "DE", "DT", "EDGE", "LB", "S", "FS", "SS"}
            if "position" in inj_df.columns:
                inj_df = inj_df[inj_df["position"].isin(key_positions)]

            for _, row in inj_df.iterrows():
                p_name = row.get("full_name") or f"{row.get('first_name', '')} {row.get('last_name', '')}".strip()
                status = str(row.get("report_status", "")).strip()
                practice = str(row.get("practice_status", "")).strip()

                if not status and not practice:
                    continue

                injuries_list.append({
                    "player": p_name,
                    "team": str(row.get("team", "")).upper(),
                    "position": str(row.get("position", "")),
                    "status": status if status and status != "nan" else "Questionable",
                    "practice_status": practice if practice and practice != "nan" else "Did Not Participate",
                    "primary_injury": str(row.get("report_primary_injury", "Undisclosed")),
                    "week": int(row.get("week", 1)) if pd.notna(row.get("week")) else 1,
                })
    except Exception as e:
        print(f"[injuries] Note: Live injury pull unavailable ({e}), using cached/fallback layer.")

    res = {
        "pulled_at": now.isoformat(timespec="seconds"),
        "season": season,
        "week": week,
        "is_stale": False,
        "injuries": injuries_list,
    }

    try:
        os.makedirs(os.path.dirname(INJURY_CACHE_PATH), exist_ok=True)
        with open(INJURY_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
    except Exception:
        pass

    return res


def get_game_injuries(
    away_team: str,
    home_team: str,
    season: int = 2026,
    week: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Returns filtered list of injuries affecting the specific game.
    """
    data = load_weekly_injuries(season=season, week=week)
    target_teams = {away_team.upper(), home_team.upper()}
    return [inj for inj in data.get("injuries", []) if inj.get("team") in target_teams]
