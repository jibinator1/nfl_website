"""
NFL Analytics & Matchup Hub - FastAPI Web Application
=====================================================
A pure NFL statistics, team analytics, matchup lab, and 18-week schedule service.
Runs 100% on public open-source NFL data with zero API key dependencies.
"""

import os
import threading
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

from backend.data_loader import NFLDataLoader
from backend.analytics import (
    compute_team_stat_overview,
    compute_matchup_highlights,
    compute_matchup_deepdive,
    compute_full_season_schedule
)

load_dotenv()

app = FastAPI(
    title="NFL Analytics & Matchup Hub",
    description="Interactive NFL statistics, volume x efficiency team rankings, matchup lab (H2H), and 18-week schedule explorer.",
    version="2.0.0"
)

# CORS setup for local development
_origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
custom_origin = os.getenv("ALLOWED_ORIGIN")
if custom_origin:
    _origins.append(custom_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')
os.makedirs(FRONTEND_DIR, exist_ok=True)

# Initialize Data Loader singleton (seasons 2023–2026)
engine = NFLDataLoader(seasons=[2023, 2024, 2025, 2026])
_data_lock = threading.Lock()

def ensure_data_loaded():
    """Ensures data is loaded before processing any serverless request on Vercel."""
    if not engine.is_loaded:
        with _data_lock:
            if not engine.is_loaded:
                engine.load_data()


@app.on_event("startup")
def startup_event():
    print("=" * 60)
    print("STARTING NFL ANALYTICS & MATCHUP HUB")
    print("Data Source: nflreadpy (Zero API Keys Required)")
    print("=" * 60)
    engine.load_data()
    print("Data loaded & analytics ready! Dashboard live at http://127.0.0.1:8000")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Dashboard loading... please check frontend/index.html</h1>")


@app.get("/api/status")
async def get_status():
    """Returns data ingestion and system status."""
    ensure_data_loaded()
    return {
        'is_loaded': engine.is_loaded,
        'seasons': engine.seasons,
        'weekly_records': len(engine.weekly) if engine.weekly is not None else 0,
        'scheduled_matchups': len(engine.schedules) if engine.schedules is not None else 0,
        'api_keys_required': False,
        'status': 'OPERATIONAL'
    }


@app.get("/api/team-stats")
async def get_team_stats(
    season: int = Query(None, description="NFL Season (e.g. 2026, 2025)"),
    start_date: str = Query(None, description="Start date YYYY-MM-DD"),
    end_date: str = Query(None, description="End date YYYY-MM-DD")
):
    ensure_data_loaded()
    """Returns 32-team rankings and volume x efficiency statistics overview."""
    try:
        overview = compute_team_stat_overview(
            engine.weekly, engine.schedules, season=season, start_date=start_date, end_date=end_date
        )
        return overview.to_dict(orient="records") if hasattr(overview, 'to_dict') else overview
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/team-highlights")
async def get_team_highlights(
    season: int = Query(None, description="NFL Season"),
    start_date: str = Query(None, description="Start date YYYY-MM-DD"),
    end_date: str = Query(None, description="End date YYYY-MM-DD"),
    team: str = Query(None, description="Filter by team code (e.g. KC, BAL)")
):
    ensure_data_loaded()
    """Returns matchup trends, tactical vulnerabilities, and explosive unit alerts."""
    try:
        res = compute_matchup_highlights(
            engine.weekly, engine.schedules, season=season, start_date=start_date, end_date=end_date
        )
        h_list = res.get('highlights', []) if isinstance(res, dict) else res
        if team and team.upper() != 'ALL':
            h_list = [h for h in h_list if h.get('team') == team.upper()]
        return {
            'season': season,
            'start_date': start_date,
            'end_date': end_date,
            'count': len(h_list),
            'highlights': h_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/schedule-weeks")
async def get_schedule_weeks(season: int = Query(2026, description="NFL Season")):
    """Returns all weeks and calendar dates in the schedule for dropdown filtering."""
    ensure_data_loaded()
    if engine.schedules is None:
        raise HTTPException(status_code=503, detail="Schedules data not yet loaded")
    s = engine.schedules[(engine.schedules['season'] == season) & (engine.schedules['game_type'] == 'REG')].copy()
    weeks = []
    for w_num in sorted(s['week'].unique()):
        w_games = s[s['week'] == w_num]
        dates = sorted(w_games['gameday'].dropna().unique().tolist())
        weeks.append({
            'week': int(w_num),
            'dates': dates,
            'game_count': len(w_games),
            'label': f"Week {w_num} ({dates[0] if dates else ''})"
        })
    return {'season': season, 'weeks': weeks}


@app.get("/api/matchup-deepdive")
async def get_matchup_deepdive_endpoint(
    home_team: str = Query(..., description="Home Team Abbreviation"),
    away_team: str = Query(..., description="Away Team Abbreviation"),
    season: int = Query(None, description="NFL Season (e.g. 2026, 2025)"),
    start_date: str = Query(None, description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(None, description="End date in YYYY-MM-DD format")
):
    """Returns unit battles, ground vs aerial matchups, WR1 tests, and scramble containment."""
    try:
        data = compute_matchup_deepdive(
            engine.weekly, engine.schedules,
            home_team=home_team.upper(),
            away_team=away_team.upper(),
            season=season,
            start_date=start_date,
            end_date=end_date
        )
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/schedule")
@app.get("/api/full-schedule")
async def get_full_schedule_endpoint(
    season: int = Query(2026, description="NFL Season (e.g. 2026, 2025)"),
    start_date: str = Query(None, description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(None, description="End date in YYYY-MM-DD format")
):
    """Returns the full 18-week schedule with per-game team stats, records, and opportunity funnels."""
    try:
        data = compute_full_season_schedule(
            engine.weekly, engine.schedules,
            season=season, start_date=start_date, end_date=end_date
        )
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="127.0.0.1", port=8000, reload=False)
