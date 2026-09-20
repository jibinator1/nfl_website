"""
NFL Analytics & Matchup Hub - FastAPI Web Application
=====================================================
A pure NFL statistics, team analytics, matchup lab, and 18-week schedule service.
Runs 100% on public open-source NFL data with zero API key dependencies.
"""

import os
import threading
from fastapi import FastAPI, APIRouter, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

from backend.data_loader import NFLDataLoader
from backend.analytics import (
    compute_team_stat_overview,
    compute_matchup_highlights,
    compute_matchup_deepdive,
    compute_full_season_schedule,
    compute_match_history
)

load_dotenv()

app = FastAPI(
    title="NFL Analytics & Matchup Hub",
    description="Interactive NFL statistics, volume x efficiency team rankings, matchup lab (H2H), and 18-week schedule explorer.",
    version="2.0.0"
)

# CORS setup for local and production Vercel environments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(ROOT_DIR, 'frontend')

# Initialize Data Loader singleton (seasons 2023-2026)
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
    print("NFL Analytics Hub API initialized.")
    ensure_data_loaded()


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
async def serve_index():
    candidates = [
        os.path.join(ROOT_DIR, "index.html"),
        os.path.join(FRONTEND_DIR, "index.html"),
        os.path.join(os.getcwd(), "index.html"),
        os.path.join(os.getcwd(), "frontend", "index.html"),
        "index.html",
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Dashboard loading... please refresh in a moment.</h1>")


# ---------------------------------------------------------------------------
# API Router (Registered under both /api and root / for seamless Vercel rewrites)
# ---------------------------------------------------------------------------
api_router = APIRouter()


@api_router.get("/status")
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


@api_router.get("/team-stats")
async def get_team_stats(
    season: int = Query(None, description="NFL Season (e.g. 2026, 2025)"),
    start_date: str = Query(None, description="Start date YYYY-MM-DD"),
    end_date: str = Query(None, description="End date YYYY-MM-DD")
):
    ensure_data_loaded()
    try:
        overview = compute_team_stat_overview(
            engine.weekly, engine.schedules, season=season, start_date=start_date, end_date=end_date
        )
        return overview.to_dict(orient="records") if hasattr(overview, 'to_dict') else overview
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/team-highlights")
async def get_team_highlights(
    season: int = Query(None, description="NFL Season"),
    start_date: str = Query(None, description="Start date YYYY-MM-DD"),
    end_date: str = Query(None, description="End date YYYY-MM-DD"),
    team: str = Query(None, description="Filter by team code (e.g. KC, BAL)")
):
    ensure_data_loaded()
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


@api_router.get("/schedule-weeks")
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


@api_router.get("/matchup-deepdive")
async def get_matchup_deepdive_endpoint(
    home_team: str = Query(..., description="Home Team Abbreviation"),
    away_team: str = Query(..., description="Away Team Abbreviation"),
    season: int = Query(None, description="NFL Season (e.g. 2026, 2025)"),
    start_date: str = Query(None, description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(None, description="End date in YYYY-MM-DD format")
):
    ensure_data_loaded()
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


@api_router.get("/schedule")
@api_router.get("/full-schedule")
async def get_full_schedule_endpoint(
    season: int = Query(2026, description="NFL Season (e.g. 2026, 2025)"),
    start_date: str = Query(None, description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(None, description="End date in YYYY-MM-DD format")
):
    ensure_data_loaded()
    try:
        data = compute_full_season_schedule(
            engine.weekly, engine.schedules,
            season=season, start_date=start_date, end_date=end_date
        )
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@api_router.get("/match-history")
async def get_match_history_endpoint(
    season: str = Query(None, description="NFL Season filter (e.g. 2026, 2025, 2024, 2023, or ALL)"),
    team: str = Query(None, description="Team code filter (e.g. KC, BAL, or ALL)"),
    opponent: str = Query(None, description="Opponent team code filter for H2H history"),
    outcome: str = Query(None, description="Outcome filter: W, L, T, or ALL"),
    limit: int = Query(250, description="Max number of games to return")
):
    """Returns historical match records, H2H matchups, spreads, and ATS results."""
    ensure_data_loaded()
    try:
        data = compute_match_history(
            engine.schedules,
            season=season,
            team=team,
            opponent=opponent,
            outcome=outcome,
            limit=limit
        )
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount the router under both /api and root /
app.include_router(api_router, prefix="/api")
app.include_router(api_router, prefix="")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="127.0.0.1", port=8000, reload=False)
