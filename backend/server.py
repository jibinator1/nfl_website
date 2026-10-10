"""
NFL Analytics & Matchup Hub - FastAPI Web Application
=====================================================
A pure NFL statistics, team analytics, matchup lab, and 18-week schedule service.
Runs 100% on public open-source NFL data with zero API key dependencies.
"""

import os
import threading
import json
import math
from typing import Any
import numpy as np
import pandas as pd
from fastapi import FastAPI, APIRouter, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse
from dotenv import load_dotenv

from backend.data_loader import NFLDataLoader
from backend.analytics import (
    compute_team_stat_overview,
    compute_matchup_highlights,
    compute_matchup_deepdive,
    compute_full_season_schedule,
    compute_match_history,
    compute_floor_streak_df
)
from backend.scheme_insights import compute_scheme_insights

load_dotenv()

def sanitize_nan(obj: Any) -> Any:
    """
    Recursively replaces NaN, Infinity, -Infinity with None so that
    JSON serialization produces valid RFC-compliant JSON ('null' instead of invalid 'NaN').
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if hasattr(obj, '__class__') and obj.__class__.__name__ in ('Query', 'Path', 'Header', 'Cookie', 'Body', 'Form', 'File', 'Param'):
        default_val = getattr(obj, 'default', None)
        return None if default_val is ... or default_val is Ellipsis else sanitize_nan(default_val)
    if isinstance(obj, dict):
        return {k: sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_nan(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_nan(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if (math.isnan(val) or math.isinf(val)) else val
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return sanitize_nan(obj.tolist())
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


class SafeJSONResponse(JSONResponse):
    """
    Custom JSONResponse that automatically replaces out-of-range floats (NaN, +/-Inf)
    with None (JSON null) before serializing, preventing 'ValueError: Out of range float
    values are not JSON compliant: nan'.
    """
    def render(self, content: Any) -> bytes:
        return json.dumps(
            sanitize_nan(content),
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")


app = FastAPI(
    title="NFL Analytics & Matchup Hub",
    description="Interactive NFL statistics, volume x efficiency team rankings, matchup lab (H2H), and 18-week schedule explorer.",
    version="2.0.0",
    default_response_class=SafeJSONResponse
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

# Mount static files and direct asset directories for frontend
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    js_dir = os.path.join(FRONTEND_DIR, 'js')
    if os.path.exists(js_dir):
        app.mount("/js", StaticFiles(directory=js_dir), name="js")
    css_dir = os.path.join(FRONTEND_DIR, 'css')
    if os.path.exists(css_dir):
        app.mount("/css", StaticFiles(directory=css_dir), name="css")
    data_dir = os.path.join(FRONTEND_DIR, 'data')
    if os.path.exists(data_dir):
        app.mount("/data", StaticFiles(directory=data_dir), name="data")

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
        os.path.join(FRONTEND_DIR, "index.html"),
        os.path.join(ROOT_DIR, "index.html"),
        os.path.join(os.getcwd(), "frontend", "index.html"),
        os.path.join(os.getcwd(), "index.html"),
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


@api_router.get("/floor-streak")
async def get_floor_streak_endpoint(
    season: int = Query(None, description="Filter by season (optional)"),
    pos: str = Query(None, description="Filter by position (QB, RB, WR, TE)"),
    date: str = Query(None, description="Filter by upcoming game date (YYYY-MM-DD)"),
    start_date: str = Query(None, description="Start date for stats history window (YYYY-MM-DD)"),
    end_date: str = Query(None, description="End date for stats history window (YYYY-MM-DD)"),
    margin: float = Query(None, description="Custom floor margin percentage (e.g. 0.30 for -30%)"),
    min_streak: int = Query(0, description="Minimum floor streak to include"),
    min_games: int = Query(4, description="Minimum games played to include player"),
):
    """
    Returns the Player Floor Breach Streak for all tracked players against a hard floor threshold line.
    Streak = consecutive most-recent games meeting or exceeding the hard floor threshold line ((1 - margin) × baseline median).
    Position-specific margins: QB=30%, RB=30%, WR=30%, TE=30% (or custom margin).
    """
    ensure_data_loaded()
    try:
        df = compute_floor_streak_df(
            engine.weekly,
            start_date=start_date,
            end_date=end_date,
            season=season,
            margin=margin
        )
        if df.empty:
            return {
                'status': 'success',
                'count': 0,
                'start_date': start_date,
                'end_date': end_date,
                'season': season,
                'players': []
            }

        if pos and pos.upper() != 'ALL':
            df = df[df['position'].str.upper() == pos.upper()]
        else:
            df = df[df['position'].str.upper().isin(['QB', 'RB', 'WR', 'TE'])]

        if date and 'date' in df.columns:
            df = df[df['date'] == date]

        df = df[df['Games_Played'] >= min_games]
        df = df[df['Floor_Streak'] >= min_streak]
        df['Floor_Threshold'] = ((1.0 - df['Margin_Used']) * df['Baseline_Median']).round(1)
        df = df.sort_values('Floor_Streak', ascending=False)
        df = df.where(pd.notnull(df), None)

        return {
            'status': 'success',
            'count': len(df),
            'start_date': start_date,
            'end_date': end_date,
            'season': season,
            'players': df.to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/export-matchups-excel")
async def export_matchups_excel_endpoint(
    season: int = Query(2026, description="NFL Season (e.g. 2026, 2025, 2024)"),
    week: str = Query("1", description="NFL Week (1 to 18, or 'all')"),
    start_date: str = Query(None, description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(None, description="End date in YYYY-MM-DD format")
):
    """Generates and downloads a multi-tab NFL Matchup Data workbook (.xlsx)."""
    ensure_data_loaded()
    try:
        from backend.excel_exporter import export_weekly_matchups_xlsx
        schedule_data = compute_full_season_schedule(
            engine.weekly, engine.schedules,
            season=season, start_date=start_date, end_date=end_date
        )
        target_week_label = f"Week_{week}" if week and str(week).lower() != 'all' else "All_Weeks"
        filename = f"NFL_Matchup_Data_{target_week_label}_{season}.xlsx"
        excel_bytes = export_weekly_matchups_xlsx(
            schedule_data=schedule_data,
            week=week,
            season=season
        )
        return Response(
            content=excel_bytes,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/scheme-insights")
async def get_scheme_insights_endpoint(
    home_team: str = Query(..., description="Home Team Abbreviation"),
    away_team: str = Query(..., description="Away Team Abbreviation"),
    season: int = Query(2026, description="NFL Season (e.g. 2026, 2025)"),
    week: int = Query(1, description="NFL Week Number")
):
    """
    Returns 5-pillar scheme analytics:
    1. Defensive Coverage Tendencies (Zone vs Man, MFO vs MFC sets)
    2. Run-Scheme Splits (Zone vs Gap rushing vs defense vulnerabilities)
    3. Positional Target Rates (Slot vs Outside WR, Inline TE, target rates vs coverage)
    4. Efficiency by Context (Scheme-specific YPC/YPR breakout spotters)
    5. Game-Flow & Tempo (Time of possession, neutral pace, red zone touch share, check-downs)
    """
    try:
        data = compute_scheme_insights(
            home_team=home_team.upper(),
            away_team=away_team.upper(),
            season=season,
            week=week
        )
        return sanitize_nan(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Mount the router under both /api and root /
app.include_router(api_router, prefix="/api")
app.include_router(api_router, prefix="")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="127.0.0.1", port=8000, reload=False)
