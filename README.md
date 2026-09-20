# NFL Analytics & Matchup Hub

[![Live Website](https://img.shields.io/badge/Live_Website-nflwebsite--ten.vercel.app-0070f3?style=for-the-badge&logo=vercel&logoColor=white)](https://nflwebsite-ten.vercel.app)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![NFLverse](https://img.shields.io/badge/Data-NFLverse_nflreadpy-013369?style=for-the-badge)](https://github.com/nflverse/nflreadpy)

An interactive, production-grade NFL analytics dashboard, historical match logs explorer, matchup lab (H2H), and 18-week schedule hub.

Live Production URL: **[https://nflwebsite-ten.vercel.app](https://nflwebsite-ten.vercel.app)**

> **Zero API Keys Required**: 100% powered by open-source NFL play-by-play, box scores, and spread lines via [`nflreadpy`](https://github.com/nflverse/nflreadpy) from the NFLverse project. No paid subscriptions, rate limits, or API keys needed.

---

## Core Features & Dashboard Tabs

### 1. Match History & H2H Logs (New)
* **Comprehensive Historical Coverage**: Explore 800+ completed NFL regular season games across the 2023, 2024, 2025, and 2026 seasons.
* **Team & Head-to-Head (H2H) Focus**: Instantly isolate any franchise's game logs or compare two specific teams head-to-head.
* **Betting & Spread Analytics**:
  * **Against The Spread (ATS)**: Track closing spreads, home/away cover results, cover percentages, and pushes.
  * **Over/Under (O/U)**: View Vegas totals, game combined points, Over/Under hit rates, and push records.
* **Performance KPI Banner**: Live calculated record (W-L-T), win %, home/away splits, average points scored vs allowed, and point differentials (+/-).
* **Game Context**: Starting quarterbacks, game dates/times, stadium venue, roof type, temperature, and wind speed.
* **Direct Matchup Lab Simulation**: Jump directly from any historical card into the Head-to-Head Simulation Lab with both teams pre-selected.

### 2. 18-Week Schedule & Matchup Explorer (2026 / 2025)
* **Full Schedule Coverage**: Interactive navigation across all 18 regular season weeks, game dates, and prime-time slots.
* **Per-Game Volume x Efficiency Table**: Head-to-head offensive vs. defensive statistical matchup cards featuring:
  * **Scoring**: Vegas Implied Team Totals.
  * **Passing**: Pass attempts, passing yards allowed, game-script neutral intent (**PROE**), yards per attempt (YPA offense vs. allowed), and **Explosive Pass Rates** (20+ yard chunk plays).
  * **Rushing**: Carries per game, rush yards allowed per game, yards per carry (YPC offense vs. allowed).
  * **Defense vs WR1s & Star RBs**: Measures each defense's average surplus/deficit allowed against opposing teams' **WR1** and lead rushers.
  * **Trenches**: Pass protection pressure allowed and defensive pressure generation rates.
* **Pass / Run Opportunity Funnel**: Dynamically models team play calling intent scaled by game pace and Vegas totals.

### 3. Yahoo Sports Team Stats & Rankings (32 Teams)
* **Comprehensive Metrics**: 25+ offensive and defensive volume & efficiency stats with league-wide `#1` to `#32` rankings.
* **Multi-Category Filtering**: Quick view toggles across All Stats, Passing, Rushing, Defense, Scoring, and Advanced Trenches.
* **Interactive Sorting**: Real-time ascending/descending sorting across any statistical column.

### 4. Matchup Lab (Head-to-Head Comparison)
* **Custom Matchup Selector**: Select any two NFL franchises to simulate unit matchups.
* **Unit Battle Radar**: Compares overall team ratings, passing offense vs. secondary, and rushing attack vs. front seven.
* **WR1 & Star RB Showdown**: Tests how defenses hold up against the opponent's primary weapons.
* **Scramble Containment**: Rates defensive vulnerability to mobile QBs.

### 5. Matchup Vulnerabilities & Trends
* Automated spotlight detection highlighting defensive pass funnels, soft run fronts, offensive red-zone efficiency, and rest advantages.

---

## Live Access & Deployment

The application is deployed on Vercel Serverless Functions (`@vercel/python`):

* **Live Dashboard**: [https://nflwebsite-ten.vercel.app](https://nflwebsite-ten.vercel.app)
* **REST API Endpoints**:
  * `GET /api/status`: Health check and cache status.
  * `GET /api/match-history`: Historical match logs, ATS records, and O/U splits.
  * `GET /api/full-schedule`: 18-week schedule with volume and efficiency cards.
  * `GET /api/team-stats`: 32-team Yahoo Sports rankings and statistical breakdowns.
  * `GET /api/team-highlights`: Defensive vulnerabilities and weekly trend spotlights.
  * `GET /api/matchup-deepdive`: Unit-by-unit head-to-head tactical matchup simulator.

---

## Local Development & Quick Start

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/jibinator1/nfl_website.git
cd nfl_website

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Launch Local Server
```bash
python run_app.py
```
The FastAPI server will boot and open your browser at **`http://localhost:8000`**.

---

## Project Structure

```
nfl_website/
├── api/
│   ├── index.py              # Vercel serverless entrypoint
│   └── data/                 # Pre-computed parquet cache files
├── backend/
│   ├── server.py             # FastAPI REST endpoints & router mounts
│   ├── data_loader.py        # nflreadpy data pipeline
│   ├── analytics.py          # Match history, stats rankings, schedule engine, H2H lab
│   └── pbp_features.py       # Advanced play-by-play aggregations
├── data/
│   └── manual_overrides.json # Trade and roster adjustments
├── frontend/
│   └── index.html            # Dark-themed Tailwind CSS dashboard
├── index.html                # Root static template for Vercel & local server
├── daily_update.py           # Automated daily stat pull & cache refresher
├── requirements.txt          # Production dependencies
├── vercel.json               # Vercel serverless configuration
├── run_app.py                # Standalone launcher
└── README.md
```

---

## Daily Data Updates (Auto-Sync to Vercel)

Because Vercel serverless functions have a read-only filesystem and execution time limits, fresh NFL stats are pulled locally using [`daily_update.py`](daily_update.py) or [`daily_update.sh`](daily_update.sh). The script fetches new games, recomputes the parquet cache, commits to GitHub, and triggers Vercel to automatically redeploy your live website!

### Run Manually
```bash
python daily_update.py
```

### Automated Daily Cron (Mac / Linux)
```bash
(crontab -l 2>/dev/null | grep -v "daily_update.sh"; echo "0 6 * * * cd "$PWD" && ./daily_update.sh >> update.log 2>&1") | crontab -
```

---

## License
MIT License. Open-source and free for non-commercial sports analytics.
