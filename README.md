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

### 5. Player Floor Streaks (Hard Line Thresholds)
* **Streak Tracking**: Tracks consecutive games where player production meets or exceeds their established hard floor threshold (-30% margin off trailing baseline median, minimum 10 yards).
* **Hard Line Invariant**: Evaluated as a fixed hard line threshold across all games in the streak rather than shifting dynamically per game.
* **Position & Metric Filters**: Supports QB (rushing yds), RB (rushing yds), WR (receiving yds), and TE (receiving yds) with customizable minimum streak and games filters.
* **Reliability Tiers**: Segregates players into Elite (8+ games), Consistent (5-7 games), Active (2-4 games), and Breached.

### 6. Matchup Vulnerabilities & Trends
* Automated spotlight detection highlighting defensive pass funnels, soft run fronts, offensive red-zone efficiency, and rest advantages.

### 7. Scheme Insights Engine (Forensic Matchup Scouting)

The Scheme Insights engine is a strictly **descriptive, forensic scouting tool** for NFL matchups. It takes raw play-by-play data and translates it into situational tendencies, historical distributions, and matchup fits. 

It explicitly rejects black-box projections, "betting picks," and fabricated narrative labels. Instead, it provides transparent historical stats, rank context, and sample sizes ($n$) so users can draw their own football conclusions. To prevent early-season noise, it utilizes a combined multi-season sample baseline.

The interface is organized into five sections:

* **Executive Scouting Header**
  * **Sample Window**: Displays the exact evaluation sample and number of games evaluated.
  * **Scout Cards**: Three quick-glance summaries covering:
    * *Passing Game*: Primary coverage shells faced vs. receiver target distribution.
    * *Trench & Ground*: Offensive Gap vs. Zone run mix matched against opposing defensive front resistance.
    * *Script & Pace*: Historical play volume and neutral pace environment.
* **Game Script & Pace**
  * **Situational Pace**: Seconds per snap in neutral situations (score within 6 points), historical plays per game, check-down frequency, and red-zone touch leaders.
  * **Coaching Intent**: Six situational boxes showing how each coaching staff changes their pass/run ratios based on the scoreboard (Leading by ≥7, Neutral, Trailing by ≥7). Rates are compared against the league median with team rank and sample size.
* **Passing Matchup: Shells & Targets**
  * **Efficiency Comparison**: Compares passing output (yards per pass attempt) directly against the defense's allowed target-depth baseline.
  * **Defensive Coverage Shells**: Frequencies for Middle-Field Open (Cover 2, 4, 6) vs. Middle-Field Closed (Cover 1, 3), overall Zone vs. Man splits, and a specific 6-shell breakdown.
  * **Receiver Alignment & Target Rates**: Alignment distributions for primary weapons (Slot, Perimeter/Outside, Inline) and target share splits against Zone vs. Man coverage.
* **Trenches & Run Game**
  * **Ground Fit Comparison**: The offense's rushing style (Gap/Power vs. Outside Zone) cross-referenced with the defensive front's allowed YPC against those specific schemes.
  * **Trench Resistance**: Compares rushing efficiency inside the tackles vs. outside the tackles against defensive allowances.
  * **Similar Opponent Analog Table**: Finds the closest historical opponent games matching the team's stylistic run/pass identity and displays how the defense actually performed in those specific games (with strict filters for sample size and distance).
* **Player Profiles & Situational Usage**
  * **Key Skill Players**: Highlights up to 6 key players (Lead RB, WR1, and primary TE/WR2 for each team).
  * **Verified Active Rosters**: Pulls directly from active weekly rosters to ensure every player actually took snaps for that team in the current season, eliminating stale or traded names.
  * **Historical Box Stats**:
    * Median production over the current sample window.
    * Interquartile range (25th percentile floor to 75th percentile ceiling).
    * Exact red-zone touch volume (inside the 20 and inside the 10).
    * Defensive context (e.g., opposing defense's Zone % or YPC allowed).

---

## Live Access & Deployment

The application is deployed on Vercel Serverless Functions (`@vercel/python`):

* **Live Dashboard**: [https://nflwebsite-ten.vercel.app](https://nflwebsite-ten.vercel.app)
* **REST API Endpoints**:
  * `GET /api/status`: Health check and cache status.
  * `GET /api/scheme-insights`: Forensic scheme tendencies, coverage shells, trench fits, and player usage distributions.
  * `GET /api/floor-streak`: Player floor streaks with hard line thresholds.
  * `GET /api/export-matchups-excel`: Generates and downloads multi-tab matchup workbooks (.xlsx).
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
│   ├── scheme_insights.py    # Multi-season forensic scheme insights & player usage profiles
│   ├── team_tendencies.py    # Situational pass/run splits & defensive coverage aggregations
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
