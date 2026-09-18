# NFL Analytics & Matchup Hub 🏈

An interactive, full-stack NFL statistics, team analytics, matchup lab (H2H), and 18-week schedule explorer.

> **Zero API Keys Required**: 100% powered by open-source NFL play-by-play and box score data via [`nflreadpy`](https://github.com/nflverse/nflreadpy) from the NFLverse project. No paid services, rate limits, or API keys required to run.

---

## 🌟 Core Features

### 1. 📅 18-Week Schedule & Matchup Explorer (2026 / 2025)
* **Full Schedule Coverage**: Interactive navigation across all 18 regular season weeks, game dates, and prime-time slots.
* **Per-Game Volume x Efficiency Table**: Head-to-head offensive vs. defensive statistical matchup cards featuring:
  * **Scoring**: Vegas Implied Team Totals.
  * **Passing**: Pass attempts, passing yards allowed, game-script neutral intent (**PROE**), yards per attempt (YPA offense vs. allowed), and **Explosive Pass Rates** (20+ yard chunk plays).
  * **Rushing**: Carries per game, rush yards allowed per game, yards per carry (YPC offense vs. allowed).
  * **Defense vs WR1s & Star RBs**: Measures each defense's average surplus/deficit allowed against opposing teams' **WR1** and lead rushers.
  * **Trenches**: Pass protection pressure allowed and defensive pressure generation rates.
* **Pass / Run Opportunity Funnel**: Dynamically models team play calling intent scaled by game pace and Vegas totals.

### 2. 📊 32-Team League Stats & Rankings
* **Comprehensive Metrics**: 25+ offensive and defensive volume & efficiency stats with league-wide `#1` to `#32` rankings.
* **Multi-Category Filtering**: Quick view toggles across All Stats, Passing, Rushing, Defense, Scoring, and Advanced Trenches.
* **Interactive Sorting**: Real-time ascending/descending sorting across any statistical column.

### 3. 🎯 Matchup Lab (Head-to-Head Comparison)
* **Custom Matchup Selector**: Select any two NFL franchises to simulate unit matchups.
* **Unit Battle Radar**: Compares overall team ratings, passing offense vs. secondary, and rushing attack vs. front seven.
* **WR1 & Star RB Showdown**: Tests how defenses hold up against the opponent's primary weapons.
* **Scramble Containment**: Rates defensive vulnerability to mobile QBs.

### 4. ⚡ Matchup Vulnerabilities & Trends
* Automated spotlight detection highlighting defensive pass funnels, soft run fronts, offensive red-zone efficiency, and rest advantages.

---

## 🚀 Quick Start

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

### 2. Launch the Application
```bash
python run_app.py
```
The FastAPI server will boot and automatically open your default browser at:
**`http://localhost:8000`**

---

## 🏗️ Architecture & Technology Stack

```
nfl_website/
├── backend/
│   ├── server.py             # FastAPI REST endpoints & HTML static server
│   ├── data_loader.py        # nflreadpy data pipeline (zero API keys)
│   ├── analytics.py          # Volume x efficiency metrics, H2H deep dive, schedule engine
│   └── pbp_features.py       # Advanced play-by-play metric aggregations
├── data/
│   ├── manual_overrides.json # Trade and injury adjustments
│   └── pbp_cache/            # Pre-computed parquet feature tables (<700KB)
├── frontend/
│   └── index.html            # Tailwind CSS + Lucide icons dark-themed dashboard
├── requirements.txt          # Python dependencies
├── run_app.py                # Standalone launcher
└── README.md
```

* **Frontend**: Vanilla HTML5, Tailwind CSS, Lucide Icons (pinned CDN), client-side dynamic DOM rendering.
* **Backend**: Python 3.10+, FastAPI, Uvicorn, Pandas, NumPy, PyArrow.
* **Data Sources**: Official NFL data via `nflreadpy` (NFLverse).

---

## 📄 License
MIT License. Open-source and free for non-commercial sports analytics.

---

## Deploy to Vercel

This repository is pre-configured for instant deployment on Vercel Serverless Functions (`@vercel/python`).

### Option 1: Automatic Deploy via GitHub (Recommended)
1. Go to [vercel.com](https://vercel.com) and click **"Add New Project"**.
2. Select your GitHub repository: **`jibinator1/nfl_website`**.
3. Keep default settings (Framework Preset: Other, Root Directory: `./`).
4. Click **Deploy**. Vercel will install `requirements.txt` and launch your live serverless app with a free HTTPS domain!

### Option 2: Deploy from Command Line
Run either:
* Double-click `deploy.bat` (Windows)
* Run `npx vercel --prod` in the project directory