import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.scheme_insights import compute_scheme_insights

games = [
    ("BAL", "DAL", 3),
    ("PIT", "CLE", 4),
    ("PHI", "JAX", 5)
]

results = {}
for away, home, wk in games:
    print(f"Computing insights for {away} @ {home} Week {wk}...", file=sys.stderr)
    res = compute_scheme_insights(home_team=home, away_team=away, season=2026, week=wk)
    results[f"{away}_at_{home}_w{wk}"] = res

with open("scratch_insights_output.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=str)

print("Saved scratch_insights_output.json successfully!", file=sys.stderr)
