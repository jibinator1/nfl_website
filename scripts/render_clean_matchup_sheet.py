import json
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

# Ensure backend can be imported if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

JSON_PATH = "scratch_insights_output.json"
if not os.path.exists(JSON_PATH):
    print(f"Error: {JSON_PATH} not found.", file=sys.stderr)
    sys.exit(1)

with open(JSON_PATH, "r", encoding="utf-8") as f:
    games_data = json.load(f)

STAT_NAME_MAP = {
    'pass_defense.overall_dropbacks.epa_per_dropback_opp_adj': 'Pass Defense EPA/dropback',
    'rush_defense.overall_rb.epa_per_carry_opp_adj': 'Rush Defense EPA/carry',
    'player_archetypes_allowed.vs_workhorse_rbs_15_plus_carries.median_rush_yards_allowed': 'Rush Yards Allowed to Workhorse RBs (≥15 carries)',
    'player_archetypes_allowed.vs_committee_rbs_under_10_carries.median_rush_yards_allowed': 'Rush Yards Allowed to Committee RBs (<10 carries)',
    'player_archetypes_allowed.vs_wr1_top_target.median_receiving_yards_allowed': 'Receiving Yards Allowed to WR1s',
    'player_archetypes_allowed.vs_high_volume_tes_5_plus_targets.median_receiving_yards_allowed': 'Receiving Yards Allowed to High-Volume TEs (≥5 targets)',
}

def clean_stat_name(path: str) -> str:
    if path in STAT_NAME_MAP:
        return STAT_NAME_MAP[path]
    parts = path.split('.')
    cleaned = [p.replace('_', ' ').replace('opp adj', '').replace('Opp Adj', '').title() for p in parts if p not in ('median', 'allowed', 'splits')]
    res = ' '.join(cleaned).strip()
    return res.replace(' (Opp-Adj)', '').replace(' Opp-Adj', '').replace(' Opp Adj', '')

def format_num(val):
    if val is None:
        return '--'
    try:
        n = float(val)
        if abs(n) < 1.0 and n != 0.0:
            return f"{n:+.3f}" if n < 0 or n > 0 else f"{n:.3f}"
        return f"{n:.1f}" if n % 1 != 0 else f"{int(n)}"
    except (ValueError, TypeError):
        return str(val)

def extract_qualifying_outliers(g, min_sample_required=1):
    away = g.get("away_team")
    home = g.get("home_team")
    week = g.get("week", 1)
    # Number of prior regular-season games played before this week
    team_games_default = {away: max(1, week - 1), home: max(1, week - 1)}
    
    sheet = g.get("matchup_stat_sheet", {})
    sections = [
        ("section_1_away_offense_tendencies", away),
        ("section_2_home_defense_vulnerabilities", home),
        ("section_3_home_offense_tendencies", home),
        ("section_4_away_defense_vulnerabilities", away),
    ]
    
    qualifying = []
    
    for sec_key, team_code in sections:
        sec_dict = sheet.get(sec_key, {})
        if not isinstance(sec_dict, dict):
            continue
            
        def walk(d, path=""):
            if isinstance(d, dict):
                r = d.get("rank")
                n_games = d.get("n_games")
                if n_games is None:
                    n_games = team_games_default.get(team_code, max(1, week - 1))

                # Rule 1: A rank with n < 6 should be suppressed, and the stat skipped entirely.
                # The filter should only run on stats that already pass the n >= 6 gate.
                if n_games < 6:
                    # Small sample rank suppression: skip entirely
                    pass
                elif r is not None and (r <= 6 or r >= 27):
                    # Check sample size
                    sample_size = d.get("sample_size")
                    # If sample_size is explicitly 0, skip (unplayed / no plays faced)
                    if sample_size is not None and sample_size < min_sample_required:
                        pass
                    else:
                        raw_val = d.get("raw_value")
                        med_val = d.get("league_median")
                        
                        # Distance from median for ranking importance (most extreme rank first)
                        extremity = abs(r - 16.5)
                        
                        qualifying.append({
                            "team": team_code,
                            "stat_name": clean_stat_name(path),
                            "raw_value": raw_val,
                            "league_median": med_val,
                            "rank": r,
                            "n_games": n_games,
                            "extremity": extremity,
                            "sample_size": sample_size,
                            "is_small_sample": d.get("is_small_sample", False)
                        })
                        
                for k, v in d.items():
                    walk(v, f"{path}.{k}" if path else k)
                    
        walk(sec_dict)
        
    # Sort by extremity (rank closest to 1 or 32 first)
    qualifying.sort(key=lambda x: x["extremity"], reverse=True)
    return qualifying

for key, g in games_data.items():
    away = g.get("away_team")
    home = g.get("home_team")
    season = g.get("season", 2026)
    week = g.get("week")
    
    print("=" * 80)
    print(f"MATCHUP: {away} @ {home} (Season {season} Week {week})")
    print("=" * 80)
    
    qualifying = extract_qualifying_outliers(g, min_sample_required=1)
    
    # Rule: If fewer than 3 stats qualify, output 'No notable outliers this week.'
    if len(qualifying) < 3:
        print("No notable outliers this week.")
    else:
        # Show at most 5 stats per matchup
        for stat in qualifying[:5]:
            t = stat["team"]
            sn = stat["stat_name"]
            rv = format_num(stat["raw_value"])
            med = format_num(stat["league_median"])
            rk = stat["rank"]
            ng = stat["n_games"]
            print(f"{t} {sn}: {rv} (league median {med}, rank #{rk} of 32, n={ng})")
            
    print()
