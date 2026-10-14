import json
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

# Ensure backend can be imported if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STAT_NAME_MAP = {
    'pass_defense.overall_dropbacks.epa_per_dropback_opp_adj': 'Pass Defense EPA/dropback',
    'pass_defense.overall_dropbacks.epa_per_dropback': 'Pass Defense EPA/dropback',
    'rush_defense.overall_rb.epa_per_carry_opp_adj': 'Rush Defense EPA/carry',
    'rush_defense.overall_rb.epa_per_carry': 'Rush Defense EPA/carry',
    'player_archetypes_allowed.vs_workhorse_rbs_15_plus_carries.median_rush_yards_allowed': 'Rush Yards Allowed to Workhorse RBs (≥15 carries)',
    'player_archetypes_allowed.vs_committee_rbs_under_10_carries.median_rush_yards_allowed': 'Rush Yards Allowed to Committee RBs (<10 carries)',
    'player_archetypes_allowed.vs_wr1_top_target.median_receiving_yards_allowed': 'Receiving Yards Allowed to WR1s',
    'player_archetypes_allowed.vs_high_volume_tes_5_plus_targets.median_receiving_yards_allowed': 'Receiving Yards Allowed to High-Volume TEs (≥5 targets)',
    'down_splits.down_1.pass_rate': '1st Down Pass Rate',
    'down_splits.down_1.rush_rate': '1st Down Rush Rate',
    'down_splits.down_2.pass_rate': '2nd Down Pass Rate',
    'down_splits.down_2.rush_rate': '2nd Down Rush Rate',
    'down_splits.down_3.pass_rate': '3rd Down Pass Rate',
    'down_splits.down_3.rush_rate': '3rd Down Rush Rate',
    'game_state_splits.leading.yards_per_play': 'Leading Yards/Play',
    'game_state_splits.leading.pass_rate': 'Leading Pass Rate',
    'game_state_splits.leading.rush_rate': 'Leading Rush Rate',
    'game_state_splits.leading.epa_per_play': 'Leading EPA/play',
    'game_state_splits.leading.success_rate': 'Leading Success Rate',
    'game_state_splits.neutral.pass_rate': 'Neutral Pass Rate',
    'game_state_splits.neutral.rush_rate': 'Neutral Rush Rate',
    'game_state_splits.trailing.pass_rate': 'Trailing Pass Rate',
    'game_state_splits.trailing.rush_rate': 'Trailing Rush Rate',
    'field_position_splits.opp_49_to_21.td_rate': 'Opp 49–21 TD Rate',
    'field_position_splits.opp_49_to_21.epa_per_play': 'Opp 49–21 EPA/play',
    'red_zone_shares.inside_20.wr_target_share': 'RZ Inside 20 WR Target Share',
    'red_zone_shares.inside_10.pass_rate': 'RZ Inside 10 Pass Rate',
    'red_zone_shares.inside_5.pass_rate': 'RZ Inside 5 Pass Rate',
    'fourth_quarter_margin.trailing_two_scores.pass_rate': 'Trailing Two Scores Pass Rate',
    'fourth_quarter_margin.one_score_game.pass_rate': 'One-Score 4th Qtr Pass Rate',
    'positions_allowed.wr.yards_per_game': 'WR Yards/Game Allowed',
    'positions_allowed.wr.epa_per_target': 'WR EPA/Target Allowed',
    'positions_allowed.rb.yards_per_target': 'RB Yards/Target Allowed',
    'positions_allowed.te.yards_per_target': 'TE Yards/Target Allowed',
    'field_position_splits.red_zone_20_to_1.pass_rate': 'Red Zone (20–1) Pass Rate',
    'game_state_splits.neutral.epa_per_play': 'Neutral EPA/play',
    'pass_defense.target_depth_allowed.short_0_to_9.epa_per_attempt': 'Short (0–9 yds) Pass EPA Allowed',
}

def clean_stat_name(path: str) -> str:
    if path in STAT_NAME_MAP:
        return STAT_NAME_MAP[path]
    parts = path.split('.')
    cleaned = [p.replace('_', ' ').replace('opp adj', '').replace('Opp Adj', '').title() for p in parts if p not in ('median', 'allowed', 'splits')]
    res = ' '.join(cleaned).strip()
    return res.replace(' (Opp-Adj)', '').replace(' Opp-Adj', '').replace(' Opp Adj', '')

def format_val(val, unit=""):
    if val is None:
        return "--"
    try:
        n = float(val)
        if unit in ("rate", "share") or "Rate" in unit or "Share" in unit:
            return f"{n * 100.0:.1f}%"
        if abs(n) < 1.0 and n != 0.0:
            return f"{n:+.3f}" if n < 0 or n > 0 else f"{n:.3f}"
        return f"{n:.1f}" if n % 1 != 0 else f"{int(n)}"
    except (ValueError, TypeError):
        return str(val)

def extract_qualifying_outliers(g, min_sample_required=1):
    away = g.get("away_team")
    home = g.get("home_team")
    week = g.get("week", 1)
    
    sheet = g.get("matchup_stat_sheet", {})
    s1 = sheet.get("section_1_away_offense_tendencies", {})
    s3 = sheet.get("section_3_home_offense_tendencies", {})
    
    team_games = {
        away: s1.get("n_games") or 19,
        home: s3.get("n_games") or 19
    }
    
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
                    n_games = team_games.get(team_code, 19)

                # Rule 1: A rank with n < 6 should be suppressed, and the stat skipped entirely.
                # The filter should only run on stats that already pass the n >= 6 gate.
                if n_games < 6:
                    pass
                elif r is not None and (r <= 6 or r >= 27):
                    sample_size = d.get("sample_size")
                    if sample_size is not None and sample_size < min_sample_required:
                        pass
                    else:
                        sn = clean_stat_name(path)
                        u = d.get("unit", "")
                        if "Rate" in sn or "Share" in sn:
                            u = "rate"
                            
                        extremity = abs(r - 16.5)
                        qualifying.append({
                            "team": team_code,
                            "stat_name": sn,
                            "raw_value": d.get("raw_value"),
                            "league_median": d.get("league_median"),
                            "unit": u,
                            "rank": r,
                            "n_games": n_games,
                            "extremity": extremity,
                        })
                        
                for k, v in d.items():
                    walk(v, f"{path}.{k}" if path else k)
                    
        walk(sec_dict)
        
    qualifying.sort(key=lambda x: x["extremity"], reverse=True)
    
    # Deduplicate binary pairs (e.g. 2nd Down Pass Rate vs 2nd Down Rush Rate)
    filtered = []
    seen_binary = set()
    for c in qualifying:
        sn = c["stat_name"]
        tm = c["team"]
        binary_key = None
        if "Pass Rate" in sn or "Rush Rate" in sn:
            group = sn.replace("Pass Rate", "").replace("Rush Rate", "").strip()
            binary_key = (tm, group)
        if binary_key and binary_key in seen_binary:
            continue
        if binary_key:
            seen_binary.add(binary_key)
        filtered.append(c)
        
    return filtered

def main():
    JSON_PATH = "scratch_insights_output.json"
    if not os.path.exists(JSON_PATH):
        print(f"Error: {JSON_PATH} not found.", file=sys.stderr)
        sys.exit(1)

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        games_data = json.load(f)

    for key, g in games_data.items():
        away = g.get("away_team")
        home = g.get("home_team")
        season = g.get("season", 2026)
        week = g.get("week")
        
        sheet = g.get("matchup_stat_sheet", {})
        s1 = sheet.get("section_1_away_offense_tendencies", {})
        s3 = sheet.get("section_3_home_offense_tendencies", {})
        qualifying = extract_qualifying_outliers(g, min_sample_required=1)
        
        ng = qualifying[0]["n_games"] if qualifying else (s1.get("n_games") or s3.get("n_games") or sheet.get("sample_window", {}).get("n_games") or 19)
        week_n = max(1, (week - 1) if week is not None else 1)
        sample_label = f"2025 season + {season} through week {week_n} (n={ng} games)"
        
        print("=" * 80)
        print(f"MATCHUP: {away} @ {home} (Season {season} Week {week})")
        print(f"Sample: {sample_label}")
        print("=" * 80)
        
        # Rule: If fewer than 3 stats qualify, output 'No notable outliers this week.'
        if len(qualifying) < 3:
            print("No notable outliers this week.")
        else:
            # Show at most 5 stats per matchup
            for stat in qualifying[:5]:
                t = stat["team"]
                sn = stat["stat_name"]
                u = stat["unit"]
                rv = format_val(stat["raw_value"], u)
                med = format_val(stat["league_median"], u)
                rk = stat["rank"]
                ng_val = stat["n_games"]
                print(f"{t} {sn}: {rv} (league median {med}, rank #{rk} of 32, n={ng_val})")
                
        print()

if __name__ == "__main__":
    main()
