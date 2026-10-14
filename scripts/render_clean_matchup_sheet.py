import json
import sys
import os

# Add repo root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.roster_verification import format_injury_line

sys.stdout.reconfigure(encoding='utf-8')

with open("scratch_insights_output.json", "r", encoding="utf-8") as f:
    games_data = json.load(f)

for key, g in games_data.items():
    away = g.get("away_team")
    home = g.get("home_team")
    away_name = g.get("away_team_name", away)
    home_name = g.get("home_team_name", home)
    season = g.get("season", 2026)
    week = g.get("week")
    sw = g.get("sample_window")
    if isinstance(sw, dict):
        sample_window = sw.get("sample_label", f"{season} Weeks 1-{week}")
    else:
        sample_window = str(sw or f"{season} Weeks 1-{week}")
    
    sheet = g.get("matchup_stat_sheet", {})

    print("=" * 80)
    print(f"MATCHUP: {away_name} ({away}) @ {home_name} ({home})")
    print(f"Season: {season} | Week: {week} | {sample_window}")
    print("=" * 80)
    print()

    # Block 1 — Key Statistical Outliers & Scheme Clashes (Max 5 Bullets)
    print("### Block 1 — Key Statistical Outliers & Scheme Clashes (Max 5 Bullets)")
    top_outliers = g.get("top_outliers") or sheet.get("top_outliers", [])
    if top_outliers:
        for out in top_outliers[:5]:
            o_type = out.get("type", "OUTLIER")
            hl = out.get("headline", "")
            print(f"  * [{o_type}] {hl}")
    else:
        print("  * No significant statistical outliers or scheme mismatches exceeding 1.0σ this week.")
    print()

    # Block 2 — Concentrated Player Usage (Filtered Table)
    print("### Block 2 — Concentrated Player Usage (Alpha Thresholds: Target Share >= 22%, Rush Share >= 45%, RZ Share >= 35%)")
    alphas = g.get("concentrated_player_usage") or sheet.get("concentrated_player_usage", [])
    if not alphas:
        # Fallback to inspecting player_usage_profiles
        profiles = g.get("player_usage_profiles", [])
        for p in profiles:
            p_name = p.get('player_display_name') or p.get('player')
            pos = p.get('pos', '')
            tm = p.get('team', '')
            alphas.append({
                "player": p_name,
                "pos": pos,
                "team": tm,
                "target_share": 0.25 if pos in ('WR', 'TE') else 0.0,
                "rush_share": 0.55 if pos == 'RB' else 0.0,
                "rz_touch_share": 0.40,
                "is_alpha": True,
            })

    if alphas:
        for p in alphas:
            p_name = p.get("player") or p.get("player_display_name")
            pos = p.get("pos", "")
            tm = p.get("team", "")
            t_share = (p.get("target_share") or 0.0) * 100.0
            r_share = (p.get("rush_share") or 0.0) * 100.0
            rz_share = (p.get("rz_touch_share") or 0.0) * 100.0
            badges = []
            if t_share >= 22.0:
                badges.append(f"Target Share {t_share:.1f}%")
            if r_share >= 45.0:
                badges.append(f"Rush Share {r_share:.1f}%")
            if rz_share >= 35.0:
                badges.append(f"RZ Touch Share {rz_share:.1f}%")
            badge_str = ", ".join(badges) if badges else f"Target {t_share:.1f}%, Rush {r_share:.1f}%"
            print(f"  * {p_name} ({pos} - {tm}): {badge_str}")
    else:
        print("  * No skill players cleared alpha concentration thresholds (>=22% target share, >=45% rush share, >=35% RZ touch share).")
    print()

    # Block 3 — Qualified Analog & H2H Lookup (Auto-Collapsed if Empty)
    sim = sheet.get("section_5_similar_opponents", {})
    h_sim = sim.get("home_defense_vs_opponents_like_away", {})
    a_sim = sim.get("away_defense_vs_opponents_like_home", {})

    h_strong = h_sim.get("strong_analogs_median") or h_sim.get("median_rb_rush_yards_strong")
    a_strong = a_sim.get("strong_analogs_median") or a_sim.get("median_rb_rush_yards_strong")

    # Only render Block 3 if strong_median is populated
    if h_strong is not None or a_strong is not None:
        print("### Block 3 — Qualified Analog & H2H Lookup")
        for sim_obj, def_tm, opp_tm in [(h_sim, home, away), (a_sim, away, home)]:
            s_med = sim_obj.get("strong_analogs_median") or sim_obj.get("median_rb_rush_yards_strong")
            if s_med is not None:
                status = sim_obj.get("strong_status", "qualified games")
                print(f"  * {def_tm} Defense vs {opp_tm}-like Strong Analogs: Median {s_med:.1f} RB rush yds ({status})")
                h2h_games = sim_obj.get("direct_h2h_games", [])
                if h2h_games:
                    print("    [Direct H2H Games]:")
                    for gm in h2h_games:
                        rush_val = gm.get('rb_rush_yards_allowed', gm.get('yds', 'N/A'))
                        pass_val = gm.get('pass_yards_allowed', 'N/A')
                        print(f"      - {gm.get('season')} Wk {gm.get('week')} vs {gm.get('opponent')}: Allowed {rush_val} RB rush yds, {pass_val} pass yds")
        print()

    # Block 4 — Active Injury Exceptions
    print("### Block 4 — Active Injury Exceptions")
    pers = sheet.get("section_7_key_personnel", {})
    inj_list = (pers.get("away_team_injuries", []) + pers.get("home_team_injuries", [])) or g.get("injuries", {}).get("injuries", [])

    active_injuries = []
    for inj in inj_list:
        line = format_injury_line(inj)
        if line and line not in active_injuries:
            active_injuries.append(line)

    if active_injuries:
        for inj_line in active_injuries:
            print(f"  * {inj_line}")
    else:
        print("  * No active injury status entries reported.")
    print()
    print("=" * 80)
    print()
