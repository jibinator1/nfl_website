import json
import sys
import os

# Add repo root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

    print("=" * 80)
    print(f"MATCHUP: {away_name} ({away}) @ {home_name} ({home})")
    print(f"Season: {season} | Week: {week} | {sample_window}")
    print("=" * 80)
    print()

    # SECTION 1 — TEAM PLAYSTYLE IDENTITY & SCHEMATIC FUNNEL (BOTH SIDES)
    print("### Section 1 — Team Playstyle Identity & Schematic Funnel")
    s1 = g.get("section_1_playstyle_matchup") or g.get("playstyle_funnel", {}).get("section_1_playstyle_matchup", {})
    
    for matchup_key, label in [
        ("away_off_vs_home_def", f"{away} Offense vs {home} Defense"),
        ("home_off_vs_away_def", f"{home} Offense vs {away} Defense"),
    ]:
        m_data = s1.get(matchup_key, {})
        if not m_data:
            continue
        print(f"[{label}]")
        def_funnel = m_data.get("defense_funnel", {})
        off_playstyle = m_data.get("offense_playstyle", {})
        funnels = m_data.get("funnel_insights", [])
        
        print(f"  * [DEFENSIVE SHELL & FUNNEL] {def_funnel.get('summary', 'N/A')}")
        print(f"  * [OFFENSIVE PLAYSTYLE] {off_playstyle.get('summary', 'N/A')}")
        
        if funnels:
            print("  * What This Opens Up (Schematic Funnels):")
            for fn in funnels:
                print(f"    - [{fn.get('name')}]: {fn.get('takeaway')}")
        print()

    # SECTION 2 — PLAYER ARCHETYPES TO TARGET
    print("### Section 2 — Player Archetypes to Target (Matchup Beneficiaries & Usage Concentration)")
    s2 = g.get("section_2_archetypes_to_target") or g.get("playstyle_funnel", {}).get("section_2_archetypes_to_target", {})
    beneficiaries = s2.get("beneficiary_players", [])
    
    if beneficiaries:
        print("  [Matchup Beneficiaries — Who Benefits & Why]:")
        for b in beneficiaries:
            p_name = b.get("player")
            pos = b.get("pos")
            tm = b.get("team")
            arch = b.get("archetype")
            ctx = b.get("role_context")
            rat = b.get("matchup_rationale")
            print(f"  * {p_name} ({pos} - {tm}) — {arch} [{ctx}]:")
            print(f"    Rationale: {rat}")
    else:
        print("  * No specific schematic archetype beneficiaries identified.")
    print()

    print("  [Concentrated Alpha Usage (Target Share >= 22%, Rush Share >= 45%, RZ Share >= 35%)]:")
    alphas = s2.get("concentrated_alpha_usage") or g.get("concentrated_player_usage", [])
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
        print("  * No skill players cleared alpha concentration thresholds.")
    print()

    # SECTION 3 — HISTORICAL STYLISTIC ANALOGS & H2H PERFORMANCE
    print("### Section 3 — Historical Stylistic Analogs & H2H Performance")
    s3 = g.get("section_3_historical_analogs") or g.get("playstyle_funnel", {}).get("section_3_historical_analogs", {})
    h_sim = s3.get("home_defense_vs_similar_away", {})
    a_sim = s3.get("away_defense_vs_similar_home", {})

    units = [
        (h_sim, home, away),
        (a_sim, away, home)
    ]
    rendered_any_analog = False
    for sim_obj, def_tm, opp_tm in units:
        s_med = sim_obj.get("strong_analogs_median") or sim_obj.get("median_rb_rush_yards_strong")
        prod = sim_obj.get("strong_production_allowed")
        h2h_games = sim_obj.get("direct_h2h_games", [])
        
        if s_med is not None or h2h_games:
            rendered_any_analog = True
            status = sim_obj.get("strong_status", "qualified games")
            if s_med is not None:
                print(f"  * {def_tm} Defense vs {opp_tm}-like Strong Analogs ({status}):")
                if prod:
                    print(f"    - Medians Allowed: {prod.get('rb_rush_yards', 'N/A')} RB rush yds | {prod.get('pass_yards', 'N/A')} pass yds | {prod.get('wr_rec_yards', 'N/A')} WR rec yds | {prod.get('te_rec_yards', 'N/A')} TE rec yds | {prod.get('qb_rush_yards', 'N/A')} QB rush yds")
                else:
                    print(f"    - Median RB Rush Yards Allowed: {s_med:.1f}")
            else:
                print(f"  * {def_tm} Defense vs {opp_tm}-like Analogs: {status}")
            
            if h2h_games:
                print("    [Direct H2H Games]:")
                for gm in h2h_games:
                    rush_val = gm.get('rb_rush_yards_allowed', gm.get('yds', 'N/A'))
                    pass_val = gm.get('pass_yards_allowed', 'N/A')
                    wr_val = gm.get('wr_rec_yards_allowed', 'N/A')
                    te_val = gm.get('te_rec_yards_allowed', 'N/A')
                    qb_val = gm.get('qb_rush_yards_allowed', 'N/A')
                    print(f"      - {gm.get('season')} Wk {gm.get('week')} vs {gm.get('opponent')}: Allowed {rush_val} RB rush yds, {pass_val} pass yds, {wr_val} WR yds, {te_val} TE yds, {qb_val} QB rush yds")
            print()

    if not rendered_any_analog:
        print("  * No qualified stylistic analogs meeting the <= 0.85σ RMSD threshold and minimum 3-opponent floor.")
        print()

    print("=" * 80)
    print()
