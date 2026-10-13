import json
import sys

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
    print(f"Season: {season} | Week: {week} | Sample Window: {sample_window}")
    print("=" * 80)
    print()

    # (1) Matchup Summary
    print("### (1) Matchup Summary")
    summary = g.get("matchup_summary", "")
    print(f"{summary}")
    print()

    # (2) Situational Tendencies
    print("### (2) Situational Tendencies")
    cov = g.get("coverage_tendencies", {})
    a_vs_h = cov.get("away_vs_home", {})
    h_vs_a = cov.get("home_vs_away", {})
    print(f"**{away} Defensive Tendencies:**")
    print(f"  * Middle-Field Open (MFO): {h_vs_a.get('def_mfo_pct', 0.0):.1f}% | Middle-Field Closed (MFC): {h_vs_a.get('def_mfc_pct', 0.0):.1f}%")
    print(f"  * Zone Coverage: {h_vs_a.get('def_zone_pct', 0.0):.1f}% | Man Coverage: {h_vs_a.get('def_man_pct', 0.0):.1f}%")
    print(f"  * Shell Archetype: {h_vs_a.get('def_archetype', 'N/A')}")
    print(f"  * Summary: {h_vs_a.get('coverage_summary', '')}")
    print()
    print(f"**{home} Defensive Tendencies:**")
    print(f"  * Middle-Field Open (MFO): {a_vs_h.get('def_mfo_pct', 0.0):.1f}% | Middle-Field Closed (MFC): {a_vs_h.get('def_mfc_pct', 0.0):.1f}%")
    print(f"  * Zone Coverage: {a_vs_h.get('def_zone_pct', 0.0):.1f}% | Man Coverage: {a_vs_h.get('def_man_pct', 0.0):.1f}%")
    print(f"  * Shell Archetype: {a_vs_h.get('def_archetype', 'N/A')}")
    print(f"  * Summary: {a_vs_h.get('coverage_summary', '')}")
    print()

    # (3) Opponent Vulnerabilities
    print("### (3) Opponent Vulnerabilities")
    run_splits = g.get("run_scheme_splits", {})
    a_run = run_splits.get("away_unit", {})
    h_run = run_splits.get("home_unit", {})
    print(f"**{away} Ground Attack vs. {home} Defense:**")
    print(f"  * {away} Offense: Zone {a_run.get('offense_zone_pct', 0):.1f}% ({a_run.get('zone_ypc', 0.0):.2f} YPC) | Gap {a_run.get('offense_gap_pct', 0):.1f}% ({a_run.get('gap_ypc', 0.0):.2f} YPC)")
    print(f"  * {home} Defense Allowed: Zone {a_run.get('opp_def_zone_ypc_allowed', 0.0):.2f} YPC [{a_run.get('opp_def_zone_rank_display', '')}] | Gap {a_run.get('opp_def_gap_ypc_allowed', 0.0):.2f} YPC [{a_run.get('opp_def_gap_rank_display', '')}]")
    print(f"  * Diagnostic: {a_run.get('comparison_summary', '')}")
    print()
    print(f"**{home} Ground Attack vs. {away} Defense:**")
    print(f"  * {home} Offense: Zone {h_run.get('offense_zone_pct', 0):.1f}% ({h_run.get('zone_ypc', 0.0):.2f} YPC) | Gap {h_run.get('offense_gap_pct', 0):.1f}% ({h_run.get('gap_ypc', 0.0):.2f} YPC)")
    print(f"  * {away} Defense Allowed: Zone {h_run.get('opp_def_zone_ypc_allowed', 0.0):.2f} YPC [{h_run.get('opp_def_zone_rank_display', '')}] | Gap {h_run.get('opp_def_gap_ypc_allowed', 0.0):.2f} YPC [{h_run.get('opp_def_gap_rank_display', '')}]")
    print(f"  * Diagnostic: {h_run.get('comparison_summary', '')}")
    print()

    # (4) Similar-Opponent Historical Lookup
    print("### (4) Similar-Opponent Historical Lookup")
    sim = sheet.get("section_5_similar_opponents", {})
    h_sim = sim.get("home_defense_vs_opponents_like_away", {})
    a_sim = sim.get("away_defense_vs_opponents_like_home", {})

    def _render_sim(sim_obj, def_tm, tgt_tm):
        print(f"**{def_tm} Defense vs. Opponents Similar to {tgt_tm}:**")
        if not sim_obj.get("render_match_table", True):
            print(f"  * {sim_obj.get('no_analogs_message')}")
        else:
            games_list = sim_obj.get("similar_games", [])
            print(f"  * Analog Games Evaluated ({len(games_list)} games under 1.50σ ceiling):")
            for gm in games_list:
                h2h_flag = " [Direct H2H]" if gm.get("is_direct_h2h") else " [Analog]"
                rush_val = gm.get('rb_rush_yards_allowed', 'N/A')
                pass_val = gm.get('pass_yards_allowed', 'N/A')
                print(f"    - {gm.get('season')} Wk {gm.get('week')} vs {gm.get('opponent')}{h2h_flag}: RMSD = {gm.get('rmsd', 0.0):.3f}σ ({gm.get('similarity_tier')}) | Allowed: {rush_val} RB rush yds, {pass_val} pass yds")
            str_med = sim_obj.get('median_rb_rush_yards_strong')
            all_med = sim_obj.get('median_rb_rush_yards_all')
            str_med_disp = f"{str_med} yds" if str_med is not None else "Suppressed (n < 3 strong analogs)"
            all_med_disp = f"{all_med} yds" if all_med is not None else "N/A"
            print(f"  * Median RB Rush Allowed: Strong Analogs = {str_med_disp} | All Evaluated Matches = {all_med_disp}")
        print()

    _render_sim(h_sim, home, away)
    _render_sim(a_sim, away, home)

    # (5) Dual Archetype Comparison
    print("### (5) Dual Archetype Comparison")
    h_dual = sim.get("home_defense_dual_archetype_splits", {})
    a_dual = sim.get("away_defense_dual_archetype_splits", {})

    def _render_dual(dual_obj, def_tm, opp_tm):
        print(f"**{def_tm} Defense vs. {opp_tm}-Profile Offenses (Ground vs. Aerial):**")
        status = dual_obj.get("can_conclude_status", "cannot_conclude")
        print(f"  * Conclusion Status: {status.upper()}")
        print(f"  * Sample Analysis: {dual_obj.get('conclusion_verdict', '')}")
        ct = dual_obj.get("contrast_table", [])
        if ct:
            for row in ct:
                arch = row.get("archetype", "")
                stat = row.get("status", "")
                str_yd = row.get("rb_rush_yards_strong")
                all_yd = row.get("rb_rush_yards_all")
                ypc = row.get("rb_ypc")
                pass_yd = row.get("pass_yards")
                print(f"    - {arch}: {stat}")
                print(f"      RB Rush Yds Allowed: Strong Med = {str_yd} yds | All Med = {all_yd} yds | YPC = {ypc} | Pass Yds = {pass_yd}")
        print()

    _render_dual(h_dual, home, away)
    _render_dual(a_dual, away, home)

    # (6) Player Usage Profiles
    print("### (6) Player Usage Profiles")
    profiles = g.get("player_usage_profiles", [])
    for p in profiles:
        p_name = p.get('player_display_name') or p.get('player')
        print(f"**{p_name} ({p.get('pos')} - {p.get('team')}) | Focus: {p.get('stat_focus')}**")
        l10 = p.get("last_10_stats", {})
        print(f"  * Last {l10.get('n_games', 0)} Games Box Stats: Median {l10.get('median')} {l10.get('unit')} (p25–p75: {l10.get('p25')}–{l10.get('p75')}, Range: {l10.get('min')}–{l10.get('max')})")
        print(f"  * Coverage Splits: {p.get('splits', {}).get('coverage_shell')}")
        print(f"  * Red Zone Opportunities: {p.get('red_zone', {}).get('summary_str')}")
        print(f"  * Opponent Defense Context: {p.get('defense_context')}")
        print()

    # (7) Pace and Tempo
    print("### (7) Pace and Tempo")
    gf = g.get("game_flow_tempo", {})
    print(f"  * {gf.get('pace_and_tempo', '')}")
    print(f"  * Situational Tendencies: {gf.get('situational_tendencies', '')}")
    print()

    # (8) Head-to-Head History
    print("### (8) Head-to-Head History")
    h2h = sheet.get("section_6_head_to_head", {})
    if h2h.get("played_recently"):
        print(f"  * {h2h.get('descriptive_note')}")
        ab = h2h.get("away_team", {})
        hb = h2h.get("home_team", {})
        print(f"  * {ab.get('team')}: {ab.get('total_plays')} plays, {ab.get('rush_yards')} rush yds, {ab.get('pass_yards')} pass yds, Turnovers: {ab.get('turnovers')}, EPA/play: {ab.get('epa_per_play')}")
        print(f"    - Leaders: Rusher {ab.get('leading_rusher')} | Receiver {ab.get('leading_receiver')}")
        print(f"  * {hb.get('team')}: {hb.get('total_plays')} plays, {hb.get('rush_yards')} rush yds, {hb.get('pass_yards')} pass yds, Turnovers: {hb.get('turnovers')}, EPA/play: {hb.get('epa_per_play')}")
        print(f"    - Leaders: Rusher {hb.get('leading_rusher')} | Receiver {hb.get('leading_receiver')}")
    else:
        print(f"  * {h2h.get('note')}")
    print()

    # (9) Reported Personnel Status
    print("### (9) Reported Personnel Status")
    pers = sheet.get("section_7_key_personnel", {})
    print(f"  * Protocol: {pers.get('note', '')}")
    a_injs = pers.get("away_team_injuries", [])
    h_injs = pers.get("home_team_injuries", [])
    if a_injs:
        print(f"  * {away} Reported Status:")
        for inj in a_injs[:5]:
            p_n = inj.get('player') or inj.get('player_name') or 'Player'
            pos = inj.get('position', '')
            st = inj.get('status', 'Active')
            pr = inj.get('practice_status', 'Full')
            inj_type = inj.get('primary_injury', '')
            print(f"    - {p_n} ({pos}): Status = {st}, Practice = {pr}" + (f", Injury = {inj_type}" if inj_type else ""))
    else:
        print(f"  * {away}: No reported injury status entries on file.")

    if h_injs:
        print(f"  * {home} Reported Status:")
        for inj in h_injs[:5]:
            p_n = inj.get('player') or inj.get('player_name') or 'Player'
            pos = inj.get('position', '')
            st = inj.get('status', 'Active')
            pr = inj.get('practice_status', 'Full')
            inj_type = inj.get('primary_injury', '')
            print(f"    - {p_n} ({pos}): Status = {st}, Practice = {pr}" + (f", Injury = {inj_type}" if inj_type else ""))
    else:
        print(f"  * {home}: No reported injury status entries on file.")
    print()
    print("=" * 80)
    print()
