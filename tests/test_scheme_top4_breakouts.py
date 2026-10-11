from backend.scheme_insights import compute_scheme_insights

def test_scheme_top4_breakouts():
    test_matchups = [
        ('HOU', 'TEN'),
        ('KC', 'BAL'),
        ('DET', 'GB'),
        ('NO', 'MIN'),
        ('PHI', 'DAL'),
        ('SF', 'LA'),
        ('BUF', 'MIA')
    ]

    for a, h in test_matchups:
        res = compute_scheme_insights(a, h)
        breakouts = res.get('efficiency_breakouts', [])
        assert 2 <= len(breakouts) <= 4, f"Expected 2-4 breakouts, got {len(breakouts)} for {a}@{h}"
        
        # Verify no duplicates
        players = [b['player'] for b in breakouts]
        assert len(players) == len(set(players)), f"Duplicates in {a}@{h}: {players}"
        
        # Verify clean structure & market-aware probabilistic metrics
        for b in breakouts:
            assert 'priority' not in b, f"Priority leaked in {b['player']}"
            assert b['player'] and b['team'] and b['pos'], f"Missing basic info in {b}"
            assert b['stat_focus'] in ('Receiving Yards', 'Receptions', 'Rushing Yards'), f"Invalid stat_focus in {b}"
            assert b.get('scheme_delta'), f"Missing scheme_delta in {b}"
            assert b.get('rationale'), f"Missing rationale in {b}"
            assert 0.44 <= b.get('win_prob', 0) <= 0.68, f"Uncalibrated win_prob {b.get('win_prob')} in {b}"
            assert b.get('win_prob_fmt', '').endswith('%'), f"Invalid win_prob_fmt in {b}"
            assert b.get('market_line'), f"Missing market_line in {b}"
            assert b.get('fair_line'), f"Missing fair_line in {b}"
            assert 'EV' in b.get('ev_pct', ''), f"Missing EV in {b}"
            assert b.get('ev_pct_num', 0) <= 14.0, f"Implausibly large EV {b.get('ev_pct_num')} in {b}"
            assert 'Grade' in b.get('confidence_tier', ''), f"Missing confidence_tier in {b}"
            assert b.get('z_score', '').endswith('z'), f"Missing z_score in {b}"

def test_hou_ten_breakouts():
    res = compute_scheme_insights('HOU', 'TEN')
    breakouts = res.get('efficiency_breakouts', [])
    player_names = [b['player'] for b in breakouts]
    
    # David Montgomery should NOT be on Houston
    assert 'D.Montgomery' not in player_names
    
    # Verify Nico Collins is present
    assert any('Collins' in name for name in player_names)

def test_pickens_on_cowboys():
    # George Pickens should NOT be on PIT in IND @ PIT
    res_pit = compute_scheme_insights('IND', 'PIT')
    pit_breakouts = res_pit.get('efficiency_breakouts', [])
    pit_players = [b['player'] for b in pit_breakouts if b['team'] == 'PIT']
    assert 'G.Pickens' not in pit_players
    
    # In DAL matchups, Pickens belongs to DAL
    res_dal = compute_scheme_insights('DAL', 'NYG')
    dal_breakouts = res_dal.get('efficiency_breakouts', [])
    dal_pickens = [b for b in dal_breakouts if b['player'] == 'G.Pickens']
    if dal_pickens:
        assert dal_pickens[0]['team'] == 'DAL'

def test_adams_on_rams():
    # Adams belongs to LA (LAR)
    res_la = compute_scheme_insights('LA', 'SF')
    la_breakouts = res_la.get('efficiency_breakouts', [])
    la_adams = [b for b in la_breakouts if b['player'] == 'D.Adams']
    if la_adams:
        assert la_adams[0]['team'] == 'LA'
    
    # In NYJ matchups, Adams is NOT on NYJ
    res_nyj = compute_scheme_insights('NYJ', 'NE')
    nyj_breakouts = res_nyj.get('efficiency_breakouts', [])
    nyj_players = [b['player'] for b in nyj_breakouts if b['team'] == 'NYJ']
    assert 'D.Adams' not in nyj_players

def test_jax_engram_and_washington():
    res = compute_scheme_insights('PHI', 'JAX')
    breakouts = res.get('efficiency_breakouts', [])
    jax_players = [b['player'] for b in breakouts if b['team'] == 'JAX']
    
    # Evan Engram must NOT be on JAX
    assert 'E.Engram' not in jax_players
    
    # Parker Washington is JAX WR1 and appears in JAX breakouts
    assert 'P.Washington' in jax_players
    
    # Brenton Strange is JAX TE1
    assert 'B.Strange' in jax_players

def test_den_engram():
    from backend.scheme_insights import get_team_scheme_profile
    den_prof = get_team_scheme_profile('DEN')
    den_targets = [t['name'] for t in den_prof['target_alignment']['key_targets']]
    assert 'E.Engram' in den_targets

    # Evan Engram should not be in JAX profile
    jax_prof = get_team_scheme_profile('JAX')
    jax_targets = [t['name'] for t in jax_prof['target_alignment']['key_targets']]
    assert 'E.Engram' not in jax_targets
    assert 'P.Washington' in jax_targets

def test_calibrated_edges_and_tiers():
    # Pickens, Nacua, Adams must have calibrated, realistic EVs (2% to 12%), not +40%
    res_dal = compute_scheme_insights('DAL', 'NYG')
    pickens = [b for b in res_dal.get('efficiency_breakouts', []) if b['player'] == 'G.Pickens']
    if pickens:
        p = pickens[0]
        assert p['win_prob'] <= 0.65, f"Pickens win_prob too high: {p['win_prob']}"
        assert p['ev_pct_num'] <= 12.0, f"Pickens EV too high: {p['ev_pct_num']}"
        assert p['fair_line_num'] <= 75.0, f"Pickens fair line too high: {p['fair_line_num']}"

    res_la = compute_scheme_insights('SF', 'LA')
    nacua = [b for b in res_la.get('efficiency_breakouts', []) if b['player'] == 'P.Nacua']
    if nacua:
        n = nacua[0]
        assert n['win_prob'] <= 0.65, f"Nacua win_prob too high: {n['win_prob']}"
        assert n['ev_pct_num'] <= 12.0, f"Nacua EV too high: {n['ev_pct_num']}"
        assert n['fair_line_num'] <= 75.0, f"Nacua fair line too high: {n['fair_line_num']}"

def test_dynamic_situational_pace_by_season():
    # 2026 (this year) for GB should be ~56.8% RZ pass rate, not static 45.7%
    res26 = compute_scheme_insights('CHI', 'GB', season=2026)
    gb_tempo_26 = res26.get('game_flow_tempo', {}).get('away_tempo', {})
    assert gb_tempo_26.get('rz_pass_rate') == 56.8, f"Expected 56.8% RZ pass rate for GB in 2026, got {gb_tempo_26.get('rz_pass_rate')}"
    assert gb_tempo_26.get('third_down_pass_rate') == 93.3

    # 2024 season should dynamically reflect 2024 data (46.8% RZ pass rate)
    res24 = compute_scheme_insights('CHI', 'GB', season=2024)
    gb_tempo_24 = res24.get('game_flow_tempo', {}).get('away_tempo', {})
    assert gb_tempo_24.get('rz_pass_rate') == 46.8, f"Expected 46.8% RZ pass rate for GB in 2024, got {gb_tempo_24.get('rz_pass_rate')}"




