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
        assert 2 <= len(breakouts) <= 6, f"Expected 2-6 breakouts, got {len(breakouts)} for {a}@{h}"
        
        # Verify no duplicates
        players = [b['player'] for b in breakouts]
        assert len(players) == len(set(players)), f"Duplicates in {a}@{h}: {players}"
        
        # Verify clean structure & descriptive statistics metrics
        for b in breakouts:
            assert 'priority' not in b, f"Priority leaked in {b['player']}"
            assert b['player'] and b['team'] and b['pos'], f"Missing basic info in {b}"
            assert b['stat_focus'] in ('Receiving Yards', 'Receptions', 'Rushing Yards'), f"Invalid stat_focus in {b}"
            assert 'last_10_stats' in b, f"Missing last_10_stats in {b}"
            assert 'median' in b['last_10_stats'], f"Missing median in {b['last_10_stats']}"
            assert 'p25' in b['last_10_stats'], f"Missing p25 in {b['last_10_stats']}"
            assert 'p75' in b['last_10_stats'], f"Missing p75 in {b['last_10_stats']}"
            assert 'n_games' in b['last_10_stats'], f"Missing n_games in {b['last_10_stats']}"
            assert 'splits' in b, f"Missing splits in {b}"
            assert 'red_zone' in b, f"Missing red_zone in {b}"
            assert 'defense_context' in b, f"Missing defense_context in {b}"

            # Strict check: verify no betting model words or residue
            prohibited = ['market_line', 'fair_line', 'confidence_tier', 'win_prob', 'ev_pct', 'verdict', 'matchup_mechanism']
            for field in prohibited:
                assert field not in b, f"Found prohibited model field '{field}' in breakout card {b['player']}"

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
    assert 'Parker Washington' in jax_players
    
    # Brenton Strange is JAX TE1
    assert 'Brenton Strange' in jax_players

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
    # Pickens and Nacua must have descriptive box stats (median, p25, p75, n_games), no betting predictions
    res_dal = compute_scheme_insights('DAL', 'NYG')
    pickens = [b for b in res_dal.get('efficiency_breakouts', []) if b['player'] == 'G.Pickens']
    if pickens:
        p = pickens[0]
        assert p['last_10_stats']['median'] > 0
        assert p['last_10_stats']['n_games'] > 0
        assert 'market_line' not in p
        assert 'win_prob' not in p

    res_la = compute_scheme_insights('SF', 'LA')
    nacua = [b for b in res_la.get('efficiency_breakouts', []) if b['player'] == 'P.Nacua']
    if nacua:
        n = nacua[0]
        assert n['last_10_stats']['median'] > 0
        assert n['last_10_stats']['n_games'] > 0
        assert 'market_line' not in n
        assert 'win_prob' not in n

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

def test_defensive_suppression_context():
    # Verify defense context is surfaced in matchups
    matchups = [('KC', 'BAL'), ('SF', 'LA'), ('DAL', 'NYG')]
    all_cards = []
    for a, h in matchups:
        res = compute_scheme_insights(a, h)
        all_cards.extend(res.get('efficiency_breakouts', []))
    assert len(all_cards) > 0
    # Verify no betting lines or verdicts exist
    for card in all_cards:
        assert 'defense_context' in card
        assert 'market_line' not in card
        assert 'verdict' not in card

