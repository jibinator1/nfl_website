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
        
        # Verify clean structure
        for b in breakouts:
            assert 'priority' not in b, f"Priority leaked in {b['player']}"
            assert b['player'] and b['team'] and b['pos'], f"Missing basic info in {b}"
            assert b['stat_focus'] in ('Receiving Yards', 'Receptions', 'Rushing Yards'), f"Invalid stat_focus in {b}"
            assert b.get('scheme_delta'), f"Missing scheme_delta in {b}"
            assert b.get('rationale'), f"Missing rationale in {b}"

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

