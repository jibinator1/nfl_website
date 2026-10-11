import json
import os
import pandas as pd
import numpy as np

# Load parquet data
df = pd.read_parquet('data/weekly_cache.parquet')
recent = df[df['season'] >= 2024].copy()

# Load scheme profiles
profile_path = 'data/scheme_profiles_32.json'
with open(profile_path, 'r', encoding='utf-8') as f:
    profiles = json.load(f)

# Known primary targets mapping for 2026/2025 NFL rosters to guarantee proper starter names
ROSTER_STARTERS = {
    'ARI': {'wr1': 'M.Harrison Jr.', 'wr2': 'M.Wilson', 'te': 'T.McBride', 'rb': 'J.Conner'},
    'ATL': {'wr1': 'D.London', 'wr2': 'D.Mooney', 'te': 'K.Pitts', 'rb': 'B.Robinson'},
    'BAL': {'wr1': 'Z.Flowers', 'wr2': 'R.Bateman', 'te': 'M.Andrews', 'rb': 'D.Henry'},
    'BUF': {'wr1': 'K.Shakir', 'wr2': 'A.Cooper', 'te': 'D.Kincaid', 'rb': 'J.Cook'},
    'CAR': {'wr1': 'A.Thielen', 'wr2': 'X.Legette', 'te': 'J.Sanders', 'rb': 'C.Hubbard'},
    'CHI': {'wr1': 'DJ Moore', 'wr2': 'R.Odunze', 'te': 'C.Kmet', 'rb': 'D.Swift'},
    'CIN': {'wr1': "J.Chase", 'wr2': 'T.Higgins', 'te': 'M.Gesicki', 'rb': 'C.Brown'},
    'CLE': {'wr1': 'J.Jeudy', 'wr2': 'E.Moore', 'te': 'D.Njoku', 'rb': 'N.Chubb'},
    'DAL': {'wr1': 'C.Lamb', 'wr2': 'G.Pickens', 'te': 'J.Ferguson', 'rb': 'R.Dowdle'},
    'DEN': {'wr1': 'C.Sutton', 'wr2': 'M.Mims Jr.', 'te': 'G.Dulcich', 'rb': 'J.Williams'},
    'DET': {'wr1': 'A.St. Brown', 'wr2': 'J.Williams', 'te': 'S.LaPorta', 'rb': 'J.Gibbs'},
    'GB':  {'wr1': 'J.Reed', 'wr2': 'C.Watson', 'te': 'T.Kraft', 'rb': 'J.Jacobs'},
    'HOU': {'wr1': 'N.Collins', 'wr2': 'T.Dell', 'te': 'D.Schultz', 'rb': 'J.Mixon'},
    'IND': {'wr1': 'M.Pittman Jr.', 'wr2': 'J.Downs', 'te': 'K.Granson', 'rb': 'J.Taylor'},
    'JAX': {'wr1': 'B.Thomas Jr.', 'wr2': 'C.Kirk', 'te': 'E.Engram', 'rb': 'T.Etienne Jr.'},
    'KC':  {'wr1': 'D.Hopkins', 'wr2': 'X.Worthy', 'te': 'T.Kelce', 'rb': 'I.Pacheco'},
    'LA':  {'wr1': 'P.Nacua', 'wr2': 'C.Kupp', 'te': 'C.Parkinson', 'rb': 'K.Williams'},
    'LAC': {'wr1': 'L.McConkey', 'wr2': 'Q.Johnston', 'te': 'W.Dissly', 'rb': 'J.Dobbins'},
    'LV':  {'wr1': 'J.Meyers', 'wr2': 'T.Tucker', 'te': 'B.Bowers', 'rb': 'A.Mattison'},
    'MIA': {'wr1': 'T.Hill', 'wr2': 'J.Waddle', 'te': 'J.Smith', 'rb': 'D.Achane'},
    'MIN': {'wr1': 'J.Jefferson', 'wr2': 'J.Addison', 'te': 'T.Hockenson', 'rb': 'A.Jones'},
    'NE':  {'wr1': 'A.Brown', 'wr2': 'D.Douglas', 'te': 'H.Henry', 'rb': 'R.Stevenson'},
    'NO':  {'wr1': 'C.Olave', 'wr2': 'R.Shaheed', 'te': 'J.Johnson', 'rb': 'A.Kamara'},
    'NYG': {'wr1': 'M.Nabers', 'wr2': 'W.Robinson', 'te': 'T.Johnson', 'rb': 'T.Tracy Jr.'},
    'NYJ': {'wr1': 'G.Wilson', 'wr2': 'D.Adams', 'te': 'T.Conklin', 'rb': 'B.Hall'},
    'PHI': {'wr1': 'D.Smith', 'wr2': 'J.Dotson', 'te': 'D.Goedert', 'rb': 'S.Barkley'},
    'PIT': {'wr1': 'C.Austin III', 'wr2': 'M.Williams', 'te': 'P.Freiermuth', 'rb': 'N.Harris'},
    'SEA': {'wr1': 'D.Metcalf', 'wr2': 'J.Smith-Njigba', 'te': 'N.Fant', 'rb': 'K.Walker III'},
    'SF':  {'wr1': 'D.Samuel', 'wr2': 'J.Jennings', 'te': 'G.Kittle', 'rb': 'C.McCaffrey'},
    'TB':  {'wr1': 'M.Evans', 'wr2': 'C.Godwin', 'te': 'C.Otton', 'rb': 'B.Irving'},
    'TEN': {'wr1': 'C.Ridley', 'wr2': 'T.Boyd', 'te': 'C.Okonkwo', 'rb': 'T.Pollard'},
    'WAS': {'wr1': 'T.McLaurin', 'wr2': 'N.Brown', 'te': 'Z.Ertz', 'rb': 'B.Robinson Jr.'}
}

# Empirical metrics by player from parquet
player_stats_cache = {}
for p_name, p_df in recent.groupby('player_name'):
    if len(p_df) >= 2:
        tgt_s = p_df['target_share'].mean()
        if pd.isna(tgt_s) or tgt_s <= 0:
            tgts = p_df['targets'].sum()
            tgt_s = min(0.35, max(0.05, tgts / (len(p_df) * 32.0)))
        air_s = p_df['air_yards_share'].mean()
        yds = p_df['receiving_yards'].sum()
        tgts = max(1, float(p_df['targets'].sum()))
        ypt = yds / tgts
        adot = max(2.0, float(p_df['receiving_air_yards'].sum() / tgts))
        
        player_stats_cache[p_name] = {
            'target_share': float(tgt_s),
            'air_yards_share': float(air_s) if not pd.isna(air_s) else 0.20,
            'ypt': float(ypt),
            'adot': float(adot),
            'games': len(p_df)
        }

print(f"Cached empirical metrics for {len(player_stats_cache)} players.")

# Specific alignments based on real play styles
SLOT_SPECIALISTS = {'K.Shakir', 'W.Robinson', 'J.Downs', 'L.McConkey', 'J.Reed', 'C.Lamb', 'A.St. Brown', 'R.Rice', 'D.Douglas'}

updated_teams = 0
for team_code, prof in profiles.items():
    starters = ROSTER_STARTERS.get(team_code, {})
    if not starters:
        continue
    
    wr1_name = starters['wr1']
    wr2_name = starters['wr2']
    te_name = starters['te']
    rb_name = starters['rb']
    
    def get_p_dat(pname, default_dict):
        aliases = {
            'C.Austin III': 'C.Austin',
            'M.Harrison Jr.': 'M.Harrison',
            'M.Mims Jr.': 'M.Mims',
            'T.Tracy Jr.': 'T.Tracy',
            'B.Robinson Jr.': 'B.Robinson',
            'B.Thomas Jr.': 'B.Thomas',
        }
        if pname in player_stats_cache:
            return player_stats_cache[pname]
        alt = aliases.get(pname)
        if alt and alt in player_stats_cache:
            return player_stats_cache[alt]
        return default_dict

    # 1. WR1 stats
    w1_dat = get_p_dat(wr1_name, {'target_share': 0.28, 'air_yards_share': 0.35, 'ypt': 8.8, 'adot': 11.2})
    w1_base = max(24.0, min(34.0, w1_dat['target_share'] * 100))
    is_w1_slot = wr1_name in SLOT_SPECIALISTS
    w1_slot = 54.0 if is_w1_slot else 26.0
    w1_out = 100.0 - w1_slot
    # WR1 alpha gets target boost vs Man
    w1_man_mult = 1.14 + (w1_dat.get('air_yards_share', 0.3) * 0.08)
    w1_tgt_man = round(w1_base * w1_man_mult, 1)
    w1_tgt_zone = round(w1_base * 0.94, 1)
    w1_yprr_m = round(max(2.40, min(3.20, (w1_dat['ypt'] * 0.28) + 0.35)), 2)
    w1_yprr_z = round(max(2.10, min(2.75, (w1_dat['ypt'] * 0.25) + 0.20)), 2)
    
    # 2. WR2 stats
    w2_dat = get_p_dat(wr2_name, {'target_share': 0.18, 'air_yards_share': 0.24, 'ypt': 8.0, 'adot': 12.5})
    if wr2_name in ('G.Pickens', 'A.Cooper', 'X.Worthy', 'M.Williams') or w2_dat.get('air_yards_share', 0) >= 0.28:
        # Co-alpha boundary deep threat (e.g. G.Pickens on Cowboys, A.Cooper on Bills)
        w2_base = max(21.0, min(26.5, w2_dat['target_share'] * 100))
        w2_slot = 20.0
        w2_out = 80.0
        w2_tgt_man = round(w2_base * 1.16, 1)
        w2_tgt_zone = round(w2_base * 0.94, 1)
        w2_yprr_m = round(max(2.60, min(3.15, (w2_dat['ypt'] * 0.28) + 0.35)), 2)
        w2_yprr_z = round(max(2.15, min(2.70, (w2_dat['ypt'] * 0.25) + 0.20)), 2)
    else:
        w2_base = max(16.0, min(23.5, w2_dat['target_share'] * 100))
        # If WR1 was slot, WR2 is boundary, or vice versa
        is_w2_slot = wr2_name in SLOT_SPECIALISTS
        w2_slot = 68.0 if is_w2_slot else 18.0
        w2_out = 100.0 - w2_slot
        # WR2 sees slight drop vs Man, slight boost vs Zone honey holes
        w2_tgt_man = round(w2_base * 0.96, 1)
        w2_tgt_zone = round(w2_base * 1.03, 1)
        w2_yprr_m = round(max(1.65, min(2.25, (w2_dat['ypt'] * 0.23) + 0.10)), 2)
        w2_yprr_z = round(max(1.60, min(2.15, (w2_dat['ypt'] * 0.22) + 0.05)), 2)
    
    # 3. TE stats
    te_dat = get_p_dat(te_name, {'target_share': 0.18, 'air_yards_share': 0.16, 'ypt': 7.5, 'adot': 7.2})
    te_base = max(14.0, min(24.0, te_dat['target_share'] * 100))
    te_slot = 44.0
    te_inline = 46.0
    te_out = 10.0
    # TEs thrive vs Zone intermediate voids
    te_tgt_zone = round(te_base * 1.16, 1)
    te_tgt_man = round(te_base * 0.85, 1)
    te_yprr_z = round(max(1.85, min(2.45, (te_dat['ypt'] * 0.26) + 0.20)), 2)
    te_yprr_m = round(max(1.45, min(1.95, (te_dat['ypt'] * 0.22) - 0.05)), 2)
    
    # 4. RB stats
    rb_dat = get_p_dat(rb_name, {'target_share': 0.12, 'air_yards_share': 0.02, 'ypt': 5.8, 'adot': 0.5})
    rb_base = max(9.0, min(16.0, rb_dat['target_share'] * 100))
    rb_slot = 10.0
    rb_out = 5.0
    rb_inline = 0.0
    # RBs get checkdowns vs Zone
    rb_tgt_zone = round(rb_base * 1.25, 1)
    rb_tgt_man = round(rb_base * 0.75, 1)
    rb_yprr_z = round(max(1.25, min(1.85, (rb_dat['ypt'] * 0.22) + 0.15)), 2)
    rb_yprr_m = round(max(0.85, min(1.30, (rb_dat['ypt'] * 0.17) - 0.05)), 2)
    
    # Assemble team's 4 key targets in proper hierarchy (WR1, WR2, TE, RB)
    new_targets = [
        {
            'name': wr1_name,
            'pos': 'WR',
            'slot_pct': w1_slot,
            'outside_pct': w1_out,
            'inline_pct': 0.0,
            'tgt_man': w1_tgt_man,
            'tgt_zone': w1_tgt_zone,
            'yprr_man': w1_yprr_m,
            'yprr_zone': w1_yprr_z
        },
        {
            'name': wr2_name,
            'pos': 'WR',
            'slot_pct': w2_slot,
            'outside_pct': w2_out,
            'inline_pct': 0.0,
            'tgt_man': w2_tgt_man,
            'tgt_zone': w2_tgt_zone,
            'yprr_man': w2_yprr_m,
            'yprr_zone': w2_yprr_z
        },
        {
            'name': te_name,
            'pos': 'TE',
            'slot_pct': te_slot,
            'outside_pct': te_out,
            'inline_pct': te_inline,
            'tgt_man': te_tgt_man,
            'tgt_zone': te_tgt_zone,
            'yprr_man': te_yprr_m,
            'yprr_zone': te_yprr_z
        },
        {
            'name': rb_name,
            'pos': 'RB',
            'slot_pct': rb_slot,
            'outside_pct': rb_out,
            'inline_pct': rb_inline,
            'tgt_man': rb_tgt_man,
            'tgt_zone': rb_tgt_zone,
            'yprr_man': rb_yprr_m,
            'yprr_zone': rb_yprr_z
        }
    ]
    
    prof['target_alignment']['key_targets'] = new_targets
    prof['target_alignment']['slot_wr_pct'] = round(float(np.mean([w1_slot, w2_slot]) * 0.65), 1)
    prof['target_alignment']['wide_wr_pct'] = round(float(np.mean([w1_out, w2_out]) * 0.65), 1)
    prof['target_alignment']['inline_te_pct'] = round(float(te_inline * 0.38), 1)
    prof['target_alignment']['backfield_rb_pct'] = round(float(rb_base), 1)
    prof['tempo_situational']['red_zone_touch_leader'] = f"{rb_name} (Lead Goal-Line Share)"
    prof['tempo_situational']['flow_verdict'] = f"Offensive tempo guided by {rb_name} red zone conversion and {prof['tempo_situational'].get('checkdown_pct', 15.0):.1f}% check-down release rate."
    
    updated_teams += 1

print(f"Successfully recalibrated key_targets for {updated_teams} teams!")

# Save to scheme_profiles_32.json
with open(profile_path, 'w', encoding='utf-8') as f:
    json.dump(profiles, f, indent=2)

print("Saved updated profiles to data/scheme_profiles_32.json")
