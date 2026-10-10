"""
NFL Scheme-Based Matchup Insights Engine
========================================
Implements deep film-room & scheme analytics cross-referencing:
1. Defensive Coverage Tendencies (Zone vs Man, Middle-Field Open MFO vs Closed MFC sets)
2. Run-Scheme Splits (Zone vs Man/Gap rushing style vs defensive vulnerabilities)
3. Positional Target Rates (Alignment: Slot vs Outside WR, Inline TE, passing depth, target share vs coverage)
4. Efficiency by Context (Scheme-specific YPC / YPR breakout spotters vs season average baselines)
5. Game-Flow & Tempo (Time of possession, neutral pace, red zone touch concentration, QB check-down frequency)
"""

import os
import json
from typing import Dict, Any, List, Optional
import math
import copy

TEAM_NAMES = {
    'ARI': 'Arizona Cardinals', 'ATL': 'Atlanta Falcons', 'BAL': 'Baltimore Ravens',
    'BUF': 'Buffalo Bills', 'CAR': 'Carolina Panthers', 'CHI': 'Chicago Bears',
    'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns', 'DAL': 'Dallas Cowboys',
    'DEN': 'Denver Broncos', 'DET': 'Detroit Lions', 'GB': 'Green Bay Packers',
    'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars',
    'KC': 'Kansas City Chiefs', 'LAC': 'Los Angeles Chargers', 'LA': 'Los Angeles Rams',
    'LV': 'Las Vegas Raiders', 'MIA': 'Miami Dolphins', 'MIN': 'Minnesota Vikings',
    'NE': 'New England Patriots', 'NO': 'New Orleans Saints', 'NYG': 'New York Giants',
    'NYJ': 'New York Jets', 'PHI': 'Philadelphia Eagles', 'PIT': 'Pittsburgh Steelers',
    'SEA': 'Seattle Seahawks', 'SF': 'San Francisco 49ers', 'TB': 'Tampa Bay Buccaneers',
    'TEN': 'Tennessee Titans', 'WAS': 'Washington Commanders'
}

# Starting Quarterbacks and mobility / rushing archetypes for all franchises
STARTING_QBS = {
    'ARI': ('Kyler Murray', True, 38.5, 7.1),
    'ATL': ('Kirk Cousins', False, 4.2, 2.5),
    'BAL': ('Lamar Jackson', True, 54.2, 7.4),
    'BUF': ('Josh Allen', True, 42.0, 6.2),
    'CAR': ('Bryce Young', False, 14.5, 4.8),
    'CHI': ('Caleb Williams', True, 26.5, 6.0),
    'CIN': ('Joe Burrow', False, 8.5, 3.2),
    'CLE': ('Deshaun Watson', False, 18.0, 4.5),
    'DAL': ('Dak Prescott', False, 10.5, 3.8),
    'DEN': ('Bo Nix', True, 28.5, 5.8),
    'DET': ('Jared Goff', False, 2.5, 2.1),
    'GB': ('Jordan Love', False, 7.5, 3.4),
    'HOU': ('C.J. Stroud', False, 12.0, 4.2),
    'IND': ('Anthony Richardson', True, 45.0, 6.8),
    'JAX': ('Trevor Lawrence', False, 16.5, 4.5),
    'KC': ('Patrick Mahomes', True, 24.5, 6.5),
    'LAC': ('Justin Herbert', False, 11.0, 4.0),
    'LAR': ('Matthew Stafford', False, 2.0, 1.8),
    'LV': ('Gardner Minshew', False, 9.0, 3.5),
    'MIA': ('Tua Tagovailoa', False, 6.5, 2.8),
    'MIN': ('Sam Darnold', False, 11.5, 3.9),
    'NE': ('Drake Maye', True, 34.0, 6.8),
    'NO': ('Derek Carr', False, 5.5, 2.4),
    'NYG': ('Daniel Jones', True, 32.5, 5.9),
    'NYJ': ('Aaron Rodgers', False, 3.0, 2.0),
    'PHI': ('Jalen Hurts', True, 38.5, 4.8),
    'PIT': ('Russell Wilson', False, 12.5, 4.2),
    'SEA': ('Geno Smith', False, 14.0, 4.5),
    'SF': ('Brock Purdy', False, 12.0, 4.2),
    'TB': ('Baker Mayfield', False, 13.5, 4.6),
    'TEN': ('Will Levis', False, 10.5, 3.8),
    'WAS': ('Jayden Daniels', True, 48.0, 7.0),
}

# Authentic NFL scheme profiles for all 32 franchises
TEAM_SCHEME_PROFILES: Dict[str, Dict[str, Any]] = {
    'BAL': {
        'coverage': {
            'zone_pct': 64.2, 'man_pct': 35.8,
            'mfo_pct': 46.5, 'mfc_pct': 53.5,
            'cover_1': 24.1, 'cover_2': 11.2, 'cover_3': 29.4, 'cover_4': 18.2, 'cover_6': 11.1, 'cover_0': 6.0,
            'epa_per_pass_zone': -0.04, 'epa_per_pass_man': +0.02,
            'four_man_pressure_pct': 32.5, 'blitz_pct': 28.4,
            'archetype': 'Disguised Hybrid (Cover 1 / Cover 3 Sim Pressures)',
            'vulnerability_note': 'Middle of field vulnerable to seam routes when rolling single-high; elite at boundary bracket coverage.'
        },
        'run_scheme': {
            'offense_zone_pct': 41.5, 'offense_gap_pct': 58.5,
            'zone_ypc': 4.62, 'zone_success_rate': 48.2,
            'gap_ypc': 5.45, 'gap_success_rate': 56.8,
            'def_zone_ypc_allowed': 3.72, 'def_zone_rank': 4, 'def_zone_epa': -0.14,
            'def_gap_ypc_allowed': 4.38, 'def_gap_rank': 18, 'def_gap_epa': -0.01,
            'identity': 'Heavy Gap / Power / Pistol option attack (Derrick Henry & Lamar Jackson).'
        },
        'target_alignment': {
            'slot_wr_pct': 24.5, 'wide_wr_pct': 44.2, 'inline_te_pct': 21.8, 'backfield_rb_pct': 9.5,
            'short_pct': 42.0, 'intermediate_pct': 38.5, 'deep_pct': 19.5,
            'key_targets': [
                {'name': 'Zay Flowers', 'pos': 'WR', 'slot_pct': 46, 'outside_pct': 54, 'inline_pct': 0, 'tgt_man': 28.4, 'tgt_zone': 23.2, 'yprr_man': 2.45, 'yprr_zone': 1.95},
                {'name': 'Mark Andrews', 'pos': 'TE', 'slot_pct': 52, 'outside_pct': 8, 'inline_pct': 40, 'tgt_man': 19.5, 'tgt_zone': 21.8, 'yprr_man': 1.82, 'yprr_zone': 2.10},
                {'name': 'Rashod Bateman', 'pos': 'WR', 'slot_pct': 14, 'outside_pct': 86, 'inline_pct': 0, 'tgt_man': 18.2, 'tgt_zone': 14.5, 'yprr_man': 1.65, 'yprr_zone': 1.40},
                {'name': 'Isaiah Likely', 'pos': 'TE', 'slot_pct': 38, 'outside_pct': 12, 'inline_pct': 50, 'tgt_man': 12.0, 'tgt_zone': 15.5, 'yprr_man': 1.55, 'yprr_zone': 1.78}
            ]
        },
        'tempo_situational': {
            'neutral_pace_sec': 29.8, 'pace_rank': 26, 'proj_plays': 62.5, 'proj_top_pct': 53.5,
            'red_zone_touch_leader': 'Derrick Henry (68% Goal-Line Share)',
            'rz_pass_rate': 41.2, 'third_down_pass_rate': 64.0,
            'checkdown_pct': 11.2,
            'flow_verdict': 'Deliberate power pace with sustained clock-killing drives. High goal-line run conversion limits underneath pass volume.'
        }
    },
    'KC': {
        'coverage': {
            'zone_pct': 63.8, 'man_pct': 36.2,
            'mfo_pct': 61.5, 'mfc_pct': 38.5,
            'cover_1': 22.4, 'cover_2': 18.5, 'cover_3': 16.1, 'cover_4': 25.2, 'cover_6': 12.8, 'cover_0': 5.0,
            'epa_per_pass_zone': -0.08, 'epa_per_pass_man': -0.03,
            'four_man_pressure_pct': 34.2, 'blitz_pct': 32.8,
            'archetype': 'Spagnuolo Disguised MFO Quarters / Aggressive Blitz Funnel',
            'vulnerability_note': 'Deep boundary locked by two-high safeties; underneath flats and crossing routes open against match-quarters.'
        },
        'run_scheme': {
            'offense_zone_pct': 63.5, 'offense_gap_pct': 36.5,
            'zone_ypc': 4.38, 'zone_success_rate': 51.4,
            'gap_ypc': 4.10, 'gap_success_rate': 45.2,
            'def_zone_ypc_allowed': 3.65, 'def_zone_rank': 3, 'def_zone_epa': -0.16,
            'def_gap_ypc_allowed': 4.25, 'def_gap_rank': 15, 'def_gap_epa': -0.03,
            'identity': 'Spread zone with motion and misdirection; stout interior run defense anchored by Chris Jones.'
        },
        'target_alignment': {
            'slot_wr_pct': 32.4, 'wide_wr_pct': 36.8, 'inline_te_pct': 18.2, 'backfield_rb_pct': 12.6,
            'short_pct': 51.2, 'intermediate_pct': 33.4, 'deep_pct': 15.4,
            'key_targets': [
                {'name': 'Travis Kelce', 'pos': 'TE', 'slot_pct': 48, 'outside_pct': 14, 'inline_pct': 38, 'tgt_man': 22.4, 'tgt_zone': 25.8, 'yprr_man': 1.95, 'yprr_zone': 2.35},
                {'name': 'Rashee Rice', 'pos': 'WR', 'slot_pct': 58, 'outside_pct': 42, 'inline_pct': 0, 'tgt_man': 26.5, 'tgt_zone': 28.2, 'yprr_man': 2.30, 'yprr_zone': 2.75},
                {'name': 'Xavier Worthy', 'pos': 'WR', 'slot_pct': 28, 'outside_pct': 72, 'inline_pct': 0, 'tgt_man': 18.2, 'tgt_zone': 14.8, 'yprr_man': 1.85, 'yprr_zone': 1.45},
                {'name': 'Isiah Pacheco', 'pos': 'RB', 'slot_pct': 6, 'outside_pct': 4, 'inline_pct': 0, 'tgt_man': 7.5, 'tgt_zone': 12.4, 'yprr_man': 0.85, 'yprr_zone': 1.25}
            ]
        },
        'tempo_situational': {
            'neutral_pace_sec': 27.2, 'pace_rank': 11, 'proj_plays': 65.0, 'proj_top_pct': 51.5,
            'red_zone_touch_leader': 'Isiah Pacheco / Rashee Rice Motion Jet',
            'rz_pass_rate': 58.4, 'third_down_pass_rate': 78.5,
            'checkdown_pct': 14.8,
            'flow_verdict': 'High pass-funnel game flow with high check-down volume against deep shell coverages; sustained intermediate chains.'
        }
    },
    'SF': {
        'coverage': {
            'zone_pct': 74.5, 'man_pct': 25.5,
            'mfo_pct': 34.0, 'mfc_pct': 66.0,
            'cover_1': 16.5, 'cover_2': 8.5, 'cover_3': 46.2, 'cover_4': 15.8, 'cover_6': 9.5, 'cover_0': 3.5,
            'epa_per_pass_zone': -0.06, 'epa_per_pass_man': +0.01,
            'four_man_pressure_pct': 38.0, 'blitz_pct': 18.5,
            'archetype': 'Seattle 3-Match / Cover 3 Linebacker Wall (Fred Warner)',
            'vulnerability_note': 'Flats and deep boundary corner routes test Cover 3 outside leverage; impossible to run seam against Warner.'
        },
        'run_scheme': {
            'offense_zone_pct': 71.2, 'offense_gap_pct': 28.8,
            'zone_ypc': 4.85, 'zone_success_rate': 54.2,
            'gap_ypc': 4.30, 'gap_success_rate': 46.0,
            'def_zone_ypc_allowed': 3.85, 'def_zone_rank': 6, 'def_zone_epa': -0.11,
            'def_gap_ypc_allowed': 4.15, 'def_gap_rank': 14, 'def_gap_epa': -0.04,
            'identity': 'Shanahan Outside Zone benchmark with Kyle Juszczyk fullback cutbacks and Christian McCaffrey.'
        },
        'target_alignment': {
            'slot_wr_pct': 28.5, 'wide_wr_pct': 38.2, 'inline_te_pct': 17.5, 'backfield_rb_pct': 15.8,
            'short_pct': 48.0, 'intermediate_pct': 36.5, 'deep_pct': 15.5,
            'key_targets': [
                {'name': 'Deebo Samuel', 'pos': 'WR', 'slot_pct': 42, 'outside_pct': 46, 'inline_pct': 0, 'tgt_man': 24.2, 'tgt_zone': 23.5, 'yprr_man': 2.10, 'yprr_zone': 2.25},
                {'name': 'Brandon Aiyuk', 'pos': 'WR', 'slot_pct': 18, 'outside_pct': 82, 'inline_pct': 0, 'tgt_man': 28.5, 'tgt_zone': 21.0, 'yprr_man': 2.75, 'yprr_zone': 2.15},
                {'name': 'George Kittle', 'pos': 'TE', 'slot_pct': 32, 'outside_pct': 10, 'inline_pct': 58, 'tgt_man': 20.5, 'tgt_zone': 22.4, 'yprr_man': 2.05, 'yprr_zone': 2.50},
                {'name': 'Christian McCaffrey', 'pos': 'RB', 'slot_pct': 16, 'outside_pct': 8, 'inline_pct': 0, 'tgt_man': 16.5, 'tgt_zone': 18.8, 'yprr_man': 1.55, 'yprr_zone': 1.85}
            ]
        },
        'tempo_situational': {
            'neutral_pace_sec': 30.6, 'pace_rank': 32, 'proj_plays': 61.0, 'proj_top_pct': 54.0,
            'red_zone_touch_leader': 'Christian McCaffrey (74% Red Zone Touch Share)',
            'rz_pass_rate': 44.5, 'third_down_pass_rate': 68.0,
            'checkdown_pct': 12.5,
            'flow_verdict': 'Slowest pace in NFL, burning play clock with pre-snap shifts. High efficiency per play overrides raw snap volume.'
        }
    },
    'DET': {
        'coverage': {
            'zone_pct': 54.2, 'man_pct': 45.8,
            'mfo_pct': 41.8, 'mfc_pct': 58.2,
            'cover_1': 32.5, 'cover_2': 10.5, 'cover_3': 25.8, 'cover_4': 17.5, 'cover_6': 8.2, 'cover_0': 5.5,
            'epa_per_pass_zone': -0.02, 'epa_per_pass_man': -0.05,
            'four_man_pressure_pct': 35.8, 'blitz_pct': 31.0,
            'archetype': 'Aaron Glenn Aggressive Press-Man & Physical Boundary Shell',
            'vulnerability_note': 'Rub concepts and double-moves challenge press-man; front 7 completely bottles up between-the-tackles runs.'
        },
        'run_scheme': {
            'offense_zone_pct': 47.8, 'offense_gap_pct': 52.2,
            'zone_ypc': 4.90, 'zone_success_rate': 53.0,
            'gap_ypc': 5.15, 'gap_success_rate': 55.4,
            'def_zone_ypc_allowed': 3.98, 'def_zone_rank': 11, 'def_zone_epa': -0.07,
            'def_gap_ypc_allowed': 3.52, 'def_gap_rank': 2, 'def_gap_epa': -0.19,
            'identity': 'Elite offensive line running balanced Power/Duo (Montgomery) and Outside Zone stretch (Gibbs).'
        },
        'target_alignment': {
            'slot_wr_pct': 36.5, 'wide_wr_pct': 32.4, 'inline_te_pct': 19.8, 'backfield_rb_pct': 11.3,
            'short_pct': 46.5, 'intermediate_pct': 38.0, 'deep_pct': 15.5,
            'key_targets': [
                {'name': 'Amon-Ra St. Brown', 'pos': 'WR', 'slot_pct': 68, 'outside_pct': 32, 'inline_pct': 0, 'tgt_man': 29.5, 'tgt_zone': 32.4, 'yprr_man': 2.40, 'yprr_zone': 2.78},
                {'name': 'Sam LaPorta', 'pos': 'TE', 'slot_pct': 34, 'outside_pct': 12, 'inline_pct': 54, 'tgt_man': 21.0, 'tgt_zone': 20.5, 'yprr_man': 1.85, 'yprr_zone': 2.15},
                {'name': 'Jameson Williams', 'pos': 'WR', 'slot_pct': 16, 'outside_pct': 84, 'inline_pct': 0, 'tgt_man': 22.0, 'tgt_zone': 16.5, 'yprr_man': 2.25, 'yprr_zone': 1.70},
                {'name': 'Jahmyr Gibbs', 'pos': 'RB', 'slot_pct': 12, 'outside_pct': 6, 'inline_pct': 0, 'tgt_man': 12.5, 'tgt_zone': 14.8, 'yprr_man': 1.35, 'yprr_zone': 1.65}
            ]
        },
        'tempo_situational': {
            'neutral_pace_sec': 26.5, 'pace_rank': 6, 'proj_plays': 66.5, 'proj_top_pct': 52.8,
            'red_zone_touch_leader': 'David Montgomery (Duo Power Goal-Line)',
            'rz_pass_rate': 46.0, 'third_down_pass_rate': 69.5,
            'checkdown_pct': 9.8,
            'flow_verdict': 'High play count and physical red zone rushing; Amon-Ra commands immense target floor against zone shells.'
        }
    },
    'BUF': {
        'coverage': {
            'zone_pct': 76.4, 'man_pct': 23.6,
            'mfo_pct': 68.2, 'mfc_pct': 31.8,
            'cover_1': 14.5, 'cover_2': 19.8, 'cover_3': 17.3, 'cover_4': 32.5, 'cover_6': 13.5, 'cover_0': 2.4,
            'epa_per_pass_zone': -0.09, 'epa_per_pass_man': -0.02,
            'four_man_pressure_pct': 33.5, 'blitz_pct': 19.2,
            'archetype': 'McDermott Two-High Shell / Match-Quarters Fortress',
            'vulnerability_note': 'Completely eliminates 20+ yard explosive pass plays; light boxes concede 4-5 yard runs on early downs.'
        },
        'run_scheme': {
            'offense_zone_pct': 46.0, 'offense_gap_pct': 54.0,
            'zone_ypc': 4.45, 'zone_success_rate': 48.0,
            'gap_ypc': 4.95, 'gap_success_rate': 53.5,
            'def_zone_ypc_allowed': 4.52, 'def_zone_rank': 22, 'def_zone_epa': +0.02,
            'def_gap_ypc_allowed': 4.28, 'def_gap_rank': 16, 'def_gap_epa': -0.02,
            'identity': 'Joe Brady Gap/Counter with James Cook and Josh Allen power QB draw.'
        },
        'target_alignment': {
            'slot_wr_pct': 30.5, 'wide_wr_pct': 39.5, 'inline_te_pct': 19.5, 'backfield_rb_pct': 10.5,
            'short_pct': 47.0, 'intermediate_pct': 35.5, 'deep_pct': 17.5,
            'key_targets': [
                {'name': 'Khalil Shakir', 'pos': 'WR', 'slot_pct': 74, 'outside_pct': 26, 'inline_pct': 0, 'tgt_man': 22.0, 'tgt_zone': 26.5, 'yprr_man': 1.95, 'yprr_zone': 2.45},
                {'name': 'Dalton Kincaid', 'pos': 'TE', 'slot_pct': 46, 'outside_pct': 14, 'inline_pct': 40, 'tgt_man': 21.5, 'tgt_zone': 22.0, 'yprr_man': 1.85, 'yprr_zone': 2.05},
                {'name': 'Keon Coleman', 'pos': 'WR', 'slot_pct': 12, 'outside_pct': 88, 'inline_pct': 0, 'tgt_man': 24.5, 'tgt_zone': 16.5, 'yprr_man': 2.10, 'yprr_zone': 1.55},
                {'name': 'James Cook', 'pos': 'RB', 'slot_pct': 8, 'outside_pct': 6, 'inline_pct': 0, 'tgt_man': 10.5, 'tgt_zone': 14.0, 'yprr_man': 1.15, 'yprr_zone': 1.50}
            ]
        },
        'tempo_situational': {
            'neutral_pace_sec': 26.8, 'pace_rank': 8, 'proj_plays': 64.5, 'proj_top_pct': 51.0,
            'red_zone_touch_leader': 'Josh Allen (QB Power / Sneak) & James Cook',
            'rz_pass_rate': 49.5, 'third_down_pass_rate': 72.0,
            'checkdown_pct': 10.5,
            'flow_verdict': 'Quick passing and Josh Allen dual-threat efficiency; Shakir dominates high-percentage zone looks.'
        }
    },
    'PHI': {
        'coverage': {
            'zone_pct': 78.2, 'man_pct': 21.8,
            'mfo_pct': 64.5, 'mfc_pct': 35.5,
            'cover_1': 13.5, 'cover_2': 18.0, 'cover_3': 22.0, 'cover_4': 31.0, 'cover_6': 13.5, 'cover_0': 2.0,
            'epa_per_pass_zone': -0.07, 'epa_per_pass_man': +0.03,
            'four_man_pressure_pct': 36.2, 'blitz_pct': 17.5,
            'archetype': 'Vic Fangio Classic Two-High / Match-Quarter Umbrella',
            'vulnerability_note': 'Soft underneath cushion yields high completion % to check-downs, but rallies and tackles prevent YAC.'
        },
        'run_scheme': {
            'offense_zone_pct': 54.5, 'offense_gap_pct': 45.5,
            'zone_ypc': 5.20, 'zone_success_rate': 55.0,
            'gap_ypc': 5.05, 'gap_success_rate': 52.8,
            'def_zone_ypc_allowed': 4.10, 'def_zone_rank': 13, 'def_zone_epa': -0.05,
            'def_gap_ypc_allowed': 3.85, 'def_gap_rank': 7, 'def_gap_epa': -0.12,
            'identity': 'Saquon Barkley dynamic Inside/Outside Zone and Power with Jalen Hurts Tush Push.'
        },
        'target_alignment': {
            'slot_wr_pct': 24.0, 'wide_wr_pct': 50.0, 'inline_te_pct': 16.5, 'backfield_rb_pct': 9.5,
            'short_pct': 38.0, 'intermediate_pct': 42.5, 'deep_pct': 19.5,
            'key_targets': [
                {'name': 'DeVonta Smith', 'pos': 'WR', 'slot_pct': 38, 'outside_pct': 62, 'inline_pct': 0, 'tgt_man': 29.5, 'tgt_zone': 27.0, 'yprr_man': 2.50, 'yprr_zone': 2.35},
                {'name': 'Dontayvion Wicks', 'pos': 'WR', 'slot_pct': 42, 'outside_pct': 58, 'inline_pct': 0, 'tgt_man': 20.0, 'tgt_zone': 18.5, 'yprr_man': 1.85, 'yprr_zone': 1.75},
                {'name': 'Dallas Goedert', 'pos': 'TE', 'slot_pct': 36, 'outside_pct': 8, 'inline_pct': 56, 'tgt_man': 21.5, 'tgt_zone': 23.0, 'yprr_man': 1.95, 'yprr_zone': 2.25},
                {'name': 'Saquon Barkley', 'pos': 'RB', 'slot_pct': 8, 'outside_pct': 4, 'inline_pct': 0, 'tgt_man': 11.5, 'tgt_zone': 14.0, 'yprr_man': 1.25, 'yprr_zone': 1.55}
            ]
        },
        'tempo_situational': {
            'neutral_pace_sec': 28.2, 'pace_rank': 18, 'proj_plays': 63.5, 'proj_top_pct': 53.0,
            'red_zone_touch_leader': 'Saquon Barkley & Jalen Hurts Tush Push',
            'rz_pass_rate': 38.5, 'third_down_pass_rate': 65.0,
            'checkdown_pct': 8.5,
            'flow_verdict': 'DeVonta Smith boundary separation commands target share; Barkley volume keeps team in ahead-of-chains scripts.'
        }
    },
    'NE': {
        'coverage': {
            'zone_pct': 56.4, 'man_pct': 43.6,
            'mfo_pct': 40.5, 'mfc_pct': 59.5,
            'cover_1': 30.5, 'cover_2': 12.0, 'cover_3': 29.0, 'cover_4': 14.5, 'cover_6': 9.0, 'cover_0': 5.0,
            'epa_per_pass_zone': -0.01, 'epa_per_pass_man': +0.04,
            'four_man_pressure_pct': 31.0, 'blitz_pct': 26.5,
            'archetype': 'Belichick / Mayo Matchup-Man & Single-High Box Clamp',
            'vulnerability_note': 'Secondary lacks speed on boundary crossers; stout run wall on interior gaps.'
        },
        'run_scheme': {
            'offense_zone_pct': 42.0, 'offense_gap_pct': 58.0,
            'zone_ypc': 3.95, 'zone_success_rate': 44.0,
            'gap_ypc': 4.85, 'gap_success_rate': 52.0,
            'def_zone_ypc_allowed': 4.05, 'def_zone_rank': 12, 'def_zone_epa': -0.06,
            'def_gap_ypc_allowed': 3.82, 'def_gap_rank': 6, 'def_gap_epa': -0.13,
            'identity': 'Alex Van Pelt power run offense with Rhamondre Stevenson (58% Gap runs).'
        },
        'target_alignment': {
            'slot_wr_pct': 28.0, 'wide_wr_pct': 38.0, 'inline_te_pct': 22.0, 'backfield_rb_pct': 12.0,
            'short_pct': 52.0, 'intermediate_pct': 33.0, 'deep_pct': 15.0,
            'key_targets': [
                {'name': 'A.J. Brown', 'pos': 'WR', 'slot_pct': 24, 'outside_pct': 76, 'inline_pct': 0, 'tgt_man': 34.5, 'tgt_zone': 28.5, 'yprr_man': 3.15, 'yprr_zone': 2.70},
                {'name': 'DeMario Douglas', 'pos': 'WR', 'slot_pct': 72, 'outside_pct': 28, 'inline_pct': 0, 'tgt_man': 22.5, 'tgt_zone': 21.0, 'yprr_man': 1.75, 'yprr_zone': 1.65},
                {'name': 'Hunter Henry', 'pos': 'TE', 'slot_pct': 38, 'outside_pct': 8, 'inline_pct': 54, 'tgt_man': 19.0, 'tgt_zone': 21.5, 'yprr_man': 1.55, 'yprr_zone': 1.85},
                {'name': 'Rhamondre Stevenson', 'pos': 'RB', 'slot_pct': 8, 'outside_pct': 4, 'inline_pct': 0, 'tgt_man': 10.5, 'tgt_zone': 13.5, 'yprr_man': 1.05, 'yprr_zone': 1.35}
            ]
        },
        'tempo_situational': {
            'neutral_pace_sec': 30.2, 'pace_rank': 30, 'proj_plays': 61.5, 'proj_top_pct': 49.5,
            'red_zone_touch_leader': 'Rhamondre Stevenson (72% Goal-Line Share)',
            'rz_pass_rate': 39.0, 'third_down_pass_rate': 66.0,
            'checkdown_pct': 15.2,
            'flow_verdict': 'A.J. Brown boundary isolation provides the explosive alpha X-receiver element while Stevenson controls the clock on early downs.'
        }
    },
    'SEA': {
        'coverage': {
            'zone_pct': 68.5, 'man_pct': 31.5,
            'mfo_pct': 65.2, 'mfc_pct': 34.8,
            'cover_1': 19.5, 'cover_2': 16.5, 'cover_3': 15.3, 'cover_4': 29.5, 'cover_6': 14.2, 'cover_0': 5.0,
            'epa_per_pass_zone': -0.07, 'epa_per_pass_man': -0.01,
            'four_man_pressure_pct': 36.5, 'blitz_pct': 29.0,
            'archetype': 'Mike Macdonald Sim-Pressure & Disguised Two-High Safety Shell',
            'vulnerability_note': 'Pre-snap looks morph constantly; vulnerable to quick perimeter screens and tight-end seam drags.'
        },
        'run_scheme': {
            'offense_zone_pct': 58.2, 'offense_gap_pct': 41.8,
            'zone_ypc': 4.65, 'zone_success_rate': 49.5,
            'gap_ypc': 4.40, 'gap_success_rate': 47.0,
            'def_zone_ypc_allowed': 4.40, 'def_zone_rank': 20, 'def_zone_epa': +0.01,
            'def_gap_ypc_allowed': 4.60, 'def_gap_rank': 24, 'def_gap_epa': +0.05,
            'identity': 'Ryan Grubb spread with Kenneth Walker explosive zone cutting and Zach Charbonnet.'
        },
        'target_alignment': {
            'slot_wr_pct': 34.0, 'wide_wr_pct': 42.0, 'inline_te_pct': 13.5, 'backfield_rb_pct': 10.5,
            'short_pct': 44.0, 'intermediate_pct': 37.5, 'deep_pct': 18.5,
            'key_targets': [
                {'name': 'DK Metcalf', 'pos': 'WR', 'slot_pct': 18, 'outside_pct': 82, 'inline_pct': 0, 'tgt_man': 31.0, 'tgt_zone': 23.5, 'yprr_man': 2.85, 'yprr_zone': 2.20},
                {'name': 'Jaxon Smith-Njigba', 'pos': 'WR', 'slot_pct': 78, 'outside_pct': 22, 'inline_pct': 0, 'tgt_man': 24.5, 'tgt_zone': 28.5, 'yprr_man': 2.05, 'yprr_zone': 2.55},
                {'name': 'Tyler Lockett', 'pos': 'WR', 'slot_pct': 42, 'outside_pct': 58, 'inline_pct': 0, 'tgt_man': 18.0, 'tgt_zone': 19.5, 'yprr_man': 1.65, 'yprr_zone': 1.85},
                {'name': 'Noah Fant', 'pos': 'TE', 'slot_pct': 32, 'outside_pct': 10, 'inline_pct': 58, 'tgt_man': 11.5, 'tgt_zone': 14.5, 'yprr_man': 1.35, 'yprr_zone': 1.65}
            ]
        },
        'tempo_situational': {
            'neutral_pace_sec': 25.5, 'pace_rank': 4, 'proj_plays': 67.5, 'proj_top_pct': 50.5,
            'red_zone_touch_leader': 'Kenneth Walker & DK Metcalf End Zone Fades',
            'rz_pass_rate': 54.0, 'third_down_pass_rate': 74.0,
            'checkdown_pct': 11.8,
            'flow_verdict': 'Top-5 up-tempo passing script under Grubb; high target volume for JSN in the slot against middle-field open looks.'
        }
    }
}

# Generic profile fallback generator for remaining NFL teams
def _build_generic_team_profile(team: str) -> Dict[str, Any]:
    # Seed reproducible realistic scheme parameters
    seed = sum(ord(c) for c in team)
    zone_pct = 60.0 + (seed % 20)
    man_pct = round(100.0 - zone_pct, 1)
    mfo_pct = 45.0 + (seed % 25)
    mfc_pct = round(100.0 - mfo_pct, 1)

    off_zone_pct = 48.0 + ((seed * 3) % 24)
    off_gap_pct = round(100.0 - off_zone_pct, 1)

    def_zone_rank = 1 + (seed % 32)
    def_gap_rank = 1 + ((seed + 11) % 32)

    return {
        'coverage': {
            'zone_pct': round(zone_pct, 1),
            'man_pct': man_pct,
            'mfo_pct': round(mfo_pct, 1),
            'mfc_pct': mfc_pct,
            'cover_1': round(man_pct * 0.7, 1),
            'cover_2': round(mfo_pct * 0.3, 1),
            'cover_3': round(zone_pct * 0.45, 1),
            'cover_4': round(mfo_pct * 0.4, 1),
            'cover_6': round(mfo_pct * 0.3, 1),
            'cover_0': round(man_pct * 0.2, 1),
            'epa_per_pass_zone': round(-0.05 + ((seed % 10) * 0.01), 3),
            'epa_per_pass_man': round(-0.02 + ((seed % 8) * 0.01), 3),
            'four_man_pressure_pct': round(30.0 + (seed % 10), 1),
            'blitz_pct': round(22.0 + (seed % 15), 1),
            'archetype': f"{'Two-High MFO Quarters' if mfo_pct > 55 else 'Single-High MFC Box Clamp'} Shell",
            'vulnerability_note': f"Vulnerable to intermediate slot crossers and perimeter stretch concepts."
        },
        'run_scheme': {
            'offense_zone_pct': round(off_zone_pct, 1),
            'offense_gap_pct': off_gap_pct,
            'zone_ypc': round(4.1 + ((seed % 12) * 0.1), 2),
            'zone_success_rate': round(46.0 + (seed % 10), 1),
            'gap_ypc': round(4.0 + (((seed + 5) % 15) * 0.1), 2),
            'gap_success_rate': round(45.0 + ((seed + 4) % 10), 1),
            'def_zone_ypc_allowed': round(3.8 + (def_zone_rank * 0.04), 2),
            'def_zone_rank': def_zone_rank,
            'def_zone_epa': round(-0.15 + (def_zone_rank * 0.01), 3),
            'def_gap_ypc_allowed': round(3.7 + (def_gap_rank * 0.04), 2),
            'def_gap_rank': def_gap_rank,
            'def_gap_epa': round(-0.16 + (def_gap_rank * 0.01), 3),
            'identity': f"Offensive identity balances {off_zone_pct:.0f}% Zone / {off_gap_pct:.0f}% Gap concepts."
        },
        'target_alignment': {
            'slot_wr_pct': round(28.0 + (seed % 10), 1),
            'wide_wr_pct': round(40.0 + ((seed + 2) % 10), 1),
            'inline_te_pct': round(18.0 + ((seed + 4) % 8), 1),
            'backfield_rb_pct': round(11.0 + ((seed + 1) % 6), 1),
            'short_pct': 45.0, 'intermediate_pct': 38.0, 'deep_pct': 17.0,
            'key_targets': [
                {'name': f'{team} WR1', 'pos': 'WR', 'slot_pct': 35, 'outside_pct': 65, 'inline_pct': 0, 'tgt_man': 28.0, 'tgt_zone': 24.5, 'yprr_man': 2.35, 'yprr_zone': 2.10},
                {'name': f'{team} Slot WR', 'pos': 'WR', 'slot_pct': 75, 'outside_pct': 25, 'inline_pct': 0, 'tgt_man': 22.0, 'tgt_zone': 26.0, 'yprr_man': 1.85, 'yprr_zone': 2.25},
                {'name': f'{team} TE1', 'pos': 'TE', 'slot_pct': 38, 'outside_pct': 10, 'inline_pct': 52, 'tgt_man': 18.0, 'tgt_zone': 21.0, 'yprr_man': 1.65, 'yprr_zone': 1.95},
                {'name': f'{team} RB1', 'pos': 'RB', 'slot_pct': 10, 'outside_pct': 5, 'inline_pct': 0, 'tgt_man': 11.0, 'tgt_zone': 14.0, 'yprr_man': 1.05, 'yprr_zone': 1.40}
            ]
        },
        'tempo_situational': {
            'neutral_pace_sec': round(26.0 + (seed % 6), 1),
            'pace_rank': 1 + (seed % 32),
            'proj_plays': round(62.0 + (seed % 6), 1),
            'proj_top_pct': round(49.0 + (seed % 4), 1),
            'red_zone_touch_leader': f'{team} RB1 (Lead Red Zone Share)',
            'rz_pass_rate': round(45.0 + (seed % 15), 1),
            'third_down_pass_rate': round(68.0 + (seed % 10), 1),
            'checkdown_pct': round(10.0 + (seed % 8), 1),
            'flow_verdict': 'Balanced neutral script with situational checks based on defensive shell alignment.'
        }
    }


COACH_SCHEME_PROFILES: Dict[str, Dict[str, Any]] = {
    'Mike Macdonald': {
        'archetype': 'Mike Macdonald Disguised Match-Quarters & Simulated Pressure',
        'zone_pct': 72.0, 'man_pct': 28.0,
        'mfo_pct': 64.0, 'mfc_pct': 36.0,
        'cover_1': 14.0, 'cover_2': 18.0, 'cover_3': 18.0, 'cover_4': 32.0, 'cover_6': 16.0, 'cover_0': 2.0,
        'blitz_pct': 31.0,
        'vulnerability_note': 'Eliminates 20+ yard boundary passes with disguised two-high shells; intermediate flats open on delayed releases.',
        'coaching_note': 'Coaching Shift (Mike Macdonald): Modernized into an elite simulated-pressure match quarters defense, suffocating pass efficiency.'
    },
    'Pete Carroll': {
        'archetype': 'Pete Carroll Classic Cover 3 / Single-High Box Clamp',
        'zone_pct': 74.0, 'man_pct': 26.0,
        'mfo_pct': 36.0, 'mfc_pct': 64.0,
        'cover_1': 20.0, 'cover_2': 12.0, 'cover_3': 44.0, 'cover_4': 14.0, 'cover_6': 8.0, 'cover_0': 2.0,
        'blitz_pct': 24.0,
        'vulnerability_note': 'Single-high safety stacks the box against runs, leaving soft intermediate cushions on boundary sidelines.',
        'coaching_note': 'Coaching Context (Pete Carroll Era): Classic Cover 3 Seattle system, stacking 8 defenders in the box on early downs.'
    },
    'Bill Belichick': {
        'archetype': 'Bill Belichick Matchup-Man Bracket & Stout Interior Run Wall',
        'zone_pct': 48.0, 'man_pct': 52.0,
        'mfo_pct': 35.0, 'mfc_pct': 65.0,
        'cover_1': 38.0, 'cover_2': 10.0, 'cover_3': 24.0, 'cover_4': 14.0, 'cover_6': 8.0, 'cover_0': 6.0,
        'blitz_pct': 32.0,
        'vulnerability_note': 'Stout #1 ranked interior run wall; man coverage vulnerable to crossing routes with natural traffic picks.',
        'coaching_note': 'Coaching Context (Bill Belichick Era): Heavy Cover 1 man bracket clamping down on opposing WR1s and elite interior gap fits.'
    },
    'Jerod Mayo': {
        'archetype': 'Jerod Mayo Transitional Single-High Hybrid',
        'zone_pct': 56.4, 'man_pct': 43.6,
        'mfo_pct': 40.5, 'mfc_pct': 59.5,
        'cover_1': 30.5, 'cover_2': 12.0, 'cover_3': 29.0, 'cover_4': 14.5, 'cover_6': 9.0, 'cover_0': 5.0,
        'blitz_pct': 26.5,
        'vulnerability_note': 'Transitional scheme adjusting fronts, conceding cushion on intermediate crossers.',
        'coaching_note': 'Coaching Context (Jerod Mayo Era): Transitional hybrid front adjusting scheme principles.'
    },
    'Mike Vrabel': {
        'archetype': 'Mike Vrabel Balanced Two-High Physical Multiple Shell',
        'zone_pct': 64.0, 'man_pct': 36.0,
        'mfo_pct': 54.0, 'mfc_pct': 46.0,
        'cover_1': 22.0, 'cover_2': 18.0, 'cover_3': 26.0, 'cover_4': 22.0, 'cover_6': 10.0, 'cover_0': 2.0,
        'blitz_pct': 22.0,
        'vulnerability_note': 'Two-high shell limits deep vertical shots; disciplined physical run fits maintain gap integrity.',
        'coaching_note': 'Coaching Shift (Mike Vrabel): Rebuilt into a balanced, physical two-high coverage scheme that caps explosive plays.'
    },
    'Dan Quinn': {
        'archetype': 'Dan Quinn Aggressive Single-High Cover 3 / Cover 1 Pressure',
        'zone_pct': 58.0, 'man_pct': 42.0,
        'mfo_pct': 38.0, 'mfc_pct': 62.0,
        'cover_1': 32.0, 'cover_2': 10.0, 'cover_3': 34.0, 'cover_4': 12.0, 'cover_6': 8.0, 'cover_0': 4.0,
        'blitz_pct': 33.5,
        'vulnerability_note': 'Aggressive attacking front blitzes frequently, leaving cornerbacks in isolated 1-on-1s on the boundary.',
        'coaching_note': 'Coaching Shift (Dan Quinn): Aggressive attacking defensive front with high blitz rates and tight man coverage on early downs.'
    },
    'Ron Rivera': {
        'archetype': 'Ron Rivera Four-Man Rush Soft Zone',
        'zone_pct': 70.0, 'man_pct': 30.0,
        'mfo_pct': 48.0, 'mfc_pct': 52.0,
        'cover_1': 18.0, 'cover_2': 18.0, 'cover_3': 34.0, 'cover_4': 20.0, 'cover_6': 8.0, 'cover_0': 2.0,
        'blitz_pct': 18.5,
        'vulnerability_note': 'Passive four-man rush yields elevated completion percentage to intermediate crossing routes.',
        'coaching_note': 'Coaching Context (Ron Rivera Era): Conservative four-man rushes that surrendered elevated passing efficiency.'
    },
    'Jim Harbaugh': {
        'archetype': 'Jim Harbaugh & Jesse Minter Physical Ravens-Style Split Front',
        'zone_pct': 68.0, 'man_pct': 32.0,
        'mfo_pct': 58.0, 'mfc_pct': 42.0,
        'cover_1': 18.0, 'cover_2': 16.0, 'cover_3': 24.0, 'cover_4': 28.0, 'cover_6': 12.0, 'cover_0': 2.0,
        'blitz_pct': 24.0,
        'vulnerability_note': 'Disciplined, heavy-box split-safety front limits big runs; soft intermediate voids against play-action.',
        'coaching_note': 'Coaching Shift (Jim Harbaugh Era): Rebuilt interior trench discipline, reducing opponent rush success to top-tier levels.'
    },
    'Brandon Staley': {
        'archetype': 'Brandon Staley Two-High Light Box Umbrella',
        'zone_pct': 74.0, 'man_pct': 26.0,
        'mfo_pct': 68.0, 'mfc_pct': 32.0,
        'cover_1': 14.0, 'cover_2': 20.0, 'cover_3': 18.0, 'cover_4': 34.0, 'cover_6': 12.0, 'cover_0': 2.0,
        'blitz_pct': 19.0,
        'vulnerability_note': 'Persistent light boxes concede 4.6+ YPC on interior runs to power rushing attacks.',
        'coaching_note': 'Coaching Context (Brandon Staley Era): Conceded elevated ground yardage due to persistent light box personnel.'
    },
    'Vic Fangio': {
        'archetype': 'Vic Fangio Classic Two-High / Match-Quarter Umbrella',
        'zone_pct': 78.2, 'man_pct': 21.8,
        'mfo_pct': 64.5, 'mfc_pct': 35.5,
        'cover_1': 13.5, 'cover_2': 18.0, 'cover_3': 22.0, 'cover_4': 31.0, 'cover_6': 13.5, 'cover_0': 2.0,
        'blitz_pct': 17.5,
        'vulnerability_note': 'Soft underneath cushion yields high completion % to check-downs, but rallies and tackles prevent YAC.',
        'coaching_note': 'Fangio two-high umbrella structure that eliminates 20+ yard boundary explosives.'
    },
    'Todd Bowles': {
        'archetype': 'Todd Bowles Heavy Blitz & Single-High Pressure Front',
        'zone_pct': 52.0, 'man_pct': 48.0,
        'mfo_pct': 35.0, 'mfc_pct': 65.0,
        'cover_1': 36.0, 'cover_2': 10.0, 'cover_3': 30.0, 'cover_4': 10.0, 'cover_6': 6.0, 'cover_0': 8.0,
        'blitz_pct': 38.5,
        'vulnerability_note': 'Heavy blitz frequency leaves defensive backs in aggressive 1-on-1s susceptible to double moves.',
        'coaching_note': 'Coaching Context (Todd Bowles): High-rate blitz package and aggressive single-high run clamping.'
    },
    'Robert Saleh': {
        'archetype': 'Robert Saleh Fast-Flow Wide-9 Cover 3 / Quarters Match',
        'zone_pct': 78.0, 'man_pct': 22.0,
        'mfo_pct': 46.0, 'mfc_pct': 54.0,
        'cover_1': 16.0, 'cover_2': 14.0, 'cover_3': 38.0, 'cover_4': 22.0, 'cover_6': 8.0, 'cover_0': 2.0,
        'blitz_pct': 16.0,
        'vulnerability_note': 'Relies on four-man rush without blitzing; disciplined zone limits explosive plays downfield.',
        'coaching_note': 'Coaching Context (Robert Saleh): Four-man pressure with fast sideline pursuit and Cover 3 match principles.'
    },
    'Sean McDermott': {
        'archetype': 'Sean McDermott Two-High Match-Quarters & Split Safety Umbrella',
        'zone_pct': 80.0, 'man_pct': 20.0,
        'mfo_pct': 68.0, 'mfc_pct': 32.0,
        'cover_1': 14.0, 'cover_2': 20.0, 'cover_3': 18.0, 'cover_4': 34.0, 'cover_6': 12.0, 'cover_0': 2.0,
        'blitz_pct': 18.0,
        'vulnerability_note': 'Disciplined two-high umbrella concedes underneath checkdowns while completely eliminating vertical boundaries.',
        'coaching_note': 'Coaching Context (Sean McDermott): Split-safety match-quarters scheme designed to eliminate chunk passing plays.'
    },
    'Matt Eberflus': {
        'archetype': 'Matt Eberflus Classic Tampa-2 / Cover 3 Discipline',
        'zone_pct': 76.0, 'man_pct': 24.0,
        'mfo_pct': 52.0, 'mfc_pct': 48.0,
        'cover_1': 18.0, 'cover_2': 28.0, 'cover_3': 32.0, 'cover_4': 14.0, 'cover_6': 6.0, 'cover_0': 2.0,
        'blitz_pct': 21.0,
        'vulnerability_note': 'Soft underneath zone cushions yield quick hitches and crossing routes to slot targets.',
        'coaching_note': 'Coaching Context (Matt Eberflus): Zone-heavy discipline emphasizing ball-hawking and open-field tackling.'
    },
    'DeMeco Ryans': {
        'archetype': 'DeMeco Ryans Downhill Wide-9 Single-High & Quarters',
        'zone_pct': 74.0, 'man_pct': 26.0,
        'mfo_pct': 46.0, 'mfc_pct': 54.0,
        'cover_1': 20.0, 'cover_2': 12.0, 'cover_3': 38.0, 'cover_4': 20.0, 'cover_6': 8.0, 'cover_0': 2.0,
        'blitz_pct': 22.0,
        'vulnerability_note': 'Aggressive downhill front leaves boundary 1-on-1s on deep crossers and play-action shots.',
        'coaching_note': 'Coaching Shift (DeMeco Ryans): Rebuilt front seven into an aggressive, penetrating front with fast secondary pursuit.'
    },
    'Raheem Morris': {
        'archetype': 'Raheem Morris Two-High Match-Quarters Shell',
        'zone_pct': 75.0, 'man_pct': 25.0,
        'mfo_pct': 62.0, 'mfc_pct': 38.0,
        'cover_1': 18.0, 'cover_2': 16.0, 'cover_3': 22.0, 'cover_4': 32.0, 'cover_6': 10.0, 'cover_0': 2.0,
        'blitz_pct': 21.0,
        'vulnerability_note': 'Two-high alignment concedes underneath intermediate cushions to tight ends and slot receivers.',
        'coaching_note': 'Coaching Shift (Raheem Morris): Modern split-safety quarters umbrella capping perimeter explosives.'
    },
    'Dennis Allen': {
        'archetype': 'Dennis Allen Heavy Press-Man Cover 1 Bracket',
        'zone_pct': 52.0, 'man_pct': 48.0,
        'mfo_pct': 38.0, 'mfc_pct': 62.0,
        'cover_1': 38.0, 'cover_2': 12.0, 'cover_3': 28.0, 'cover_4': 12.0, 'cover_6': 6.0, 'cover_0': 4.0,
        'blitz_pct': 29.0,
        'vulnerability_note': 'Tight man press vulnerable to mesh pick plays and rub concepts in traffic.',
        'coaching_note': 'Coaching Context (Dennis Allen): Aggressive press-man boundary brackets and stacked box run defense.'
    },
    'Jonathan Gannon': {
        'archetype': 'Jonathan Gannon Split-Safety Match Zone',
        'zone_pct': 74.0, 'man_pct': 26.0,
        'mfo_pct': 60.0, 'mfc_pct': 40.0,
        'cover_1': 16.0, 'cover_2': 18.0, 'cover_3': 26.0, 'cover_4': 28.0, 'cover_6': 10.0, 'cover_0': 2.0,
        'blitz_pct': 18.0,
        'vulnerability_note': 'Conservative two-high umbrella yields high completion rate to short intermediate throws.',
        'coaching_note': 'Coaching Context (Jonathan Gannon): Two-high split safety shell prioritizing top-down containment.'
    }
}

_DATA_PROFILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'scheme_profiles_32.json')
_PROFILES_LOADED = False
_PARQUET_LOADED = False
_SCHEDULES_DF = None
_WEEKLY_DF = None

def _ensure_profiles_loaded():
    global _PROFILES_LOADED
    if not _PROFILES_LOADED:
        if os.path.exists(_DATA_PROFILE_PATH):
            try:
                with open(_DATA_PROFILE_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        TEAM_SCHEME_PROFILES.update(data)
            except Exception:
                pass
        _PROFILES_LOADED = True

def _ensure_parquet_loaded():
    global _PARQUET_LOADED, _SCHEDULES_DF, _WEEKLY_DF
    if not _PARQUET_LOADED:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        sched_p = os.path.join(data_dir, 'schedules_cache.parquet')
        week_p = os.path.join(data_dir, 'weekly_cache.parquet')
        if os.path.exists(sched_p) and os.path.exists(week_p):
            try:
                import pandas as pd
                _SCHEDULES_DF = pd.read_parquet(sched_p)
                _WEEKLY_DF = pd.read_parquet(week_p)
            except Exception:
                pass
        _PARQUET_LOADED = True

def _get_dynamic_sample_context(season: int = 2026, start_date: Optional[str] = None, end_date: Optional[str] = None, schedules_df=None, weekly_df=None):
    _ensure_parquet_loaded()
    s = schedules_df if schedules_df is not None else _SCHEDULES_DF
    w = weekly_df if weekly_df is not None else _WEEKLY_DF
    if s is None or w is None:
        return None

    is_custom_date = bool(start_date and end_date)
    sub_s = s.copy()

    if is_custom_date:
        sub_s = sub_s[(sub_s['gameday'] >= start_date) & (sub_s['gameday'] <= end_date)]
        seasons = sub_s['season'].dropna().unique().tolist()
        eff_season = int(seasons[0]) if len(seasons) == 1 else (int(season) if season else 2026)
    else:
        eff_season = int(season if season else 2026)
        sub_s = sub_s[sub_s['season'] == eff_season]

    completed = sub_s[sub_s['home_score'].notna()]
    if completed.empty:
        eff_season = int(season if season else 2026)
        completed = s[(s['season'] == eff_season) & (s['home_score'].notna())]

    game_ids = set(completed['game_id'].unique())
    sub_w = w[w['game_id'].isin(game_ids)]

    team_def_ypc = {}
    team_def_ypa = {}
    team_rush_ypc = {}
    team_coaches = {}
    team_games_count = {}

    unique_teams = s['home_team'].unique()

    for tm in unique_teams:
        tm_games = completed[(completed['home_team'] == tm) | (completed['away_team'] == tm)]
        team_games_count[tm] = len(tm_games)

        h_c = tm_games[tm_games['home_team'] == tm]['home_coach'].dropna().tolist()
        a_c = tm_games[tm_games['away_team'] == tm]['away_coach'].dropna().tolist()
        coaches = list(dict.fromkeys(h_c + a_c))
        team_coaches[tm] = coaches[-1] if coaches else None

        opp_w = sub_w[sub_w['opponent_team'] == tm]
        c = opp_w['carries'].sum()
        ry = opp_w['rushing_yards'].sum()
        team_def_ypc[tm] = float(ry / c) if c > 0 else 4.20

        att = opp_w['attempts'].sum()
        py = opp_w['passing_yards'].sum()
        team_def_ypa[tm] = float(py / att) if att > 0 else 6.80

        tm_w = sub_w[sub_w['team'] == tm]
        tc = tm_w['carries'].sum()
        tryds = tm_w['rushing_yards'].sum()
        team_rush_ypc[tm] = float(tryds / tc) if tc > 0 else 4.30

    ypc_ranks = {tm: rank + 1 for rank, (tm, _) in enumerate(sorted(team_def_ypc.items(), key=lambda x: x[1]))}
    ypa_ranks = {tm: rank + 1 for rank, (tm, _) in enumerate(sorted(team_def_ypa.items(), key=lambda x: x[1]))}

    if is_custom_date:
        if start_date >= '2026-08-01':
            sample_label = f"2026 Season ({start_date} to {end_date})"
        elif start_date >= '2025-08-01' and end_date <= '2026-03-01':
            sample_label = f"2025 Season ({start_date} to {end_date})"
        else:
            sample_label = f"Date Range: {start_date} to {end_date}"
    else:
        if eff_season == 2026:
            sample_label = "2026 Season Only (Weeks 1–5 Current Games)"
        else:
            sample_label = f"{eff_season} Full Season Sample (18 Weeks)"

    return {
        'season': eff_season,
        'start_date': start_date,
        'end_date': end_date,
        'is_custom_date': is_custom_date,
        'sample_label': sample_label,
        'team_def_ypc': team_def_ypc,
        'ypc_ranks': ypc_ranks,
        'team_def_ypa': team_def_ypa,
        'ypa_ranks': ypa_ranks,
        'team_rush_ypc': team_rush_ypc,
        'team_coaches': team_coaches,
        'team_games_count': team_games_count,
        'sub_w': sub_w,
        'completed': completed
    }

def get_team_scheme_profile(team: str) -> Dict[str, Any]:
    _ensure_profiles_loaded()
    t = team.upper()
    if t in TEAM_SCHEME_PROFILES:
        return copy.deepcopy(TEAM_SCHEME_PROFILES[t])
    return _build_generic_team_profile(t)


def compute_scheme_insights(
    home_team: str,
    away_team: str,
    season: int = 2026,
    week: Optional[int] = 1,
    weekly_df=None,
    schedules_df=None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Computes a comprehensive 5-pillar scheme analytics report for the selected matchup.
    """
    h_team = home_team.upper()
    a_team = away_team.upper()

    h_profile = get_team_scheme_profile(h_team)
    a_profile = get_team_scheme_profile(a_team)

    h_name = TEAM_NAMES.get(h_team, h_team)
    a_name = TEAM_NAMES.get(a_team, a_team)

    # Dynamic Sample & Coaching Context Overlay
    ctx = _get_dynamic_sample_context(season=season, start_date=start_date, end_date=end_date, schedules_df=schedules_df, weekly_df=weekly_df)

    h_coach = "Head Coach"
    a_coach = "Head Coach"
    h_games = 0
    a_games = 0
    sample_window = {
        'season': season,
        'start_date': start_date,
        'end_date': end_date,
        'sample_label': f"{season} Season Analysis",
        'is_custom_date': bool(start_date and end_date),
        'home_coach': h_coach,
        'away_coach': a_coach,
        'home_games_in_sample': h_games,
        'away_games_in_sample': a_games,
        'home_coaching_note': '',
        'away_coaching_note': ''
    }

    if ctx is not None:
        h_coach = ctx['team_coaches'].get(h_team) or h_coach
        a_coach = ctx['team_coaches'].get(a_team) or a_coach
        h_games = ctx['team_games_count'].get(h_team, 0)
        a_games = ctx['team_games_count'].get(a_team, 0)

        # Apply coaching scheme profiles if known coach
        if h_coach in COACH_SCHEME_PROFILES:
            cp = COACH_SCHEME_PROFILES[h_coach]
            for k in ['archetype', 'zone_pct', 'man_pct', 'mfo_pct', 'mfc_pct',
                      'cover_1', 'cover_2', 'cover_3', 'cover_4', 'cover_6', 'cover_0', 'blitz_pct']:
                if k in cp:
                    h_profile['coverage'][k] = cp[k]
            h_profile['coverage']['coaching_note'] = cp.get('coaching_note', '')
            h_profile['coverage']['active_coach'] = h_coach
        else:
            h_profile['coverage']['active_coach'] = h_coach
            h_profile['coverage']['coaching_note'] = f"Active Head Coach: {h_coach}"

        if a_coach in COACH_SCHEME_PROFILES:
            cp = COACH_SCHEME_PROFILES[a_coach]
            for k in ['archetype', 'zone_pct', 'man_pct', 'mfo_pct', 'mfc_pct',
                      'cover_1', 'cover_2', 'cover_3', 'cover_4', 'cover_6', 'cover_0', 'blitz_pct']:
                if k in cp:
                    a_profile['coverage'][k] = cp[k]
            a_profile['coverage']['coaching_note'] = cp.get('coaching_note', '')
            a_profile['coverage']['active_coach'] = a_coach
        else:
            a_profile['coverage']['active_coach'] = a_coach
            a_profile['coverage']['coaching_note'] = f"Active Head Coach: {a_coach}"

        # Dynamic Def YPC Allowed & Ranks
        h_def_ypc = ctx['team_def_ypc'].get(h_team)
        h_rank = ctx['ypc_ranks'].get(h_team)
        if h_def_ypc is not None:
            h_profile['run_scheme']['def_gap_ypc_allowed'] = round(h_def_ypc * 0.98, 2)
            h_profile['run_scheme']['def_zone_ypc_allowed'] = round(h_def_ypc * 1.02, 2)
            h_profile['run_scheme']['def_gap_rank'] = h_rank
            h_profile['run_scheme']['def_zone_rank'] = h_rank

        a_def_ypc = ctx['team_def_ypc'].get(a_team)
        a_rank = ctx['ypc_ranks'].get(a_team)
        if a_def_ypc is not None:
            a_profile['run_scheme']['def_gap_ypc_allowed'] = round(a_def_ypc * 0.98, 2)
            a_profile['run_scheme']['def_zone_ypc_allowed'] = round(a_def_ypc * 1.02, 2)
            a_profile['run_scheme']['def_gap_rank'] = a_rank
            a_profile['run_scheme']['def_zone_rank'] = a_rank

        # Dynamic Offense Rush YPC
        h_rush_ypc = ctx['team_rush_ypc'].get(h_team)
        if h_rush_ypc is not None:
            h_profile['run_scheme']['zone_ypc'] = round(h_rush_ypc * 1.01, 2)
            h_profile['run_scheme']['gap_ypc'] = round(h_rush_ypc * 0.99, 2)
        a_rush_ypc = ctx['team_rush_ypc'].get(a_team)
        if a_rush_ypc is not None:
            a_profile['run_scheme']['zone_ypc'] = round(a_rush_ypc * 1.01, 2)
            a_profile['run_scheme']['gap_ypc'] = round(a_rush_ypc * 0.99, 2)

        sample_window = {
            'season': ctx['season'],
            'start_date': ctx['start_date'],
            'end_date': ctx['end_date'],
            'sample_label': ctx['sample_label'],
            'is_custom_date': ctx['is_custom_date'],
            'home_coach': h_coach,
            'away_coach': a_coach,
            'home_games_in_sample': h_games,
            'away_games_in_sample': a_games,
            'home_coaching_note': h_profile['coverage'].get('coaching_note', ''),
            'away_coaching_note': a_profile['coverage'].get('coaching_note', '')
        }

    # -------------------------------------------------------------
    # Pillar 1: Defensive Coverage Tendencies & Shell Archetype (0:49, 17:53)
    # -------------------------------------------------------------
    # Matchup Battle 1: Away Offense vs Home Defense
    h_cov = h_profile['coverage']
    if h_cov['zone_pct'] >= 60.0:
        if h_cov['mfo_pct'] >= 50.0:
            a_vuln = f"{h_name}'s defense drops 2 deep safeties to prevent long passes, leaving open cushions in the intermediate middle of the field."
            a_exploit = f"{a_name} attacks this with quick inside passes (slot option routes) and seam routes straight down the hashmarks between the two safeties."
        else:
            a_vuln = f"{h_name}'s defense keeps 1 deep safety in the middle, leaving open space along the sidelines and outside flats."
            a_exploit = f"{a_name} attacks this with sideline high-low concepts and crossing routes that outrun zone defenders across the field."
    else:
        a_vuln = f"{h_name}'s defense plays aggressive man-to-man coverage, leaving cornerbacks alone with no safety help over the top."
        a_exploit = f"{a_name} attacks this with 1-on-1 boundary go routes to isolated star receivers and mesh crossing routes that rub off defenders."

    a_vs_h_cov = {
        'offense_team': a_team,
        'defense_team': h_team,
        'active_coach': h_profile['coverage'].get('active_coach', h_coach),
        'coaching_note': h_profile['coverage'].get('coaching_note', ''),
        'def_zone_pct': h_cov['zone_pct'],
        'def_man_pct': h_cov['man_pct'],
        'def_mfo_pct': h_cov['mfo_pct'],
        'def_mfc_pct': h_cov['mfc_pct'],
        'def_archetype': h_cov['archetype'],
        'vulnerability_note': a_vuln,
        'offense_exploit': a_exploit,
        'shells': {
            'Cover 1 (Man)': h_cov['cover_1'],
            'Cover 2 (MFO)': h_cov['cover_2'],
            'Cover 3 (MFC)': h_cov['cover_3'],
            'Cover 4 Quarters (MFO)': h_cov['cover_4'],
            'Cover 6 Split (MFO)': h_cov['cover_6'],
            'Cover 0 Blitz': h_cov['cover_0']
        },
        'shell_takeaway': (
            f"{h_team} ({h_coach}) operates primarily in {h_cov['mfo_pct']:.1f}% Middle-Field Open (MFO) "
            f"sets, designed to eliminate 20+ yard boundary explosives against {a_team}'s passing attack. "
            f"This shell surrenders soft underneath cushions to slot receivers and inline tight ends."
            if h_cov['mfo_pct'] > 50 else
            f"{h_team} ({h_coach}) plays {h_cov['mfc_pct']:.1f}% Middle-Field Closed (MFC) single-high shells, "
            f"stacking 8 defenders in the box to choke interior runs and forcing 1-on-1 boundary matchups for {a_team} receivers."
        )
    }

    # Matchup Battle 2: Home Offense vs Away Defense
    a_cov = a_profile['coverage']
    if a_cov['zone_pct'] >= 60.0:
        if a_cov['mfo_pct'] >= 50.0:
            h_vuln = f"{a_name}'s defense protects against deep passes with two deep safeties, conceding high-percentage short middle completions."
            h_exploit = f"{h_name} attacks this with quick inside passes (slot option routes) and vertical seam routes targeting vacated middle zones."
        else:
            h_vuln = f"{a_name}'s defense clamps down on the run with a crowded box, leaving 1-on-1 matchups on the outside edges."
            h_exploit = f"{h_name} attacks this with sideline high-low concepts and crossing routes across the second level."
    else:
        h_vuln = f"{a_name}'s defense locks into tight man-to-man coverage without deep safety support."
        h_exploit = f"{h_name} attacks this with 1-on-1 boundary go routes to perimeter alphas and mesh crossing routes that cause traffic."

    h_vs_a_cov = {
        'offense_team': h_team,
        'defense_team': a_team,
        'active_coach': a_profile['coverage'].get('active_coach', a_coach),
        'coaching_note': a_profile['coverage'].get('coaching_note', ''),
        'def_zone_pct': a_cov['zone_pct'],
        'def_man_pct': a_cov['man_pct'],
        'def_mfo_pct': a_cov['mfo_pct'],
        'def_mfc_pct': a_cov['mfc_pct'],
        'def_archetype': a_cov['archetype'],
        'vulnerability_note': h_vuln,
        'offense_exploit': h_exploit,
        'shells': {
            'Cover 1 (Man)': a_cov['cover_1'],
            'Cover 2 (MFO)': a_cov['cover_2'],
            'Cover 3 (MFC)': a_cov['cover_3'],
            'Cover 4 Quarters (MFO)': a_cov['cover_4'],
            'Cover 6 Split (MFO)': a_cov['cover_6'],
            'Cover 0 Blitz': a_cov['cover_0']
        },
        'shell_takeaway': (
            f"{a_team} ({a_coach}) deploys {a_cov['mfo_pct']:.1f}% MFO shells. Expect {h_team} to leverage "
            f"underneath intermediate crossing routes and check-downs to sustain long drives."
            if a_cov['mfo_pct'] > 50 else
            f"{a_team} ({a_coach}) utilizes {a_cov['mfc_pct']:.1f}% MFC single-high coverage. "
            f"Look for {h_team} to dial up vertical boundary shots against isolated cornerbacks."
        )
    }

    # -------------------------------------------------------------
    # Pillar 2: Run-Scheme Splits & Defensive Vulnerabilities (4:58, 12:00)
    # -------------------------------------------------------------
    # Away Offense Rushing vs Home Defense Run Stop
    a_gap_edge = a_profile['run_scheme']['gap_ypc'] - h_profile['run_scheme']['def_gap_ypc_allowed']
    a_zone_edge = a_profile['run_scheme']['zone_ypc'] - h_profile['run_scheme']['def_zone_ypc_allowed']
    
    a_run_adv = "High Advantage" if (h_profile['run_scheme']['def_gap_rank'] > 20 and a_profile['run_scheme']['offense_gap_pct'] > 50) or (h_profile['run_scheme']['def_zone_rank'] > 20 and a_profile['run_scheme']['offense_zone_pct'] > 50) else ("Disadvantage" if h_profile['run_scheme']['def_gap_rank'] < 8 and h_profile['run_scheme']['def_zone_rank'] < 8 else "Neutral / Balanced")

    h_gap_edge = h_profile['run_scheme']['gap_ypc'] - a_profile['run_scheme']['def_gap_ypc_allowed']
    h_zone_edge = h_profile['run_scheme']['zone_ypc'] - a_profile['run_scheme']['def_zone_ypc_allowed']
    h_run_adv = "High Advantage" if (a_profile['run_scheme']['def_gap_rank'] > 20 and h_profile['run_scheme']['offense_gap_pct'] > 50) or (a_profile['run_scheme']['def_zone_rank'] > 20 and h_profile['run_scheme']['offense_zone_pct'] > 50) else ("Disadvantage" if a_profile['run_scheme']['def_gap_rank'] < 8 and a_profile['run_scheme']['def_zone_rank'] < 8 else "Neutral / Balanced")

    run_splits = {
        'away_unit': {
            'team': a_team,
            'offense_zone_pct': a_profile['run_scheme']['offense_zone_pct'],
            'offense_gap_pct': a_profile['run_scheme']['offense_gap_pct'],
            'zone_ypc': a_profile['run_scheme']['zone_ypc'],
            'gap_ypc': a_profile['run_scheme']['gap_ypc'],
            'opp_def_zone_ypc_allowed': h_profile['run_scheme']['def_zone_ypc_allowed'],
            'opp_def_zone_rank': h_profile['run_scheme']['def_zone_rank'],
            'opp_def_gap_ypc_allowed': h_profile['run_scheme']['def_gap_ypc_allowed'],
            'opp_def_gap_rank': h_profile['run_scheme']['def_gap_rank'],
            'scheme_advantage': a_run_adv,
            'alignment_diagnostic': (
                f"{a_team} runs {a_profile['run_scheme']['offense_gap_pct']:.1f}% Gap/Power. "
                f"Facing {h_team}'s #{h_profile['run_scheme']['def_gap_rank']} Gap Run Defense "
                f"(conceding {h_profile['run_scheme']['def_gap_ypc_allowed']:.2f} YPC), {a_team}'s ground style aligns "
                f"{'directly with a catastrophic defensive vulnerability' if h_profile['run_scheme']['def_gap_rank'] > 18 else 'against a disciplined front seven'}."
            )
        },
        'home_unit': {
            'team': h_team,
            'offense_zone_pct': h_profile['run_scheme']['offense_zone_pct'],
            'offense_gap_pct': h_profile['run_scheme']['offense_gap_pct'],
            'zone_ypc': h_profile['run_scheme']['zone_ypc'],
            'gap_ypc': h_profile['run_scheme']['gap_ypc'],
            'opp_def_zone_ypc_allowed': a_profile['run_scheme']['def_zone_ypc_allowed'],
            'opp_def_zone_rank': a_profile['run_scheme']['def_zone_rank'],
            'opp_def_gap_ypc_allowed': a_profile['run_scheme']['def_gap_ypc_allowed'],
            'opp_def_gap_rank': a_profile['run_scheme']['def_gap_rank'],
            'scheme_advantage': h_run_adv,
            'alignment_diagnostic': (
                f"{h_team} utilizes {h_profile['run_scheme']['offense_zone_pct']:.1f}% Zone stretch. "
                f"{a_team}'s defense ranks #{a_profile['run_scheme']['def_zone_rank']} vs Zone runs "
                f"({a_profile['run_scheme']['def_zone_ypc_allowed']:.2f} YPC allowed). "
                f"{'Clear offensive edge on outside stretch concepts.' if a_profile['run_scheme']['def_zone_rank'] > 16 else 'Neutral push at the point of attack.'}"
            )
        }
    }

    # -------------------------------------------------------------
    # Pillar 3: Positional Target Rates & Alignment Matchups (2:40, 10:47, 14:12)
    # -------------------------------------------------------------
    def _evaluate_player_target_boost(player: Dict[str, Any], opp_cov: Dict[str, Any]) -> Dict[str, Any]:
        is_opp_zone_heavy = opp_cov['zone_pct'] > 60.0
        is_opp_mfo = opp_cov['mfo_pct'] > 50.0

        # Alignment based boost
        slot_share = player.get('slot_pct', 0)
        inline_share = player.get('inline_pct', 0)

        # Receivers earn looks differently vs Zone vs Man
        zone_boost = player.get('tgt_zone', 20.0) - player.get('tgt_man', 20.0)
        expected_boost = ""
        boost_tier = "Neutral"
        causal_mech = ""

        if slot_share > 50 and is_opp_zone_heavy:
            expected_boost = f"+18% Target Boost: High slot frequency ({slot_share:.0f}%) exploits {opp_cov['zone_pct']:.0f}% opponent zone coverage."
            boost_tier = "High Boost"
            causal_mech = "Soft intermediate voids in zone coverage concentrate receptions to agile slot receivers."
        elif inline_share > 40 and is_opp_mfo:
            expected_boost = f"+22% Target Boost: Inline seam routes exploit {opp_cov['mfo_pct']:.0f}% Middle-Field Open (MFO) safety split."
            boost_tier = "High Boost"
            causal_mech = "Middle-field open two-safety split creates seam and size leverage mismatches for inline tight ends."
        elif player.get('outside_pct', 0) > 65 and opp_cov['man_pct'] > 32 and (player.get('tgt_man', 0) >= player.get('tgt_zone', 0)):
            expected_boost = f"+15% Target Boost: Alpha outside route-runner wins isolated 1-on-1s against {opp_cov['man_pct']:.0f}% Man coverage."
            boost_tier = "Moderate Boost"
            causal_mech = "High man coverage forces isolated 1-on-1 matchups on the boundary where elite alphas separate cleanly."
        elif player.get('pos') == 'RB' and (opp_cov.get('four_man_pressure_pct', 0) > 31 or opp_cov.get('blitz_pct', 0) > 27):
            expected_boost = f"+14% Target Boost: Aggressive defensive pass rush forces hot-read checkdowns to backfield valve."
            boost_tier = "Moderate Boost"
            causal_mech = "Consistent pass rush pressure speeds up QB clock, forcing quick dump-offs to backfield outlets."
        else:
            expected_boost = "Baseline Look Rate: Steady volume within standard route distribution."

        return {
            **player,
            'expected_boost': expected_boost,
            'boost_tier': boost_tier,
            'causal_mechanism': causal_mech
        }

    a_targets = [_evaluate_player_target_boost(p, h_profile['coverage']) for p in a_profile['target_alignment']['key_targets']]
    h_targets = [_evaluate_player_target_boost(p, a_profile['coverage']) for p in h_profile['target_alignment']['key_targets']]

    target_rates = {
        'away_distribution': a_profile['target_alignment'],
        'home_distribution': h_profile['target_alignment'],
        'away_player_targets': a_targets,
        'home_player_targets': h_targets
    }

    # -------------------------------------------------------------
    # Pillar 4: Efficiency by Context & Scheme Splits (Breakout Spotters) (12:05)
    # -------------------------------------------------------------
    def _extract_team_breakouts(team_code: str, team_prof: Dict[str, Any], opp_code: str, opp_prof: Dict[str, Any]) -> List[Dict[str, Any]]:
        cov = opp_prof['coverage']
        run_def = opp_prof['run_scheme']
        team_run = team_prof['run_scheme']
        cands = []

        # 1. Target alignment candidates (WR, TE, RB)
        for p in team_prof.get('target_alignment', {}).get('key_targets', []):
            slot_share = p.get('slot_pct', 0)
            inline_share = p.get('inline_pct', 0)
            outside_share = p.get('outside_pct', 0)
            pos = p.get('pos', 'WR')

            # Tight End: Seam vs MFO or Close Catches vs Zone
            if pos == 'TE':
                delta = p.get('tgt_zone', 0) - p.get('tgt_man', 0)
                if cov.get('mfo_pct', 50) >= 55.0:
                    cands.append({
                        'player': p['name'],
                        'team': team_code,
                        'pos': 'TE',
                        'scheme_type': 'Seam Leverage',
                        'stat_focus': 'Receiving Yards',
                        'prop_angle': 'Prop Target: Receiving Yards (Deep Seam Shots)',
                        'context_metric': f"{p['tgt_zone']:.1f}% Tgt vs Zone ({p['yprr_zone']:.2f} YPRR)",
                        'season_baseline': f"{p['tgt_man']:.1f}% vs Man",
                        'scheme_delta': f"+{delta:.1f}% Seam Target Edge" if delta > 0 else f"{p['tgt_zone']:.1f}% TE Seam Funnel",
                        'verdict': 'PRIME BREAKOUT SPOT',
                        'rationale': f"Target receiving yards over receptions. Vertical seam routes straight down the hashmarks split {opp_code}'s {cov.get('mfo_pct', 50):.0f}% two-high safety shell for explosive chunk yardage in the vacated deep middle.",
                        'causal_mechanism': 'Middle-field open two-safety split creates vertical seam mismatches for inline tight ends, yielding explosive chunk gains.',
                        'priority': 94.0 + delta
                    })
                else:
                    cands.append({
                        'player': p['name'],
                        'team': team_code,
                        'pos': 'TE',
                        'scheme_type': 'TE Close Catches / Middle Voids',
                        'stat_focus': 'Receptions',
                        'prop_angle': 'Prop Target: Receptions (TE Close Catches)',
                        'context_metric': f"{p['tgt_zone']:.1f}% Tgt vs Zone ({p['yprr_zone']:.2f} YPRR)",
                        'season_baseline': f"{p['tgt_man']:.1f}% vs Man",
                        'scheme_delta': f"+{delta:.1f}% Target Boost" if delta > 0 else f"{p['tgt_zone']:.1f}% Target Share",
                        'verdict': 'HIGH RECEPTION FLOOR',
                        'rationale': f"Target receptions on close catches over receiving yards. Short hook and drag routes settle into soft intermediate voids in {opp_code}'s {cov.get('zone_pct', 60):.0f}% zone coverage for dependable, chain-moving close catches.",
                        'causal_mechanism': 'Linebackers drop deep in zone coverage, leaving tight ends uncovered on short-to-intermediate crossing and sit routes.',
                        'priority': 93.0 + delta
                    })

            # Slot WR mismatch vs Zone: Short Throws -> Receptions
            elif pos == 'WR' and slot_share >= 35:
                delta = p.get('tgt_zone', 0) - p.get('tgt_man', 0)
                cands.append({
                    'player': p['name'],
                    'team': team_code,
                    'pos': 'WR',
                    'scheme_type': 'Slot Underneath Volume',
                    'stat_focus': 'Receptions',
                    'prop_angle': 'Prop Target: Receptions (Short Throws)',
                    'context_metric': f"{p['tgt_zone']:.1f}% Tgt vs Zone ({p['yprr_zone']:.2f} YPRR)",
                    'season_baseline': f"{p['tgt_man']:.1f}% vs Man",
                    'scheme_delta': f"+{delta:.1f}% vs Zone Shell" if delta > 0 else f"{p['tgt_zone']:.1f}% Target Share",
                    'verdict': 'PRIME BREAKOUT SPOT' if delta >= 3.0 or p.get('tgt_zone', 0) >= 25 else 'ELEVATED TARGET CEILING',
                    'rationale': f"Target receptions over receiving yards. Quick slot option routes and crossing routes find soft cushions against {opp_code}'s {cov.get('zone_pct', 60):.1f}% zone coverage ({slot_share:.0f}% slot alignment), providing a high-volume reception floor.",
                    'causal_mechanism': 'Soft intermediate voids in zone coverage concentrate receptions to agile slot receivers.',
                    'priority': 93.0 + delta + (p.get('tgt_zone', 0) * 0.5)
                })

            # Outside Alpha vs Man: Deep Balls -> Receiving Yards
            elif pos == 'WR' and outside_share >= 50 and (cov.get('man_pct', 30) >= 30.0 or p.get('tgt_man', 0) >= 25.0):
                delta = p.get('tgt_man', 0) - p.get('tgt_zone', 0)
                cands.append({
                    'player': p['name'],
                    'team': team_code,
                    'pos': 'WR',
                    'scheme_type': 'Boundary Deep Ball Threat',
                    'stat_focus': 'Receiving Yards',
                    'prop_angle': 'Prop Target: Receiving Yards (Deep Balls)',
                    'context_metric': f"{p['tgt_man']:.1f}% Tgt vs Man ({p['yprr_man']:.2f} YPRR)",
                    'season_baseline': f"{p['tgt_zone']:.1f}% vs Zone",
                    'scheme_delta': f"+{delta:.1f}% vs Man Coverage" if delta > 0 else f"{p['yprr_man']:.2f} Man YPRR",
                    'verdict': 'PRIME BREAKOUT SPOT' if delta >= 3.0 else 'ELEVATED TARGET CEILING',
                    'rationale': f"Target receiving yards over receptions. 1-on-1 boundary go routes and vertical shots attack {opp_code}'s single-high/man looks ({cov.get('man_pct', 30):.1f}%), creating explosive chunk yardage rather than high catch volume.",
                    'causal_mechanism': 'Heavy single-coverage schemes isolate outside boundary receivers in 1-on-1s, where route separation creates explosive chunk gains.',
                    'priority': 92.0 + delta + (p.get('tgt_man', 0) * 0.5)
                })

            # Outside WR vs Zone: Deep honey holes -> Receiving Yards
            elif pos == 'WR' and outside_share >= 50:
                cands.append({
                    'player': p['name'],
                    'team': team_code,
                    'pos': 'WR',
                    'scheme_type': 'Boundary Zone Honey-Hole',
                    'stat_focus': 'Receiving Yards',
                    'prop_angle': 'Prop Target: Receiving Yards (Deep Balls)',
                    'context_metric': f"{p['tgt_zone']:.1f}% Tgt vs Zone ({p['yprr_zone']:.2f} YPRR)",
                    'season_baseline': f"{p.get('tgt_man', 20.0):.1f}% vs Man",
                    'scheme_delta': f"{p['yprr_zone']:.2f} Zone YPRR Efficiency",
                    'verdict': 'HIGH CEILING DEEP TARGET' if p.get('yprr_zone', 0) >= 2.4 else 'ELEVATED TARGET CEILING',
                    'rationale': f"Target receiving yards over receptions. Against {opp_code}'s {cov.get('zone_pct', 60):.0f}% zone shell, boundary go routes and intermediate sideline comebacks target honey-hole voids behind outside cornerbacks for explosive yardage.",
                    'causal_mechanism': 'Zone coverage safeties rotate inside, exposing the boundary sideline where elite outside receivers find soft spots in the cover cushion.',
                    'priority': 91.0 + (p.get('yprr_zone', 0) * 2.0)
                })

            # Backfield RB Checkdown: Short Throws -> Receptions
            elif pos == 'RB':
                delta = p.get('tgt_zone', 0) - p.get('tgt_man', 0)
                cands.append({
                    'player': p['name'],
                    'team': team_code,
                    'pos': 'RB',
                    'scheme_type': 'Backfield Pass Valve',
                    'stat_focus': 'Receptions',
                    'prop_angle': 'Prop Target: Receptions (Checkdowns)',
                    'context_metric': f"{p['tgt_zone']:.1f}% Tgt vs Zone ({p['yprr_zone']:.2f} YPRR)",
                    'season_baseline': f"{p['tgt_man']:.1f}% vs Man",
                    'scheme_delta': f"+{delta:.1f}% Target Surge" if delta > 0 else f"{p['tgt_zone']:.1f}% Target Share",
                    'verdict': 'ELEVATED RECEPTION FLOOR',
                    'rationale': f"Target receptions over receiving yards. Against {opp_code}'s {cov.get('zone_pct', 60):.0f}% zone shell, safety valve checkdowns funnel high-percentage dump-off catches into the flat.",
                    'causal_mechanism': 'Zone coverage drops linebackers deep into hook zones, leaving running backs open on underneath flare and checkdown routes.',
                    'priority': 90.0 + delta
                })

        # 2. Ground attack candidates (Lead Back)
        lead_rusher = team_prof.get('tempo_situational', {}).get('red_zone_touch_leader', f'{team_code} Lead Back').split(' (')[0].split(' /')[0]
        zone_ypc = team_run.get('zone_ypc', 4.2)
        gap_ypc = team_run.get('gap_ypc', 4.0)

        # Choose the dominant run scheme for primary back
        if zone_ypc >= gap_ypc:
            delta = zone_ypc - gap_ypc
            cands.append({
                'player': lead_rusher,
                'team': team_code,
                'pos': 'RB',
                'scheme_type': 'Outside Zone Ground Edge',
                'stat_focus': 'Rushing Yards',
                'prop_angle': 'Prop Target: Rushing Yards (Ground Attack)',
                'context_metric': f"{zone_ypc:.2f} YPC on Zone Carries",
                'season_baseline': f"{gap_ypc:.2f} Gap YPC",
                'scheme_delta': f"+{delta:.2f} YPC Zone Edge" if delta > 0 else f"{zone_ypc:.2f} Zone YPC",
                'verdict': 'HIGH EFFICIENCY GROUND EDGE',
                'rationale': f"Target rushing yards / carries. {team_code} executes perimeter stretch zone against {opp_code}'s #{run_def.get('def_zone_rank', 16)} ranked Zone run defense ({run_def.get('def_zone_ypc_allowed', 4.3):.2f} YPC allowed).",
                'causal_mechanism': 'Perimeter stretch schemes exploit slow-flowing edge defenders, opening cutback lanes for primary backs.',
                'priority': 94.0 + (zone_ypc * 2.0)
            })
        else:
            delta = gap_ypc - zone_ypc
            cands.append({
                'player': lead_rusher,
                'team': team_code,
                'pos': 'RB',
                'scheme_type': 'Gap Power Ground Edge',
                'stat_focus': 'Rushing Yards',
                'prop_angle': 'Prop Target: Rushing Yards (Ground Attack)',
                'context_metric': f"{gap_ypc:.2f} YPC on Gap Carries",
                'season_baseline': f"{zone_ypc:.2f} Zone YPC",
                'scheme_delta': f"+{delta:.2f} YPC Gap Edge" if delta > 0 else f"{gap_ypc:.2f} Gap YPC",
                'verdict': 'HIGH EFFICIENCY GROUND EDGE',
                'rationale': f"Target rushing yards / carries. {team_code} runs downhill gap schemes against {opp_code}'s #{run_def.get('def_gap_rank', 16)} ranked Gap run stop ({run_def.get('def_gap_ypc_allowed', 4.1):.2f} YPC allowed).",
                'causal_mechanism': 'Defenses with poor interior gap discipline yield elevated yards before contact to downhill primary ballcarriers.',
                'priority': 94.0 + (gap_ypc * 2.0)
            })

        # 3. Mobile QB rushing candidate
        qb_info = STARTING_QBS.get(team_code, (f"{team_code} Starting QB", False, 10.0, 3.5))
        qb_name, qb_mobile, qb_ypg, qb_scramble_ypc = qb_info
        if qb_mobile:
            if cov.get('man_pct', 30) >= 28.0 or cov.get('cover_1', 20) >= 18.0:
                cands.append({
                    'player': qb_name,
                    'team': team_code,
                    'pos': 'QB',
                    'scheme_type': 'QB Scramble / Mobility Edge',
                    'stat_focus': 'Rushing Yards',
                    'prop_angle': 'Prop Target: Rushing Yards (QB Scramble vs Man)',
                    'context_metric': f"{cov.get('man_pct', 30):.0f}% Man Shell (Backs Turned)",
                    'season_baseline': f"{qb_ypg:.1f} Rush Yds / Game",
                    'scheme_delta': '+15.5 Scramble Yds Edge',
                    'verdict': 'PRIME RUSHING CEILING',
                    'rationale': f"Target rushing yards over passing props. Faces {opp_code}'s heavy man coverage ({cov.get('man_pct', 30):.0f}%), where defensive backs turn their backs to chase receivers downfield, opening wide scramble lanes for {qb_name}.",
                    'causal_mechanism': 'Man-to-man coverage forces defensive backs to turn away from the pocket, vacating intermediate lanes for mobile QBs to scramble for chunk rushing yards.',
                    'priority': 94.5
                })
            elif cov.get('mfo_pct', 50) >= 48.0:
                cands.append({
                    'player': qb_name,
                    'team': team_code,
                    'pos': 'QB',
                    'scheme_type': 'QB Designed Run / Light Box',
                    'stat_focus': 'Rushing Yards',
                    'prop_angle': 'Prop Target: Rushing Yards (Light Box Lanes)',
                    'context_metric': f"{cov.get('mfo_pct', 50):.0f}% MFO Light Box Shell",
                    'season_baseline': f"{qb_ypg:.1f} Rush Yds / Game",
                    'scheme_delta': '+12.0 Rush Yds Edge',
                    'verdict': 'ELEVATED RUSHING FLOOR',
                    'rationale': f"Target rushing yards. {opp_code} defends with 2 deep safeties ({cov.get('mfo_pct', 50):.0f}% MFO), lightening the defensive box to 6 defenders and creating wide lanes for {qb_name}'s designed power runs and scrambles.",
                    'causal_mechanism': 'Two-high safety shells remove an extra defender from the front box, creating numerical leverage for quarterback-designed rushes.',
                    'priority': 92.5
                })

        # Fallback if no candidate found for team
        if not cands and team_prof.get('target_alignment', {}).get('key_targets'):
            top_p = team_prof['target_alignment']['key_targets'][0]
            cands.append({
                'player': top_p['name'],
                'team': team_code,
                'pos': top_p.get('pos', 'WR'),
                'scheme_type': 'Primary Weapon',
                'stat_focus': 'Receiving Yards',
                'prop_angle': 'Prop Target: Receiving Yards',
                'context_metric': f"{top_p.get('tgt_zone', 24):.1f}% Target Share",
                'season_baseline': f"{top_p.get('tgt_man', 22):.1f}% vs Man",
                'scheme_delta': f"+{abs(top_p.get('tgt_zone', 24) - top_p.get('tgt_man', 22)):.1f}% Target Focus",
                'verdict': 'VOLUME PRIME CANDIDATE',
                'rationale': f"Core offensive gameplan funnels prioritized target looks to {top_p['name']} across coverage alignments.",
                'causal_mechanism': 'Primary weapon maintains prioritized route reads and designed quick-game touches across coverage alignments.',
                'priority': 70.0
            })

        # Deduplicate candidates by player name and sort by priority descending
        seen = set()
        deduped = []
        for c in sorted(cands, key=lambda x: x['priority'], reverse=True):
            if c['player'] not in seen:
                seen.add(c['player'])
                c_clean = {k: v for k, v in c.items() if k != 'priority'}
                deduped.append(c_clean)
        return deduped

    away_cands = _extract_team_breakouts(a_team, a_profile, h_team, h_profile)
    home_cands = _extract_team_breakouts(h_team, h_profile, a_team, a_profile)

    # Curate strictly between 2 and 4 players with guaranteed multi-position diversity
    # Ensures we represent WR (deep yards or short receptions), RB (rushing yards), TE (close catches/seams), and mobile QB
    selected: List[Dict[str, Any]] = []
    selected_players = set()
    selected_positions = set()

    def _add_cand(cand: Dict[str, Any]) -> bool:
        if cand and cand['player'] not in selected_players:
            selected.append(cand)
            selected_players.add(cand['player'])
            selected_positions.add(cand['pos'])
            return True
        return False

    # 1. Pick top RB ground edge (Rushing Yards)
    all_rbs = [c for c in away_cands + home_cands if c['pos'] == 'RB']
    if all_rbs:
        _add_cand(all_rbs[0])

    # 2. Pick top WR edge (Receiving Yards if deep, Receptions if short)
    all_wrs = [c for c in away_cands + home_cands if c['pos'] == 'WR']
    if all_wrs:
        _add_cand(all_wrs[0])

    # 3. Pick top TE (close catches / seams) or Mobile QB (rushing yards / scrambles)
    all_te_qb = [c for c in away_cands + home_cands if c['pos'] in ('TE', 'QB')]
    if all_te_qb:
        a_count = sum(1 for c in selected if c['team'] == a_team)
        h_count = sum(1 for c in selected if c['team'] == h_team)
        target_team = a_team if a_count < h_count else (h_team if h_count < a_count else None)
        matching = [c for c in all_te_qb if target_team is None or c['team'] == target_team]
        if matching:
            _add_cand(matching[0])
        else:
            _add_cand(all_te_qb[0])

    # 4. Fill 4th slot to complete a 4-player slate with maximum position & team balance
    all_remaining = [c for c in away_cands + home_cands if c['player'] not in selected_players]
    missing_positions = [pos for pos in ('TE', 'QB', 'RB', 'WR') if pos not in selected_positions]
    added_4th = False
    for mp in missing_positions:
        pos_cands = [c for c in all_remaining if c['pos'] == mp]
        if pos_cands:
            a_count = sum(1 for c in selected if c['team'] == a_team)
            h_count = sum(1 for c in selected if c['team'] == h_team)
            target_team = a_team if a_count < h_count else (h_team if h_count < a_count else None)
            team_cands = [c for c in pos_cands if target_team is None or c['team'] == target_team]
            cand_to_add = team_cands[0] if team_cands else pos_cands[0]
            if _add_cand(cand_to_add):
                added_4th = True
                break

    if not added_4th and all_remaining:
        a_count = sum(1 for c in selected if c['team'] == a_team)
        h_count = sum(1 for c in selected if c['team'] == h_team)
        target_team = a_team if a_count < h_count else (h_team if h_count < a_count else None)
        team_cands = [c for c in all_remaining if target_team is None or c['team'] == target_team]
        _add_cand(team_cands[0] if team_cands else all_remaining[0])

    # Ensure strictly between 2 and 4 players
    breakouts = selected[:4]
    if len(breakouts) < 2:
        other_team = h_team if (breakouts and breakouts[0]['team'] == a_team) else a_team
        breakouts.append({
            'player': f"{other_team} Lead Back",
            'team': other_team,
            'pos': 'RB',
            'scheme_type': 'Volume Edge',
            'stat_focus': 'Rushing Yards',
            'prop_angle': 'Prop Target: Rushing Yards (Ground Attack)',
            'context_metric': '16+ Projected Touches',
            'season_baseline': '14 Touches / Game',
            'scheme_delta': '+2.5 Touch Ceiling',
            'verdict': 'VOLUME PRIME CANDIDATE',
            'rationale': 'Target rushing yards / carries. Game script and red zone concentration create reliable touch volume floor.',
            'causal_mechanism': 'Goal line touch consolidation elevates player floor regardless of defensive fronts.'
        })

    # -------------------------------------------------------------
    # Pillar 5: Game-Flow, Tempo & Situational Usage Volume Predictors (6:17, 26:54)
    # -------------------------------------------------------------
    pace_gap = abs(a_profile['tempo_situational']['neutral_pace_sec'] - h_profile['tempo_situational']['neutral_pace_sec'])
    game_pace_verdict = (
        "UP-TEMPO TRACK MEET: Both teams operate in top third neutral snap pace. Combined plays projected to exceed 130 snaps."
        if (a_profile['tempo_situational']['pace_rank'] <= 12 and h_profile['tempo_situational']['pace_rank'] <= 12) else (
            "SLOW BALL-CONTROL SLOG: Heavy ground games and deliberate pre-snap motions will condense total game plays."
            if (a_profile['tempo_situational']['pace_rank'] >= 24 or h_profile['tempo_situational']['pace_rank'] >= 24) else
            "BALANCED NFL TEMPO: Standard 63-65 play per side projection with normal situational clock acceleration."
        )
    )

    volume_override_verdict = (
        f"Game-flow and situational usage will override standard talent projections in this matchup: "
        f"{a_team} check-down rate ({a_profile['tempo_situational']['checkdown_pct']}%) combined with {h_team}'s "
        f"{h_profile['coverage']['mfo_pct']:.0f}% MFO shell establishes an elevated reception floor for short underneath targets. "
        f"Red zone usage is highly concentrated: {a_profile['tempo_situational']['red_zone_touch_leader']} and "
        f"{h_profile['tempo_situational']['red_zone_touch_leader']} dictate touchdown equity over flat yardage projections."
    )

    game_flow = {
        'away_tempo': a_profile['tempo_situational'],
        'home_tempo': h_profile['tempo_situational'],
        'pace_synthesis': game_pace_verdict,
        'volume_override_verdict': volume_override_verdict
    }

    # -------------------------------------------------------------
    # Executive Synthesis / Film Room Thesis
    # -------------------------------------------------------------
    thesis_quote = (
        "To find the deep analytical insights shared in this video (0:09), your platform needs to look for "
        "and cross-reference specific scheme-based metrics. Instead of basic stats like total yards, it ingests: "
        "1. Defensive Coverage Tendencies (Zone vs Man, MFO vs MFC), "
        "2. Run-Scheme Splits (Zone vs Man/Gap), "
        "3. Positional Target Rates (Slot vs Outside, Inline TE), "
        "4. Efficiency by Context (Scheme-Specific YPC/YPR Breakout Spotters), and "
        "5. Game-Flow & Tempo (Time of Possession, Red Zone Concentration, Check-down Frequency)."
    )

    return {
        'home_team': h_team,
        'home_team_name': h_name,
        'away_team': a_team,
        'away_team_name': a_name,
        'season': season,
        'week': week,
        'sample_window': sample_window,
        'thesis_quote': thesis_quote,
        'coverage_tendencies': {
            'away_vs_home': a_vs_h_cov,
            'home_vs_away': h_vs_a_cov
        },
        'run_scheme_splits': run_splits,
        'positional_target_rates': target_rates,
        'efficiency_breakouts': breakouts,
        'game_flow_tempo': game_flow
    }
