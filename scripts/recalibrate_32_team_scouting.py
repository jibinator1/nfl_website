"""
scripts/recalibrate_32_team_scouting.py
======================================
Computes comprehensive, empirical scouting metrics for all 32 NFL franchises
using combined 2025 + 2026 play-by-play data and participation data:
  1. Passing YPA Offense & Defense Allowed (with ranks 1-32)
  2. Game-State Pass Rates & Ranks (Leading, Neutral, Trailing)
  3. Defensive Box Counts (Light Box <= 6, Heavy Box >= 8, YPC allowed) & Offense YPC vs Light/Heavy
  4. Pass Depth EPA/play allowed & Success Rate allowed (Behind LOS, Short, Intermediate, Deep)
  5. Pressure Rate without Blitzing (was_pressure == 1 and rushers <= 4) vs Blitz Rate
  6. Positional Targets Allowed (WR, TE, RB) and Yards Per Target (YPT)
  7. Run Splits & Directional Tackle Splits (Inside vs Outside)

Outputs:
  - data/team_tendencies_32.json
  - api/data/team_tendencies_32.json
  - Updates data/scheme_profiles_32.json
  - Updates api/data/scheme_profiles_32.json
"""

import os
import sys
import json
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
API_DATA_DIR = os.path.join(ROOT_DIR, "api", "data")
PBP_DIR = os.path.join(DATA_DIR, "pbp_cache")

def run():
    print("Loading 2025 + 2026 clean play-by-play data...")
    pbp25_path = os.path.join(PBP_DIR, "pbp_2025.parquet")
    pbp26_path = os.path.join(PBP_DIR, "pbp_2026.parquet")
    
    dfs = []
    if os.path.exists(pbp25_path):
        dfs.append(pd.read_parquet(pbp25_path))
    if os.path.exists(pbp26_path):
        dfs.append(pd.read_parquet(pbp26_path))
        
    if not dfs:
        raise RuntimeError("No PBP data found in data/pbp_cache!")
        
    pbp = pd.concat(dfs, ignore_index=True)
    pbp = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna()].copy()
    if "qb_kneel" in pbp.columns:
        pbp = pbp[pbp["qb_kneel"] == 0]
    if "qb_spike" in pbp.columns:
        pbp = pbp[pbp["qb_spike"] == 0]
        
    print(f"Loaded {len(pbp)} total clean regular plays across 2025+2026.")
    
    # Load participation data for 2025 (box counts, pressure without blitz)
    print("Loading participation data for box counts and pressure metrics...")
    try:
        import nflreadpy as nfl
        part = nfl.load_participation(seasons=[2025]).to_pandas()
        part_cols = ["nflverse_game_id", "play_id", "defenders_in_box", "number_of_pass_rushers", "was_pressure"]
        merged_part = pd.merge(
            pbp,
            part[part_cols],
            left_on=["game_id", "play_id"],
            right_on=["nflverse_game_id", "play_id"],
            how="left"
        )
    except Exception as e:
        print(f"Warning: Failed loading participation via nflreadpy ({e}). Using existing PBP fields.")
        merged_part = pbp.copy()
        if "defenders_in_box" not in merged_part.columns:
            merged_part["defenders_in_box"] = np.nan
        if "number_of_pass_rushers" not in merged_part.columns:
            merged_part["number_of_pass_rushers"] = np.nan
        if "was_pressure" not in merged_part.columns:
            merged_part["was_pressure"] = np.nan

    # Load weekly stats for player position map
    from backend.team_tendencies import get_player_position_map
    pos_map = get_player_position_map()
    merged_part["pos"] = merged_part["receiver_player_id"].map(pos_map).fillna("Unknown")

    teams = sorted(merged_part["posteam"].dropna().unique().tolist())
    print(f"Computing metrics for all {len(teams)} teams...")

    # Pre-calculate play states
    diff = merged_part["score_differential"].fillna(0)
    merged_part["_state"] = np.where(diff >= 7, "leading", np.where(diff <= -7, "trailing", "neutral"))
    merged_part["_is_pass"] = (merged_part["play_type"] == "pass").astype(int)
    merged_part["_is_run"] = (merged_part["play_type"] == "run").astype(int)

    # 1. Offense and Defense Passing YPA
    off_ypa_map = {}
    off_ypa_n = {}
    def_ypa_map = {}
    def_ypa_n = {}
    
    for tm in teams:
        op = merged_part[(merged_part["posteam"] == tm) & (merged_part["_is_pass"] == 1)]
        dp = merged_part[(merged_part["defteam"] == tm) & (merged_part["_is_pass"] == 1)]
        off_ypa_map[tm] = float(op["yards_gained"].sum() / len(op)) if len(op) > 0 else 6.5
        off_ypa_n[tm] = len(op)
        def_ypa_map[tm] = float(dp["yards_gained"].sum() / len(dp)) if len(dp) > 0 else 6.5
        def_ypa_n[tm] = len(dp)

    off_ypa_ranked = sorted(off_ypa_map.items(), key=lambda x: x[1], reverse=True)
    off_ypa_ranks = {tm: r + 1 for r, (tm, _) in enumerate(off_ypa_ranked)}
    def_ypa_ranked = sorted(def_ypa_map.items(), key=lambda x: x[1])  # lower is better for defense
    def_ypa_ranks = {tm: r + 1 for r, (tm, _) in enumerate(def_ypa_ranked)}
    med_off_ypa = float(np.median(list(off_ypa_map.values())))
    med_def_ypa = float(np.median(list(def_ypa_map.values())))

    # 2. Game-State Pass Rates and Ranks
    game_state_data = {}
    for st in ["leading", "neutral", "trailing"]:
        st_sub = merged_part[merged_part["_state"] == st]
        st_rates = {}
        st_counts = {}
        for tm in teams:
            tm_p = st_sub[st_sub["posteam"] == tm]
            st_rates[tm] = float(tm_p["_is_pass"].mean()) if len(tm_p) > 0 else 0.50
            st_counts[tm] = len(tm_p)
        st_ranked = sorted(st_rates.items(), key=lambda x: x[1], reverse=True)
        st_ranks = {tm: r + 1 for r, (tm, _) in enumerate(st_ranked)}
        st_med = float(np.median(list(st_rates.values())))
        game_state_data[st] = {
            "rates": st_rates,
            "counts": st_counts,
            "ranks": st_ranks,
            "median": round(st_med, 3),
            "p25": round(float(np.percentile(list(st_rates.values()), 25)), 3),
            "p75": round(float(np.percentile(list(st_rates.values()), 75)), 3),
        }

    # 3. Defensive Box Counts & Offense Box Counts
    def_box_data = {}
    off_box_data = {}
    for tm in teams:
        # Defense facing runs
        d_runs = merged_part[(merged_part["defteam"] == tm) & (merged_part["_is_run"] == 1) & (merged_part["defenders_in_box"].notna())]
        d_tot = len(d_runs)
        if d_tot > 0:
            d_light = d_runs[d_runs["defenders_in_box"] <= 6]
            d_heavy = d_runs[d_runs["defenders_in_box"] >= 8]
            light_pct = round(len(d_light) / d_tot * 100.0, 1)
            heavy_pct = round(len(d_heavy) / d_tot * 100.0, 1)
            ypc_light = round(float(d_light["yards_gained"].mean()), 2) if len(d_light) > 0 else 4.2
            ypc_heavy = round(float(d_heavy["yards_gained"].mean()), 2) if len(d_heavy) > 0 else 3.2
            def_box_data[tm] = {
                "light_box_rate": light_pct,
                "heavy_box_rate": heavy_pct,
                "ypc_vs_light_box": ypc_light,
                "ypc_vs_heavy_box": ypc_heavy,
                "carries_light": len(d_light),
                "carries_heavy": len(d_heavy),
                "total_run_snaps": d_tot,
            }
        else:
            def_box_data[tm] = {
                "light_box_rate": 55.0,
                "heavy_box_rate": 8.0,
                "ypc_vs_light_box": 4.30,
                "ypc_vs_heavy_box": 3.10,
                "carries_light": 0,
                "carries_heavy": 0,
                "total_run_snaps": 0,
            }

        # Offense running
        o_runs = merged_part[(merged_part["posteam"] == tm) & (merged_part["_is_run"] == 1) & (merged_part["defenders_in_box"].notna())]
        if len(o_runs) > 0:
            o_light = o_runs[o_runs["defenders_in_box"] <= 6]
            o_heavy = o_runs[o_runs["defenders_in_box"] >= 8]
            off_box_data[tm] = {
                "rush_ypc_vs_light_box": round(float(o_light["yards_gained"].mean()), 2) if len(o_light) > 0 else 4.3,
                "rush_ypc_vs_heavy_box": round(float(o_heavy["yards_gained"].mean()), 2) if len(o_heavy) > 0 else 3.1,
                "carries_light": len(o_light),
                "carries_heavy": len(o_heavy),
            }
        else:
            off_box_data[tm] = {
                "rush_ypc_vs_light_box": 4.30,
                "rush_ypc_vs_heavy_box": 3.10,
                "carries_light": 0,
                "carries_heavy": 0,
            }

    # 4. Pressure Rate without Blitzing
    pressure_no_blitz_data = {}
    for tm in teams:
        dp = merged_part[(merged_part["defteam"] == tm) & (merged_part["_is_pass"] == 1) & (merged_part["number_of_pass_rushers"].notna())]
        if len(dp) > 0:
            no_blitz = dp[dp["number_of_pass_rushers"] <= 4]
            blitz = dp[dp["number_of_pass_rushers"] >= 5]
            pnb_rate = round(float((no_blitz["was_pressure"] == 1).mean() * 100.0), 1) if len(no_blitz) > 0 else 24.0
            blitz_rate = round(float(len(blitz) / len(dp) * 100.0), 1)
            pressure_no_blitz_data[tm] = {
                "pressure_no_blitz_pct": pnb_rate,
                "blitz_pct": blitz_rate,
                "no_blitz_snaps": len(no_blitz),
                "total_pass_snaps": len(dp),
            }
        else:
            pressure_no_blitz_data[tm] = {
                "pressure_no_blitz_pct": 24.5,
                "blitz_pct": 26.0,
                "no_blitz_snaps": 0,
                "total_pass_snaps": 0,
            }

    # 5. Pass Depth EPA & SR Allowed
    pass_depth_allowed_data = {}
    for tm in teams:
        dp = merged_part[(merged_part["defteam"] == tm) & (merged_part["_is_pass"] == 1) & (merged_part["air_yards"].notna())]
        depth_splits = {}
        for d_key, cond in [
            ("behind_los", dp["air_yards"] < 0),
            ("short", (dp["air_yards"] >= 0) & (dp["air_yards"] < 10)),
            ("intermediate", (dp["air_yards"] >= 10) & (dp["air_yards"] < 20)),
            ("deep", dp["air_yards"] >= 20),
        ]:
            sub = dp[cond]
            n_att = len(sub)
            if n_att > 0:
                depth_splits[d_key] = {
                    "epa_per_attempt": round(float(sub["epa"].mean()), 3),
                    "success_rate": round(float(sub["success"].mean() * 100.0), 1),
                    "yards_per_attempt": round(float(sub["yards_gained"].mean()), 2),
                    "sample_attempts": n_att,
                }
            else:
                depth_splits[d_key] = {
                    "epa_per_attempt": 0.0,
                    "success_rate": 45.0,
                    "yards_per_attempt": 6.0,
                    "sample_attempts": 0,
                }
        pass_depth_allowed_data[tm] = depth_splits

    # 6. Positional Targets Allowed (WR, TE, RB) and YPT
    pos_targets_allowed_data = {}
    for tm in teams:
        dp = merged_part[(merged_part["defteam"] == tm) & (merged_part["_is_pass"] == 1) & (merged_part["pos"].isin(["WR", "TE", "RB"]))]
        tot_tgts = len(dp)
        pos_splits = {}
        for pos_key in ["WR", "TE", "RB"]:
            sub = dp[dp["pos"] == pos_key]
            n_sub = len(sub)
            if n_sub > 0:
                share = round(float(n_sub / tot_tgts * 100.0), 1) if tot_tgts > 0 else 0.0
                ypt = round(float(sub["yards_gained"].sum() / n_sub), 2)
                epa_tgt = round(float(sub["epa"].mean()), 3)
                pos_splits[pos_key.lower()] = {
                    "target_share": share,
                    "yards_per_target": ypt,
                    "epa_per_target": epa_tgt,
                    "sample_targets": n_sub,
                }
            else:
                pos_splits[pos_key.lower()] = {
                    "target_share": 33.3,
                    "yards_per_target": 7.0,
                    "epa_per_target": 0.0,
                    "sample_targets": 0,
                }
        pos_targets_allowed_data[tm] = pos_splits

    # 7. Directional Tackle Gaps (Inside vs Outside)
    tackle_gaps_data = {}
    inside_ypc_map = {}
    outside_ypc_map = {}
    for tm in teams:
        dr = merged_part[(merged_part["defteam"] == tm) & (merged_part["_is_run"] == 1)]
        ins = dr[(dr["run_location"] == "middle") | (dr["run_gap"] == "guard")]
        out = dr[(dr["run_location"].isin(["left", "right"])) & (dr["run_gap"].isin(["tackle", "end"]))]
        ins_ypc = round(float(ins["yards_gained"].mean()), 2) if len(ins) > 0 else 3.86
        out_ypc = round(float(out["yards_gained"].mean()), 2) if len(out) > 0 else 4.55
        inside_ypc_map[tm] = ins_ypc
        outside_ypc_map[tm] = out_ypc
        tackle_gaps_data[tm] = {
            "inside_tackles_ypc": ins_ypc,
            "inside_carries": len(ins),
            "outside_tackles_ypc": out_ypc,
            "outside_carries": len(out),
        }

    ins_ranked = sorted(inside_ypc_map.items(), key=lambda x: x[1])
    ins_ranks = {tm: r + 1 for r, (tm, _) in enumerate(ins_ranked)}
    out_ranked = sorted(outside_ypc_map.items(), key=lambda x: x[1])
    out_ranks = {tm: r + 1 for r, (tm, _) in enumerate(out_ranked)}
    for tm in teams:
        tackle_gaps_data[tm]["inside_rank"] = ins_ranks[tm]
        tackle_gaps_data[tm]["outside_rank"] = out_ranks[tm]

    # Assemble complete 32-team JSON database
    compiled_32 = {}
    for tm in teams:
        n_games = int(merged_part[merged_part["posteam"] == tm]["game_id"].nunique())
        
        # Build Game State Splits dictionary for this team
        g_splits = {}
        for st in ["leading", "neutral", "trailing"]:
            st_info = game_state_data[st]
            raw_v = round(st_info["rates"][tm], 3)
            rk = st_info["ranks"][tm]
            cnt = st_info["counts"][tm]
            med = st_info["median"]
            diff_m = round(raw_v - med, 3)
            diff_pct = round((diff_m / med) * 100.0, 1) if med > 0 else 0.0
            
            g_splits[st] = {
                "pass_rate": {
                    "raw_value": raw_v,
                    "league_median": med,
                    "league_p25": st_info["p25"],
                    "league_p75": st_info["p75"],
                    "rank": rk if n_games >= 6 else None,
                    "rank_display": f"Rank #{rk} (n={n_games} games)" if n_games >= 6 else f"{raw_v*100:.1f}% (league median {med*100:.1f}%, n={n_games} games -- rank suppressed, small sample)",
                    "rank_suppressed": bool(n_games < 6),
                    "n_games": n_games,
                    "total_teams": 32,
                    "rank_direction": "higher is better",
                    "diff_from_median": diff_m,
                    "diff_pct": diff_pct,
                    "sample_size": cnt,
                    "is_small_sample": bool(cnt < 40),
                    "unit": "rate",
                }
            }

        compiled_32[tm] = {
            "team": tm,
            "n_games": n_games,
            "offense": {
                "passing_ypa": {
                    "raw_value": round(off_ypa_map[tm], 2),
                    "league_median": round(med_off_ypa, 2),
                    "rank": off_ypa_ranks[tm],
                    "rank_display": f"Rank #{off_ypa_ranks[tm]} (n={n_games} games)",
                    "sample_attempts": off_ypa_n[tm],
                },
                "game_state_splits": g_splits,
                "box_counts": off_box_data[tm],
            },
            "defense": {
                "passing_ypa_allowed": {
                    "raw_value": round(def_ypa_map[tm], 2),
                    "league_median": round(med_def_ypa, 2),
                    "rank": def_ypa_ranks[tm],
                    "rank_display": f"Rank #{def_ypa_ranks[tm]} (n={n_games} games)",
                    "sample_attempts": def_ypa_n[tm],
                },
                "pressure_without_blitz": pressure_no_blitz_data[tm],
                "pass_depth_allowed": pass_depth_allowed_data[tm],
                "positional_targets_allowed": pos_targets_allowed_data[tm],
                "box_counts": def_box_data[tm],
                "directional_gaps": tackle_gaps_data[tm],
            }
        }

    # Write out data/team_tendencies_32.json and api/data/team_tendencies_32.json
    out_paths = [
        os.path.join(DATA_DIR, "team_tendencies_32.json"),
        os.path.join(API_DATA_DIR, "team_tendencies_32.json"),
    ]
    for p in out_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(compiled_32, f, indent=2)
        print(f"Saved 32-team scouting database to {p}")

    # Also augment scheme_profiles_32.json
    sp_paths = [
        os.path.join(DATA_DIR, "scheme_profiles_32.json"),
        os.path.join(API_DATA_DIR, "scheme_profiles_32.json"),
    ]
    for sp_path in sp_paths:
        if os.path.exists(sp_path):
            try:
                with open(sp_path, "r", encoding="utf-8") as f:
                    sp_data = json.load(f)
                for tm, d_entry in compiled_32.items():
                    if tm in sp_data:
                        sp_data[tm]["scouting_tendencies"] = d_entry
                with open(sp_path, "w", encoding="utf-8") as f:
                    json.dump(sp_data, f, indent=2)
                print(f"Augmented {sp_path} with comprehensive scouting tendencies.")
            except Exception as e:
                print(f"Warning: Failed augmenting {sp_path}: {e}")

    print("Successfully finished recalibrating 32-team scouting database!")

if __name__ == "__main__":
    run()
