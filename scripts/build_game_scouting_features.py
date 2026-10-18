"""
Build Game-Level Scouting Features Parquet
==========================================
Aggregates play-by-play and participation into a lightweight per-game dataset:
- Enables instant (< 2ms) dynamic date filtering across all 32 teams.
- Captures game-state splits (Ahead, Neutral, Trailing), box counts, pass depth, pressure without blitz, and directional gaps per game.
- Size: ~100 KB total, perfectly suited for serverless deployment and local execution.
"""

import os
import sys
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PBP_DIR = os.path.join(ROOT_DIR, "data", "pbp_cache")
DATA_DIR = os.path.join(ROOT_DIR, "data")
API_DATA_DIR = os.path.join(ROOT_DIR, "api", "data")


def main():
    print("Building game-level scouting features table...")
    schedules_path = os.path.join(DATA_DIR, "schedules_cache.parquet")
    weekly_path = os.path.join(DATA_DIR, "weekly_cache.parquet")
    
    if not os.path.exists(schedules_path):
        raise FileNotFoundError(f"{schedules_path} not found")
        
    sched = pd.read_parquet(schedules_path)
    sched = sched[sched["home_score"].notna()].copy()
    game_meta = {}
    for _, row in sched.iterrows():
        game_meta[row["game_id"]] = {
            "season": int(row["season"]),
            "week": int(row["week"]),
            "gameday": str(row["gameday"]),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
        }
        
    weekly = pd.read_parquet(weekly_path)
    pos_map = dict(zip(weekly["player_id"].dropna(), weekly["position"].dropna()))

    pbp_files = [
        os.path.join(PBP_DIR, "pbp_2023.parquet"),
        os.path.join(PBP_DIR, "pbp_2024.parquet"),
        os.path.join(PBP_DIR, "pbp_2025.parquet"),
        os.path.join(PBP_DIR, "pbp_2026.parquet"),
    ]
    dfs = []
    for f in pbp_files:
        if os.path.exists(f):
            print(f"Loading {os.path.basename(f)}...")
            dfs.append(pd.read_parquet(f))
            
    pbp = pd.concat(dfs, ignore_index=True)
    pbp = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna()].copy()
    if "qb_kneel" in pbp.columns:
        pbp = pbp[pbp["qb_kneel"] == 0]
    if "qb_spike" in pbp.columns:
        pbp = pbp[pbp["qb_spike"] == 0]

    # Load participation data if available
    try:
        import nflreadpy as nfl
        part = nfl.load_participation(seasons=[2024, 2025]).to_pandas()
        part_cols = ["nflverse_game_id", "play_id", "defenders_in_box", "number_of_pass_rushers", "was_pressure"]
        merged = pd.merge(
            pbp,
            part[part_cols],
            left_on=["game_id", "play_id"],
            right_on=["nflverse_game_id", "play_id"],
            how="left"
        )
    except Exception as e:
        print(f"Warning: nflreadpy participation failed ({e}), using fallback column defaults.")
        merged = pbp.copy()
        if "defenders_in_box" not in merged.columns:
            merged["defenders_in_box"] = np.nan
        if "number_of_pass_rushers" not in merged.columns:
            merged["number_of_pass_rushers"] = np.nan
        if "was_pressure" not in merged.columns:
            merged["was_pressure"] = np.nan

    merged["pos"] = merged["receiver_player_id"].map(pos_map).fillna("Unknown")
    diff = merged["score_differential"].fillna(0)
    merged["_state"] = np.where(diff >= 7, "leading", np.where(diff <= -7, "trailing", "neutral"))
    merged["_is_pass"] = (merged["play_type"] == "pass").astype(int)
    merged["_is_run"] = (merged["play_type"] == "run").astype(int)

    # Compile by game and team
    all_game_ids = [gid for gid in merged["game_id"].unique() if gid in game_meta]
    print(f"Processing {len(all_game_ids)} games across all seasons...")

    records = []
    for gid in all_game_ids:
        gm = game_meta[gid]
        g_plays = merged[merged["game_id"] == gid]
        h_team = gm["home_team"]
        a_team = gm["away_team"]

        for tm, opp, is_home in [(h_team, a_team, True), (a_team, h_team, False)]:
            o_plays = g_plays[g_plays["posteam"] == tm]
            d_plays = g_plays[g_plays["defteam"] == tm]

            o_pass = o_plays[o_plays["_is_pass"] == 1]
            o_run = o_plays[o_plays["_is_run"] == 1]
            d_pass = d_plays[d_plays["_is_pass"] == 1]
            d_run = d_plays[d_plays["_is_run"] == 1]

            # Offense basic
            pass_att = len(o_pass)
            pass_yds = float(o_pass["yards_gained"].sum()) if pass_att > 0 else 0.0
            pass_epa = float(o_pass["epa"].sum()) if pass_att > 0 else 0.0
            rush_att = len(o_run)
            rush_yds = float(o_run["yards_gained"].sum()) if rush_att > 0 else 0.0
            rush_epa = float(o_run["epa"].sum()) if rush_att > 0 else 0.0

            # Defense basic
            def_pass_att = len(d_pass)
            def_pass_yds = float(d_pass["yards_gained"].sum()) if def_pass_att > 0 else 0.0
            def_pass_epa = float(d_pass["epa"].sum()) if def_pass_att > 0 else 0.0
            def_rush_att = len(d_run)
            def_rush_yds = float(d_run["yards_gained"].sum()) if def_rush_att > 0 else 0.0
            def_rush_epa = float(d_run["epa"].sum()) if def_rush_att > 0 else 0.0

            # Game-state splits (Offense)
            lead_pass = len(o_pass[o_pass["_state"] == "leading"])
            lead_rush = len(o_run[o_run["_state"] == "leading"])
            neut_pass = len(o_pass[o_pass["_state"] == "neutral"])
            neut_rush = len(o_run[o_run["_state"] == "neutral"])
            trail_pass = len(o_pass[o_pass["_state"] == "trailing"])
            trail_rush = len(o_run[o_run["_state"] == "trailing"])

            # Box counts
            o_box = o_run[o_run["defenders_in_box"].notna()]
            o_c_light = len(o_box[o_box["defenders_in_box"] <= 6])
            o_y_light = float(o_box[o_box["defenders_in_box"] <= 6]["yards_gained"].sum())
            o_c_heavy = len(o_box[o_box["defenders_in_box"] >= 8])
            o_y_heavy = float(o_box[o_box["defenders_in_box"] >= 8]["yards_gained"].sum())

            d_box = d_run[d_run["defenders_in_box"].notna()]
            d_c_light = len(d_box[d_box["defenders_in_box"] <= 6])
            d_y_light = float(d_box[d_box["defenders_in_box"] <= 6]["yards_gained"].sum())
            d_c_heavy = len(d_box[d_box["defenders_in_box"] >= 8])
            d_y_heavy = float(d_box[d_box["defenders_in_box"] >= 8]["yards_gained"].sum())
            d_box_total = len(d_box)

            # Pressure without blitz (defense)
            d_rushers = d_pass[d_pass["number_of_pass_rushers"].notna()]
            d_nb = d_rushers[d_rushers["number_of_pass_rushers"] <= 4]
            d_blitz = d_rushers[d_rushers["number_of_pass_rushers"] >= 5]
            d_pnb = int((d_nb["was_pressure"] == 1).sum()) if len(d_nb) > 0 else 0

            # Pass depth allowed (defense): Behind LOS (<0), Short (0-5), Interim (6-15), Deep (16+)
            d_ay = d_pass[d_pass["air_yards"].notna()]
            d_blos = d_ay[d_ay["air_yards"] < 0]
            d_short = d_ay[(d_ay["air_yards"] >= 0) & (d_ay["air_yards"] <= 5)]
            d_inter = d_ay[(d_ay["air_yards"] >= 6) & (d_ay["air_yards"] <= 15)]
            d_deep = d_ay[d_ay["air_yards"] >= 16]

            # Offense pass depth: Behind LOS (<0), Short (0-5), Interim (6-15), Deep (16+)
            o_ay = o_pass[o_pass["air_yards"].notna()]
            o_blos = o_ay[o_ay["air_yards"] < 0]
            o_short = o_ay[(o_ay["air_yards"] >= 0) & (o_ay["air_yards"] <= 5)]
            o_inter = o_ay[(o_ay["air_yards"] >= 6) & (o_ay["air_yards"] <= 15)]
            o_deep = o_ay[o_ay["air_yards"] >= 16]

            # Offense situational pass depth by game state
            o_lead_ay = o_ay[o_ay["_state"] == "leading"]
            o_neut_ay = o_ay[o_ay["_state"] == "neutral"]
            o_trail_ay = o_ay[o_ay["_state"] == "trailing"]

            lead_att_blos = len(o_lead_ay[o_lead_ay["air_yards"] < 0])
            lead_att_short = len(o_lead_ay[(o_lead_ay["air_yards"] >= 0) & (o_lead_ay["air_yards"] <= 5)])
            lead_att_inter = len(o_lead_ay[(o_lead_ay["air_yards"] >= 6) & (o_lead_ay["air_yards"] <= 15)])
            lead_att_deep = len(o_lead_ay[o_lead_ay["air_yards"] >= 16])

            neut_att_blos = len(o_neut_ay[o_neut_ay["air_yards"] < 0])
            neut_att_short = len(o_neut_ay[(o_neut_ay["air_yards"] >= 0) & (o_neut_ay["air_yards"] <= 5)])
            neut_att_inter = len(o_neut_ay[(o_neut_ay["air_yards"] >= 6) & (o_neut_ay["air_yards"] <= 15)])
            neut_att_deep = len(o_neut_ay[o_neut_ay["air_yards"] >= 16])

            trail_att_blos = len(o_trail_ay[o_trail_ay["air_yards"] < 0])
            trail_att_short = len(o_trail_ay[(o_trail_ay["air_yards"] >= 0) & (o_trail_ay["air_yards"] <= 5)])
            trail_att_inter = len(o_trail_ay[(o_trail_ay["air_yards"] >= 6) & (o_trail_ay["air_yards"] <= 15)])
            trail_att_deep = len(o_trail_ay[o_trail_ay["air_yards"] >= 16])

            # Positional targets allowed
            d_tgt_wr = len(d_pass[d_pass["pos"] == "WR"])
            d_yds_wr = float(d_pass[d_pass["pos"] == "WR"]["yards_gained"].sum())
            d_tgt_te = len(d_pass[d_pass["pos"] == "TE"])
            d_yds_te = float(d_pass[d_pass["pos"] == "TE"]["yards_gained"].sum())
            d_tgt_rb = len(d_pass[d_pass["pos"] == "RB"])
            d_yds_rb = float(d_pass[d_pass["pos"] == "RB"]["yards_gained"].sum())

            # Directional gaps (defense)
            d_ins = d_run[(d_run["run_location"] == "middle") | (d_run["run_gap"] == "guard")]
            d_out = d_run[(d_run["run_location"].isin(["left", "right"])) & (d_run["run_gap"].isin(["tackle", "end"]))]
            d_c_ins = len(d_ins)
            d_y_ins = float(d_ins["yards_gained"].sum())
            d_c_out = len(d_out)
            d_y_out = float(d_out["yards_gained"].sum())

            # Directional gaps (offense)
            o_ins = o_run[(o_run["run_location"] == "middle") | (o_run["run_gap"] == "guard")]
            o_out = o_run[(o_run["run_location"].isin(["left", "right"])) & (o_run["run_gap"].isin(["tackle", "end"]))]
            o_c_ins = len(o_ins)
            o_y_ins = float(o_ins["yards_gained"].sum())
            o_c_out = len(o_out)
            o_y_out = float(o_out["yards_gained"].sum())

            records.append({
                "game_id": gid,
                "season": gm["season"],
                "week": gm["week"],
                "gameday": gm["gameday"],
                "team": tm,
                "opponent": opp,
                "is_home": is_home,
                
                # Offense
                "pass_att": pass_att,
                "pass_yds": pass_yds,
                "pass_epa": pass_epa,
                "rush_att": rush_att,
                "rush_yds": rush_yds,
                "rush_epa": rush_epa,
                
                # Defense
                "def_pass_att": def_pass_att,
                "def_pass_yds": def_pass_yds,
                "def_pass_epa": def_pass_epa,
                "def_rush_att": def_rush_att,
                "def_rush_yds": def_rush_yds,
                "def_rush_epa": def_rush_epa,
                
                # Game State Splits
                "lead_pass": lead_pass,
                "lead_rush": lead_rush,
                "neut_pass": neut_pass,
                "neut_rush": neut_rush,
                "trail_pass": trail_pass,
                "trail_rush": trail_rush,

                # Game State Pass Depth (Offense)
                "lead_att_blos": lead_att_blos,
                "lead_att_short": lead_att_short,
                "lead_att_inter": lead_att_inter,
                "lead_att_deep": lead_att_deep,
                "neut_att_blos": neut_att_blos,
                "neut_att_short": neut_att_short,
                "neut_att_inter": neut_att_inter,
                "neut_att_deep": neut_att_deep,
                "trail_att_blos": trail_att_blos,
                "trail_att_short": trail_att_short,
                "trail_att_inter": trail_att_inter,
                "trail_att_deep": trail_att_deep,
                
                # Box Counts
                "o_c_light": o_c_light,
                "o_y_light": o_y_light,
                "o_c_heavy": o_c_heavy,
                "o_y_heavy": o_y_heavy,
                "d_c_light": d_c_light,
                "d_y_light": d_y_light,
                "d_c_heavy": d_c_heavy,
                "d_y_heavy": d_y_heavy,
                "d_box_total": d_box_total,
                
                # Pressure Without Blitz
                "d_pass_rush_snaps": len(d_rushers),
                "d_no_blitz_snaps": len(d_nb),
                "d_blitz_snaps": len(d_blitz),
                "d_pressure_no_blitz": d_pnb,
                
                # Pass Depth Defense
                "d_att_blos": len(d_blos),
                "d_yds_blos": float(d_blos["yards_gained"].sum()),
                "d_epa_blos": float(d_blos["epa"].sum()),
                "d_sr_blos": float(d_blos["success"].sum()),
                
                "d_att_short": len(d_short),
                "d_yds_short": float(d_short["yards_gained"].sum()),
                "d_epa_short": float(d_short["epa"].sum()),
                "d_sr_short": float(d_short["success"].sum()),
                
                "d_att_inter": len(d_inter),
                "d_yds_inter": float(d_inter["yards_gained"].sum()),
                "d_epa_inter": float(d_inter["epa"].sum()),
                "d_sr_inter": float(d_inter["success"].sum()),
                
                "d_att_deep": len(d_deep),
                "d_yds_deep": float(d_deep["yards_gained"].sum()),
                "d_epa_deep": float(d_deep["epa"].sum()),
                "d_sr_deep": float(d_deep["success"].sum()),
                
                # Pass Depth Offense
                "o_att_blos": len(o_blos),
                "o_yds_blos": float(o_blos["yards_gained"].sum()),
                "o_epa_blos": float(o_blos["epa"].sum()),
                "o_sr_blos": float(o_blos["success"].sum()),
                
                "o_att_short": len(o_short),
                "o_yds_short": float(o_short["yards_gained"].sum()),
                "o_epa_short": float(o_short["epa"].sum()),
                "o_sr_short": float(o_short["success"].sum()),
                
                "o_att_inter": len(o_inter),
                "o_yds_inter": float(o_inter["yards_gained"].sum()),
                "o_epa_inter": float(o_inter["epa"].sum()),
                "o_sr_inter": float(o_inter["success"].sum()),
                
                "o_att_deep": len(o_deep),
                "o_yds_deep": float(o_deep["yards_gained"].sum()),
                "o_epa_deep": float(o_deep["epa"].sum()),
                "o_sr_deep": float(o_deep["success"].sum()),
                
                # Positional Targets Allowed
                "d_tgt_wr": d_tgt_wr,
                "d_yds_wr": d_yds_wr,
                "d_tgt_te": d_tgt_te,
                "d_yds_te": d_yds_te,
                "d_tgt_rb": d_tgt_rb,
                "d_yds_rb": d_yds_rb,
                
                # Gaps
                "d_c_ins": d_c_ins,
                "d_y_ins": d_y_ins,
                "d_c_out": d_c_out,
                "d_y_out": d_y_out,
                "o_c_ins": o_c_ins,
                "o_y_ins": o_y_ins,
                "o_c_out": o_c_out,
                "o_y_out": o_y_out,
            })

    out_df = pd.DataFrame(records)
    print(f"Generated scouting features DataFrame: {out_df.shape}")
    
    # Save to data/ and api/data/
    out_path_1 = os.path.join(DATA_DIR, "game_scouting_features.parquet")
    out_path_2 = os.path.join(API_DATA_DIR, "game_scouting_features.parquet")
    out_df.to_parquet(out_path_1, index=False)
    out_df.to_parquet(out_path_2, index=False)
    
    size_kb = os.path.getsize(out_path_1) / 1024
    print(f"Saved {out_path_1} and {out_path_2} ({size_kb:.1f} KB). Successfully compiled!")


if __name__ == "__main__":
    main()
