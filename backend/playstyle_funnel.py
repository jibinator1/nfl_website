"""
backend/playstyle_funnel.py

Playstyle & Player Archetype Funnel Engine.
Translates raw defensive shells and offensive tendencies into actionable football scouting:
1. Section 1 — Team Playstyle Identity & Schematic Funnel (Both Sides):
   - Defensive Shell & Funnel Profile (MFO Two-High vs. MFC Single-High, Zone vs. Man, Blitz, Underneath comp %, Deep rate allowed, QB scramble yds allowed).
   - Offensive Playstyle Profile (Underneath <10 air yds %, Deep shot %, Shotgun %, Neutral pass split, Run Gap vs. Zone, Mobile QB threat).
   - Schematic Funnel Translation ("What This Opens Up"):
     * Deep-Shell Underneath Funnel (The Puka Underneath Rule)
     * QB Scramble & Rush Lane Exposure (The Josh Allen Scramble Rule)
     * Hot-Read & Checkdown Funnel (The TE / RB Checkdown Rule)
     * Boundary X-Receiver Isolation (The Outside Vertical Rule)
     * Ground Scheme Attack (Gap Grinder vs. Outside Zone Slasher)
2. Section 2 — Player Archetypes to Target (Matchup Beneficiaries):
   - Concrete "Who Benefits & Why" mapped to active roster players (Slot WR, Mobile QB, Pass-catching RB, Seam TE, Lead Rusher).
   - Integrated with Alpha Usage Concentration (Target Share >= 22%, Rush Share >= 45%, RZ Touch Share >= 35%).
3. Section 3 — Historical Stylistic Analogs:
   - Multivariate production allowed to top-8 stylistic clones (RB rush yds, Pass yds, WR rec yds, TE rec yds, QB rush yds).
   - Direct Head-to-Head games.
"""
from __future__ import annotations

import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from backend.scheme_insights import STARTING_QBS, get_team_scheme_profile, TEAM_NAMES
from backend.team_tendencies import load_clean_pbp, get_player_position_map
from backend.roster_verification import get_roster_profiles
from backend.similar_opponents import find_similar_opponent_games


# -----------------------------------------------------------------------------
# 1. Defensive Shell & Funnel Profile
# -----------------------------------------------------------------------------
def compute_defensive_funnel(
    def_team: str,
    season: int = 2026,
    up_to_week: Optional[int] = None,
    pbp_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Computes a defense's coverage shell tendencies and where they bleed production:
    - Coverage Shell: MFO (Two-High) vs MFC (Single-High), Zone vs Man, Blitz.
    - Funnel Bleed: Throws <10 air yds (completion % and volume share), deep shots allowed,
      RB target share, TE target share, and QB scramble/rush yards allowed per game.
    """
    t = def_team.upper()
    prof = get_team_scheme_profile(t)
    cov = prof.get("coverage", {})
    run_s = prof.get("run_scheme", {})

    zone_pct = float(cov.get("zone_pct", 65.0))
    man_pct = float(cov.get("man_pct", 35.0))
    mfo_pct = float(cov.get("mfo_pct", 50.0))
    mfc_pct = float(cov.get("mfc_pct", 50.0))
    blitz_pct = float(cov.get("blitz_pct", 25.0))
    def_gap_ypc = float(run_s.get("def_gap_ypc_allowed", 4.20))
    def_zone_ypc = float(run_s.get("def_zone_ypc_allowed", 4.15))

    if mfo_pct >= 50.0:
        shell_archetype = "Two-High / Split-Safety Shell (Cover 2 / Cover 4 Quarters)"
    else:
        shell_archetype = "Single-High / Loaded Box (Cover 1 / Cover 3)"

    # Pull empirical PBP metrics
    df = pbp_df if pbp_df is not None else load_clean_pbp(season=season, up_to_week=up_to_week)
    pos_map = get_player_position_map()

    def_df = df[df["defteam"] == t].copy() if not df.empty and "defteam" in df.columns else pd.DataFrame()

    if not def_df.empty:
        def_df["_rec_pos"] = def_df["receiver_player_id"].map(pos_map).fillna("Unknown")
        def_df["_rush_pos"] = def_df["rusher_player_id"].map(pos_map).fillna("Unknown")
        passes = def_df[def_df["play_type"] == "pass"]
        tot_passes = len(passes)

        if tot_passes >= 10:
            short_p = passes[passes["air_yards"] < 10]
            short_share = round(len(short_p) / tot_passes * 100.0, 1)
            short_comp_pct = round(float(short_p["complete_pass"].fillna(0).mean()) * 100.0, 1)

            deep_p = passes[passes["air_yards"] >= 20]
            deep_share = round(len(deep_p) / tot_passes * 100.0, 1)

            rb_p = passes[passes["_rec_pos"] == "RB"]
            rb_share = round(len(rb_p) / tot_passes * 100.0, 1)

            te_p = passes[passes["_rec_pos"] == "TE"]
            te_share = round(len(te_p) / tot_passes * 100.0, 1)
        else:
            short_share = 62.0
            short_comp_pct = 75.0
            deep_share = 13.0
            rb_share = 16.0
            te_share = 18.0

        qb_runs = def_df[(def_df["play_type"] == "run") & ((def_df["_rush_pos"] == "QB") | (def_df.get("qb_scramble", 0) == 1))]
        n_g = max(int(def_df["game_id"].nunique()), 1)
        qb_rush_ypg = round(float(qb_runs["yards_gained"].sum()) / n_g, 1)
        qb_rush_att_pg = round(float(len(qb_runs)) / n_g, 1)
    else:
        short_share = 62.0
        short_comp_pct = 75.0
        deep_share = 13.0
        rb_share = 16.0
        te_share = 18.0
        qb_rush_ypg = 22.0
        qb_rush_att_pg = 3.0
        n_g = 1

    summary_str = (
        f"{t} Defense: {shell_archetype} — {mfo_pct:.0f}% MFO (Two-High), {zone_pct:.0f}% Zone, {blitz_pct:.0f}% Blitz; "
        f"funnels {short_share:.0f}% of passes underneath <10 air yds ({short_comp_pct:.1f}% comp); concedes {qb_rush_ypg:.1f} QB rush yds/g."
    )

    return {
        "team": t,
        "team_name": TEAM_NAMES.get(t, t),
        "coverage_shell": {
            "mfo_pct": mfo_pct,
            "mfc_pct": mfc_pct,
            "zone_pct": zone_pct,
            "man_pct": man_pct,
            "blitz_pct": blitz_pct,
            "shell_archetype": shell_archetype,
        },
        "funnel_bleed": {
            "short_target_share": short_share,
            "short_comp_pct": short_comp_pct,
            "deep_shot_share_allowed": deep_share,
            "rb_target_share_allowed": rb_share,
            "te_target_share_allowed": te_share,
            "qb_rush_ypg_allowed": qb_rush_ypg,
            "qb_rush_att_pg_allowed": qb_rush_att_pg,
            "gap_ypc_allowed": def_gap_ypc,
            "zone_ypc_allowed": def_zone_ypc,
        },
        "summary": summary_str,
        "n_games": n_g,
    }


# -----------------------------------------------------------------------------
# 2. Offensive Playstyle Profile
# -----------------------------------------------------------------------------
def compute_offensive_playstyle(
    off_team: str,
    season: int = 2026,
    up_to_week: Optional[int] = None,
    pbp_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Computes an offense's playstyle fingerprint:
    - Dropback & Depth: Quick-game <10 air yds %, Deep shot %, Neutral pass %, Shotgun %.
    - Run Identity: Gap % vs Zone %, primary ground concept.
    - QB Threat: Scramble / designed rushing volume and mobility status.
    """
    t = off_team.upper()
    prof = get_team_scheme_profile(t)
    run_s = prof.get("run_scheme", {})
    targ_s = prof.get("target_alignment", {})

    off_gap_pct = float(run_s.get("offense_gap_pct", 45.0))
    off_zone_pct = float(run_s.get("offense_zone_pct", 55.0))

    if off_gap_pct >= 48.0:
        primary_run = "Gap / Power heavy"
        primary_run_concept = "Gap"
    else:
        primary_run = "Zone / Stretch heavy"
        primary_run_concept = "Zone"

    qb_name, is_mobile, rush_yds_pg, rush_att_pg = STARTING_QBS.get(t, (f"{t} QB", False, 10.0, 3.0))

    df = pbp_df if pbp_df is not None else load_clean_pbp(season=season, up_to_week=up_to_week)
    off_df = df[df["posteam"] == t].copy() if not df.empty and "posteam" in df.columns else pd.DataFrame()

    if not off_df.empty:
        passes = off_df[off_df["play_type"] == "pass"]
        tot_passes = len(passes)

        if tot_passes >= 10:
            short_p = passes[passes["air_yards"] < 10]
            short_share = round(len(short_p) / tot_passes * 100.0, 1)

            deep_p = passes[passes["air_yards"] >= 20]
            deep_share = round(len(deep_p) / tot_passes * 100.0, 1)
        else:
            short_share = float(targ_s.get("short_pct", 65.0))
            deep_share = float(targ_s.get("deep_pct", 15.0))

        shotgun_pct = round(float(off_df["shotgun"].fillna(0).mean()) * 100.0, 1)

        neut = off_df[off_df["score_differential"].abs() <= 6]
        if len(neut) >= 10:
            neutral_pass_pct = round(float((neut["play_type"] == "pass").mean()) * 100.0, 1)
        else:
            neutral_pass_pct = round(float((off_df["play_type"] == "pass").mean()) * 100.0, 1)

        n_g = max(int(off_df["game_id"].nunique()), 1)
    else:
        short_share = float(targ_s.get("short_pct", 65.0))
        deep_share = float(targ_s.get("deep_pct", 15.0))
        shotgun_pct = 70.0
        neutral_pass_pct = 57.0
        n_g = 1

    if is_mobile or rush_att_pg >= 3.5:
        qb_summary = f"{qb_name} is an elite mobile rushing/scramble weapon ({rush_att_pg:.1f} att/g, {rush_yds_pg:.1f} yds/g)"
    else:
        qb_summary = f"{qb_name} operates primarily from the pocket ({rush_att_pg:.1f} att/g)"

    summary_str = (
        f"{t} Offense: High short-area target concentration ({short_share:.0f}% <10 air yds, {deep_share:.0f}% deep shots); "
        f"{primary_run} ({off_gap_pct:.1f}% Gap / {off_zone_pct:.1f}% Zone); {qb_summary}."
    )

    return {
        "team": t,
        "team_name": TEAM_NAMES.get(t, t),
        "dropback_profile": {
            "pass_short_under_10_pct": short_share,
            "deep_shot_rate": deep_share,
            "neutral_pass_pct": neutral_pass_pct,
            "shotgun_pct": shotgun_pct,
        },
        "run_identity": {
            "gap_pct": off_gap_pct,
            "zone_pct": off_zone_pct,
            "primary_run": primary_run,
            "primary_run_concept": primary_run_concept,
        },
        "qb_threat": {
            "qb_name": qb_name,
            "is_mobile": is_mobile,
            "rush_att_per_game": rush_att_pg,
            "rush_yds_per_game": rush_yds_pg,
            "summary": qb_summary,
        },
        "summary": summary_str,
        "n_games": n_g,
    }


# -----------------------------------------------------------------------------
# 3. Schematic Funnel & "What This Opens Up" Translation
# -----------------------------------------------------------------------------
def evaluate_schematic_funnels(
    off_playstyle: Dict[str, Any],
    def_funnel: Dict[str, Any],
    off_team: str,
    def_team: str,
) -> List[Dict[str, Any]]:
    """
    Translates defensive coverage shells and pass-rush habits into concrete
    football cause-and-effect ("What This Opens Up").
    """
    funnels: List[Dict[str, Any]] = []

    def_cov = def_funnel.get("coverage_shell", {})
    def_bleed = def_funnel.get("funnel_bleed", {})
    off_drop = off_playstyle.get("dropback_profile", {})
    off_run = off_playstyle.get("run_identity", {})
    off_qb = off_playstyle.get("qb_threat", {})

    # Rule 1: The "Puka Underneath" Rule (Deep-Shell / Soft Zone Funnel)
    # Trigger: Two-High (MFO >= 48%) or Zone >= 62% AND defense funnels high underneath completion/volume
    if (def_cov.get("mfo_pct", 50) >= 48.0 or def_cov.get("zone_pct", 65) >= 62.0) and (
        def_bleed.get("short_comp_pct", 70) >= 72.0 or def_bleed.get("short_target_share", 50) >= 54.0
    ):
        funnels.append({
            "rule_id": "DEEP_SHELL_UNDERNEATH",
            "archetype_key": "SLOT_UNDERNEATH_WR",
            "archetype_title": "Slot / Underneath PPR Receiver",
            "name": "Deep-Shell Underneath Funnel",
            "priority": 1,
            "takeaway": (
                f"{def_team} plays top-down coverage ({def_cov.get('mfo_pct', 50):.0f}% Two-High MFO, {def_cov.get('zone_pct', 65):.0f}% Zone) "
                f"to cap vertical shots, conceding {def_bleed.get('short_comp_pct', 75):.1f}% completions on throws <10 air yds. "
                f"Funnels heavy target volume and YAC opportunities to slot receivers and underneath separators."
            ),
        })

    # Rule 2: The "Josh Allen Scramble" Rule (Man Coverage / Pass-Rush Lane Funnel)
    # Trigger: High Man (>= 28%), High Blitz (>= 25%), or Conceding >= 25 QB rush yds/g AND QB is mobile
    if off_qb.get("is_mobile") and (
        def_cov.get("man_pct", 30) >= 28.0 or def_cov.get("blitz_pct", 25) >= 24.0 or def_bleed.get("qb_rush_ypg_allowed", 20) >= 24.0
    ):
        funnels.append({
            "rule_id": "QB_SCRAMBLE_LANES",
            "archetype_key": "MOBILE_QB",
            "archetype_title": "Mobile QB Rushing / Scramble Threat",
            "name": "QB Scramble & Rush Lane Exposure",
            "priority": 2,
            "takeaway": (
                f"{def_team} defense ({def_cov.get('man_pct', 30):.0f}% Man, {def_cov.get('blitz_pct', 25):.0f}% Blitz) "
                f"turns defensive backs' eyes downfield and vacates underneath second-level zones, conceding {def_bleed.get('qb_rush_ypg_allowed', 25):.1f} QB rush yds/game. "
                f"Creates wide-open scramble and designed running lanes for {off_qb.get('qb_name')}."
            ),
        })

    # Rule 3: The "TE / RB Checkdown" Rule (Blitz / Pressure Funnel)
    # Trigger: High Blitz (>= 25%) or High RB target share allowed (>= 17%) or High TE target share (>= 18%)
    if def_cov.get("blitz_pct", 25) >= 24.0 or def_bleed.get("rb_target_share_allowed", 16) >= 17.0 or def_bleed.get("te_target_share_allowed", 18) >= 19.0:
        funnels.append({
            "rule_id": "HOT_READ_CHECKDOWN",
            "archetype_key": "PASS_CATCHING_RB_TE",
            "archetype_title": "Receiving Back / Seam TE",
            "name": "Hot-Read & Checkdown Funnel",
            "priority": 3,
            "takeaway": (
                f"{def_team}'s pressure rate ({def_cov.get('blitz_pct', 25):.0f}% Blitz) forces immediate hot reads, "
                f"funneling rapid checkdowns to running backs ({def_bleed.get('rb_target_share_allowed', 16):.1f}% target share conceded) "
                f"and intermediate seams to tight ends."
            ),
        })

    # Rule 4: The "Outside Vertical" Rule (Single-High / Man Funnel)
    # Trigger: High Single-High (MFC >= 48%) and Man >= 28%
    if def_cov.get("mfc_pct", 50) >= 48.0 and def_cov.get("man_pct", 30) >= 28.0:
        funnels.append({
            "rule_id": "BOUNDARY_ISOLATION",
            "archetype_key": "BOUNDARY_X_WR",
            "archetype_title": "Boundary X-Receiver / Deep Shot Threat",
            "name": "Boundary X-Receiver Isolation Window",
            "priority": 4,
            "takeaway": (
                f"{def_team} lives in Single-High ({def_cov.get('mfc_pct', 50):.0f}% MFC) and aggressive Man coverage, "
                f"leaving boundary cornerbacks in true 1-on-1 isolation without over-the-top safety help."
            ),
        })

    # Rule 5: Ground Scheme Exploitation (Gap vs Zone)
    primary_concept = off_run.get("primary_run_concept", "Gap")
    if primary_concept == "Gap" and def_bleed.get("gap_ypc_allowed", 4.2) >= 4.20:
        funnels.append({
            "rule_id": "GROUND_GAP_CLASH",
            "archetype_key": "GAP_GRINDER_RB",
            "archetype_title": "North-South Gap Grinder",
            "name": "Interior Gap / Power Attack Exploitation",
            "priority": 5,
            "takeaway": (
                f"{off_team} leans heavily into downhill Gap/Power concepts ({off_run.get('gap_pct', 50):.1f}%) "
                f"directly into a {def_team} interior front conceding {def_bleed.get('gap_ypc_allowed', 4.2):.2f} YPC on Gap carries."
            ),
        })
    elif primary_concept == "Zone" and def_bleed.get("zone_ypc_allowed", 4.15) >= 4.20:
        funnels.append({
            "rule_id": "GROUND_ZONE_CLASH",
            "archetype_key": "ZONE_SLASHER_RB",
            "archetype_title": "Outside Zone Slasher",
            "name": "Perimeter Zone / Stretch Attack Exploitation",
            "priority": 5,
            "takeaway": (
                f"{off_team} establishes outside Zone/Stretch rushing ({off_run.get('zone_pct', 50):.1f}%) "
                f"against a {def_team} defense allowing {def_bleed.get('zone_ypc_allowed', 4.15):.2f} YPC on Zone carries."
            ),
        })

    funnels.sort(key=lambda x: x["priority"])
    return funnels


# -----------------------------------------------------------------------------
# 4. Player Archetypes to Target Matcher
# -----------------------------------------------------------------------------
def match_target_archetypes(
    funnels: List[Dict[str, Any]],
    off_team: str,
    def_team: str,
    season: int = 2026,
    up_to_week: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Matches triggered schematic funnels to real, active skill players on off_team.
    Returns concrete 'Who Feasts & Why' recommendations.
    """
    t = off_team.upper()
    active_roster = get_roster_profiles(t, season=season, up_to_week=up_to_week)
    scheme_prof = get_team_scheme_profile(t)
    key_targets = scheme_prof.get("target_alignment", {}).get("key_targets", [])
    qb_name, is_mobile, rush_yds_pg, rush_att_pg = STARTING_QBS.get(t, (f"{t} QB", False, 10.0, 3.0))

    # Helper maps
    slot_leaders = {kt["name"]: kt.get("slot_pct", 0) for kt in key_targets if "slot_pct" in kt}
    outside_leaders = {kt["name"]: kt.get("outside_pct", 0) for kt in key_targets if "outside_pct" in kt}

    targets: List[Dict[str, Any]] = []
    seen_players = set()

    for fn in funnels:
        arch_key = fn.get("archetype_key")
        arch_title = fn.get("archetype_title")

        # 1. Slot / Underneath Receiver
        if arch_key == "SLOT_UNDERNEATH_WR":
            # Find best slot / underneath receiver
            best_p = None
            best_score = -1.0
            for p in active_roster:
                pos = p.get("pos")
                name = p.get("player_display_name", "")
                if pos in ("WR", "TE"):
                    t_share = p.get("target_share") or 0.0
                    slot_pct = slot_leaders.get(name, 35.0)
                    score = t_share * 100.0 + (slot_pct * 0.4)
                    if score > best_score:
                        best_score = score
                        best_p = p

            if best_p:
                p_name = best_p.get("player_display_name")
                pos = best_p.get("pos")
                t_pct = (best_p.get("target_share") or 0.0) * 100.0
                slot_info = f", {slot_leaders[p_name]}% slot alignment" if p_name in slot_leaders else ""
                targets.append({
                    "player": p_name,
                    "team": t,
                    "pos": pos,
                    "archetype": arch_title,
                    "role_context": f"{t_pct:.1f}% Target Share{slot_info}",
                    "matchup_rationale": f"Commands primary underneath target volume against {def_team}'s soft two-high zone shell that concedes high-percentage short completions.",
                })
                seen_players.add(p_name)

        # 2. Mobile QB Scramble / Rushing
        elif arch_key == "MOBILE_QB":
            if is_mobile or rush_att_pg >= 3.5:
                targets.append({
                    "player": qb_name,
                    "team": t,
                    "pos": "QB",
                    "archetype": arch_title,
                    "role_context": f"{rush_att_pg:.1f} Carries/g • {rush_yds_pg:.1f} Rush Yds/g",
                    "matchup_rationale": f"Faces {def_team}'s heavy man/blitz concepts where defenders turn their backs downfield, opening wide scramble lanes.",
                })
                seen_players.add(qb_name)

        # 3. Receiving Back / Seam TE
        elif arch_key == "PASS_CATCHING_RB_TE":
            # Prioritize TE or pass-catching RB
            best_te = next((p for p in active_roster if p.get("pos") == "TE" and (p.get("target_share") or 0.0) >= 0.12), None)
            best_rb = next((p for p in active_roster if p.get("pos") == "RB" and (p.get("target_share") or 0.0) >= 0.08), None)
            candidate = best_te or best_rb

            if candidate and candidate.get("player_display_name") not in seen_players:
                p_name = candidate.get("player_display_name")
                pos = candidate.get("pos")
                t_pct = (candidate.get("target_share") or 0.0) * 100.0
                targets.append({
                    "player": p_name,
                    "team": t,
                    "pos": pos,
                    "archetype": arch_title,
                    "role_context": f"{t_pct:.1f}% Target Share",
                    "matchup_rationale": f"Serves as the primary quick-release safety valve against {def_team}'s aggressive blitz and pressure front.",
                })
                seen_players.add(p_name)

        # 4. Boundary X-Receiver
        elif arch_key == "BOUNDARY_X_WR":
            best_wr = None
            best_score = -1.0
            for p in active_roster:
                if p.get("pos") == "WR":
                    name = p.get("player_display_name", "")
                    out_pct = outside_leaders.get(name, 50.0)
                    t_share = p.get("target_share") or 0.0
                    score = t_share * 100.0 + (out_pct * 0.3)
                    if score > best_score and name not in seen_players:
                        best_score = score
                        best_wr = p

            if best_wr:
                p_name = best_wr.get("player_display_name")
                t_pct = (best_wr.get("target_share") or 0.0) * 100.0
                out_info = f", {outside_leaders[p_name]}% perimeter alignment" if p_name in outside_leaders else ""
                targets.append({
                    "player": p_name,
                    "team": t,
                    "pos": "WR",
                    "archetype": arch_title,
                    "role_context": f"{t_pct:.1f}% Target Share{out_info}",
                    "matchup_rationale": f"Isolated on perimeter vertical routes with true 1-on-1 boundary matchups against {def_team}'s single-high safety coverage.",
                })
                seen_players.add(p_name)

        # 5. Lead Rusher (Gap Grinder / Zone Slasher)
        elif arch_key in ("GAP_GRINDER_RB", "ZONE_SLASHER_RB"):
            lead_rb = next((p for p in active_roster if p.get("pos") == "RB" and (p.get("rush_share") or 0.0) >= 0.40), None)
            if lead_rb and lead_rb.get("player_display_name") not in seen_players:
                p_name = lead_rb.get("player_display_name")
                r_pct = (lead_rb.get("rush_share") or 0.0) * 100.0
                rz_pct = (lead_rb.get("rz_touch_share") or 0.0) * 100.0
                targets.append({
                    "player": p_name,
                    "team": t,
                    "pos": "RB",
                    "archetype": arch_title,
                    "role_context": f"{r_pct:.1f}% Rush Share • {rz_pct:.1f}% RZ Touch Share",
                    "matchup_rationale": f"Workhorse ball-carrier directly attacking {def_team}'s primary front weakness in neutral and positive scripts.",
                })
                seen_players.add(p_name)

    return targets


# -----------------------------------------------------------------------------
# 5. Full Matchup Assembly (The 3-Section Engine)
# -----------------------------------------------------------------------------
def build_playstyle_funnel_matchup(
    away_team: str,
    home_team: str,
    season: int = 2026,
    up_to_week: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Assembles the 3-section Playstyle & Player Archetype Funnel report:
      Section 1 — Team Playstyle Identity & Schematic Funnel (Both Sides)
      Section 2 — Player Archetypes to Target (Matchup Beneficiaries & Usage Concentration)
      Section 3 — Historical Stylistic Analogs (Multivariate Production Allowed & H2H)
    """
    a_team = away_team.upper()
    h_team = home_team.upper()

    pbp = load_clean_pbp(season=season, up_to_week=up_to_week)

    # 1. Compute profiles for both sides
    away_off = compute_offensive_playstyle(a_team, season=season, up_to_week=up_to_week, pbp_df=pbp)
    home_def = compute_defensive_funnel(h_team, season=season, up_to_week=up_to_week, pbp_df=pbp)

    home_off = compute_offensive_playstyle(h_team, season=season, up_to_week=up_to_week, pbp_df=pbp)
    away_def = compute_defensive_funnel(a_team, season=season, up_to_week=up_to_week, pbp_df=pbp)

    # 2. Evaluate schematic funnels
    away_funnels = evaluate_schematic_funnels(away_off, home_def, a_team, h_team)
    home_funnels = evaluate_schematic_funnels(home_off, away_def, h_team, a_team)

    # 3. Match player archetypes to target
    away_targets = match_target_archetypes(away_funnels, a_team, h_team, season=season, up_to_week=up_to_week)
    home_targets = match_target_archetypes(home_funnels, h_team, a_team, season=season, up_to_week=up_to_week)
    all_targets = away_targets + home_targets

    # 4. Concentrated Alpha usage
    away_roster = get_roster_profiles(a_team, season=season, up_to_week=up_to_week)
    home_roster = get_roster_profiles(h_team, season=season, up_to_week=up_to_week)
    alpha_usage = [p for p in (away_roster + home_roster) if p.get("is_alpha")]

    # 5. Stylistic analogs with multivariate production
    home_sim = find_similar_opponent_games(a_team, h_team, season=season, up_to_week=up_to_week, n_similar=5)
    away_sim = find_similar_opponent_games(h_team, a_team, season=season, up_to_week=up_to_week, n_similar=5)

    return {
        "away_team": a_team,
        "home_team": h_team,
        "season": season,
        "up_to_week": up_to_week,
        # Section 1: Playstyle & Schematic Funnel
        "section_1_playstyle_matchup": {
            "away_off_vs_home_def": {
                "offense_team": a_team,
                "defense_team": h_team,
                "offense_playstyle": away_off,
                "defense_funnel": home_def,
                "funnel_insights": away_funnels,
            },
            "home_off_vs_away_def": {
                "offense_team": h_team,
                "defense_team": a_team,
                "offense_playstyle": home_off,
                "defense_funnel": away_def,
                "funnel_insights": home_funnels,
            },
        },
        # Section 2: Player Archetypes to Target
        "section_2_archetypes_to_target": {
            "beneficiary_players": all_targets,
            "away_targets": away_targets,
            "home_targets": home_targets,
            "concentrated_alpha_usage": alpha_usage,
        },
        # Section 3: Historical Stylistic Analogs
        "section_3_historical_analogs": {
            "home_defense_vs_similar_away": home_sim,
            "away_defense_vs_similar_home": away_sim,
        },
    }
