"""
backend/synthesis.py

Part 6: Matchup Synthesis & Tactical Wrap-Up Engine.
Generates testable matchup summary and film-room contingency notes:
- Notable matchup stats in this game, sorted by difference from league median
- Each stat displays: metric, size vs. league average (in SD), and sample-size flag
- Includes a dedicated 'What Would Change This' contingency section (injuries, game script shifts)
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional


def build_matchup_synthesis(
    cov_eval_away: Dict[str, Any],
    cov_eval_home: Dict[str, Any],
    rush_eval_away: Dict[str, Any],
    rush_eval_home: Dict[str, Any],
    tempo_eval: Dict[str, Any],
    key_players: Optional[List[Dict[str, Any]]] = None,
    key_picks: Optional[List[Dict[str, Any]]] = None,
    injury_context: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Synthesizes the analytical pillars into notable matchup stats and contingencies.
    """
    players_list = key_players if key_players is not None else (key_picks or [])
    stats = []

    # 1. Rushing / Trench Matchup Stat
    rush_z_away = abs(rush_eval_away.get("composite_z", 0.0))
    rush_z_home = abs(rush_eval_home.get("composite_z", 0.0))

    off_away = rush_eval_away.get("offense_team") or rush_eval_away.get("team", "Away")
    def_away = rush_eval_away.get("defense_team") or cov_eval_away.get("defense_team", "Home")
    off_home = rush_eval_home.get("offense_team") or rush_eval_home.get("team", "Home")
    def_home = rush_eval_home.get("defense_team") or cov_eval_home.get("defense_team", "Away")

    if rush_z_away >= rush_z_home and rush_z_away >= 0.40:
        stats.append({
            "pillar": "Run Game & Fronts",
            "metric": f"{off_away} Ground Push vs. {def_away} Front",
            "size_vs_league": f"{rush_eval_away.get('composite_z', 0.0):+.2f} SD vs. League Avg",
            "confidence": "Low Sample (n < 40)" if rush_eval_away.get("is_low_sample") else "Robust Sample",
            "detail": rush_eval_away.get("diagnostic", ""),
            "importance": rush_z_away,
        })
    elif rush_z_home >= 0.40:
        stats.append({
            "pillar": "Run Game & Fronts",
            "metric": f"{off_home} Ground Push vs. {def_home} Front",
            "size_vs_league": f"{rush_eval_home.get('composite_z', 0.0):+.2f} SD vs. League Avg",
            "confidence": "Low Sample (n < 40)" if rush_eval_home.get("is_low_sample") else "Robust Sample",
            "detail": rush_eval_home.get("diagnostic", ""),
            "importance": rush_z_home,
        })
    else:
        stats.append({
            "pillar": "Run Game & Fronts",
            "metric": f"Balanced Ground Attack: {off_away} vs. {def_away}",
            "size_vs_league": "Neutral Trench Matchup",
            "confidence": "Standard Sample",
            "detail": "Neither rushing front holds a dominant mismatch at the line of scrimmage.",
            "importance": 0.20,
        })

    # 2. Coverage Secondary Stat
    def_tm_away = cov_eval_away.get("defense_team", "Defense")
    def_tm_home = cov_eval_home.get("defense_team", "Defense")
    shell_away = cov_eval_away.get("shell_label") or ("MFO Two-High" if cov_eval_away.get("def_mfo_pct", 50) >= 50 else "MFC Single-High")
    shell_home = cov_eval_home.get("shell_label") or ("MFO Two-High" if cov_eval_home.get("def_mfo_pct", 50) >= 50 else "MFC Single-High")

    man_away = cov_eval_away.get("man_rate") or cov_eval_away.get("def_man_pct", 35.0) / 100.0
    man_home = cov_eval_home.get("man_rate") or cov_eval_home.get("def_man_pct", 35.0) / 100.0
    mfo_away = cov_eval_away.get("mfo_rate") or cov_eval_away.get("def_mfo_pct", 50.0) / 100.0
    mfo_home = cov_eval_home.get("mfo_rate") or cov_eval_home.get("def_mfo_pct", 50.0) / 100.0

    cov_importance = max(abs(man_away - 0.35), abs(man_home - 0.35)) * 4.0
    stats.append({
        "pillar": "Coverage Structure",
        "metric": f"Secondary Shells: {def_tm_away} ({shell_away}) vs. {def_tm_home} ({shell_home})",
        "size_vs_league": f"MFO Split: {mfo_away:.0%} & {mfo_home:.0%}",
        "confidence": "Low Sample" if cov_eval_away.get("is_low_sample") else "Robust Sample",
        "detail": (
            "Two-high split safeties cap perimeter deep passes and direct targets inside to slot/tight end options."
            if mfo_away >= 0.50 and mfo_home >= 0.50 else
            "Single-high closed shells feature 1-on-1 boundary coverage for perimeter wide receivers."
        ),
        "importance": cov_importance,
    })

    # 3. Tempo / Environment Stat
    combined_plays = tempo_eval.get("combined_avg_plays", tempo_eval.get("combined_proj_plays", 128.0))
    pace_importance = abs(combined_plays - 128.0) / 4.0
    stats.append({
        "pillar": "Pace & Environment",
        "metric": f"Game Script: {tempo_eval.get('pace_tag', 'Balanced Pace')} (~{combined_plays} Plays)",
        "size_vs_league": f"{tempo_eval.get('spread_line', 0.0):+.1f} Spread / {tempo_eval.get('total_line', 45.0):.1f} Total",
        "confidence": "Neutral Script Baseline",
        "detail": tempo_eval.get("script_note", tempo_eval.get("volume_override_verdict", "")),
        "importance": pace_importance,
    })

    # Sort notable stats strictly by deviation from league median (importance)
    stats.sort(key=lambda x: x["importance"], reverse=True)

    # Contingencies: What would change this
    contingencies = [
        "Early two-score deficit: If a trailing team falls behind by 14+ points, neutral ground scripts are typically abandoned for dropbacks.",
        "Turnover-induced field position: Short-field drives reduce play volume while concentrating red-zone touch conversion.",
    ]
    if injury_context:
        for inj in injury_context[:2]:
            contingencies.append(f"Injury Alert: {inj.get('player')} ({inj.get('position')}) status ({inj.get('status')}) reflects altered player availability.")

    return {
        "notable_matchup_stats": stats,
        "ranked_drivers": stats,
        "contingencies": contingencies,
        "key_players_summary": [
            f"{p.get('player')} ({p.get('team')} {p.get('pos')}): {p.get('stat_focus', '')} - {p.get('last_10_stats', {}).get('summary_str', p.get('season_stat', ''))}"
            for p in players_list
        ],
    }
