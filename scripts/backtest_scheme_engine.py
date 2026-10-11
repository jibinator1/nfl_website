"""
NFL Scheme Insights Engine Backtesting Framework
=================================================
Validates the calibrated probability, pricing, and breakout recommendations
against historical game outcomes across multiple seasons (2023, 2024, 2025).

Metrics computed:
1. Empirical Hit Rate vs Model Win Probability (Calibration)
2. Wilson 95% Confidence Intervals per Tier and Stat Focus
3. Brier Score (Mean Squared Probability Error)
4. Flat Stake ROI at standard -110 juice (implied 52.38%)
5. Breakdown by Confidence Tier (Grade A, Grade B, Grade C)
6. Breakdown by Stat Focus (Receiving Yards, Receptions, Rushing Yards)
7. Fits and persists post-hoc isotonic calibration curves
"""

import os
import sys
import json
import math
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.scheme_insights import compute_scheme_insights
from backend.probability_calibration import fit_calibration


def wilson_ci(wins: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Computes Wilson score interval with continuity correction for binomial proportions."""
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2.0 * n)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / n + z**2 / (4.0 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def run_backtest(seasons: List[int] = [2023, 2024, 2025], max_weeks: int = 18):
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    sched_path = os.path.join(data_dir, 'schedules_cache.parquet')
    week_path = os.path.join(data_dir, 'weekly_cache.parquet')

    if not os.path.exists(sched_path) or not os.path.exists(week_path):
        print(f"Error: Parquet data files not found in {data_dir}")
        return

    schedules_df = pd.read_parquet(sched_path)
    weekly_df = pd.read_parquet(week_path)

    results = []
    seen_props = set()

    print(f"Starting Scheme Engine Backtest for seasons {seasons} (Weeks 1-{max_weeks})...")

    for season in seasons:
        season_sched = schedules_df[(schedules_df['season'] == season) & (schedules_df['home_score'].notna())]
        completed_weeks = sorted(season_sched['week'].dropna().unique())

        for wk in completed_weeks:
            if wk > max_weeks:
                continue

            wk_games = season_sched[season_sched['week'] == wk]
            for _, g in wk_games.iterrows():
                h_tm = g['home_team']
                a_tm = g['away_team']
                g_id = g['game_id']

                try:
                    insights = compute_scheme_insights(
                        home_team=h_tm,
                        away_team=a_tm,
                        season=season,
                        week=wk,
                        weekly_df=weekly_df,
                        schedules_df=schedules_df
                    )
                except Exception as e:
                    continue

                breakouts = insights.get('efficiency_breakouts', [])
                if not breakouts:
                    continue

                game_w = weekly_df[weekly_df['game_id'] == g_id]

                for b in breakouts:
                    p_name = b['player']
                    pos = b['pos']
                    stat_focus = b['stat_focus']
                    mkt_line = b['market_line_num']
                    fair_line = b['fair_line_num']
                    win_prob = b['win_prob']
                    tier = b['confidence_tier']
                    team = b['team']

                    # Deduplicate by (game_id, player, stat_focus)
                    dedup_key = (g_id, p_name, stat_focus)
                    if dedup_key in seen_props:
                        continue
                    seen_props.add(dedup_key)

                    # Find actual player row in this game
                    p_rows = game_w[(game_w['player_name'] == p_name) | (game_w['player_display_name'] == p_name)]
                    if p_rows.empty:
                        # Try partial / alias matching
                        short_p = p_name.split('.')[-1].strip()
                        p_rows = game_w[(game_w['team'] == team) & (game_w['player_name'].str.endswith(short_p))]

                    if p_rows.empty:
                        continue

                    actual_row = p_rows.iloc[0]

                    if stat_focus == 'Receiving Yards':
                        actual_val = float(actual_row.get('receiving_yards', 0.0) or 0.0)
                        is_over = True
                    elif stat_focus == 'Receptions':
                        actual_val = float(actual_row.get('receptions', 0.0) or 0.0)
                        is_over = True
                    elif stat_focus == 'Rushing Yards':
                        actual_val = float(actual_row.get('rushing_yards', 0.0) or 0.0)
                        is_over = ('Under' not in tier and 'Caution' not in tier)
                    else:
                        continue

                    if is_over:
                        won = 1 if actual_val > mkt_line else (0 if actual_val < mkt_line else -1)
                    else:
                        won = 1 if actual_val < mkt_line else (0 if actual_val > mkt_line else -1)

                    if won == -1:  # Push
                        profit = 0.0
                    elif won == 1:
                        profit = 100.0 / 110.0  # +0.909 units at -110
                    else:
                        profit = -1.0  # -1.0 unit loss

                    results.append({
                        'season': season,
                        'week': wk,
                        'game_id': g_id,
                        'player': p_name,
                        'team': team,
                        'pos': pos,
                        'stat_focus': stat_focus,
                        'market_line': mkt_line,
                        'fair_line': fair_line,
                        'actual_val': actual_val,
                        'win_prob': win_prob,
                        'confidence_tier': tier,
                        'won': won,
                        'profit': profit
                    })

    if not results:
        print("No matched props found for backtest.")
        return None

    df_res = pd.DataFrame(results)
    graded = df_res[df_res['won'] >= 0].copy()

    total_bets = len(graded)
    wins = int((graded['won'] == 1).sum())
    losses = int((graded['won'] == 0).sum())
    pushes = int((df_res['won'] == -1).sum())

    hit_rate = (wins / total_bets) * 100.0 if total_bets > 0 else 0.0
    avg_pred_prob = graded['win_prob'].mean() * 100.0
    brier_score = ((graded['win_prob'] - graded['won']) ** 2).mean()
    net_units = graded['profit'].sum()
    roi_pct = (net_units / total_bets) * 100.0 if total_bets > 0 else 0.0
    ov_lo, ov_hi = wilson_ci(wins, total_bets)

    print("\n" + "=" * 75)
    print("NFL SCHEME INSIGHTS ENGINE -- MULTI-SEASON HISTORICAL BACKTEST REPORT")
    print("=" * 75)
    print(f"Sample Seasons       : {seasons} (All {max_weeks} Weeks)")
    print(f"Total Bets Analyzed  : {total_bets} (Pushes: {pushes})")
    print(f"Overall Record       : {wins}W - {losses}L")
    print(f"Empirical Hit Rate   : {hit_rate:.2f}% (95% CI [{ov_lo*100:.1f}%, {ov_hi*100:.1f}%])")
    print(f"Average Model Win Prob: {avg_pred_prob:.2f}%")
    print(f"Calibration Delta    : {hit_rate - avg_pred_prob:+.2f}%")
    print(f"Brier Score          : {brier_score:.4f} (Benchmark: < 0.2500 is predictive)")
    print(f"Net Units (-110)     : {net_units:+.2f} units")
    print(f"Flat Stake ROI       : {roi_pct:+.2f}%")
    print("=" * 75)

    print("\n[+] BREAKDOWN BY CONFIDENCE TIER:")
    print("-" * 75)
    print(f"{'Tier':<32} {'Bets':<7} {'Record':<12} {'Hit Rate':<10} {'95% CI':<18} {'ROI%':<8}")
    print("-" * 75)
    for tier, t_df in graded.groupby('confidence_tier'):
        t_wins = int((t_df['won'] == 1).sum())
        t_losses = int((t_df['won'] == 0).sum())
        n = len(t_df)
        t_hr = (t_wins / n) * 100.0 if n > 0 else 0.0
        t_roi = (t_df['profit'].sum() / n) * 100.0 if n > 0 else 0.0
        lo, hi = wilson_ci(t_wins, n)
        ci_str = f"[{lo*100:.1f}%, {hi*100:.1f}%]"
        print(f"{tier:<32} {n:<7} {t_wins}W-{t_losses}L{'':<4} {t_hr:.1f}%{'':<4} {ci_str:<18} {t_roi:+.1f}%")

    print("\n[*] BREAKDOWN BY STAT FOCUS:")
    print("-" * 75)
    print(f"{'Stat Focus':<25} {'Bets':<7} {'Record':<12} {'Hit Rate':<10} {'95% CI':<18} {'ROI%':<8}")
    print("-" * 75)
    for stat, s_df in graded.groupby('stat_focus'):
        s_wins = int((s_df['won'] == 1).sum())
        s_losses = int((s_df['won'] == 0).sum())
        n = len(s_df)
        s_hr = (s_wins / n) * 100.0 if n > 0 else 0.0
        s_roi = (s_df['profit'].sum() / n) * 100.0 if n > 0 else 0.0
        lo, hi = wilson_ci(s_wins, n)
        ci_str = f"[{lo*100:.1f}%, {hi*100:.1f}%]"
        print(f"{stat:<25} {n:<7} {s_wins}W-{s_losses}L{'':<4} {s_hr:.1f}%{'':<4} {ci_str:<18} {s_roi:+.1f}%")

    print("=" * 75 + "\n")

    # Collect and fit isotonic post-hoc recalibration
    records_by_stat: Dict[str, List[Tuple[float, int]]] = {}
    for _, row in graded.iterrows():
        sf = str(row['stat_focus'])
        if sf not in records_by_stat:
            records_by_stat[sf] = []
        records_by_stat[sf].append((float(row['win_prob']), int(row['won'])))

    print("[*] FITTING POST-HOC ISOTONIC CALIBRATION CURVES ACROSS 3 SEASONS...")
    fit_calibration(records_by_stat)
    print("=" * 75 + "\n")

    return df_res


if __name__ == '__main__':
    run_backtest(seasons=[2023, 2024, 2025], max_weeks=18)
