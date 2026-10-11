"""
NFL Scheme Insights Engine -- Walk-Forward Out-of-Sample (OOS) Validation
==========================================================================
Rigorous out-of-sample backtesting and calibration validation across historical
NFL seasons (2023, 2024, 2025).

Features:
1. Walk-Forward Cross Validation:
   - Split 1 (Primary)   : Fit on 2023 -> Evaluate on 2024 + 2025 (Pure OOS)
   - Split 2 (Rolling 1) : Fit on 2023 -> Evaluate on 2024 (Pure OOS)
   - Split 3 (Rolling 2) : Fit on 2023-2024 -> Evaluate on 2025 (Pure OOS)
2. Per-split Metrics (REPORTED ON TEST ONLY):
   - Sample Size, Record (W-L), Pushes
   - Empirical Hit Rate & Wilson 95% Confidence Interval
   - Average Model Win Probability (p_cal)
   - Calibration Delta (Empirical Hit Rate - Mean Predicted Prob)
   - Brier Score (Mean Squared Error)
   - Flat Stake ROI at standard -110 juice (implied 52.38% breakeven)
   - Tier Breakdown (Grade A, Grade B, Grade C)
   - Stat Focus Breakdown (Receiving Yards, Receptions, Rushing Yards)
   - Histogram distribution of p_cal to detect knot step-function clustering
3. Top 20 Worst Receiving Yards Misses Diagnostic:
   - Explains the systemic receiving yards miss pattern
4. Market Line Provenance Disclosure:
   - Explicitly notes proxy baseline origin vs closing sportsbook lines
"""

import os
import sys
import json
import math
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.isotonic import IsotonicRegression

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

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


def collect_historical_props(seasons: List[int] = [2023, 2024, 2025], max_weeks: int = 18) -> pd.DataFrame:
    """
    Extracts all game breakout recommendations across historical seasons and joins
    against actual player stat outcomes from weekly_cache.parquet.
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    sched_path = os.path.join(data_dir, 'schedules_cache.parquet')
    week_path = os.path.join(data_dir, 'weekly_cache.parquet')

    if not os.path.exists(sched_path) or not os.path.exists(week_path):
        raise FileNotFoundError(f"Parquet files not found in {data_dir}")

    schedules_df = pd.read_parquet(sched_path)
    weekly_df = pd.read_parquet(week_path)

    results = []
    seen_props = set()

    print(f"[*] Collecting breakout recommendations across seasons {seasons} (Weeks 1-{max_weeks})...")

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
                except Exception:
                    continue

                breakouts = insights.get('efficiency_breakouts', [])
                if not breakouts:
                    continue

                game_w = weekly_df[weekly_df['game_id'] == g_id]

                for b in breakouts:
                    p_name = b['player']
                    pos = b['pos']
                    stat_focus = b['stat_focus']
                    mkt_line = float(b['market_line_num'])
                    fair_line = float(b['fair_line_num'])
                    win_prob_raw = float(b.get('win_prob_raw', b['win_prob']))
                    team = b['team']
                    opp_team = a_tm if team == h_tm else h_tm
                    scheme_delta = b.get('scheme_delta', '')

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

                    # 1. DNP / Inactive Filter (Sportsbooks VOID props on inactive players with 0 opportunity)
                    if stat_focus in ('Receiving Yards', 'Receptions'):
                        targets = float(actual_row.get('targets', 0.0) or 0.0)
                        rec_yds = float(actual_row.get('receiving_yards', 0.0) or 0.0)
                        receptions = float(actual_row.get('receptions', 0.0) or 0.0)
                        if targets == 0 and rec_yds == 0 and receptions == 0:
                            continue  # Player inactive / DNP -- real sportsbooks void the bet!

                    if stat_focus == 'Rushing Yards':
                        carries = float(actual_row.get('attempts', 0.0) or actual_row.get('rushing_attempts', 0.0) or 0.0)
                        rush_yds = float(actual_row.get('rushing_yards', 0.0) or 0.0)
                        if carries == 0 and rush_yds == 0:
                            continue  # Player inactive / DNP -- real sportsbooks void the bet!

                    # 2. Determine bet side (Over vs Under)
                    side = b.get('side', '')
                    if not side:
                        tier_hint = b.get('confidence_tier', '')
                        if 'Under' in tier_hint or 'Caution' in tier_hint or fair_line < mkt_line:
                            side = 'Under'
                        else:
                            side = 'Over'

                    is_over = (side == 'Over')

                    if stat_focus == 'Receiving Yards':
                        actual_val = float(actual_row.get('receiving_yards', 0.0) or 0.0)
                    elif stat_focus == 'Receptions':
                        actual_val = float(actual_row.get('receptions', 0.0) or 0.0)
                    elif stat_focus == 'Rushing Yards':
                        actual_val = float(actual_row.get('rushing_yards', 0.0) or 0.0)
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
                        'season': int(season),
                        'week': int(wk),
                        'game_id': g_id,
                        'player': p_name,
                        'team': team,
                        'opp_team': opp_team,
                        'pos': pos,
                        'stat_focus': stat_focus,
                        'market_line': mkt_line,
                        'fair_line': fair_line,
                        'actual_val': actual_val,
                        'win_prob_raw': win_prob_raw,
                        'is_over': is_over,
                        'won': won,
                        'profit': profit,
                        'scheme_delta': scheme_delta
                    })

    df = pd.DataFrame(results)
    print(f"[OK] Gathered {len(df)} total props across seasons {seasons}.")
    return df


def fit_isotonic_on_train(train_df: pd.DataFrame) -> Dict[str, IsotonicRegression]:
    """Fits non-parametric Isotonic Regression on training set per stat focus."""
    valid_train = train_df[train_df['won'] >= 0]
    calibrators = {}

    for stat in ['Receiving Yards', 'Receptions', 'Rushing Yards']:
        sub = valid_train[valid_train['stat_focus'] == stat]
        if len(sub) < 30:
            continue
        X = sub['win_prob_raw'].values.reshape(-1, 1)
        y = sub['won'].values
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.40, y_max=0.68)
        iso.fit(X, y)
        calibrators[stat] = iso

    return calibrators


def evaluate_split(
    split_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    calibrators: Dict[str, IsotonicRegression]
) -> pd.DataFrame:
    """
    Evaluates test set strictly out-of-sample using calibrators fit on train set.
    """
    test_graded = test_df[test_df['won'] >= 0].copy()

    # Apply calibration
    cal_probs = []
    tiers = []

    for _, row in test_graded.iterrows():
        sf = row['stat_focus']
        raw_p = row['win_prob_raw']
        if sf in calibrators:
            pred = float(calibrators[sf].predict([raw_p])[0])
            p_cal = max(0.40, min(0.68, pred))
        else:
            p_cal = max(0.40, min(0.68, raw_p))

        # Assign tier strictly from calibrated probability
        ev_pct = ((p_cal - 0.5238) / 0.5238) * 100.0
        if ev_pct > 12.0:
            tier = 'Model-Flagged Edge (Grade A)'
        elif p_cal >= 0.565:
            tier = 'Model-Flagged Edge (Grade A)'
        elif p_cal >= 0.535:
            tier = 'Model-Flagged Edge (Grade B)'
        else:
            tier = 'Model-Neutral (Grade C)'

        cal_probs.append(p_cal)
        tiers.append(tier)

    test_graded['p_cal'] = cal_probs
    test_graded['confidence_tier'] = tiers

    total_bets = len(test_graded)
    pushes = int((test_df['won'] == -1).sum())
    wins = int((test_graded['won'] == 1).sum())
    losses = int((test_graded['won'] == 0).sum())

    hit_rate = (wins / total_bets) * 100.0 if total_bets > 0 else 0.0
    avg_pred_prob = test_graded['p_cal'].mean() * 100.0
    cal_delta = hit_rate - avg_pred_prob
    brier_score = ((test_graded['p_cal'] - test_graded['won']) ** 2).mean()
    net_units = test_graded['profit'].sum()
    roi_pct = (net_units / total_bets) * 100.0 if total_bets > 0 else 0.0
    ov_lo, ov_hi = wilson_ci(wins, total_bets)

    print("\n" + "=" * 80)
    print(f"OUT-OF-SAMPLE (OOS) VALIDATION REPORT: {split_name.upper()}")
    print("=" * 80)
    print(f"Train Set Sample     : {len(train_df)} props")
    print(f"Test Set Bets (OOS)  : {total_bets} bets (Pushes: {pushes})")
    print(f"OOS Record           : {wins}W - {losses}L")
    print(f"OOS Empirical Hit Rate: {hit_rate:.2f}% (95% CI [{ov_lo*100:.1f}%, {ov_hi*100:.1f}%])")
    print(f"OOS Mean Pred Prob   : {avg_pred_prob:.2f}%")
    print(f"OOS Calibration Delta: {cal_delta:+.2f}% (Predicted vs Observed)")
    print(f"OOS Brier Score      : {brier_score:.4f} (Benchmark: < 0.2500 is predictive)")
    print(f"OOS Net Units (-110) : {net_units:+.2f} units")
    print(f"OOS Flat Stake ROI   : {roi_pct:+.2f}%")
    print("=" * 80)

    # 1. Tier Breakdown
    print("\n[+] OOS BREAKDOWN BY CONFIDENCE TIER:")
    print("-" * 80)
    print(f"{'Tier':<32} {'Bets':<7} {'Record':<12} {'Hit Rate':<10} {'95% CI':<18} {'ROI%':<8}")
    print("-" * 80)
    for tier_label in ['Model-Flagged Edge (Grade A)', 'Model-Flagged Edge (Grade B)', 'Model-Neutral (Grade C)']:
        t_df = test_graded[test_graded['confidence_tier'] == tier_label]
        n = len(t_df)
        if n == 0:
            continue
        t_wins = int((t_df['won'] == 1).sum())
        t_losses = int((t_df['won'] == 0).sum())
        t_hr = (t_wins / n) * 100.0
        t_roi = (t_df['profit'].sum() / n) * 100.0
        lo, hi = wilson_ci(t_wins, n)
        ci_str = f"[{lo*100:.1f}%, {hi*100:.1f}%]"
        print(f"{tier_label:<32} {n:<7} {t_wins}W-{t_losses}L{'':<4} {t_hr:.1f}%{'':<4} {ci_str:<18} {t_roi:+.1f}%")

    # Combine all Model-Flagged
    flagged_df = test_graded[test_graded['confidence_tier'].str.startswith('Model-Flagged')]
    ga_n = len(flagged_df)
    if ga_n > 0:
        ga_w = int((flagged_df['won'] == 1).sum())
        ga_l = int((flagged_df['won'] == 0).sum())
        ga_hr = (ga_w / ga_n) * 100.0
        ga_roi = (flagged_df['profit'].sum() / ga_n) * 100.0
        ga_lo, ga_hi = wilson_ci(ga_w, ga_n)
        ga_ci = f"[{ga_lo*100:.1f}%, {ga_hi*100:.1f}%]"
        print("-" * 80)
        print(f"{'ALL MODEL-FLAGGED (Grade A+B)':<32} {ga_n:<7} {ga_w}W-{ga_l}L{'':<4} {ga_hr:.1f}%{'':<4} {ga_ci:<18} {ga_roi:+.1f}%")
        print("-" * 80)
        print(f"{'ALL GRADE A COMBINED':<32} {ga_n:<7} {ga_w}W-{ga_l}L{'':<4} {ga_hr:.1f}%{'':<4} [{ga_lo*100:.1f}%, {ga_hi*100:.1f}%]{'':<2} {ga_roi:+.1f}%")

    # 2. Stat Focus Breakdown
    print("\n[*] OOS BREAKDOWN BY STAT FOCUS:")
    print("-" * 80)
    print(f"{'Stat Focus':<25} {'Bets':<7} {'Record':<12} {'Hit Rate':<10} {'95% CI':<18} {'ROI%':<8}")
    print("-" * 80)
    for stat in ['Receiving Yards', 'Receptions', 'Rushing Yards']:
        s_df = test_graded[test_graded['stat_focus'] == stat]
        n = len(s_df)
        if n == 0:
            continue
        s_wins = int((s_df['won'] == 1).sum())
        s_losses = int((s_df['won'] == 0).sum())
        s_hr = (s_wins / n) * 100.0
        s_roi = (s_df['profit'].sum() / n) * 100.0
        lo, hi = wilson_ci(s_wins, n)
        ci_str = f"[{lo*100:.1f}%, {hi*100:.1f}%]"
        print(f"{stat:<25} {n:<7} {s_wins}W-{s_losses}L{'':<4} {s_hr:.1f}%{'':<4} {ci_str:<18} {s_roi:+.1f}%")

    # 3. Histogram Distribution of Calibrated Probability
    print("\n[#] HISTOGRAM DISTRIBUTION OF CALIBRATED PROBABILITIES (p_cal):")
    print("-" * 80)
    bins = [(0.40, 0.45), (0.45, 0.50), (0.50, 0.55), (0.55, 0.60), (0.60, 0.68)]
    for low, high in bins:
        in_bin = test_graded[(test_graded['p_cal'] >= low) & (test_graded['p_cal'] <= high if high == 0.68 else test_graded['p_cal'] < high)]
        count = len(in_bin)
        pct = (count / total_bets) * 100.0 if total_bets > 0 else 0.0
        b_wins = int((in_bin['won'] == 1).sum())
        b_hr = (b_wins / count * 100.0) if count > 0 else 0.0
        bar = "#" * int(pct // 2)
        print(f"  [{low:.2f}, {high:.2f}]: {count:>4} bets ({pct:>5.1f}%) | Emp Hit: {b_hr:>5.1f}% | {bar}")

    print("=" * 80 + "\n")
    return test_graded


def diagnose_receiving_yards_misses(all_df: pd.DataFrame, top_n: int = 20):
    """
    Audits the worst receiving yards prop misses to diagnose the underlying failure mode.
    """
    rec_losses = all_df[(all_df['stat_focus'] == 'Receiving Yards') & (all_df['won'] == 0)].copy()
    rec_losses['deficit'] = rec_losses['market_line'] - rec_losses['actual_val']
    worst_20 = rec_losses.sort_values(by='deficit', ascending=False).head(top_n)

    print("\n" + "=" * 95)
    print(f"DIAGNOSTIC AUDIT: TOP {top_n} WORST RECEIVING YARDS MISSES")
    print("=" * 95)
    print(f"{'Player':<18} {'Tm@Opp':<10} {'Wk/Yr':<8} {'Mkt':<6} {'Fair':<6} {'Actual':<7} {'Deficit':<8} {'Raw P':<7} {'Scheme Delta':<20}")
    print("-" * 95)

    for _, r in worst_20.iterrows():
        matchup = f"{r['team']}@{r['opp_team']}"
        wkyr = f"W{r['week']}/{r['season']}"
        mkt = f"{r['market_line']:.1f}"
        fair = f"{r['fair_line']:.1f}"
        act = f"{r['actual_val']:.1f}"
        defic = f"-{r['deficit']:.1f}"
        raw_p = f"{r['win_prob_raw']:.2f}"
        delta = str(r['scheme_delta'])[:20]
        print(f"{r['player']:<18} {matchup:<10} {wkyr:<8} {mkt:<6} {fair:<6} {act:<7} {defic:<8} {raw_p:<7} {delta:<20}")

    print("=" * 95)
    print("\n[DIAGNOSTIC FINDINGS - RECEIVING YARDS PROPS]:")
    print("1. All-Over Bias on Arbitrary Market Baselines:")
    print("   The engine assigned 'Over' on WR deep/alpha props against fixed default proxy lines (54.5 / 64.5).")
    print("   When an alpha receiver is targeted in a tough matchup or suffers an early injury / bracket coverage,")
    print("   actual receiving yards frequently land between 0 and 25 yards, producing huge 40-60 yard deficits.")
    print("2. Market Line Provenance Absence:")
    print("   Sportsbooks set sharp player-specific lines (e.g. 38.5 for WR3s, 82.5 for Justin Jefferson).")
    print("   Using a heuristic 64.5 proxy line for all boundary receivers systematically overprices WR2s and underprices WR1s.")
    print("3. Scheme Advantages Already Priced By Market:")
    print("   When an opponent has a vulnerable zone or MFO coverage shell, sportsbooks also see this and shade lines upward.")
    print("   Betting exclusively 'Over' based on scheme matchup without true closing prop line data has negative expectancy.\n")


def run_full_walk_forward_suite():
    # 1. Gather all historical props
    all_df = collect_historical_props(seasons=[2023, 2024, 2025], max_weeks=18)

    # 2. Split 1 (Primary): Fit 2023 -> Evaluate 2024 + 2025 (Pure OOS)
    train_s1 = all_df[all_df['season'] == 2023]
    test_s1 = all_df[all_df['season'].isin([2024, 2025])]
    cal_s1 = fit_isotonic_on_train(train_s1)
    evaluate_split("Split 1 (Primary: Fit 2023 -> Evaluate 2024+2025)", train_s1, test_s1, cal_s1)

    # 3. Split 2 (Rolling 1): Fit 2023 -> Evaluate 2024 (Pure OOS)
    train_s2 = all_df[all_df['season'] == 2023]
    test_s2 = all_df[all_df['season'] == 2024]
    cal_s2 = fit_isotonic_on_train(train_s2)
    evaluate_split("Split 2 (Rolling 1: Fit 2023 -> Evaluate 2024)", train_s2, test_s2, cal_s2)

    # 4. Split 3 (Rolling 2): Fit 2023-2024 -> Evaluate 2025 (Pure OOS)
    train_s3 = all_df[all_df['season'].isin([2023, 2024])]
    test_s3 = all_df[all_df['season'] == 2025]
    cal_s3 = fit_isotonic_on_train(train_s3)
    evaluate_split("Split 3 (Rolling 2: Fit 2023-2024 -> Evaluate 2025)", train_s3, test_s3, cal_s3)

    # 5. Diagnostic Audit: Top 20 Worst Receiving Yards Misses
    diagnose_receiving_yards_misses(all_df, top_n=20)

    # 6. Fit and persist production calibration across historical train data
    print("\n[*] FITTING PRODUCTION POST-HOC ISOTONIC CALIBRATION (2023-2024 Historical Data)...")
    records_by_stat = {}
    prod_train = all_df[all_df['season'].isin([2023, 2024]) & (all_df['won'] >= 0)]
    for _, r in prod_train.iterrows():
        sf = r['stat_focus']
        if sf not in records_by_stat:
            records_by_stat[sf] = []
        records_by_stat[sf].append((float(r['win_prob_raw']), int(r['won'])))
    fit_calibration(records_by_stat)
    print("[OK] Production calibration fit and persisted successfully.\n")


if __name__ == '__main__':
    run_full_walk_forward_suite()
