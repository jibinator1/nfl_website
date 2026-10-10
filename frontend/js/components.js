/**
 * NFL Prediction & Analytics Hub - UI Component Library
 * ====================================================
 * Reusable, high-performance UI components with clean monochrome aesthetics
 * (gray, black, white), interactive simulation tools, and zero layout shift.
 */

// Polyfill Math.erf if missing in the browser environment
if (!Math.erf) {
    Math.erf = function(x) {
        const sign = (x >= 0) ? 1 : -1;
        const absX = Math.abs(x);
        const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429;
        const p = 0.3275911;
        const t = 1.0 / (1.0 + p * absX);
        const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-absX * absX);
        return sign * y;
    };
}

// Global active prop stored for the interactive simulation modal
let currentModalProp = null;

const components = {

    /**
     * Render Game Matchup Spread & Totals Cards
     */
    renderSpreadCards(games) {
        if (!games || games.length === 0) {
            return `
                <div class="col-span-full p-8 text-center glass-panel">
                    <i data-lucide="calendar-x" class="w-8 h-8 mx-auto text-zinc-500 mb-2"></i>
                    <p class="text-sm text-zinc-400">No scheduled games found for this date range.</p>
                </div>
            `;
        }

        return games.map(g => {
            const homeSpread = parseFloat(g.spread_line) || 0;
            const vegasTotal = parseFloat(g.total_line) || 44.0;
            const predMargin = parseFloat(g.pred_margin) || 0;
            const predTotal = parseFloat(g.pred_total) || vegasTotal;
            const homeTeam = g.home_team || 'HOME';
            const awayTeam = g.away_team || 'AWAY';
            const gameDate = g.gameday || g.date || 'TBD';

            // In nflverse schedule data, spread_line > 0 means the home team is favored.
            // In American sports betting notation:
            // Favorite has minus line (-7), Underdog has plus line (+7).
            const homeLineStr = homeSpread === 0 ? 'PK' : (homeSpread > 0 ? `-${homeSpread}` : `+${Math.abs(homeSpread)}`);
            const awayLineStr = homeSpread === 0 ? 'PK' : (homeSpread > 0 ? `+${homeSpread}` : `-${Math.abs(homeSpread)}`);

            // Favorite team & spread display for Vegas Line
            let vegasLineDisplay = 'PK';
            if (homeSpread > 0) {
                vegasLineDisplay = `${homeTeam} -${homeSpread}`;
            } else if (homeSpread < 0) {
                vegasLineDisplay = `${awayTeam} -${Math.abs(homeSpread)}`;
            }

            // Check if model has generated predictions for this game
            const hasModelPred = g.is_model_trained !== false && g.pred_home_margin !== null && g.pred_home_margin !== undefined;

            // Model prediction line display
            let predLineDisplay = '<span class="text-zinc-500 font-sans text-xs">Not run yet</span>';
            if (hasModelPred) {
                if (predMargin > 0) {
                    predLineDisplay = `${homeTeam} -${predMargin.toFixed(1)}`;
                } else if (predMargin < 0) {
                    predLineDisplay = `${awayTeam} -${Math.abs(predMargin).toFixed(1)}`;
                } else {
                    predLineDisplay = 'EVEN';
                }
            }

            const spreadDiff = predMargin - homeSpread;
            const homeCovers = spreadDiff > 0;
            const pickTeam = homeCovers ? homeTeam : awayTeam;
            const pickLine = homeCovers ? homeLineStr : awayLineStr;
            const spreadEdge = Math.abs(spreadDiff).toFixed(1);

            const totalDiff = predTotal - vegasTotal;
            const totalCall = totalDiff > 0 ? 'OVER' : 'UNDER';
            const totalEdge = Math.abs(totalDiff).toFixed(1);

            return `
                <div class="glass-panel p-4 rounded-xl border border-zinc-800 hover:border-zinc-600 transition duration-200 space-y-3">
                    <div class="flex items-center justify-between text-xs text-zinc-400 border-b border-zinc-800 pb-2">
                        <span class="font-mono flex items-center gap-1.5">
                            <i data-lucide="clock" class="w-3.5 h-3.5 text-zinc-300"></i>
                            ${gameDate}
                        </span>
                        <span class="px-2 py-0.5 rounded bg-zinc-800 text-[10px] font-mono text-zinc-300">
                            Wk ${g.week || '1'}
                        </span>
                    </div>

                    <!-- Teams Row -->
                    <div class="grid grid-cols-2 gap-3 py-1">
                        <div class="space-y-1">
                            <div class="text-[11px] text-zinc-400 font-medium">AWAY</div>
                            <div class="text-base font-bold font-heading text-white flex items-center gap-2">
                                <span>${awayTeam}</span>
                            </div>
                        </div>
                        <div class="space-y-1 text-right">
                            <div class="text-[11px] text-zinc-400 font-medium">HOME</div>
                            <div class="text-base font-bold font-heading text-white flex items-center justify-end gap-2">
                                <span>${homeTeam}</span>
                            </div>
                        </div>
                    </div>

                    <!-- Vegas Slate Context -->
                    <div class="bg-black/40 rounded-lg p-2.5 space-y-2 text-xs font-mono border border-zinc-800/80">
                        <div class="flex items-center justify-between">
                            <span class="text-zinc-400">Vegas Line:</span>
                            <span class="text-zinc-200 font-semibold">${vegasLineDisplay} (Total ${vegasTotal})</span>
                        </div>
                        <div class="flex items-center justify-between">
                            <span class="text-zinc-400">Matchup:</span>
                            <span class="text-white font-medium">${awayTeam} at ${homeTeam}</span>
                        </div>
                    </div>

                    <!-- Matchup Context -->
                    <div class="grid grid-cols-2 gap-2 pt-1">
                        <div class="bg-zinc-900 border border-zinc-700/80 rounded-lg p-2 text-center">
                            <div class="text-[10px] text-zinc-400 uppercase font-bold">Game Slate</div>
                            <div class="text-xs font-bold text-white font-mono mt-0.5">${homeTeam} vs ${awayTeam}</div>
                            <div class="text-[10px] text-zinc-400 mt-0.5">Week ${g.week || '1'}</div>
                        </div>
                        <div class="bg-zinc-900 border border-zinc-700/80 rounded-lg p-2 text-center">
                            <div class="text-[10px] text-zinc-400 uppercase font-bold">Implied Total</div>
                            <div class="text-xs font-bold text-white font-mono mt-0.5">${vegasTotal} O/U</div>
                            <div class="text-[10px] text-zinc-400 mt-0.5">Vegas Consensus</div>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    },

    /**
     * Render Player Props Table (Supports both XGBoost and MLR Starters)
     */
    renderPropsTable(props, isMLR = false, isTrained = true) {
        if (!props || props.length === 0) {
            const hasDateFilter = typeof state !== 'undefined' && Boolean(state.selectedDate);
            const hasAnyModelProps = typeof state !== 'undefined' && Boolean(
                (state.tdPredictions?.props && state.tdPredictions.props.length > 0) ||
                (state.mlrPredictions?.props && state.mlrPredictions.props.length > 0) ||
                (state.xgbPredictions?.props && state.xgbPredictions.props.length > 0)
            );
            if (!isTrained && !isMLR && !hasDateFilter && !hasAnyModelProps) {
                return `
                    <tr>
                        <td colspan="11" class="p-10 text-center">
                            <div class="flex flex-col items-center justify-center gap-3 max-w-md mx-auto">
                                <div class="w-10 h-10 rounded-full bg-zinc-800 border border-zinc-700 flex items-center justify-center text-zinc-300">
                                    <i data-lucide="play" class="w-4 h-4 ml-0.5 fill-current"></i>
                                </div>
                                <div>
                                    <h4 class="text-sm font-semibold text-zinc-200">Model Ready to Run</h4>
                                    <p class="text-xs text-zinc-400 mt-1">The prediction model has not been run yet. Click <span class="text-white font-medium">RUN MODEL</span> in the toolbar above whenever you are ready to train and generate predictions.</p>
                                </div>
                            </div>
                        </td>
                    </tr>
                `;
            }
            const dateNotice = (typeof state !== 'undefined' && state.selectedDate) ? ` for ${state.selectedDate}` : '';
            return `
                <tr>
                    <td colspan="11" class="p-8 text-center text-zinc-400">
                        <i data-lucide="search-x" class="w-8 h-8 mx-auto text-zinc-500 mb-2"></i>
                        <p class="text-sm font-medium text-zinc-300">No player props match your current filters${dateNotice}.</p>
                        <div class="flex items-center justify-center gap-2 mt-3">
                            ${(typeof state !== 'undefined' && state.filterConf !== 'ALL') ? `<button onclick="setConfFilter('ALL')" class="btn-secondary text-[11px] py-1 px-2.5">Set Confidence to ALL</button>` : ''}
                            ${(typeof state !== 'undefined' && state.selectedDate) ? `<button onclick="clearPredictionsDateFilter()" class="btn-secondary text-[11px] py-1 px-2.5">Show All Dates</button>` : ''}
                        </div>
                    </td>
                </tr>
            `;
        }

        return props.map((p, idx) => {
            const player = p.Player || p.player || 'Unknown';
            const team = p.Team || p.team || '';
            const pos = p.Pos || p.pos || '';
            const role = p.Role || p.role || (pos ? `${pos}1` : 'STARTER');
            const matchup = p.Matchup || p.matchup || '';
            const stat = p.Stat || p.stat || 'Yards';
            const isTD = stat === 'Anytime TD' || stat === 'Anytime Touchdown' || p.Bet !== undefined;

            const line = isTD ? (p.Line !== undefined && p.Line !== null ? String(p.Line) : '1') : parseFloat(p.Line ?? p.line ?? 0).toFixed(1);
            const pred = isTD ? `${parseFloat(p.Prediction ?? p.PRED ?? 0).toFixed(1)}%` : parseFloat(p.PRED ?? p.pred ?? 0).toFixed(1);
            const diffFromLine = (parseFloat(p.PRED ?? p.pred ?? 0) - parseFloat(p.Line ?? p.line ?? 0)).toFixed(1);
            const diffDisplay = `${parseFloat(diffFromLine) >= 0 ? '+' : ''}${diffFromLine}${isTD ? '%' : ''}`;
            const call = isTD ? (p.Bet || 'YES (To Score)') : (p['O/U'] || p.ou || (parseFloat(pred) >= parseFloat(line) ? 'OVER' : 'UNDER')).toUpperCase();
            
            const rawProb = p.Model_Prob ?? p.Win_Prob ?? p.win_prob ?? (call === 'OVER' ? 0.58 : 0.54);
            const pOver = (call === 'OVER' ? rawProb : (1.0 - rawProb));
            const pOverPct = (pOver * 100).toFixed(0);
            const bandLow = Math.max(5, Math.round((pOver - 0.09) * 100));
            const bandHigh = Math.min(95, Math.round((pOver + 0.09) * 100));
            const probBand = `[${bandLow}–${bandHigh}%]`;

            // Direction: FAVORABLE / NEUTRAL / UNFAVORABLE
            const direction = (p.Direction || p.direction || (parseFloat(diffFromLine) >= 3.0 ? 'FAVORABLE' : (parseFloat(diffFromLine) <= -3.0 ? 'UNFAVORABLE' : 'NEUTRAL'))).toUpperCase();
            let dirBadge = 'badge-med';
            if (direction === 'FAVORABLE') dirBadge = 'badge-high';
            else if (direction === 'UNFAVORABLE') dirBadge = 'badge-low';
            const actual = p.Actual ?? p.actual;
            const isCorrect = p.Correct ?? p.correct;

            const callBadge = (isTD || call === 'OVER') ? 'badge-over' : 'badge-under';

            let attribText = '';
            if (p.Attribution && p.Attribution.length > 0) {
                const topA = p.Attribution[0];
                const impactVal = topA.impact ?? topA.contrib ?? 0;
                const numImpact = typeof impactVal === 'number' ? impactVal : (parseFloat(impactVal) || 0);
                const sign = numImpact >= 0 ? '+' : '';
                attribText = `<span class="text-[11px] font-mono text-zinc-300" title="${topA.name || topA.feature || ''}">${sign}${numImpact.toFixed(1)} ${topA.name || topA.feature || ''}</span>`;
            } else if (p.shap_top3 && p.shap_top3.length > 0) {
                const topS = p.shap_top3[0];
                const val = topS.contrib ?? topS.shap_value ?? topS.impact ?? 0;
                const numVal = typeof val === 'number' ? val : (parseFloat(val) || 0);
                const sign = numVal >= 0 ? '+' : '';
                attribText = `<span class="text-[11px] font-mono text-zinc-300" title="${topS.feature || ''}">${sign}${numVal.toFixed(1)} ${topS.feature || ''}</span>`;
            } else {
                attribText = `<span class="text-[11px] text-zinc-500 font-mono">Baseline Fit</span>`;
            }

            let actualBadge = '<span class="text-zinc-500 text-xs">-</span>';
            if (actual !== undefined && actual !== null && actual !== '') {
                const actNum = parseFloat(actual).toFixed(1);
                if (isCorrect === 'Yes') {
                    actualBadge = `<span class="font-mono text-white font-bold text-xs" title="Actual: ${actNum}">✓ ${actNum} (Hit)</span>`;
                } else if (isCorrect === 'No') {
                    actualBadge = `<span class="font-mono text-zinc-400 font-bold text-xs" title="Actual: ${actNum}">✗ ${actNum} (Miss)</span>`;
                } else {
                    actualBadge = `<span class="font-mono text-zinc-300 text-xs">${actNum}</span>`;
                }
            }

            // Floor Breach Streak cell (only populated for TD rows)
            let floorStreakCell = '<span class="text-zinc-600 text-xs">—</span>';
            if (isTD && p.Floor_Streak !== undefined && p.Floor_Streak !== null) {
                const streak = parseInt(p.Floor_Streak, 10);
                const baseMed = p.Baseline_Median != null ? parseFloat(p.Baseline_Median).toFixed(1) : '?';
                const statLabel = p.Floor_Stat ? p.Floor_Stat.replace(/_/g, ' ') : 'yds';
                const tooltip = `${streak} games without floor breach | Baseline: ${baseMed} ${statLabel}`;
                const streakColor = streak >= 8 ? 'text-zinc-100 font-bold' : (streak >= 5 ? 'text-zinc-300 font-medium' : 'text-zinc-400');
                floorStreakCell = `
                    <div class="flex flex-col items-center" title="${tooltip}">
                        <span class="font-mono text-xs ${streakColor}">${streak}</span>
                        <span class="text-[10px] text-zinc-500 font-mono">${baseMed} ${statLabel.split(' ')[0]}</span>
                    </div>`;
            }

            return `
                <tr class="clickable hover:bg-zinc-800/50 transition duration-150 cursor-pointer" onclick="openPlayerModalByIndex(${idx})">
                    <!-- Player & Role -->
                    <td>
                        <div class="flex items-center gap-2">
                            <span class="badge badge-role font-mono text-[10px]">${role}</span>
                            <div>
                                <div class="font-semibold text-white flex items-center gap-1.5">
                                    <span>${player}</span>
                                    <span class="text-[10px] font-mono text-zinc-400">${team}</span>
                                </div>
                                <div class="text-[11px] text-zinc-400 font-mono">${matchup}</div>
                            </div>
                        </div>
                    </td>

                    <!-- Stat Category -->
                    <td>
                        <span class="text-xs font-medium text-zinc-300">${stat}</span>
                    </td>

                    <!-- Sportsbook Line -->
                    <td class="text-right font-mono font-semibold text-zinc-300">
                        ${line}
                    </td>

                    <!-- Model PRED -->
                    <td class="text-right font-mono font-bold text-white">
                        ${pred}
                    </td>

                    <!-- Pred - Line -->
                    <td class="text-right font-mono font-bold text-white">
                        ${diffDisplay}
                    </td>

                    <!-- Call (O/U) -->
                    <td class="text-center">
                        <span class="badge ${callBadge}">${call}</span>
                    </td>

                    <!-- p(OVER) -->
                    <td class="text-right font-mono text-zinc-200">
                        ${pOverPct}%
                    </td>

                    <!-- 95% Uncertainty Band -->
                    <td class="text-center font-mono text-zinc-400 text-xs">
                        ${probBand}
                    </td>

                    <!-- Direction -->
                    <td class="text-center">
                        <span class="badge ${dirBadge}">${direction}</span>
                    </td>

                    <!-- Driver / Attribution OR Floor Streak (TD mode) -->
                    <td class="text-center">
                        ${isTD ? floorStreakCell : attribText}
                    </td>

                    <!-- Actual Outcome -->
                    <td class="text-center font-mono">
                        ${actualBadge}
                    </td>
                </tr>
            `;
        }).join('');
    },


    /**
     * Render Player Hard Line Floor Streaks Table
     */
    renderFloorStreakTable(players) {
        if (!players || players.length === 0) {
            return `
                <tr>
                    <td colspan="11" class="p-8 text-center text-zinc-400">
                        <div class="flex flex-col items-center justify-center gap-2">
                            <i data-lucide="info" class="w-6 h-6 text-zinc-500"></i>
                            <p class="text-sm">No players match the selected position, streak, or search criteria.</p>
                        </div>
                    </td>
                </tr>
            `;
        }

        return players.map((p, idx) => {
            const rank = idx + 1;
            const name = p.player_name || p.Player || 'Unknown';
            const pos = (p.position || p.Pos || 'WR').toUpperCase();
            const team = p.team || p.Team || 'NFL';
            const streak = parseInt(p.Floor_Streak ?? 0, 10);
            const baseMed = parseFloat(p.Baseline_Median ?? 0).toFixed(1);
            const margin = parseFloat(p.Margin_Used ?? 0.50);
            const marginPct = `-${Math.round(margin * 100)}%`;
            const threshold = parseFloat(p.Floor_Threshold ?? (baseMed * (1.0 - margin))).toFixed(1);
            const statCol = p.Floor_Stat || (pos === 'QB' ? 'rushing_yards' : 'receiving_yards');
            const statLabel = statCol.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            const games = p.Games_Played || 0;

            // Tier styling (sleek, dark, monochrome)
            let tierBadge = '';
            let streakColor = 'text-zinc-300';
            let barWidth = Math.min(100, Math.round((streak / 12) * 100));
            let barColor = 'bg-zinc-500';

            if (streak >= 8) {
                streakColor = 'text-zinc-100 font-bold';
                barColor = 'bg-zinc-300';
                tierBadge = `<span class="badge font-mono text-[10px] bg-zinc-800 text-zinc-200 border-zinc-700">Elite (8+)</span>`;
            } else if (streak >= 5) {
                streakColor = 'text-zinc-200 font-semibold';
                barColor = 'bg-zinc-400';
                tierBadge = `<span class="badge font-mono text-[10px] bg-zinc-850 text-zinc-300 border-zinc-800">Consistent (5-7)</span>`;
            } else if (streak >= 2) {
                streakColor = 'text-zinc-300';
                barColor = 'bg-zinc-600';
                tierBadge = `<span class="badge font-mono text-[10px] bg-zinc-900 text-zinc-400 border-zinc-800">Active (2-4)</span>`;
            } else {
                streakColor = 'text-zinc-500';
                barColor = 'bg-zinc-800';
                tierBadge = `<span class="badge font-mono text-[10px] bg-zinc-900/60 text-zinc-500 border-zinc-850">Breached (0-1)</span>`;
            }

            const posBadgeClass = 'bg-zinc-900 text-zinc-300 border-zinc-800';
            const lastBreach = p.Last_Breach || 'None in record';
            const lastBreachGame = p.Last_Breach_Game || 'None';
            const breachTooltip = lastBreach !== 'None in record'
                ? `${streak} games without a floor breach. Last breach: ${lastBreach}`
                : `Flawless: 0 floor breaches across all ${games} games recorded`;
            const matchup = p.matchup || (p.opponent ? (p.is_home ? `${team} vs ${p.opponent}` : `${team} @ ${p.opponent}`) : '');

            return `
                <tr class="hover:bg-zinc-800/40 transition">
                    <td class="text-center font-mono text-xs text-zinc-500">${rank}</td>
                    <td>
                        <div class="flex items-center gap-2">
                            <span class="font-bold text-white font-heading">${name}</span>
                        </div>
                    </td>
                    <td class="text-center">
                        <span class="badge font-mono text-[10px] ${posBadgeClass}">${pos}</span>
                    </td>
                    <td class="text-center font-mono text-xs font-semibold text-zinc-300">
                        <div>${team}</div>
                        ${matchup ? `<div class="text-[10px] text-zinc-500 font-normal font-sans tracking-tight">${matchup}</div>` : ''}
                    </td>
                    <td class="text-xs text-zinc-300 font-mono">
                        <span class="inline-flex items-center gap-1.5">
                            <span class="w-1.5 h-1.5 rounded-full bg-zinc-500"></span>
                            <span>${statLabel}</span>
                        </span>
                    </td>
                    <td class="text-right font-mono text-xs font-bold text-zinc-200">
                        ${baseMed} <span class="text-[10px] text-zinc-500">yds</span>
                    </td>
                    <td class="text-center">
                        <span class="text-[11px] font-mono text-zinc-400 bg-zinc-900 px-1.5 py-0.5 rounded border border-zinc-800">${marginPct}</span>
                    </td>
                    <td class="text-right font-mono text-xs font-semibold text-zinc-300">
                        ≥ ${threshold} <span class="text-[10px] text-zinc-500">yds</span>
                    </td>
                    <td class="text-center" title="${breachTooltip}">
                        <div class="flex flex-col items-center">
                            <span class="font-mono text-xs ${streakColor}">${streak} <span class="text-[10px] text-zinc-500 font-normal">games</span></span>
                            <div class="w-14 bg-zinc-900 border border-zinc-800/80 rounded-full h-1 mt-1 overflow-hidden">
                                <div class="${barColor} h-1 rounded-full" style="width: ${barWidth}%"></div>
                            </div>
                            ${lastBreachGame !== 'None' ? `<span class="text-[9px] text-zinc-500 font-mono mt-0.5" title="${breachTooltip}">Breached: ${lastBreachGame}</span>` : `<span class="text-[9px] text-zinc-600 font-mono mt-0.5">Flawless</span>`}
                        </div>
                    </td>
                    <td class="text-center">${tierBadge}</td>
                    <td class="text-center font-mono text-xs text-zinc-400">${games}</td>
                </tr>
            `;
        }).join('');
    },

    /**
     * Render Matchup Lab (H2H Deep Dive)
     */
    renderMatchupDeepDive(data) {
        if (!data || data.error) {
            return `
                <div class="p-8 text-center text-zinc-400 glass-panel">
                    <i data-lucide="alert-circle" class="w-8 h-8 mx-auto text-zinc-500 mb-2"></i>
                    <p class="text-sm font-semibold text-zinc-300">${data?.error || 'Select teams above and click "Simulate Matchup" to inspect head-to-head battle.'}</p>
                </div>
            `;
        }

        const home = data.home_team || data.home_stats?.team || 'HOME';
        const away = data.away_team || data.away_stats?.team || 'AWAY';
        const hStats = data.home_stats || {};
        const aStats = data.away_stats || {};
        const battles = data.battles || [];
        const xFactors = data.x_factors || [];
        const homeLeaders = data.home_leaders || {};
        const awayLeaders = data.away_leaders || {};
        const defStars = data.defense_vs_stars || {};
        const spotlights = defStars.spotlights || [];
        const funnel = data.funnel || {};

        // Helper to get battle verdict badge class
        const getBattleBadgeClass = (level) => {
            switch (level) {
                case 'heavy_off': return 'badge-heavy-off';
                case 'slight_off': return 'badge-slight-off';
                case 'slight_def': return 'badge-slight-def';
                case 'heavy_def': return 'badge-heavy-def';
                default: return 'badge-battle-neutral';
            }
        };

        // Helper to get spotlight level badge class
        const getSpotlightBadgeClass = (level) => {
            switch (level) {
                case 'lockdown': return 'badge-lockdown';
                case 'vulnerable': return 'badge-vulnerable';
                default: return 'badge-battle-neutral';
            }
        };

        // Helper to format record or games played
        const formatRecord = (st) => {
            if (st.record) return st.record;
            if (st.wins !== undefined && st.losses !== undefined) return `${st.wins}-${st.losses}`;
            return `${st.games_played || 0} GP`;
        };

        return `
            <div class="space-y-6">
                <!-- 1. HEAD TO HEAD SCORECARD & EFFICIENCY HEADER -->
                <div class="grid grid-cols-1 md:grid-cols-11 gap-3 items-center">
                    <!-- Away Team Card -->
                    <div class="md:col-span-5 p-4 bg-zinc-900/90 border border-zinc-800 rounded-xl space-y-3 h2h-card">
                        <div class="flex items-center justify-between">
                            <span class="text-[11px] font-bold tracking-wider uppercase px-2 py-0.5 rounded bg-zinc-800 text-zinc-400">AWAY</span>
                            <span class="text-xs font-mono text-zinc-400">${formatRecord(aStats)}</span>
                        </div>
                        <div class="flex items-baseline justify-between">
                            <div class="text-3xl font-extrabold font-heading text-white tracking-tight">${away}</div>
                            <div class="text-right">
                                <div class="text-xs font-mono font-bold text-zinc-200">${aStats.pts_per_game ? aStats.pts_per_game.toFixed(1) : '--'} PPG</div>
                                <div class="text-[10px] text-zinc-400">Rank #${aStats.rank_pts_scored || aStats.rank_pts || '--'} Offense</div>
                            </div>
                        </div>
                        <div class="grid grid-cols-3 gap-2 pt-2 border-t border-zinc-800/80 text-center">
                            <div class="p-1.5 bg-zinc-950/60 rounded">
                                <div class="text-[10px] text-zinc-400">Yards/G</div>
                                <div class="text-xs font-mono font-bold text-zinc-200">${(aStats.total_yds_per_game || aStats.yards_per_game) ? (aStats.total_yds_per_game || aStats.yards_per_game).toFixed(0) : '--'}</div>
                            </div>
                            <div class="p-1.5 bg-zinc-950/60 rounded">
                                <div class="text-[10px] text-zinc-400">EPA/Play</div>
                                <div class="text-xs font-mono font-bold ${(aStats.epa_per_play ?? 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}">
                                    ${(aStats.epa_per_play ?? 0) >= 0 ? '+' : ''}${(aStats.epa_per_play ?? 0).toFixed(3)}
                                </div>
                            </div>
                            <div class="p-1.5 bg-zinc-950/60 rounded">
                                <div class="text-[10px] text-zinc-400">TO Margin</div>
                                <div class="text-xs font-mono font-bold ${(aStats.turnover_diff ?? 0) >= 0 ? 'text-zinc-200' : 'text-rose-400'}">
                                    ${(aStats.turnover_diff ?? 0) >= 0 ? '+' : ''}${aStats.turnover_diff ?? 0}
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Matchup VS Divider -->
                    <div class="md:col-span-1 text-center py-2">
                        <div class="inline-flex flex-col items-center justify-center w-10 h-10 rounded-full bg-zinc-800/80 border border-zinc-700/60 shadow-lg">
                            <span class="text-xs font-bold text-zinc-300 font-heading">VS</span>
                        </div>
                    </div>

                    <!-- Home Team Card -->
                    <div class="md:col-span-5 p-4 bg-zinc-900/90 border border-zinc-800 rounded-xl space-y-3 h2h-card">
                        <div class="flex items-center justify-between">
                            <span class="text-xs font-mono text-zinc-400">${formatRecord(hStats)}</span>
                            <span class="text-[11px] font-bold tracking-wider uppercase px-2 py-0.5 rounded bg-zinc-800 text-zinc-400">HOME</span>
                        </div>
                        <div class="flex items-baseline justify-between">
                            <div class="text-left">
                                <div class="text-xs font-mono font-bold text-zinc-200">${hStats.pts_per_game ? hStats.pts_per_game.toFixed(1) : '--'} PPG</div>
                                <div class="text-[10px] text-zinc-400">Rank #${hStats.rank_pts_scored || hStats.rank_pts || '--'} Offense</div>
                            </div>
                            <div class="text-3xl font-extrabold font-heading text-white tracking-tight">${home}</div>
                        </div>
                        <div class="grid grid-cols-3 gap-2 pt-2 border-t border-zinc-800/80 text-center">
                            <div class="p-1.5 bg-zinc-950/60 rounded">
                                <div class="text-[10px] text-zinc-400">Yards/G</div>
                                <div class="text-xs font-mono font-bold text-zinc-200">${(hStats.total_yds_per_game || hStats.yards_per_game) ? (hStats.total_yds_per_game || hStats.yards_per_game).toFixed(0) : '--'}</div>
                            </div>
                            <div class="p-1.5 bg-zinc-950/60 rounded">
                                <div class="text-[10px] text-zinc-400">EPA/Play</div>
                                <div class="text-xs font-mono font-bold ${(hStats.epa_per_play ?? 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}">
                                    ${(hStats.epa_per_play ?? 0) >= 0 ? '+' : ''}${(hStats.epa_per_play ?? 0).toFixed(3)}
                                </div>
                            </div>
                            <div class="p-1.5 bg-zinc-950/60 rounded">
                                <div class="text-[10px] text-zinc-400">TO Margin</div>
                                <div class="text-xs font-mono font-bold ${(hStats.turnover_diff ?? 0) >= 0 ? 'text-zinc-200' : 'text-rose-400'}">
                                    ${(hStats.turnover_diff ?? 0) >= 0 ? '+' : ''}${hStats.turnover_diff ?? 0}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 2. TACTICAL UNIT BATTLES -->
                <div class="space-y-3">
                    <div class="flex items-center justify-between">
                        <div class="flex items-center gap-2">
                            <i data-lucide="swords" class="w-4 h-4 text-zinc-400"></i>
                            <h3 class="text-xs uppercase font-bold text-zinc-300 tracking-wider">Tactical Unit Battles & Scheme Clashes</h3>
                        </div>
                        <span class="text-[11px] font-mono text-zinc-400">${battles.length} matchups</span>
                    </div>

                    <div class="grid grid-cols-1 gap-2.5">
                        ${battles.map(b => `
                            <div class="p-3 bg-zinc-900 border border-zinc-800/80 rounded-xl flex flex-col md:flex-row md:items-center justify-between gap-3 h2h-card">
                                <div class="space-y-1">
                                    <div class="flex items-center gap-2">
                                        <span class="text-xs font-bold text-white">${b.stat}</span>
                                        <span class="text-[11px] font-mono text-zinc-400">${b.offense} vs ${b.defense}</span>
                                    </div>
                                    <div class="flex items-center gap-3 text-xs text-zinc-400 font-mono">
                                        <span>Offense Rank: <strong class="text-zinc-200">#${b.off_rank}</strong></span>
                                        <span>•</span>
                                        <span>Defense Rank: <strong class="text-zinc-200">#${b.def_rank}</strong></span>
                                        <span>•</span>
                                        <span>Net Gap: <strong class="${b.diff > 0 ? 'text-emerald-400' : b.diff < 0 ? 'text-rose-400' : 'text-zinc-400'}">${b.diff > 0 ? '+' + b.diff : b.diff}</strong></span>
                                    </div>
                                </div>
                                <div class="shrink-0 flex items-center gap-2">
                                    <span class="px-2.5 py-1 rounded text-xs font-semibold ${getBattleBadgeClass(b.level)}">
                                        ${b.verdict}
                                    </span>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>

                <!-- 3. DYNAMIC OPPORTUNITY FUNNEL -->
                ${funnel.home && funnel.away ? `
                    <div class="space-y-3">
                        <div class="flex items-center gap-2">
                            <i data-lucide="filter" class="w-4 h-4 text-zinc-400"></i>
                            <h3 class="text-xs uppercase font-bold text-zinc-300 tracking-wider">Projected Play Distribution & Opportunity Funnel</h3>
                        </div>

                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                            ${[funnel.away, funnel.home].map((f, idx) => {
                                const teamCode = f.team || (idx === 0 ? away : home);
                                const passPct = Math.round((f.projected_pass_share || 0.55) * 100);
                                const rushPct = 100 - passPct;
                                return `
                                    <div class="p-4 bg-zinc-900 border border-zinc-800 rounded-xl space-y-3">
                                        <div class="flex items-center justify-between">
                                            <span class="font-bold text-xs text-zinc-200">${teamCode} Offensive Script</span>
                                            <span class="text-[10px] font-semibold px-2 py-0.5 rounded bg-zinc-800 text-zinc-300 border border-zinc-700">
                                                ${f.funnel_verdict || 'Balanced Script'}
                                            </span>
                                        </div>

                                        <div>
                                            <div class="flex justify-between text-xs text-zinc-400 font-mono mb-1">
                                                <span>Pass: ${f.projected_pass_plays ? f.projected_pass_plays.toFixed(1) : '--'} att (${passPct}%)</span>
                                                <span>Rush: ${f.projected_rush_plays ? f.projected_rush_plays.toFixed(1) : '--'} car (${rushPct}%)</span>
                                            </div>
                                            <div class="funnel-bar-container">
                                                <div class="funnel-bar-pass" style="width: ${passPct}%"></div>
                                                <div class="funnel-bar-rush" style="width: ${rushPct}%"></div>
                                            </div>
                                        </div>

                                        <div class="text-[11px] text-zinc-400 space-y-1 pt-1">
                                            ${f.pass_matchup_desc ? `<p class="leading-relaxed"><strong class="text-zinc-300">Passing Scheme:</strong> ${f.pass_matchup_desc}</p>` : ''}
                                            ${f.rush_matchup_desc ? `<p class="leading-relaxed"><strong class="text-zinc-300">Ground Attack:</strong> ${f.rush_matchup_desc}</p>` : ''}
                                        </div>
                                    </div>
                                `;
                            }).join('')}
                        </div>
                    </div>
                ` : ''}

                <!-- 4. TACTICAL GAME KEYS & X-FACTORS -->
                ${xFactors.length > 0 ? `
                    <div class="space-y-3">
                        <div class="flex items-center gap-2">
                            <i data-lucide="zap" class="w-4 h-4 text-zinc-400"></i>
                            <h3 class="text-xs uppercase font-bold text-zinc-300 tracking-wider">Tactical Game Keys & X-Factors</h3>
                        </div>

                        <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                            ${xFactors.map(xf => `
                                <div class="p-3.5 bg-zinc-900 border border-zinc-800/80 rounded-xl space-y-2 h2h-card">
                                    <div class="flex items-center gap-2">
                                        <i data-lucide="${xf.icon || 'zap'}" class="w-4 h-4 text-zinc-400 shrink-0"></i>
                                        <span class="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-zinc-800 text-zinc-300">${xf.tag}</span>
                                    </div>
                                    <div class="text-xs font-bold text-white">${xf.title}</div>
                                    <p class="text-xs text-zinc-400 leading-relaxed">${xf.detail}</p>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}

                <!-- 5. STARTING LEADERS & SKILL COMPARISON -->
                <div class="space-y-3">
                    <div class="flex items-center gap-2">
                        <i data-lucide="users" class="w-4 h-4 text-zinc-400"></i>
                        <h3 class="text-xs uppercase font-bold text-zinc-300 tracking-wider">Starting Leaders & Skill Production</h3>
                    </div>

                    <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                        <!-- Quarterbacks -->
                        <div class="p-3.5 bg-zinc-900 border border-zinc-800 rounded-xl space-y-2.5">
                            <div class="flex items-center justify-between text-xs font-bold text-zinc-300 border-b border-zinc-800/80 pb-1.5">
                                <span>Quarterback Matchup</span>
                                <span class="text-[10px] text-zinc-400">QB</span>
                            </div>
                            <div class="space-y-2 text-xs">
                                <div>
                                    <div class="font-semibold text-white flex items-center justify-between">
                                        <span>${awayLeaders.qb?.player_display_name || `${away} QB`}</span>
                                        <span class="text-[10px] text-zinc-400 font-mono">${away}</span>
                                    </div>
                                    <div class="text-[11px] text-zinc-400 font-mono">
                                        ${awayLeaders.qb?.pass_yds_per_game ?? '--'} YPG • ${awayLeaders.qb?.pass_tds ?? 0} TDs • ${awayLeaders.qb?.comp_pct ? (awayLeaders.qb.comp_pct * 100).toFixed(0) + '%' : '--'}
                                        ${awayLeaders.qb?.rush_yds_per_game >= 10 ? ` • ${awayLeaders.qb.rush_yds_per_game} Rush YPG` : ''}
                                    </div>
                                </div>
                                <div class="border-t border-zinc-800/50 pt-1.5">
                                    <div class="font-semibold text-white flex items-center justify-between">
                                        <span>${homeLeaders.qb?.player_display_name || `${home} QB`}</span>
                                        <span class="text-[10px] text-zinc-400 font-mono">${home}</span>
                                    </div>
                                    <div class="text-[11px] text-zinc-400 font-mono">
                                        ${homeLeaders.qb?.pass_yds_per_game ?? '--'} YPG • ${homeLeaders.qb?.pass_tds ?? 0} TDs • ${homeLeaders.qb?.comp_pct ? (homeLeaders.qb.comp_pct * 100).toFixed(0) + '%' : '--'}
                                        ${homeLeaders.qb?.rush_yds_per_game >= 10 ? ` • ${homeLeaders.qb.rush_yds_per_game} Rush YPG` : ''}
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Running Back 1 -->
                        <div class="p-3.5 bg-zinc-900 border border-zinc-800 rounded-xl space-y-2.5">
                            <div class="flex items-center justify-between text-xs font-bold text-zinc-300 border-b border-zinc-800/80 pb-1.5">
                                <span>Lead Back (RB1) Matchup</span>
                                <span class="text-[10px] text-zinc-400">RB1</span>
                            </div>
                            <div class="space-y-2 text-xs">
                                <div>
                                    <div class="font-semibold text-white flex items-center justify-between">
                                        <span>${awayLeaders.rb?.player_display_name || `${away} RB1`}</span>
                                        <span class="text-[10px] text-zinc-400 font-mono">${away}</span>
                                    </div>
                                    <div class="text-[11px] text-zinc-400 font-mono">
                                        ${awayLeaders.rb?.yds_per_game ?? '--'} Rush YPG • ${awayLeaders.rb?.carries_per_game ?? '--'} Car/G • ${awayLeaders.rb?.rush_tds ?? 0} TDs
                                    </div>
                                </div>
                                <div class="border-t border-zinc-800/50 pt-1.5">
                                    <div class="font-semibold text-white flex items-center justify-between">
                                        <span>${homeLeaders.rb?.player_display_name || `${home} RB1`}</span>
                                        <span class="text-[10px] text-zinc-400 font-mono">${home}</span>
                                    </div>
                                    <div class="text-[11px] text-zinc-400 font-mono">
                                        ${homeLeaders.rb?.yds_per_game ?? '--'} Rush YPG • ${homeLeaders.rb?.carries_per_game ?? '--'} Car/G • ${homeLeaders.rb?.rush_tds ?? 0} TDs
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Wide Receiver 1 -->
                        <div class="p-3.5 bg-zinc-900 border border-zinc-800 rounded-xl space-y-2.5">
                            <div class="flex items-center justify-between text-xs font-bold text-zinc-300 border-b border-zinc-800/80 pb-1.5">
                                <span>Target Leader (WR1) Matchup</span>
                                <span class="text-[10px] text-zinc-400">WR1</span>
                            </div>
                            <div class="space-y-2 text-xs">
                                <div>
                                    <div class="font-semibold text-white flex items-center justify-between">
                                        <span>${awayLeaders.wr?.player_display_name || `${away} WR1`}</span>
                                        <span class="text-[10px] text-zinc-400 font-mono">${away}</span>
                                    </div>
                                    <div class="text-[11px] text-zinc-400 font-mono">
                                        ${awayLeaders.wr?.yds_per_game ?? '--'} Rec YPG • ${awayLeaders.wr?.tgt_per_game ?? '--'} Tgt/G • ${awayLeaders.wr?.rec_tds ?? 0} TDs
                                    </div>
                                </div>
                                <div class="border-t border-zinc-800/50 pt-1.5">
                                    <div class="font-semibold text-white flex items-center justify-between">
                                        <span>${homeLeaders.wr?.player_display_name || `${home} WR1`}</span>
                                        <span class="text-[10px] text-zinc-400 font-mono">${home}</span>
                                    </div>
                                    <div class="text-[11px] text-zinc-400 font-mono">
                                        ${homeLeaders.wr?.yds_per_game ?? '--'} Rec YPG • ${homeLeaders.wr?.tgt_per_game ?? '--'} Tgt/G • ${homeLeaders.wr?.rec_tds ?? 0} TDs
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 6. POSITIONAL MATCHUP SPOTLIGHTS & DEFENSIVE TESTS -->
                ${spotlights.length > 0 ? `
                    <div class="space-y-3">
                        <div class="flex items-center justify-between">
                            <div class="flex items-center gap-2">
                                <i data-lucide="shield-check" class="w-4 h-4 text-zinc-400"></i>
                                <h3 class="text-xs uppercase font-bold text-zinc-300 tracking-wider">Defensive Matchup Spotlights (RB1, WR1 & QB Scramble Profiles)</h3>
                            </div>
                            <span class="text-[11px] font-mono text-zinc-400">${spotlights.length} profiles</span>
                        </div>

                        <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                            ${spotlights.map(s => `
                                <div class="p-3.5 bg-zinc-900 border border-zinc-800/80 rounded-xl space-y-2 h2h-card">
                                    <div class="flex items-center justify-between">
                                        <span class="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-zinc-800 text-zinc-300">
                                            ${s.tag}
                                        </span>
                                        <span class="px-2 py-0.5 rounded text-[10px] font-semibold ${getSpotlightBadgeClass(s.level)}">
                                            ${s.verdict}
                                        </span>
                                    </div>
                                    <div class="text-xs font-bold text-white">${s.title}</div>
                                    <div class="font-mono text-[11px] text-zinc-300 bg-zinc-950/60 px-2.5 py-1.5 rounded border border-zinc-800/60">
                                        ${s.stat_line}
                                    </div>
                                    <p class="text-xs text-zinc-400 leading-relaxed">${s.detail}</p>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}
            </div>
        `;
    },

    /**
     * Render Full Season Schedule by Week
     */
    renderSchedule(scheduleData, selectedWeek = '1') {
        if (!scheduleData || !scheduleData.weeks || scheduleData.weeks.length === 0) {
            return `
                <div class="col-span-full p-8 text-center glass-panel">
                    <i data-lucide="calendar-x" class="w-8 h-8 mx-auto text-zinc-500 mb-2"></i>
                    <p class="text-sm text-zinc-400">No scheduled games found for this season.</p>
                </div>
            `;
        }

        let targetWeeks = scheduleData.weeks;
        if (selectedWeek && selectedWeek.toString().toLowerCase() !== 'all') {
            const wNum = parseInt(selectedWeek, 10);
            targetWeeks = scheduleData.weeks.filter(w => w.week === wNum);
            if (targetWeeks.length === 0) {
                targetWeeks = [scheduleData.weeks[0]];
            }
        }

        const getRankBadge = (rank, highIsGood = true) => {
            if (!rank) return '';
            const r = parseInt(rank, 10);
            if (isNaN(r)) return '';
            if (r <= 8) return 'rank-top';
            if (r >= 25) return 'rank-bot';
            return 'rank-mid';
        };

        const fmt1Dec = (val) => {
            if (val == null || val === '') return '--';
            const n = parseFloat(val);
            return isNaN(n) ? val : n.toFixed(1);
        };

        const fmtDiffYds = (val) => {
            if (val == null || val === '') return '--';
            const n = parseFloat(val);
            if (isNaN(n)) return val;
            return (n > 0 ? `+${n.toFixed(1)}` : n.toFixed(1)) + ' yds';
        };

        const fmtPct = (val) => {
            if (val == null || val === '') return '--';
            const n = parseFloat(val);
            if (isNaN(n)) return val;
            return `${n.toFixed(1)}%`;
        };

        const fmtSignedPct = (val) => {
            if (val == null || val === '') return '--';
            const n = parseFloat(val);
            if (isNaN(n)) return val;
            return (n > 0 ? `+${n.toFixed(1)}%` : `${n.toFixed(1)}%`);
        };

        const STAT_EXPLANATIONS = {
            'Vegas Implied Total': 'Projected points based on consensus sportsbook spread and over/under total.',
            'Pass Att/G': 'Average pass attempts per game thrown by offense. Core measure of passing volume and pace.',
            'Pass Yds Allowed/G': 'Passing yards conceded per game by defense. Ranked #1 (stingiest) to #32 (most allowed).',
            'PROE (Neutral Pass Intent)': 'Pass Rate Over Expected in neutral situations (within 7 pts, outside 2-minute drills). Reflects true coaching pass desire.',
            'YPA (Offense)': 'Yards Per Attempt: Total passing yards divided by pass attempts. Measures chunk play efficiency.',
            'YPA Allowed': 'Yards Per Attempt conceded by pass defense. Lower is better (restricts explosive gains).',
            'Explosive Pass Rate (20+ Yds)': 'Percentage of offensive passes gaining 20+ yards. Highlights high-ceiling vertical air attacks.',
            'Opp Explosive Pass Allowed': 'Percentage of opponent passes yielding 20+ yards. Tests secondary vulnerability to deep shots.',
            'Carries/G': 'Average rushing attempts per game by offense. Measures ground volume and commitment to run.',
            'Rush Yds Allowed/G': 'Rushing yards surrendered per game by defense. Ranked #1 (stoutest run wall) to #32 (most porous).',
            'YPC (Offense)': 'Yards Per Carry: Total rushing yards divided by carries. Evaluates offensive line run push and RB efficiency.',
            'YPC Allowed': 'Yards Per Carry surrendered on defense. Measures point-of-attack front-seven run stopping.',
            'vs RB1s': 'Median rushing yards allowed to primary RBs (≥45% rush share) relative to player baseline, capped for outlier stability. Negative indicates lockdown defense.',
            'vs RB2s': 'Median scrimmage yards allowed to secondary/committee backs (15%–44% rush share) relative to baseline.',
            'vs WR1s': 'Median receiving yards allowed to opponent alpha WRs (≥20% target share) relative to player baseline, capped for outliers. Negative indicates elite CB shadow/lockdown.',
            'vs WR2s': 'Median receiving yards conceded to secondary WRs (10%–20% target share) relative to baseline. Flags defenses that bracket WR1s but bleed yards to secondary targets.',
            'vs TE1s': 'Median receiving yards allowed to opponent starting tight ends (TE1) relative to player median baseline, capped for outliers. Negative indicates elite coverage against primary seam/red-zone threats.',
            'vs TE2s': 'Median receiving yards conceded to secondary tight ends (TE2) relative to player median baseline. Highlights vulnerabilities against 12-personnel (2-TE sets).',
            'Def Pressure Rate Gen': 'Percentage of opponent dropbacks where defense generates a QB hurry, knockdown, or sack.',
            'Pass Pro Pressure Allowed': 'Percentage of offensive dropbacks allowing pressure. Lower is better (cleaner pocket for QB).',
            'Sacks Generated/G': 'Average sacks recorded per game by defense. Measures pass rush finishing ability.'
        };

        const formatRow = (label, awayVal, awayRank, homeVal, homeRank, highIsGood = true, awayN = null, homeN = null) => {
            const awayNTag = awayN != null ? `<span class="text-[9px] text-zinc-500 font-mono ml-0.5">(n=${awayN})</span>` : '';
            const homeNTag = homeN != null ? `<span class="text-[9px] text-zinc-500 font-mono mr-0.5">(n=${homeN})</span>` : '';
            const awayBadge = awayRank ? `<span class="text-[10px] px-1.5 py-0.5 rounded font-mono font-semibold ${getRankBadge(awayRank, highIsGood)}">#${awayRank}</span>` : '';
            const homeBadge = homeRank ? `<span class="text-[10px] px-1.5 py-0.5 rounded font-mono font-semibold ${getRankBadge(homeRank, highIsGood)}">#${homeRank}</span>` : '';
            
            const explanation = STAT_EXPLANATIONS[label] || '';

            return `
                <div class="grid grid-cols-3 items-center py-1.5 border-b border-zinc-800/80 text-xs hover:bg-zinc-800/40 px-2 rounded transition">
                    <!-- Away Stat -->
                    <div class="flex items-center gap-1.5 text-left font-mono">
                        ${awayBadge}
                        <span class="text-zinc-200 font-bold">${awayVal != null ? awayVal : '--'}</span>
                        ${awayNTag}
                    </div>
                    <!-- Metric Label with 2s Delayed Info Box -->
                    <div class="text-center text-zinc-400 text-[11px] font-medium uppercase tracking-tight">
                        <span class="stat-info-trigger" onclick="event.stopPropagation();" title="${explanation ? '' : label}">
                            ${label}
                            ${explanation ? `
                                <div class="stat-info-box">
                                    <div class="font-bold text-zinc-100 mb-1 flex items-center gap-1.5 border-b border-zinc-700/60 pb-1">
                                        <i data-lucide="info" class="w-3 h-3 text-zinc-400 shrink-0"></i>
                                        <span>${label}</span>
                                    </div>
                                    <div class="text-zinc-300 text-[10.5px] leading-relaxed">
                                        ${explanation}
                                    </div>
                                </div>
                            ` : ''}
                        </span>
                    </div>
                    <!-- Home Stat -->
                    <div class="flex items-center justify-end gap-1.5 text-right font-mono">
                        ${homeNTag}
                        <span class="text-zinc-200 font-bold">${homeVal != null ? homeVal : '--'}</span>
                        ${homeBadge}
                    </div>
                </div>
            `;
        };

        return targetWeeks.map(w => {
            const games = w.games || [];
            return `
                <div class="space-y-4">
                    <div class="flex flex-wrap items-center justify-between border-b border-zinc-800 pb-2 gap-2">
                        <div class="flex flex-wrap items-center gap-2">
                            <span class="badge badge-high text-xs font-bold font-mono">Week ${w.week}</span>
                            <span class="text-xs text-zinc-400 font-mono">${w.date_range_str || (w.dates ? w.dates.join(', ') : '')}</span>
                            ${scheduleData.stats_baseline_note ? `<span class="badge bg-zinc-900 border border-zinc-700 text-emerald-400 text-[10px] font-mono flex items-center gap-1"><i data-lucide="history" class="w-3 h-3 text-emerald-400 shrink-0"></i><span>${scheduleData.stats_baseline_note}</span></span>` : ''}
                        </div>
                        <span class="text-xs text-zinc-400 font-mono">${games.length} Matchups (${w.completed_count || 0} Final)</span>
                    </div>

                    <div class="grid grid-cols-1 xl:grid-cols-2 gap-4">
                        ${games.map(g => {
                            const isFinal = g.is_final;
                            const a = g.away_stats || {};
                            const h = g.home_stats || {};

                            let awayRec = g.away_record || '0-0';
                            let homeRec = g.home_record || '0-0';
                            if (awayRec === '0-0' && g.away_prior_record) awayRec = `0-0 (${g.away_prior_record})`;
                            if (homeRec === '0-0' && g.home_prior_record) homeRec = `0-0 (${g.home_prior_record})`;

                            let oddsBadgeHtml = '';
                            if (isFinal) {
                                oddsBadgeHtml = `<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 flex items-center gap-1"><span class="w-1.5 h-1.5 rounded-full bg-emerald-400"></span><span>FINAL</span></span>`;
                            } else if (g.spread_line != null) {
                                const spreadText = g.spread_line === 0 ? 'PK' : (g.spread_line > 0 ? `${g.home_team} -${g.spread_line}` : `${g.away_team} -${Math.abs(g.spread_line)}`);
                                oddsBadgeHtml = `<span class="px-2 py-0.5 rounded text-[10px] font-semibold bg-zinc-800 text-zinc-300 border border-zinc-700 font-mono">${spreadText}${g.total_line ? ' • O/U ' + g.total_line : ''}</span>`;
                            } else {
                                oddsBadgeHtml = `<span class="px-2 py-0.5 rounded text-[10px] font-semibold bg-zinc-800 text-zinc-400 border border-zinc-700">UPCOMING</span>`;
                            }

                            const venue = [g.stadium, g.roof, g.temp ? `${g.temp}°F` : null, g.wind ? `${g.wind}mph wind` : null].filter(Boolean).join(' • ');

                            return `
                                <div class="glass-panel rounded-xl p-4 sm:p-5 border border-zinc-800 space-y-4 hover:border-purple-500/60 hover:bg-zinc-900/40 transition shadow-sm h2h-card cursor-pointer group" onclick="inspectMatchupInsights('${g.home_team}', '${g.away_team}', ${g.week || w.week || 1})" title="Click matchup to view Scheme Insights">
                                    <!-- Game Info Header Bar -->
                                    <div class="flex items-center justify-between text-xs text-zinc-400 border-b border-zinc-800 pb-2.5">
                                        <div class="flex items-center gap-2">
                                            <span class="px-1.5 py-0.5 rounded text-[10px] font-bold uppercase bg-zinc-800 text-zinc-300 border border-zinc-700 font-mono">
                                                W${g.week || w.week || 1}
                                            </span>
                                            <span class="font-medium text-zinc-300">
                                                ${g.weekday ? g.weekday + ', ' : ''}${g.gameday}
                                            </span>
                                            <span class="text-zinc-600">•</span>
                                            <span class="font-mono text-zinc-400">${g.gametime ? g.gametime + ' ET' : 'TBD'}</span>
                                        </div>
                                        <div class="flex items-center gap-2">
                                            ${oddsBadgeHtml}
                                            <span class="hidden sm:inline-flex items-center gap-1 text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-purple-500/15 text-purple-300 border border-purple-500/30 group-hover:bg-purple-500/25 transition">
                                                <i data-lucide="brain" class="w-3 h-3 text-purple-400"></i>
                                                <span>Scheme Insights</span>
                                            </span>
                                        </div>
                                    </div>

                                    <!-- Matchup Banner / Scoreboard (Click to view Scheme Insights) -->
                                    <div class="grid grid-cols-12 items-center py-2 px-2.5 rounded-lg bg-zinc-900/60 group-hover:bg-zinc-800/90 border border-transparent group-hover:border-purple-500/40 transition">
                                        <!-- Away Team (5 cols) -->
                                        <div class="col-span-5 flex items-center gap-2.5 min-w-0">
                                            <div class="w-10 h-10 rounded-xl bg-zinc-800 border border-zinc-700 flex items-center justify-center font-extrabold text-white text-sm font-mono shadow-inner shrink-0 group-hover:border-purple-400/50 transition">
                                                ${g.away_team}
                                            </div>
                                            <div class="min-w-0 flex-1">
                                                <h4 class="font-bold text-white text-xs sm:text-sm truncate group-hover:text-purple-200 transition" title="${g.away_team_name}">${g.away_team_name}</h4>
                                                <div class="text-[11px] text-zinc-400 flex items-center flex-wrap gap-x-1.5 gap-y-0.5 mt-0.5">
                                                    <span class="text-zinc-400 font-medium">Away</span>
                                                    <span class="text-zinc-600">•</span>
                                                    <span class="font-mono text-zinc-300 font-semibold whitespace-nowrap">${awayRec}</span>
                                                    ${a.def_overall_character ? `<span class="text-[9px] px-1.5 py-0.5 rounded font-mono font-semibold whitespace-nowrap ${a.def_overall_character === 'Elite Defense' ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30' : (a.def_overall_character === 'Vulnerable Defense' ? 'bg-rose-500/15 text-rose-400 border border-rose-500/30' : 'bg-zinc-800 text-zinc-400 border border-zinc-700')}">${a.def_overall_character}</span>` : ''}
                                                </div>
                                            </div>
                                        </div>

                                        <!-- Center vs / Scores (2 cols) -->
                                        <div class="col-span-2 text-center flex flex-col items-center justify-center">
                                            ${isFinal ? `
                                                <div class="flex items-center justify-center gap-1.5 font-mono font-extrabold text-lg sm:text-xl text-white">
                                                    <span class="${(g.away_score || 0) > (g.home_score || 0) ? 'text-emerald-400 font-black' : 'text-zinc-300'}">${Math.round(g.away_score)}</span>
                                                    <span class="text-zinc-600 text-xs">-</span>
                                                    <span class="${(g.home_score || 0) > (g.away_score || 0) ? 'text-emerald-400 font-black' : 'text-zinc-300'}">${Math.round(g.home_score)}</span>
                                                </div>
                                                <span class="text-[9px] font-bold font-mono text-emerald-400 tracking-wider uppercase">FINAL</span>
                                            ` : `
                                                <div class="w-7 h-7 mx-auto rounded-full bg-zinc-800 border border-zinc-700 flex items-center justify-center font-bold text-zinc-400 text-xs">
                                                    @
                                                </div>
                                            `}
                                        </div>

                                        <!-- Home Team (5 cols) -->
                                        <div class="col-span-5 flex items-center justify-end gap-2.5 text-right min-w-0">
                                            <div class="min-w-0 flex-1">
                                                <h4 class="font-bold text-white text-xs sm:text-sm truncate group-hover:text-purple-200 transition" title="${g.home_team_name}">${g.home_team_name}</h4>
                                                <div class="text-[11px] text-zinc-400 flex items-center justify-end flex-wrap gap-x-1.5 gap-y-0.5 mt-0.5">
                                                    ${h.def_overall_character ? `<span class="text-[9px] px-1.5 py-0.5 rounded font-mono font-semibold whitespace-nowrap ${h.def_overall_character === 'Elite Defense' ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30' : (h.def_overall_character === 'Vulnerable Defense' ? 'bg-rose-500/15 text-rose-400 border border-rose-500/30' : 'bg-zinc-800 text-zinc-400 border border-zinc-700')}">${h.def_overall_character}</span><span class="text-zinc-600">•</span>` : ''}
                                                    <span class="font-mono text-zinc-300 font-semibold whitespace-nowrap">${homeRec}</span>
                                                    <span class="text-zinc-600">•</span>
                                                    <span class="text-zinc-400 font-medium">Home</span>
                                                </div>
                                            </div>
                                            <div class="w-10 h-10 rounded-xl bg-zinc-800 border border-zinc-700 flex items-center justify-center font-extrabold text-white text-sm font-mono shadow-inner shrink-0 group-hover:border-purple-400/50 transition">
                                                ${g.home_team}
                                            </div>
                                        </div>
                                    </div>

                                    <!-- Per-Game Team Stat Comparison Table (Volume x Efficiency) -->
                                    <div class="bg-zinc-950/80 rounded-xl border border-zinc-800 p-3 space-y-1">
                                        <div class="grid grid-cols-3 text-[10px] uppercase font-bold text-zinc-500 border-b border-zinc-800 pb-1.5 px-2">
                                            <div class="text-left font-mono">${g.away_team} STAT & RANK</div>
                                            <div class="text-center">
                                                <span class="stat-info-trigger">
                                                    PER-GAME STAT
                                                    <div class="stat-info-box">
                                                        <div class="font-bold text-zinc-100 mb-1 flex items-center gap-1.5 border-b border-zinc-700/60 pb-1 text-left">
                                                            <i data-lucide="info" class="w-3 h-3 text-zinc-400 shrink-0"></i>
                                                            <span>Per-Game Team Stats (Median Baselines)</span>
                                                        </div>
                                                        <div class="text-zinc-300 text-[10.5px] leading-relaxed text-left normal-case">
                                                            Head-to-head offensive volume & efficiency vs defensive resistance. Positional defense metrics (vs RB1, RB2, WR1, WR2, TE1, TE2) utilize player median baselines to isolate true defensive containment from single-game blowout outliers. Hover over any individual stat below for its exact definition.
                                                        </div>
                                                    </div>
                                                </span>
                                            </div>
                                            <div class="text-right font-mono">${g.home_team} STAT & RANK</div>
                                        </div>

                                        <!-- 1. Scoring (Vegas Implied Total) -->
                                        ${g.implied_away_total != null || g.implied_home_total != null ? formatRow('Vegas Implied Total', fmt1Dec(g.implied_away_total), null, fmt1Dec(g.implied_home_total), null, true) : ''}

                                        <!-- 2. Passing: Volume & Game-Script-Neutral Intent -->
                                        ${formatRow('Pass Att/G', fmt1Dec(a.pass_att_per_game), a.rank_pass_att, fmt1Dec(h.pass_att_per_game), h.rank_pass_att, true, a.games_played, h.games_played)}
                                        ${formatRow('Pass Yds Allowed/G', fmt1Dec(a.pass_yds_allowed_per_game), a.rank_pass_allowed, fmt1Dec(h.pass_yds_allowed_per_game), h.rank_pass_allowed, false)}
                                        ${a.proe != null || h.proe != null ? formatRow('PROE (Neutral Pass Intent)', fmtSignedPct(a.proe), a.rank_proe, fmtSignedPct(h.proe), h.rank_proe, true, a.neutral_plays_sample, h.neutral_plays_sample) : ''}

                                        <!-- 3. Passing: Efficiency & Explosiveness (20+ Yd Chunk Plays) -->
                                        ${formatRow('YPA (Offense)', fmt1Dec(a.yds_per_att), a.rank_ypa, fmt1Dec(h.yds_per_att), h.rank_ypa, true)}
                                        ${formatRow('YPA Allowed', fmt1Dec(a.opp_yds_per_att), a.rank_opp_ypa, fmt1Dec(h.opp_yds_per_att), h.rank_opp_ypa, false)}
                                        ${a.exp_pass_rate != null || h.exp_pass_rate != null ? formatRow('Explosive Pass Rate (20+ Yds)', fmtPct(a.exp_pass_rate), a.rank_exp_pass_rate, fmtPct(h.exp_pass_rate), h.rank_exp_pass_rate, true, a.total_pass_att_sample, h.total_pass_att_sample) : ''}
                                        ${a.opp_exp_pass_rate != null || h.opp_exp_pass_rate != null ? formatRow('Opp Explosive Pass Allowed', fmtPct(a.opp_exp_pass_rate), a.rank_opp_exp_pass_rate, fmtPct(h.opp_exp_pass_rate), h.rank_opp_exp_pass_rate, false, a.opp_pass_att_sample, h.opp_pass_att_sample) : ''}

                                        <!-- 4. Rushing: Volume -->
                                        ${formatRow('Carries/G', fmt1Dec(a.carries_per_game), a.rank_carries, fmt1Dec(h.carries_per_game), h.rank_carries, true, a.games_played, h.games_played)}
                                        ${formatRow('Rush Yds Allowed/G', fmt1Dec(a.rush_yds_allowed_per_game), a.rank_rush_allowed, fmt1Dec(h.rush_yds_allowed_per_game), h.rank_rush_allowed, false)}

                                        <!-- 5. Rushing: Efficiency -->
                                        ${formatRow('YPC (Offense)', fmt1Dec(a.yds_per_carry), a.rank_ypc, fmt1Dec(h.yds_per_carry), h.rank_ypc, true)}
                                        ${formatRow('YPC Allowed', fmt1Dec(a.opp_yds_per_carry), a.rank_opp_ypc, fmt1Dec(h.opp_yds_per_carry), h.rank_opp_ypc, false)}

                                        <!-- 6. Defense vs Positional Roles (RB1, RB2, WR1, WR2, TE1, TE2) -->
                                        ${a.def_star_rb_diff != null || h.def_star_rb_diff != null ? formatRow('vs RB1s', fmtDiffYds(a.def_star_rb_diff), a.rank_def_star_rb, fmtDiffYds(h.def_star_rb_diff), h.rank_def_star_rb, false) : ''}
                                        ${a.def_rb2_diff != null || h.def_rb2_diff != null ? formatRow('vs RB2s', fmtDiffYds(a.def_rb2_diff), a.rank_def_rb2, fmtDiffYds(h.def_rb2_diff), h.rank_def_rb2, false) : ''}
                                        ${a.def_star_wr_diff != null || h.def_star_wr_diff != null ? formatRow('vs WR1s', fmtDiffYds(a.def_star_wr_diff), a.rank_def_star_wr, fmtDiffYds(h.def_star_wr_diff), h.rank_def_star_wr, false) : ''}
                                        ${a.def_wr2_diff != null || h.def_wr2_diff != null ? formatRow('vs WR2s', fmtDiffYds(a.def_wr2_diff), a.rank_def_wr2, fmtDiffYds(h.def_wr2_diff), h.rank_def_wr2, false) : ''}
                                        ${a.def_te1_diff != null || h.def_te1_diff != null ? formatRow('vs TE1s', fmtDiffYds(a.def_te1_diff), a.rank_def_te1, fmtDiffYds(h.def_te1_diff), h.rank_def_te1, false) : ''}
                                        ${a.def_te2_diff != null || h.def_te2_diff != null ? formatRow('vs TE2s', fmtDiffYds(a.def_te2_diff), a.rank_def_te2, fmtDiffYds(h.def_te2_diff), h.rank_def_te2, false) : ''}

                                        <!-- 7. Trenches: Pressure Generation & Pass Protection Leak -->
                                        ${a.pressure_rate_generated != null || h.pressure_rate_generated != null ? formatRow('Def Pressure Rate Gen', fmtPct(a.pressure_rate_generated), a.rank_pressure_rate_generated, fmtPct(h.pressure_rate_generated), h.rank_pressure_rate_generated, true, a.opp_pass_att_sample, h.opp_pass_att_sample) : ''}
                                        ${a.pressure_rate_allowed != null || h.pressure_rate_allowed != null ? formatRow('Pass Pro Pressure Allowed', fmtPct(a.pressure_rate_allowed), a.rank_pressure_rate_allowed, fmtPct(h.pressure_rate_allowed), h.rank_pressure_rate_allowed, false, a.total_pass_att_sample, h.total_pass_att_sample) : ''}
                                        ${formatRow('Sacks Generated/G', fmt1Dec(a.sacks_forced_per_game), a.rank_sacks_forced, fmt1Dec(h.sacks_forced_per_game), h.rank_sacks_forced, true)}
                                    </div>

                                    <!-- Pass / Run Opportunity Funnel Card (Scaled by Pace & Vegas Total) -->
                                    ${g.funnel ? `
                                    <div class="bg-zinc-950/80 rounded-xl border border-zinc-800 p-3 space-y-2">
                                        <div class="flex items-center justify-between text-[10px] font-bold uppercase tracking-wider text-zinc-400 border-b border-zinc-800 pb-1.5 px-1">
                                            <span class="flex items-center gap-1.5">
                                                <i data-lucide="split" class="w-3 h-3 text-zinc-400"></i>
                                                <span>Pass / Run Opportunity Funnel</span>
                                            </span>
                                            <span class="font-mono text-[9px] text-zinc-300 bg-zinc-900 px-2 py-0.5 rounded border border-zinc-800">
                                                Game Pace: ${((g.funnel.away?.proj_total_plays || 63) + (g.funnel.home?.proj_total_plays || 63)).toFixed(0)} Plays
                                            </span>
                                        </div>
                                        <div class="grid grid-cols-2 gap-2 text-xs">
                                            <!-- Away Team Funnel -->
                                            <div class="bg-zinc-900 p-2.5 rounded-lg border border-zinc-800/80 space-y-1.5">
                                                <div class="flex items-center justify-between">
                                                    <span class="font-bold text-zinc-200 font-mono text-[11px]">${g.away_team} Offense</span>
                                                    <div class="flex items-center gap-1">
                                                        <span class="text-[9px] px-1.5 py-0.5 rounded font-mono font-bold bg-zinc-800 text-zinc-300 border border-zinc-700">${g.funnel.away?.proj_total_plays || '--'} Plays</span>
                                                        <span class="text-[9px] px-1.5 py-0.5 rounded font-mono font-semibold ${g.funnel.away?.off_tendency === 'PASS_HEAVY' ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30' : (g.funnel.away?.off_tendency === 'RUN_HEAVY' ? 'bg-amber-500/15 text-amber-400 border border-amber-500/30' : 'bg-zinc-800 text-zinc-400')}">${g.funnel.away?.off_tendency || 'BALANCED'}</span>
                                                    </div>
                                                </div>
                                                <div class="text-[11px] text-zinc-400 flex justify-between font-mono">
                                                    <span>Pass: <strong class="text-zinc-200">${g.funnel.away?.proj_pass_att || '--'} att</strong> (${g.funnel.away?.pass_rate || 55}%)</span>
                                                    <span>Rush: <strong class="text-zinc-200">${g.funnel.away?.proj_carries || '--'} car</strong> (${g.funnel.away?.run_rate || 45}%)</span>
                                                </div>
                                                <div class="w-full bg-zinc-800 h-1.5 rounded-full overflow-hidden flex">
                                                    <div class="bg-cyan-500 h-full" style="width: ${g.funnel.away?.pass_rate || 55}%"></div>
                                                    <div class="bg-amber-500 h-full" style="width: ${g.funnel.away?.run_rate || 45}%"></div>
                                                </div>
                                                <div class="text-[9px] text-zinc-400 pt-0.5 flex items-center justify-between">
                                                    <span>Opp D Funnel:</span>
                                                    <span class="font-semibold text-zinc-300 font-mono">${g.funnel.away?.def_funnel || 'Balanced'}</span>
                                                </div>
                                            </div>
                                            <!-- Home Team Funnel -->
                                            <div class="bg-zinc-900 p-2.5 rounded-lg border border-zinc-800/80 space-y-1.5">
                                                <div class="flex items-center justify-between">
                                                    <span class="font-bold text-zinc-200 font-mono text-[11px]">${g.home_team} Offense</span>
                                                    <div class="flex items-center gap-1">
                                                        <span class="text-[9px] px-1.5 py-0.5 rounded font-mono font-bold bg-zinc-800 text-zinc-300 border border-zinc-700">${g.funnel.home?.proj_total_plays || '--'} Plays</span>
                                                        <span class="text-[9px] px-1.5 py-0.5 rounded font-mono font-semibold ${g.funnel.home?.off_tendency === 'PASS_HEAVY' ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30' : (g.funnel.home?.off_tendency === 'RUN_HEAVY' ? 'bg-amber-500/15 text-amber-400 border border-amber-500/30' : 'bg-zinc-800 text-zinc-400')}">${g.funnel.home?.off_tendency || 'BALANCED'}</span>
                                                    </div>
                                                </div>
                                                <div class="text-[11px] text-zinc-400 flex justify-between font-mono">
                                                    <span>Pass: <strong class="text-zinc-200">${g.funnel.home?.proj_pass_att || '--'} att</strong> (${g.funnel.home?.pass_rate || 55}%)</span>
                                                    <span>Rush: <strong class="text-zinc-200">${g.funnel.home?.proj_carries || '--'} car</strong> (${g.funnel.home?.run_rate || 45}%)</span>
                                                </div>
                                                <div class="w-full bg-zinc-800 h-1.5 rounded-full overflow-hidden flex">
                                                    <div class="bg-cyan-500 h-full" style="width: ${g.funnel.home?.pass_rate || 55}%"></div>
                                                    <div class="bg-amber-500 h-full" style="width: ${g.funnel.home?.run_rate || 45}%"></div>
                                                </div>
                                                <div class="text-[9px] text-zinc-400 pt-0.5 flex items-center justify-between">
                                                    <span>Opp D Funnel:</span>
                                                    <span class="font-semibold text-zinc-300 font-mono">${g.funnel.home?.def_funnel || 'Balanced'}</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    ` : ''}

                                    <!-- Venue & Weather Line -->
                                    <div class="flex flex-wrap items-center justify-between text-[11px] text-zinc-400 pt-0.5 px-1">
                                        <div class="flex items-center gap-1.5 truncate">
                                            <i data-lucide="map-pin" class="w-3 h-3 text-zinc-400"></i>
                                            <span class="truncate">${venue || 'NFL Stadium'}</span>
                                        </div>
                                    </div>

                                    <!-- Action Buttons: Scheme Insights & Matchup Lab -->
                                    <div class="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1">
                                        <button onclick="event.stopPropagation(); inspectMatchupInsights('${g.home_team}', '${g.away_team}', ${g.week || w.week || 1})" class="py-2 px-3 rounded-lg bg-purple-600/30 hover:bg-purple-600 text-purple-200 hover:text-white text-xs font-semibold flex items-center justify-center gap-2 border border-purple-500/50 hover:border-purple-400 transition shadow-sm cursor-pointer">
                                            <i data-lucide="brain" class="w-3.5 h-3.5 text-purple-300"></i>
                                            <span>Scheme Insights</span>
                                        </button>
                                        <button onclick="event.stopPropagation(); inspectMatchupInLab('${g.home_team}', '${g.away_team}')" class="py-2 px-3 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-200 hover:text-white text-xs font-semibold flex items-center justify-center gap-2 border border-zinc-700 transition cursor-pointer">
                                            <i data-lucide="crosshair" class="w-3.5 h-3.5 text-zinc-300"></i>
                                            <span>Matchup Lab (H2H)</span>
                                        </button>
                                    </div>
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>
            `;
        }).join('');
    },

    /**
     * Render Scheme-Based Matchup Insights (Coverage, Run Splits, Target Rates, Breakouts, Game-Flow)
     */
    renderSchemeInsights(data) {
        if (!data || data.error) {
            return `
                <div class="p-8 text-center text-zinc-400 glass-panel">
                    <i data-lucide="alert-circle" class="w-8 h-8 mx-auto text-purple-400 mb-2"></i>
                    <p class="text-sm font-semibold text-zinc-300">${data?.error || 'No scheme insight data found for this matchup.'}</p>
                    <p class="text-xs text-zinc-500 mt-1">Please select another matchup from the dropdown above.</p>
                </div>
            `;
        }

        const home = data.home_team || 'HOME';
        const homeName = data.home_team_name || home;
        const away = data.away_team || 'AWAY';
        const awayName = data.away_team_name || away;
        const week = data.week || 1;
        const season = data.season || 2026;

        const cov = data.coverage_tendencies || {};
        const aVsH = cov.away_vs_home || {};
        const hVsA = cov.home_vs_away || {};

        const runSplits = data.run_scheme_splits || {};
        const aRun = runSplits.away_unit || {};
        const hRun = runSplits.home_unit || {};

        const targets = data.positional_target_rates || {};
        const aTargets = targets.away_player_targets || [];
        const hTargets = targets.home_player_targets || [];
        const aDist = targets.away_distribution || {};
        const hDist = targets.home_distribution || {};

        const breakouts = data.efficiency_breakouts || [];
        const gameFlow = data.game_flow_tempo || {};
        const aTempo = gameFlow.away_tempo || {};
        const hTempo = gameFlow.home_tempo || {};

        const getRankClass = (rank) => {
            if (!rank) return '';
            const r = parseInt(rank, 10);
            if (r <= 8) return 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30';
            if (r >= 22) return 'bg-rose-500/15 text-rose-400 border border-rose-500/30';
            return 'bg-zinc-800 text-zinc-300 border border-zinc-700';
        };

        const getAdvClass = (adv) => {
            if (adv === 'High Advantage') return 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40';
            if (adv === 'Disadvantage') return 'bg-rose-500/20 text-rose-300 border border-rose-500/40';
            return 'bg-zinc-800 text-zinc-300 border border-zinc-700';
        };

        const renderCoverageCard = (covUnit, offTeam, defTeam) => {
            const shells = covUnit.shells || {};
            const zonePct = covUnit.def_zone_pct || 65;
            const manPct = covUnit.def_man_pct || 35;
            const mfoPct = covUnit.def_mfo_pct || 55;
            const mfcPct = covUnit.def_mfc_pct || 45;

            return `
                <div class="glass-panel p-4 sm:p-5 rounded-xl border border-zinc-800 space-y-4 hover:border-zinc-700 transition">
                    <div class="flex flex-wrap items-center justify-between border-b border-zinc-800 pb-3 gap-2">
                        <div class="flex items-center gap-2">
                            <span class="px-2 py-0.5 rounded text-xs font-bold font-mono bg-purple-500/20 text-purple-300 border border-purple-500/30">
                                ${offTeam} PASS vs ${defTeam} COVERAGE
                            </span>
                        </div>
                        <span class="text-xs text-zinc-400 font-mono font-semibold">
                            ${covUnit.def_archetype || 'Hybrid Shell'}
                        </span>
                    </div>

                    <!-- 1. Zone vs Man Split Bar -->
                    <div class="space-y-1.5">
                        <div class="flex items-center justify-between text-xs font-mono">
                            <span class="text-purple-300 font-semibold flex items-center gap-1.5">
                                <span class="w-2 h-2 rounded-full bg-purple-400"></span>
                                <span>Zone Coverage: <strong>${zonePct.toFixed(1)}%</strong></span>
                            </span>
                            <span class="text-cyan-300 font-semibold flex items-center gap-1.5">
                                <span>Man Coverage: <strong>${manPct.toFixed(1)}%</strong></span>
                                <span class="w-2 h-2 rounded-full bg-cyan-400"></span>
                            </span>
                        </div>
                        <div class="w-full bg-zinc-800 h-2.5 rounded-full overflow-hidden flex shadow-inner">
                            <div class="bg-purple-500 h-full transition-all duration-500" style="width: ${zonePct}%" title="${defTeam} Zone Coverage: ${zonePct.toFixed(1)}%"></div>
                            <div class="bg-cyan-500 h-full transition-all duration-500" style="width: ${manPct}%" title="${defTeam} Man Coverage: ${manPct.toFixed(1)}%"></div>
                        </div>
                    </div>

                    <!-- 2. Middle-Field Open (MFO) vs Middle-Field Closed (MFC) Sets -->
                    <div class="space-y-1.5">
                        <div class="flex items-center justify-between text-xs font-mono">
                            <span class="text-emerald-300 font-semibold flex items-center gap-1.5">
                                <span class="w-2 h-2 rounded-full bg-emerald-400"></span>
                                <span>MFO (Two-High / Split Safeties): <strong>${mfoPct.toFixed(1)}%</strong></span>
                            </span>
                            <span class="text-amber-300 font-semibold flex items-center gap-1.5">
                                <span>MFC (Single-High / 8-in-Box): <strong>${mfcPct.toFixed(1)}%</strong></span>
                                <span class="w-2 h-2 rounded-full bg-amber-400"></span>
                            </span>
                        </div>
                        <div class="w-full bg-zinc-800 h-2.5 rounded-full overflow-hidden flex shadow-inner">
                            <div class="bg-emerald-500 h-full transition-all duration-500" style="width: ${mfoPct}%" title="${defTeam} Middle-Field Open: ${mfoPct.toFixed(1)}%"></div>
                            <div class="bg-amber-500 h-full transition-all duration-500" style="width: ${mfcPct}%" title="${defTeam} Middle-Field Closed: ${mfcPct.toFixed(1)}%"></div>
                        </div>
                    </div>

                    <!-- Specific Shell Distribution Chips -->
                    <div class="space-y-1.5 pt-1">
                        <span class="text-[10px] text-zinc-400 uppercase font-semibold tracking-wider">Defensive Coverage Shell Distribution:</span>
                        <div class="grid grid-cols-3 sm:grid-cols-6 gap-1.5 text-center text-[10px] font-mono">
                            ${Object.entries(shells).map(([shellName, rate]) => `
                                <div class="p-1.5 rounded bg-zinc-900 border border-zinc-800">
                                    <div class="text-zinc-400 truncate text-[9px]">${shellName.split(' ')[0]} ${shellName.split(' ')[1] || ''}</div>
                                    <div class="font-bold text-zinc-100 mt-0.5">${rate != null ? rate.toFixed(1) + '%' : '--'}</div>
                                </div>
                            `).join('')}
                        </div>
                    </div>

                    <!-- Film Room Tactical Note -->
                    <div class="p-3 rounded-lg bg-zinc-900/80 border border-zinc-800/90 text-xs text-zinc-300 flex items-start gap-2">
                        <i data-lucide="shield-alert" class="w-4 h-4 text-purple-400 shrink-0 mt-0.5"></i>
                        <div class="leading-relaxed">
                            <strong class="text-purple-300 font-semibold">Film Room Scheme Takeaway:</strong>
                            <span class="ml-1">${covUnit.shell_takeaway || 'Disguised shell challenges outside boundary separation.'}</span>
                        </div>
                    </div>
                </div>
            `;
        };

        const renderRunCard = (rUnit, oppTeam) => {
            const zRate = rUnit.offense_zone_pct || 50;
            const gRate = rUnit.offense_gap_pct || 50;
            return `
                <div class="glass-panel p-4 sm:p-5 rounded-xl border border-zinc-800 space-y-4 hover:border-zinc-700 transition">
                    <div class="flex flex-wrap items-center justify-between border-b border-zinc-800 pb-3 gap-2">
                        <span class="px-2 py-0.5 rounded text-xs font-bold font-mono bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                            ${rUnit.team} RUSH OFFENSE vs ${oppTeam} RUN DEFENSE
                        </span>
                        <span class="badge ${getAdvClass(rUnit.scheme_advantage)} text-[10px] font-mono">
                            ${rUnit.scheme_advantage}
                        </span>
                    </div>

                    <!-- Offense Run Style Split -->
                    <div class="space-y-1.5">
                        <div class="flex items-center justify-between text-xs font-mono">
                            <span class="text-cyan-300 font-semibold flex items-center gap-1.5">
                                <span class="w-2 h-2 rounded-full bg-cyan-400"></span>
                                <span>Zone Run: <strong>${zRate.toFixed(1)}%</strong> (${rUnit.zone_ypc?.toFixed(2) || '--'} YPC)</span>
                            </span>
                            <span class="text-amber-300 font-semibold flex items-center gap-1.5">
                                <span>Man/Gap: <strong>${gRate.toFixed(1)}%</strong> (${rUnit.gap_ypc?.toFixed(2) || '--'} YPC)</span>
                                <span class="w-2 h-2 rounded-full bg-amber-400"></span>
                            </span>
                        </div>
                        <div class="w-full bg-zinc-800 h-2.5 rounded-full overflow-hidden flex shadow-inner">
                            <div class="bg-cyan-500 h-full transition-all duration-500" style="width: ${zRate}%" title="Zone Runs: ${zRate.toFixed(1)}%"></div>
                            <div class="bg-amber-500 h-full transition-all duration-500" style="width: ${gRate}%" title="Man/Gap Runs: ${gRate.toFixed(1)}%"></div>
                        </div>
                    </div>

                    <!-- Defense Vulnerabilities by Scheme -->
                    <div class="grid grid-cols-2 gap-2 text-xs font-mono">
                        <div class="p-2.5 rounded-lg bg-zinc-900 border border-zinc-800 space-y-1">
                            <div class="text-[10px] text-zinc-400 uppercase font-semibold">Opp D vs Zone Runs</div>
                            <div class="flex items-center justify-between">
                                <span class="text-zinc-200 font-bold">${rUnit.opp_def_zone_ypc_allowed?.toFixed(2) || '--'} YPC allowed</span>
                                <span class="px-1.5 py-0.5 rounded text-[10px] font-bold ${getRankClass(rUnit.opp_def_zone_rank)}">#${rUnit.opp_def_zone_rank || '--'}</span>
                            </div>
                        </div>
                        <div class="p-2.5 rounded-lg bg-zinc-900 border border-zinc-800 space-y-1">
                            <div class="text-[10px] text-zinc-400 uppercase font-semibold">Opp D vs Gap/Power Runs</div>
                            <div class="flex items-center justify-between">
                                <span class="text-zinc-200 font-bold">${rUnit.opp_def_gap_ypc_allowed?.toFixed(2) || '--'} YPC allowed</span>
                                <span class="px-1.5 py-0.5 rounded text-[10px] font-bold ${getRankClass(rUnit.opp_def_gap_rank)}">#${rUnit.opp_def_gap_rank || '--'}</span>
                            </div>
                        </div>
                    </div>

                    <!-- Alignment Diagnostic -->
                    <div class="p-3 rounded-lg bg-zinc-900/80 border border-zinc-800/90 text-xs text-zinc-300 flex items-start gap-2">
                        <i data-lucide="split" class="w-4 h-4 text-cyan-400 shrink-0 mt-0.5"></i>
                        <div class="leading-relaxed">
                            <strong class="text-cyan-300 font-semibold">Scheme Alignment Diagnostic:</strong>
                            <span class="ml-1">${rUnit.alignment_diagnostic || 'Ground attack meets standard run front.'}</span>
                        </div>
                    </div>
                </div>
            `;
        };

        const renderTargetTable = (pTargets, teamLabel) => {
            return `
                <div class="table-container glass-panel rounded-xl border border-zinc-800 overflow-x-auto">
                    <table class="sports-table w-full text-left text-xs">
                        <thead>
                            <tr class="text-[10.5px] uppercase tracking-wider text-zinc-400 border-b border-zinc-800 bg-zinc-900/60 font-semibold">
                                <th class="py-2.5 px-3">Player / Pos</th>
                                <th class="py-2.5 px-2 text-center">Alignment Profile</th>
                                <th class="py-2.5 px-2 text-center">vs Man Share</th>
                                <th class="py-2.5 px-2 text-center">vs Zone Share</th>
                                <th class="py-2.5 px-2 text-center">YPRR (Scheme)</th>
                                <th class="py-2.5 px-3 text-left">Coverage Look Projection</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-zinc-800/60">
                            ${pTargets.map(p => `
                                <tr class="hover:bg-zinc-800/40 transition font-mono">
                                    <td class="py-2.5 px-3 font-sans font-bold text-zinc-100 flex items-center gap-1.5">
                                        <span>${p.name}</span>
                                        <span class="badge badge-med text-[10px] font-mono">${p.pos}</span>
                                    </td>
                                    <td class="py-2.5 px-2 text-center text-[11px] text-zinc-300">
                                        ${p.slot_pct > 40 ? `<span class="badge bg-purple-500/20 text-purple-300 border border-purple-500/40 text-[10px]">Slot ${p.slot_pct}%</span>` : (p.inline_pct > 30 ? `<span class="badge bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 text-[10px]">Inline TE ${p.inline_pct}%</span>` : `<span class="badge bg-zinc-800 text-zinc-300 border border-zinc-700 text-[10px]">Outside ${p.outside_pct}%</span>`)}
                                    </td>
                                    <td class="py-2.5 px-2 text-center text-zinc-300 font-bold">${p.tgt_man?.toFixed(1)}%</td>
                                    <td class="py-2.5 px-2 text-center text-purple-300 font-bold">${p.tgt_zone?.toFixed(1)}%</td>
                                    <td class="py-2.5 px-2 text-center text-emerald-400 font-bold">${p.yprr_zone?.toFixed(2) || '--'}</td>
                                    <td class="py-2.5 px-3 text-left font-sans text-[11px]">
                                        <span class="${p.boost_tier === 'High Boost' ? 'text-emerald-300 font-bold' : (p.boost_tier === 'Moderate Boost' ? 'text-cyan-300 font-semibold' : 'text-zinc-400')}">
                                            ${p.expected_boost || 'Steady target looks.'}
                                        </span>
                                    </td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            `;
        };

        return `
            <div class="space-y-6">
                <!-- Top Matchup Summary Card -->
                <div class="glass-panel p-5 rounded-xl border border-zinc-800 space-y-4">
                    <div class="flex flex-wrap items-center justify-between border-b border-zinc-800 pb-3 gap-3">
                        <div class="flex items-center gap-3">
                            <span class="badge bg-purple-500/20 text-purple-300 border border-purple-500/40 text-xs font-mono font-bold">
                                Week ${week} Matchup
                            </span>
                            <span class="text-xs text-zinc-400 font-mono font-semibold">
                                ${season} NFL Regular Season
                            </span>
                        </div>
                        <div class="flex items-center gap-2">
                            <span class="badge bg-zinc-900 border border-zinc-700 text-zinc-300 text-xs font-mono">
                                Scheme Battle Analysis Active
                            </span>
                        </div>
                    </div>

                    <!-- Teams Banner -->
                    <div class="grid grid-cols-12 items-center py-2 gap-3">
                        <!-- Away Team (5 cols) -->
                        <div class="col-span-5 flex items-center gap-3 min-w-0">
                            <div class="w-12 h-12 rounded-xl bg-zinc-800 border border-zinc-700 flex items-center justify-center font-black text-white text-base font-mono shadow-inner shrink-0">
                                ${away}
                            </div>
                            <div class="min-w-0 flex-1">
                                <h3 class="font-bold text-white text-base truncate">${awayName}</h3>
                                <div class="text-xs text-zinc-400 flex items-center gap-2 mt-0.5">
                                    <span class="font-mono text-zinc-300 font-semibold">${away} (Away)</span>
                                    <span class="text-zinc-600">•</span>
                                    <span class="badge bg-purple-500/10 text-purple-300 border border-purple-500/30 text-[10px] font-mono truncate">${aVsH.def_archetype ? aVsH.def_archetype.split('/')[0] : 'Hybrid Shell'}</span>
                                </div>
                            </div>
                        </div>

                        <!-- Center vs (2 cols) -->
                        <div class="col-span-2 text-center flex flex-col items-center justify-center">
                            <div class="w-8 h-8 mx-auto rounded-full bg-zinc-800 border border-zinc-700 flex items-center justify-center font-bold text-zinc-400 text-xs">
                                @
                            </div>
                            <span class="text-[9px] font-mono text-zinc-500 uppercase mt-1">SCHEME H2H</span>
                        </div>

                        <!-- Home Team (5 cols) -->
                        <div class="col-span-5 flex items-center justify-end gap-3 text-right min-w-0">
                            <div class="min-w-0 flex-1">
                                <h3 class="font-bold text-white text-base truncate">${homeName}</h3>
                                <div class="text-xs text-zinc-400 flex items-center justify-end gap-2 mt-0.5">
                                    <span class="badge bg-purple-500/10 text-purple-300 border border-purple-500/30 text-[10px] font-mono truncate">${hVsA.def_archetype ? hVsA.def_archetype.split('/')[0] : 'Hybrid Shell'}</span>
                                    <span class="text-zinc-600">•</span>
                                    <span class="font-mono text-zinc-300 font-semibold">${home} (Home)</span>
                                </div>
                            </div>
                            <div class="w-12 h-12 rounded-xl bg-zinc-800 border border-zinc-700 flex items-center justify-center font-black text-white text-base font-mono shadow-inner shrink-0">
                                ${home}
                            </div>
                        </div>
                    </div>
                </div>

                <!-- ============================================================= -->
                <!-- PILLAR 1: DEFENSIVE COVERAGE TENDENCIES (0:49, 17:53) -->
                <!-- ============================================================= -->
                <div class="space-y-3">
                    <div class="flex items-center justify-between border-b border-zinc-800 pb-2">
                        <div class="flex items-center gap-2">
                            <div class="p-1.5 rounded-lg bg-purple-500/20 text-purple-400 border border-purple-500/30">
                                <i data-lucide="shield" class="w-4 h-4"></i>
                            </div>
                            <div>
                                <h3 class="text-sm font-bold text-white uppercase tracking-wider font-heading">
                                    1. Defensive Coverage Tendencies & Shell Archetypes
                                </h3>
                                <p class="text-[11px] text-zinc-400">Zone vs Man rates & Middle-Field Open (MFO) vs Middle-Field Closed (MFC) sets</p>
                            </div>
                        </div>
                        <span class="badge bg-purple-500/20 text-purple-300 border border-purple-500/40 text-[10px] font-mono">
                            TIMESTAMPS: 0:49, 17:53
                        </span>
                    </div>

                    <div class="grid grid-cols-1 xl:grid-cols-2 gap-4">
                        ${renderCoverageCard(aVsH, away, home)}
                        ${renderCoverageCard(hVsA, home, away)}
                    </div>
                </div>

                <!-- ============================================================= -->
                <!-- PILLAR 2: RUN-SCHEME SPLITS & DEFENSIVE VULNERABILITIES (4:58, 12:00) -->
                <!-- ============================================================= -->
                <div class="space-y-3">
                    <div class="flex items-center justify-between border-b border-zinc-800 pb-2">
                        <div class="flex items-center gap-2">
                            <div class="p-1.5 rounded-lg bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">
                                <i data-lucide="split" class="w-4 h-4"></i>
                            </div>
                            <div>
                                <h3 class="text-sm font-bold text-white uppercase tracking-wider font-heading">
                                    2. Run-Scheme Splits & Front-Seven Alignment
                                </h3>
                                <p class="text-[11px] text-zinc-400">Zone vs Man/Gap rushing style aligned with defense specific vulnerabilities</p>
                            </div>
                        </div>
                        <span class="badge bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 text-[10px] font-mono">
                            TIMESTAMPS: 4:58, 12:00
                        </span>
                    </div>

                    <div class="grid grid-cols-1 xl:grid-cols-2 gap-4">
                        ${renderRunCard(aRun, home)}
                        ${renderRunCard(hRun, away)}
                    </div>
                </div>

                <!-- ============================================================= -->
                <!-- PILLAR 3: POSITIONAL TARGET RATES & ALIGNMENT (2:40, 10:47, 14:12) -->
                <!-- ============================================================= -->
                <div class="space-y-3">
                    <div class="flex items-center justify-between border-b border-zinc-800 pb-2">
                        <div class="flex items-center gap-2">
                            <div class="p-1.5 rounded-lg bg-amber-500/20 text-amber-400 border border-amber-500/30">
                                <i data-lucide="crosshair" class="w-4 h-4"></i>
                            </div>
                            <div>
                                <h3 class="text-sm font-bold text-white uppercase tracking-wider font-heading">
                                    3. Positional Target Rates & Alignment Matchups
                                </h3>
                                <p class="text-[11px] text-zinc-400">Target frequency by alignment (Slot vs Outside WR, Inline TE) & depth vs coverage</p>
                            </div>
                        </div>
                        <span class="badge bg-amber-500/20 text-amber-300 border border-amber-500/40 text-[10px] font-mono">
                            TIMESTAMPS: 2:40, 10:47, 14:12
                        </span>
                    </div>

                    <!-- Alignment Overview Distribution Chips -->
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                        <div class="glass-panel p-3.5 rounded-xl border border-zinc-800 space-y-2">
                            <span class="text-xs font-bold text-white font-mono flex items-center gap-1.5">
                                <i data-lucide="target" class="w-3.5 h-3.5 text-purple-400"></i>
                                <span>${away} Target Distribution by Alignment</span>
                            </span>
                            <div class="grid grid-cols-4 gap-2 text-center text-xs font-mono">
                                <div class="p-2 rounded bg-zinc-900 border border-zinc-800">
                                    <div class="text-[9px] text-zinc-400">SLOT WR</div>
                                    <div class="font-bold text-purple-300 mt-0.5">${aDist.slot_wr_pct || 28}%</div>
                                </div>
                                <div class="p-2 rounded bg-zinc-900 border border-zinc-800">
                                    <div class="text-[9px] text-zinc-400">OUTSIDE WR</div>
                                    <div class="font-bold text-zinc-200 mt-0.5">${aDist.wide_wr_pct || 42}%</div>
                                </div>
                                <div class="p-2 rounded bg-zinc-900 border border-zinc-800">
                                    <div class="text-[9px] text-zinc-400">INLINE TE</div>
                                    <div class="font-bold text-emerald-300 mt-0.5">${aDist.inline_te_pct || 18}%</div>
                                </div>
                                <div class="p-2 rounded bg-zinc-900 border border-zinc-800">
                                    <div class="text-[9px] text-zinc-400">BACKFIELD</div>
                                    <div class="font-bold text-cyan-300 mt-0.5">${aDist.backfield_rb_pct || 12}%</div>
                                </div>
                            </div>
                        </div>

                        <div class="glass-panel p-3.5 rounded-xl border border-zinc-800 space-y-2">
                            <span class="text-xs font-bold text-white font-mono flex items-center gap-1.5">
                                <i data-lucide="target" class="w-3.5 h-3.5 text-purple-400"></i>
                                <span>${home} Target Distribution by Alignment</span>
                            </span>
                            <div class="grid grid-cols-4 gap-2 text-center text-xs font-mono">
                                <div class="p-2 rounded bg-zinc-900 border border-zinc-800">
                                    <div class="text-[9px] text-zinc-400">SLOT WR</div>
                                    <div class="font-bold text-purple-300 mt-0.5">${hDist.slot_wr_pct || 28}%</div>
                                </div>
                                <div class="p-2 rounded bg-zinc-900 border border-zinc-800">
                                    <div class="text-[9px] text-zinc-400">OUTSIDE WR</div>
                                    <div class="font-bold text-zinc-200 mt-0.5">${hDist.wide_wr_pct || 42}%</div>
                                </div>
                                <div class="p-2 rounded bg-zinc-900 border border-zinc-800">
                                    <div class="text-[9px] text-zinc-400">INLINE TE</div>
                                    <div class="font-bold text-emerald-300 mt-0.5">${hDist.inline_te_pct || 18}%</div>
                                </div>
                                <div class="p-2 rounded bg-zinc-900 border border-zinc-800">
                                    <div class="text-[9px] text-zinc-400">BACKFIELD</div>
                                    <div class="font-bold text-cyan-300 mt-0.5">${hDist.backfield_rb_pct || 12}%</div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Target Tables -->
                    <div class="space-y-3">
                        <div class="text-xs font-bold text-zinc-300 font-mono flex items-center gap-2">
                            <span class="w-2 h-2 rounded-full bg-purple-400"></span>
                            <span>${away} Key Weapons vs ${home} Defensive Shell</span>
                        </div>
                        ${renderTargetTable(aTargets, away)}

                        <div class="text-xs font-bold text-zinc-300 font-mono flex items-center gap-2 pt-2">
                            <span class="w-2 h-2 rounded-full bg-cyan-400"></span>
                            <span>${home} Key Weapons vs ${away} Defensive Shell</span>
                        </div>
                        ${renderTargetTable(hTargets, home)}
                    </div>
                </div>

                <!-- ============================================================= -->
                <!-- PILLAR 4: EFFICIENCY BY CONTEXT (BREAKOUT SPOTTERS) (12:05) -->
                <!-- ============================================================= -->
                <div class="space-y-3">
                    <div class="flex items-center justify-between border-b border-zinc-800 pb-2">
                        <div class="flex items-center gap-2">
                            <div class="p-1.5 rounded-lg bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                                <i data-lucide="sparkles" class="w-4 h-4"></i>
                            </div>
                            <div>
                                <h3 class="text-sm font-bold text-white uppercase tracking-wider font-heading">
                                    4. Efficiency by Context & Scheme Splits (Breakout Spotters)
                                </h3>
                                <p class="text-[11px] text-zinc-400">Yards per carry/reception broken down by scheme to spot players statistically primed for breakouts</p>
                            </div>
                        </div>
                        <span class="badge bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 text-[10px] font-mono">
                            TIMESTAMP: 12:05
                        </span>
                    </div>

                    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                        ${breakouts.map(b => `
                            <div class="glass-panel p-4 rounded-xl border border-emerald-500/30 bg-emerald-950/10 space-y-3 shadow-lg relative overflow-hidden">
                                <div class="flex items-start justify-between gap-2">
                                    <div>
                                        <div class="flex items-center gap-2">
                                            <h4 class="font-bold text-white text-sm">${b.player}</h4>
                                            <span class="badge badge-med text-[10px] font-mono">${b.team} • ${b.pos}</span>
                                        </div>
                                        <span class="text-[10px] text-zinc-400 font-mono">${b.scheme_type}</span>
                                    </div>
                                    <span class="badge bg-emerald-500/25 text-emerald-300 border border-emerald-500/40 text-[10px] font-bold uppercase tracking-wider">
                                        ${b.verdict}
                                    </span>
                                </div>

                                <div class="grid grid-cols-2 gap-2 text-xs font-mono bg-zinc-900/80 p-2.5 rounded-lg border border-zinc-800">
                                    <div>
                                        <div class="text-[9px] text-zinc-400">Scheme Metric</div>
                                        <div class="font-bold text-emerald-300 text-xs">${b.context_metric}</div>
                                    </div>
                                    <div>
                                        <div class="text-[9px] text-zinc-400">Season Baseline</div>
                                        <div class="font-bold text-zinc-400 text-xs">${b.season_baseline}</div>
                                    </div>
                                </div>

                                <div class="flex items-center justify-between text-xs font-mono">
                                    <span class="text-zinc-400 text-[11px]">Contextual Scheme Delta:</span>
                                    <span class="badge bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 font-bold">${b.scheme_delta}</span>
                                </div>

                                <p class="text-[11px] text-zinc-300 leading-relaxed border-t border-zinc-800/80 pt-2">
                                    ${b.rationale}
                                </p>
                                ${b.causal_mechanism ? `
                                    <div class="text-[10px] text-purple-300 font-mono bg-purple-950/30 p-2 rounded-lg border border-purple-800/40 flex items-start gap-1.5 leading-snug">
                                        <i data-lucide="microscope" class="w-3.5 h-3.5 text-purple-400 shrink-0 mt-0.5"></i>
                                        <span><strong class="text-purple-200">Causal Mechanism:</strong> ${b.causal_mechanism}</span>
                                    </div>
                                ` : ''}
                            </div>
                        `).join('')}
                    </div>
                </div>

                <!-- ============================================================= -->
                <!-- PILLAR 5: GAME-FLOW, TEMPO & SITUATIONAL USAGE (6:17, 26:54) -->
                <!-- ============================================================= -->
                <div class="space-y-3">
                    <div class="flex items-center justify-between border-b border-zinc-800 pb-2">
                        <div class="flex items-center gap-2">
                            <div class="p-1.5 rounded-lg bg-rose-500/20 text-rose-400 border border-rose-500/30">
                                <i data-lucide="clock" class="w-4 h-4"></i>
                            </div>
                            <div>
                                <h3 class="text-sm font-bold text-white uppercase tracking-wider font-heading">
                                    5. Game-Flow, Tempo & Situational Usage Volume Predictors
                                </h3>
                                <p class="text-[11px] text-zinc-400">Time of possession, red zone participation & check-down frequency overriding flat projections</p>
                            </div>
                        </div>
                        <span class="badge bg-rose-500/20 text-rose-300 border border-rose-500/40 text-[10px] font-mono">
                            TIMESTAMPS: 6:17, 26:54
                        </span>
                    </div>

                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <!-- Away Tempo & Situational -->
                        <div class="glass-panel p-4 rounded-xl border border-zinc-800 space-y-3">
                            <div class="flex items-center justify-between border-b border-zinc-800 pb-2">
                                <span class="font-bold text-white text-xs font-mono">${away} Tempo & Situational Metrics</span>
                                <span class="badge badge-med text-[10px] font-mono">Pace Rank #${aTempo.pace_rank || 16}</span>
                            </div>
                            <div class="grid grid-cols-3 gap-2 text-center text-xs font-mono">
                                <div class="p-2 rounded bg-zinc-900 border border-zinc-800">
                                    <div class="text-[9px] text-zinc-400">NEUTRAL PACE</div>
                                    <div class="font-bold text-zinc-200 mt-0.5">${aTempo.neutral_pace_sec || 27.5}s</div>
                                </div>
                                <div class="p-2 rounded bg-zinc-900 border border-zinc-800">
                                    <div class="text-[9px] text-zinc-400">PROJ PLAYS</div>
                                    <div class="font-bold text-cyan-300 mt-0.5">${aTempo.proj_plays || 64}</div>
                                </div>
                                <div class="p-2 rounded bg-zinc-900 border border-zinc-800">
                                    <div class="text-[9px] text-zinc-400">CHECK-DOWN %</div>
                                    <div class="font-bold text-purple-300 mt-0.5">${aTempo.checkdown_pct || 11}%</div>
                                </div>
                            </div>
                            <div class="p-2.5 rounded-lg bg-zinc-900/80 border border-zinc-800 text-xs text-zinc-300 space-y-1">
                                <div class="text-[10px] text-zinc-400 uppercase font-semibold">Red Zone Touch Concentration</div>
                                <div class="font-semibold text-emerald-300">${aTempo.red_zone_touch_leader || 'Lead RB Goal Line Touch Share'}</div>
                            </div>
                        </div>

                        <!-- Home Tempo & Situational -->
                        <div class="glass-panel p-4 rounded-xl border border-zinc-800 space-y-3">
                            <div class="flex items-center justify-between border-b border-zinc-800 pb-2">
                                <span class="font-bold text-white text-xs font-mono">${home} Tempo & Situational Metrics</span>
                                <span class="badge badge-med text-[10px] font-mono">Pace Rank #${hTempo.pace_rank || 16}</span>
                            </div>
                            <div class="grid grid-cols-3 gap-2 text-center text-xs font-mono">
                                <div class="p-2 rounded bg-zinc-900 border border-zinc-800">
                                    <div class="text-[9px] text-zinc-400">NEUTRAL PACE</div>
                                    <div class="font-bold text-zinc-200 mt-0.5">${hTempo.neutral_pace_sec || 27.5}s</div>
                                </div>
                                <div class="p-2 rounded bg-zinc-900 border border-zinc-800">
                                    <div class="text-[9px] text-zinc-400">PROJ PLAYS</div>
                                    <div class="font-bold text-cyan-300 mt-0.5">${hTempo.proj_plays || 64}</div>
                                </div>
                                <div class="p-2 rounded bg-zinc-900 border border-zinc-800">
                                    <div class="text-[9px] text-zinc-400">CHECK-DOWN %</div>
                                    <div class="font-bold text-purple-300 mt-0.5">${hTempo.checkdown_pct || 11}%</div>
                                </div>
                            </div>
                            <div class="p-2.5 rounded-lg bg-zinc-900/80 border border-zinc-800 text-xs text-zinc-300 space-y-1">
                                <div class="text-[10px] text-zinc-400 uppercase font-semibold">Red Zone Touch Concentration</div>
                                <div class="font-semibold text-emerald-300">${hTempo.red_zone_touch_leader || 'Lead RB Goal Line Touch Share'}</div>
                            </div>
                        </div>
                    </div>

                    <!-- Volume Override Verdict Card -->
                    <div class="glass-panel p-4 rounded-xl border border-rose-900/40 bg-rose-950/15 text-xs text-zinc-300 space-y-2">
                        <div class="flex items-center gap-2 text-rose-300 font-bold font-heading uppercase tracking-wider text-xs">
                            <i data-lucide="zap" class="w-4 h-4 text-rose-400"></i>
                            <span>Volume Override Verdict (Tempo & Script vs Flat Projections)</span>
                        </div>
                        <p class="leading-relaxed text-zinc-200">
                            ${gameFlow.volume_override_verdict || 'Game tempo and check-down frequency establish critical target and carry volume deviations from standard season projections.'}
                        </p>
                    </div>
                </div>

                <!-- ============================================================= -->
                <!-- EXECUTIVE FILM ROOM SUMMARY / KEY BETTING & PROJECTION EDGES -->
                <!-- ============================================================= -->
                <div class="glass-panel p-5 rounded-xl border border-purple-800/40 bg-purple-950/20 space-y-3">
                    <div class="flex items-center gap-2">
                        <i data-lucide="brain" class="w-5 h-5 text-purple-400"></i>
                        <h3 class="text-sm font-bold text-white uppercase tracking-wider font-heading">
                            Executive Film Room Verdict & Matchup Conclusions
                        </h3>
                    </div>
                    <p class="text-xs text-zinc-300 leading-relaxed">
                        By examining scheme tendencies rather than raw baseline stats, this matchup reveals clear tactical asymmetries:
                        ${aRun.scheme_advantage === 'High Advantage' ? `${away}'s ground scheme possesses a commanding run-blocking advantage against ${home}'s gap-leak front.` : `${home}'s front seven is stoutly positioned against ${away}'s primary running concept.`}
                        Against ${home}'s ${cov.away_vs_home?.def_mfo_pct ? cov.away_vs_home.def_mfo_pct.toFixed(0) : '60'}% two-high shell, look for underneath targets to beat their baseline receiving floors, while vertical boundary props carry heightened variance.
                    </p>
                </div>
            </div>
        `;
    }
};

/**
 * Jump directly from a game card into Matchup Lab
 */
function inspectMatchupInLab(home, away) {
    const homeSelect = document.getElementById('h2h-home-select');
    const awaySelect = document.getElementById('h2h-away-select');
    if (homeSelect) homeSelect.value = home;
    if (awaySelect) awaySelect.value = away;
    if (window.switchTab) window.switchTab('h2h');
}
window.inspectMatchupInLab = inspectMatchupInLab;

/**
 * Open Player Prop Deep-Dive Modal with Interactive Simulation & Attribution
 */
function openPlayerModalByIndex(idx) {
    if (!window.activePropsList || !window.activePropsList[idx]) return;
    const prop = window.activePropsList[idx];
    currentModalProp = prop;

    const modal = document.getElementById('player-modal');
    if (!modal) return;

    const player = prop.Player || prop.player || 'Unknown';
    const team = prop.Team || prop.team || 'NFL';
    const opp = prop.Opponent || prop.opponent || prop.opponent_team || 'OPP';
    const pos = prop.Pos || prop.pos || '';
    const role = prop.Role || prop.role || `${pos}1`;
    const stat = prop.Stat || prop.stat || 'Yards';
    const line = parseFloat(prop.Line ?? prop.line ?? 0);
    const pred = parseFloat(prop.PRED ?? prop.pred ?? 0);
    const sigma = parseFloat(prop.Sigma ?? prop.sigma ?? 25.0);
    const call = (prop['O/U'] || prop.ou || (pred >= line ? 'OVER' : 'UNDER')).toUpperCase();
    const edge = Math.abs(pred - line).toFixed(1);

    document.getElementById('modal-player-name').textContent = player;
    document.getElementById('modal-player-sub').textContent = `${role} • ${team} vs ${opp} • ${stat}`;
    document.getElementById('modal-pred-val').textContent = pred.toFixed(1);
    document.getElementById('modal-line-val').textContent = line.toFixed(1);
    document.getElementById('modal-edge-val').textContent = `+${edge} yds`;
    document.getElementById('modal-call-badge').className = `badge ${call === 'OVER' ? 'badge-over' : 'badge-under'}`;
    document.getElementById('modal-call-badge').textContent = call;

    const slider = document.getElementById('sim-line-slider');
    slider.min = Math.max(0, Math.floor(line - 30));
    slider.max = Math.ceil(line + 30);
    slider.step = 0.5;
    slider.value = line;
    document.getElementById('sim-line-display').textContent = line.toFixed(1);

    recalcSimulation(line, pred, sigma);
    renderAttributionBreakdown(prop);

    modal.classList.add('active');
    if (window.lucide) lucide.createIcons({ root: modal });
}

/**
 * Dynamic Slider Recalculation for Edge & EV
 */
function onSimSliderChange(newVal) {
    if (!currentModalProp) return;
    const simLine = parseFloat(newVal);
    const pred = parseFloat(currentModalProp.PRED ?? currentModalProp.pred ?? 0);
    const sigma = parseFloat(currentModalProp.Sigma ?? currentModalProp.sigma ?? 25.0);
    document.getElementById('sim-line-display').textContent = simLine.toFixed(1);
    recalcSimulation(simLine, pred, sigma);
}

function recalcSimulation(lineVal, predVal, sigmaVal) {
    const diff = predVal - lineVal;
    const diffStr = `${diff >= 0 ? '+' : ''}${diff.toFixed(1)} yds`;
    const simCall = diff >= 0 ? 'OVER' : 'UNDER';

    const z = diff / Math.max(3.0, sigmaVal);
    const normalCdf = (val) => 0.5 * (1 + Math.erf(val / Math.SQRT2));
    let pOver = normalCdf(z);
    pOver = Math.min(0.95, Math.max(0.05, pOver));

    const bandLow = Math.max(5, Math.round((pOver - 0.09) * 100));
    const bandHigh = Math.min(95, Math.round((pOver + 0.09) * 100));
    const direction = diff >= 3.0 ? 'FAVORABLE' : (diff <= -3.0 ? 'UNFAVORABLE' : 'NEUTRAL');

    const elEdge = document.getElementById('sim-edge-result');
    if (elEdge) elEdge.textContent = diffStr;
    const elProb = document.getElementById('sim-winprob-result');
    if (elProb) elProb.textContent = `${(pOver * 100).toFixed(0)}%`;
    const elEv = document.getElementById('sim-ev-result');
    if (elEv) elEv.textContent = `[${bandLow}–${bandHigh}%]`;
    const elKelly = document.getElementById('sim-kelly-result');
    if (elKelly) elKelly.textContent = direction;
}

function renderAttributionBreakdown(prop) {
    const container = document.getElementById('modal-attrib-container');
    if (!container) return;

    let items = [];
    if (prop.Attribution && prop.Attribution.length > 0) {
        items = prop.Attribution.map(a => {
            const rawImpact = a.impact ?? a.contrib ?? 0;
            return {
                name: a.name || a.feature || '',
                impact: typeof rawImpact === 'number' ? rawImpact : (parseFloat(rawImpact) || 0),
                val: a.raw_value
            };
        });
    } else if (prop.shap_top3 && prop.shap_top3.length > 0) {
        items = prop.shap_top3.map(s => {
            const rawImpact = s.contrib ?? s.shap_value ?? s.impact ?? 0;
            return {
                name: s.feature || '',
                impact: typeof rawImpact === 'number' ? rawImpact : (parseFloat(rawImpact) || 0),
                val: s.feature_value ?? s.contrib ?? 0
            };
        });
    }

    if (items.length === 0) {
        container.innerHTML = '<div class="text-xs text-zinc-500 text-center py-2">Baseline statistical projection.</div>';
        return;
    }

    container.innerHTML = items.map(it => {
        const impNum = typeof it.impact === 'number' ? it.impact : (parseFloat(it.impact) || 0);
        const isPos = impNum >= 0;
        const sign = isPos ? '+' : '';
        const color = isPos ? 'text-white' : 'text-zinc-400';
        const barColor = isPos ? 'bg-zinc-300' : 'bg-zinc-600';
        const widthPct = Math.min(100, Math.abs(impNum) * 4);

        return `
            <div class="space-y-1">
                <div class="flex items-center justify-between text-xs font-mono">
                    <span class="text-zinc-300">${it.name}</span>
                    <span class="${color} font-bold">${sign}${impNum.toFixed(1)} yds</span>
                </div>
                <div class="w-full bg-zinc-800 rounded-full h-1.5 overflow-hidden">
                    <div class="${barColor} h-full rounded-full" style="width: ${widthPct}%"></div>
                </div>
            </div>
        `;
    }).join('');
}

function closeModal() {
    const modal = document.getElementById('player-modal');
    if (modal) modal.classList.remove('active');
}

window.components = components;
window.openPlayerModalByIndex = openPlayerModalByIndex;
window.onSimSliderChange = onSimSliderChange;
window.closeModal = closeModal;
