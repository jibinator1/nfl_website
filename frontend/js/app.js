/**
 * NFL Prediction & Analytics Hub - Application Controller
 * ========================================================
 * Manages global application state, tab navigation, search/filter pipelines,
 * and seamless background data synchronization.
 */

// Application State Store
const state = {
    activeTab: 'predictions',
    modelView: 'xgb', // 'xgb' (XGBoost Ensemble) or 'mlr' (ElasticNet Starters)
    season: 2026,
    week: '4',
    startDate: '2026-09-01',
    endDate: '2026-10-07',
    propsDate: '2026-10-04',
    includeProps: true,
    isModelTrained: true,
    
    // Cached datasets
    xgbPredictions: null,
    mlrPredictions: null,
    tdPredictions: null,
    floorStreaks: [],

    // Schedule configuration
    scheduleSeason: 2026,
    scheduleWeek: '4',
    scheduleData: null,
    scheduleWeeks: [],

    // Predictions tab date & games cache
    selectedDate: '2026-10-04',
    allGames: [],
    fullSeasonGames: [],
    dateToWeekMap: {},

    // Predictions filter & search criteria
    searchQuery: '',
    filterPos: 'ALL',
    filterCall: 'ALL',
    filterConf: 'ALL',
    filterDirection: 'ALL',

    // Floor Streak tab state
    floorSortCol: 'Floor_Streak',
    floorSortDir: 'desc',
    floorFilterPos: 'ALL',
    floorFilterStreak: 5,
    floorFilterMinGames: 4,
    floorSearchQuery: '',
    floorSelectedDate: '',
    floorStartDate: '2026-09-01',
    floorEndDate: new Date().toISOString().substring(0, 10),
    floorDatePreset: 'season26',

    // Schedule stats history range
    scheduleStartDate: '2026-09-01',
    scheduleEndDate: new Date().toISOString().substring(0, 10),
    scheduleDatePreset: 'season26',

    // Criteria Matches viewer state
    criteriaSeason: 2025,
    criteriaWeek: 8,
    criteriaSelected: 'weak_rush_def',
    criteriaData: null,
    criteriaLoading: false,

    // Game Highlights state
    highlightsDate: '2025-11-03',
    highlightsScope: 'week', // 'week' (full slate) or 'date' (selected date only)
    highlightsData: null,
    highlightsLoading: false,

    // Scheme-Based Insights state
    insightsHomeTeam: 'SEA',
    insightsAwayTeam: 'NE',
    insightsSeason: 2026,
    insightsWeek: 1,
    insightsData: null,
    insightsLoading: false,
    insightsCache: {}
};

// 32 NFL Franchise Full Names Map
const NFL_TEAM_NAMES = {
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
};

// Global active props list reference for modal lookups
window.activePropsList = [];

/**
 * Initialize Application on DOM Ready
 */
document.addEventListener('DOMContentLoaded', async () => {
    initDefaultDates();
    // Sync state.modelView from the dropdown's current DOM value
    // (browser may cache the last selected value across page loads)
    const engineSelect = document.getElementById('model-engine-select');
    if (engineSelect && engineSelect.value) {
        state.modelView = engineSelect.value;
    }
    // In TD mode, default to HIGH confidence filter
    if (state.modelView === 'td') {
        state.filterConf = 'HIGH';
    }
    setupEventListeners();
    updateRunModelButtonLabel();
    const redoSelect = document.getElementById('redo-props-select');
    if (redoSelect) {
        state.includeProps = (redoSelect.value === 'true');
    }
    await checkModelStatus();
    await loadCurrentTab();
});

/**
 * Sets initial dates
 */
function initDefaultDates() {
    const todayStr = new Date().toISOString().substring(0, 10);
    state.selectedDate = '2026-10-04';
    state.season = 2026;
    state.scheduleSeason = 2026;

    // Default stats date range for Floor Streaks: Sept 1, 2026 to current date
    state.floorStartDate = '2026-09-01';
    state.floorEndDate = todayStr;
    state.floorDatePreset = 'season26';

    const floorStartInput = document.getElementById('floor-start-date');
    const floorEndInput = document.getElementById('floor-end-date');
    if (floorStartInput) floorStartInput.value = state.floorStartDate;
    if (floorEndInput) floorEndInput.value = state.floorEndDate;

    // Default stats date range for Schedule & Matchups: Sept 1, 2026 to current date
    state.scheduleStartDate = '2026-09-01';
    state.scheduleEndDate = todayStr;
    state.scheduleDatePreset = 'season26';

    const scheduleStartInput = document.getElementById('schedule-start-date');
    const scheduleEndInput = document.getElementById('schedule-end-date');
    if (scheduleStartInput) scheduleStartInput.value = state.scheduleStartDate;
    if (scheduleEndInput) scheduleEndInput.value = state.scheduleEndDate;

    const datePicker = document.getElementById('predictions-date-picker');
    if (datePicker) {
        datePicker.value = state.selectedDate;
    }

    const highlightsPicker = document.getElementById('highlights-date-picker');
    if (highlightsPicker) {
        highlightsPicker.value = state.highlightsDate;
    }
}

/**
 * Register Event Listeners
 */
function setupEventListeners() {
    const searchInput = document.getElementById('props-search-input');
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            state.searchQuery = e.target.value.toLowerCase().trim();
            filterAndRenderProps();
        });
    }
}

/**
 * Check Backend Model Status on Launch
 */
async function checkModelStatus() {
    try {
        const st = await api.getModelStatus();
        if (state.modelView === 'xgb' && st.is_trained !== undefined) {
            state.isModelTrained = Boolean(st.is_trained);
        }
        const badge = document.getElementById('model-status-badge');
        if (badge) {
            if (st.is_trained) {
                badge.className = 'badge badge-over text-[11px]';
                badge.textContent = `Trained (${st.architecture || 'level_cv'})`;
            } else {
                badge.className = 'badge badge-med text-[11px]';
                badge.textContent = 'Ready to Train';
            }
        }
    } catch (e) {
        console.warn('Could not query model status:', e);
    }
}

/**
 * Switch Active Tab
 */
async function switchTab(tabId) {
    if (state.activeTab === tabId) return;
    state.activeTab = tabId;

    // Update Tab Buttons UI
    document.querySelectorAll('.nav-tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    const activeBtn = document.getElementById(`tab-btn-${tabId}`);
    if (activeBtn) activeBtn.classList.add('active');

    // Update Section Panels
    document.querySelectorAll('.tab-panel').forEach(panel => {
        panel.classList.add('hidden');
    });
    const activePanel = document.getElementById(`tab-${tabId}`);
    if (activePanel) activePanel.classList.remove('hidden');

    window.scrollTo({ top: 0, behavior: 'smooth' });

    await loadCurrentTab();
    if (window.lucide) lucide.createIcons();
}

/**
 * Toggle Model View via Dropdown Selector
 */
async function setModelView(mode) {
    if (state.modelView === mode) return;
    state.modelView = mode;

    // Sync dropdown UI
    const select = document.getElementById('model-engine-select');
    if (select && select.value !== mode) {
        select.value = mode;
    }

    // Sync export dropdown UI to active model
    const exportSelect = document.getElementById('export-xlsx-select');
    if (exportSelect) {
        if (mode === 'mlr') exportSelect.value = 'mlr';
        else if (mode === 'td') exportSelect.value = 'td';
        else exportSelect.value = 'cv';
    }

    // Relabel the 10th column header depending on active model
    const driversHeader = document.getElementById('props-col-drivers');
    if (driversHeader) {
        if (mode === 'td') {
            driversHeader.textContent = 'Floor Streak';
            driversHeader.title = 'Consecutive games without production falling below player\'s own rolling baseline. 🔥 = 8+, 📈 = 5+, ⚠️ = 0–1';
        } else {
            driversHeader.textContent = 'Key Drivers';
            driversHeader.title = '';
        }
    }

    // Clear the selected date when switching engines — each engine covers
    // different game dates, so keeping a stale date causes empty tables.
    state.selectedDate = '';
    const datePicker = document.getElementById('predictions-date-picker');
    if (datePicker) {
        datePicker.value = '';
        datePicker.classList.remove('border-zinc-400');
    }

    // Auto-set confidence filter: HIGH only in TD mode, ALL for other modes
    if (mode === 'td' && state.filterConf === 'ALL') {
        state.filterConf = 'HIGH';
    } else if (mode !== 'td') {
        state.filterConf = 'ALL';
    }
    // Sync confidence filter chip UI
    document.querySelectorAll('[data-conf-filter]').forEach(el => {
        el.classList.toggle('active', el.getAttribute('data-conf-filter') === state.filterConf);
    });

    // Update RUN MODEL button label to match the active engine
    updateRunModelButtonLabel();

    await loadPredictionsTab();
}

/**
 * Updates the RUN MODEL button label/icon to reflect the active engine.
 */
function updateRunModelButtonLabel() {
    const btn = document.getElementById('btn-run-model');
    if (!btn) return;
    const mode = state.modelView || 'xgb';
    let label = 'RUN MODEL';
    if (mode === 'td') label = 'RUN TD MODEL';
    else if (mode === 'mlr') label = 'RUN MLR';
    const span = btn.querySelector('span');
    if (span) span.textContent = label;
}

/**
 * Handles Redo Player Props dropdown change
 */
function onRedoPropsChange(val) {
    state.includeProps = (val === 'true' || val === true);
    const select = document.getElementById('redo-props-select');
    if (select) {
        select.value = state.includeProps ? 'true' : 'false';
    }
    const msg = state.includeProps
        ? 'Player props will be recalculated when running model.'
        : 'Player props recalculation disabled on model run (running games only).';
    showToast(msg, state.includeProps ? 'success' : 'info');
}

/**
 * Dispatcher to Load Data for Current Tab
 */
async function loadCurrentTab() {
    try {
        if (state.activeTab === 'predictions') {
            await loadPredictionsTab();
        } else if (state.activeTab === 'floor-streak') {
            await loadFloorStreakTab();
        } else if (state.activeTab === 'h2h') {
            await loadH2HTab();
        } else if (state.activeTab === 'schedule') {
            await loadScheduleTab();
        } else if (state.activeTab === 'insights') {
            await loadInsightsTab();
        } else if (state.activeTab === 'criteria') {
            await loadCriteriaTab();
        } else if (state.activeTab === 'highlights') {
            await loadHighlightsTab();
        }
    } catch (err) {
        showToast(err.message || 'Error loading tab data', 'error');
    }
}

/**
 * Tab 1: Predictions & Props Loader
 */
/**
 * Format date string YYYY-MM-DD to friendly human readable date
 */
function formatDateDisplay(dateStr) {
    if (!dateStr) return '';
    try {
        const parts = dateStr.split('-');
        if (parts.length === 3) {
            const y = parseInt(parts[0], 10);
            const m = parseInt(parts[1], 10);
            const d = parseInt(parts[2], 10);
            const dt = new Date(Date.UTC(y, m - 1, d, 12, 0, 0));
            return dt.toLocaleDateString('en-US', {
                weekday: 'short',
                month: 'short',
                day: 'numeric',
                year: 'numeric',
                timeZone: 'UTC'
            });
        }
        return dateStr;
    } catch (_) {
        return dateStr;
    }
}

/**
 * Render quick-select game date chips based on loaded slate
function renderPredictionsDateChips() {
    // Date buttons removed per user request: single overall date selector controls slate
}

async function ensureSeasonScheduleLoaded(targetSeason = null) {
    const s = targetSeason || state.season || 2026;
    if (state.fullSeasonGames && state.fullSeasonGames.length > 0 && state.loadedScheduleSeason === s) return;
    try {
        const sched = await api.getSchedule({ season: s });
        const flat = [];
        const dateMap = {};
        if (sched && sched.weeks && Array.isArray(sched.weeks)) {
            sched.weeks.forEach(w => {
                const wNum = w.week;
                if (Array.isArray(w.games)) {
                    w.games.forEach(g => {
                        flat.push(g);
                        const d = (g.gameday || g.date || '').substring(0, 10);
                        if (d) dateMap[d] = wNum;
                    });
                }
            });
        } else if (sched && Array.isArray(sched.games)) {
            sched.games.forEach(g => {
                flat.push(g);
                const d = (g.gameday || g.date || '').substring(0, 10);
                if (d) dateMap[d] = g.week || 1;
            });
        }
        state.fullSeasonGames = flat;
        state.dateToWeekMap = dateMap;
        state.loadedScheduleSeason = s;
    } catch (e) {
        console.warn('Could not load full season schedule:', e);
    }
}

/**
 * Tab 1: Predictions & Props Loader
 */
async function loadPredictionsTab() {
    const tbody = document.getElementById('props-table-body');
    const gamesGrid = document.getElementById('games-grid');

    if (tbody) {
        tbody.innerHTML = `
            <tr>
                <td colspan="11" class="p-8 text-center text-slate-400">
                    <div class="inline-flex items-center gap-2 text-zinc-200 font-mono text-sm">
                        <i data-lucide="loader-2" class="w-4 h-4 animate-spin"></i>
                        <span>Loading ${(() => { const sel = document.getElementById('model-engine-select'); return sel ? sel.options[sel.selectedIndex]?.text || state.modelView : state.modelView; })()} predictions...</span>
                    </div>
                </td>
            </tr>
        `;
        if (window.lucide) lucide.createIcons({ root: tbody });
    }

    try {
        const currentDate = state.selectedDate || '2026-10-04';
        state.selectedDate = currentDate;
        const dateYear = parseInt(currentDate.substring(0, 4), 10);
        state.season = (dateYear >= 2022) ? dateYear : 2026;
        state.scheduleSeason = state.season;

        const datePicker = document.getElementById('predictions-date-picker');
        if (datePicker && datePicker.value !== state.selectedDate) {
            datePicker.value = state.selectedDate;
        }

        if (state.modelView === 'xgb') {
            const data = await api.getPredictions({
                date: state.selectedDate,
                season: state.season
            });
            if (data?.is_trained !== undefined) {
                state.isModelTrained = Boolean(data.is_trained);
            } else if (data?.props?.length > 0) {
                state.isModelTrained = true;
            }
            state.xgbPredictions = data;
            state.allGames = data?.games || [];

            renderPredictionsGames();
            filterAndRenderProps();
        } else if (state.modelView === 'td') {
            // Anytime Touchdown Scorer View
            const data = await api.getTDPredictions({
                season: state.season,
                week: state.week || 1
            });
            state.tdPredictions = data;
            if (data?.is_trained !== undefined) {
                state.isModelTrained = Boolean(data.is_trained);
            } else if (data?.props?.length > 0) {
                state.isModelTrained = true;
            }

            if (state.fullSeasonGames && state.fullSeasonGames.length > 0) {
                state.allGames = state.week ? state.fullSeasonGames.filter(g => String(g.week) === String(state.week)) : state.fullSeasonGames;
            }

            renderPredictionsDateChips();
            renderPredictionsGames();
            filterAndRenderProps();
        } else {
            // MLR Starters View
            const data = await api.getMLRPredictions({
                season: state.season,
                week: state.week || (state.season === 2026 ? 4 : 1)
            });
            state.mlrPredictions = data;
            state.isModelTrained = (data?.props && data.props.length > 0) || Boolean(data?.status === 'success');

            if (state.fullSeasonGames && state.fullSeasonGames.length > 0) {
                state.allGames = state.week ? state.fullSeasonGames.filter(g => String(g.week) === String(state.week)) : state.fullSeasonGames;
            }

            renderPredictionsDateChips();
            renderPredictionsGames();
            filterAndRenderProps();
        }
    } catch (err) {
        console.error('Failed to load predictions:', err);
        if (tbody) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="11" class="p-8 text-center text-red-400">
                        <i data-lucide="alert-triangle" class="w-6 h-6 mx-auto mb-2 text-red-400"></i>
                        <p class="text-sm font-semibold">Failed to load predictions: ${err.message || 'Unknown error'}</p>
                        <p class="text-xs text-zinc-500 mt-1">Check if the model is trained or try clicking "Rerun Lines".</p>
                        <button onclick="loadPredictionsTab()" class="mt-3 px-3 py-1 bg-zinc-800 hover:bg-zinc-700 text-xs text-white rounded border border-zinc-700 transition">
                            Retry
                        </button>
                    </td>
                </tr>
            `;
        }
        if (gamesGrid) {
            gamesGrid.innerHTML = `
                <div class="col-span-full p-8 text-center glass-panel rounded-xl border border-red-900/40">
                    <i data-lucide="alert-triangle" class="w-6 h-6 mx-auto mb-2 text-red-400"></i>
                    <p class="text-sm text-red-400 font-medium">Failed to load game matchups.</p>
                    <p class="text-xs text-zinc-500 mt-1">${err.message || 'Error communicating with backend'}</p>
                </div>
            `;
        }
        showToast(err.message || 'Failed to load predictions', 'error');
    } finally {
        if (window.lucide) lucide.createIcons();
    }
}

/**
 * Render Game Spread Cards filtered by selected date
 */
function renderPredictionsGames() {
    const gamesGrid = document.getElementById('games-grid');
    const titleEl = document.getElementById('matchups-section-title');
    const badgeEl = document.getElementById('matchups-count-badge');
    if (!gamesGrid) return;

    const targetDate = state.selectedDate || '2026-10-04';
    // STRICT FILTER: Only show games on the selected date
    const displayedGames = (state.allGames || []).filter(g => {
        const gDate = (g.gameday || g.date || '').substring(0, 10);
        return gDate === targetDate;
    });

    if (titleEl) {
        titleEl.textContent = `Matchups for ${formatDateDisplay(targetDate)}`;
    }
    if (badgeEl) {
        badgeEl.textContent = `${displayedGames.length} Game${displayedGames.length === 1 ? '' : 's'}`;
        badgeEl.className = displayedGames.length > 0 ? 'badge badge-high text-[10px]' : 'badge badge-low text-[10px]';
    }

    if (displayedGames.length === 0) {
        gamesGrid.innerHTML = `
            <div class="col-span-full p-8 text-center glass-panel rounded-xl border border-zinc-800">
                <i data-lucide="calendar-x" class="w-8 h-8 mx-auto text-zinc-500 mb-2"></i>
                <p class="text-sm text-zinc-300 font-medium">No scheduled matchups found for ${formatDateDisplay(targetDate)}.</p>
                <p class="text-xs text-zinc-500 mt-1">Select another date using the date selector above.</p>
            </div>
        `;
    } else {
        gamesGrid.innerHTML = components.renderSpreadCards(displayedGames);
    }

    if (window.lucide) lucide.createIcons({ root: gamesGrid });
}

/**
 * Filter & Render Props Board based on current state filters and selected date
 */
function filterAndRenderProps() {
    const tbody = document.getElementById('props-table-body');
    if (!tbody) return;

    let rawList = [];
    const isTD = state.modelView === 'td';
    const isMLR = state.modelView === 'mlr';

    if (isTD) {
        rawList = state.tdPredictions?.props || [];
    } else if (isMLR) {
        rawList = state.mlrPredictions?.props || [];
    } else {
        rawList = state.xgbPredictions?.props || [];
    }

    // Determine teams playing on selected date for fallback matching
    let dateTeams = null;
    if (state.selectedDate) {
        dateTeams = new Set();
        (state.allGames || []).forEach(g => {
            const gDate = (g.gameday || g.date || '').substring(0, 10);
            if (gDate === state.selectedDate) {
                if (g.home_team) dateTeams.add(g.home_team.toUpperCase());
                if (g.away_team) dateTeams.add(g.away_team.toUpperCase());
            }
        });
    }

    const filtered = rawList.filter(p => {
        const player = (p.Player || p.player || '').toLowerCase();
        const team = (p.Team || p.team || '').toUpperCase();
        const pos = (p.Pos || p.pos || '').toUpperCase();
        const role = (p.Role || p.role || '').toUpperCase();
        const call = (p['O/U'] || p.ou || '').toUpperCase();
        const conf = (p['Confidence Level'] || p.confidence || '').toUpperCase();
        const edge = parseFloat(p.Edge ?? p.edge ?? 0);
        const pDate = (p.Date || p.date || p.gameday || '').substring(0, 10);

        // Date Filter: show props matching selected date or for teams playing on selected date
        if (state.selectedDate) {
            const matchesExactDate = (/^\d{4}-\d{2}-\d{2}$/.test(pDate) && pDate === state.selectedDate);
            const matchesTeamSlate = Boolean(dateTeams && dateTeams.size > 0 && dateTeams.has(team));
            if (!matchesExactDate && !matchesTeamSlate) {
                return false;
            }
        }

        // Search Query
        if (state.searchQuery && !player.includes(state.searchQuery) && !team.toLowerCase().includes(state.searchQuery)) {
            return false;
        }

        // Position Filter
        if (state.filterPos !== 'ALL' && pos !== state.filterPos) {
            return false;
        }

        // Call Filter
        if (state.filterCall !== 'ALL' && call !== state.filterCall) {
            return false;
        }

        // Direction Filter (e.g. FAVORABLE)
        const direction = (p.Direction || p.direction || '').toUpperCase();
        if (state.filterDirection && state.filterDirection !== 'ALL' && direction !== state.filterDirection) {
            return false;
        }

        return true;
    });

    // Sort picks best to worst (prioritize HIGH confidence, highest EV/Edge, highest model prob)
    const confRank = { 'HIGH': 3, 'MED': 2, 'LOW': 1, 'NONE': 0 };
    filtered.sort((a, b) => {
        const confA = confRank[(a['Confidence Level'] || a.confidence || '').toUpperCase()] ?? 0;
        const confB = confRank[(b['Confidence Level'] || b.confidence || '').toUpperCase()] ?? 0;
        if (confB !== confA) return confB - confA;

        const evA = parseFloat(a.EV_Pct ?? a.ev ?? a.EV ?? 0);
        const evB = parseFloat(b.EV_Pct ?? b.ev ?? b.EV ?? 0);
        if (Math.abs(evB - evA) >= 0.1) return evB - evA;

        const edgeA = Math.abs(parseFloat(a.Edge ?? a.edge ?? a['True Edge'] ?? 0));
        const edgeB = Math.abs(parseFloat(b.Edge ?? b.edge ?? b['True Edge'] ?? 0));
        if (Math.abs(edgeB - edgeA) >= 0.1) return edgeB - edgeA;

        const predA = parseFloat(a.Prediction ?? a.PRED ?? a.pred ?? 0);
        const predB = parseFloat(b.Prediction ?? b.PRED ?? b.pred ?? 0);
        if (Math.abs(predB - predA) >= 0.1) return predB - predA;

        const streakA = parseInt(a.Floor_Streak || 0, 10);
        const streakB = parseInt(b.Floor_Streak || 0, 10);
        return streakB - streakA;
    });

    // Store in global window for modal lookups
    window.activePropsList = filtered;

    // Update Counter Badges & Directional Skew Monitor (Bug 7)
    const countBadge = document.getElementById('props-count-badge');
    if (countBadge) countBadge.textContent = `${filtered.length} Props`;

    const ouBadge = document.getElementById('props-ou-split-badge');
    const skewBadge = document.getElementById('props-skew-warning-badge');
    const skewText = document.getElementById('props-skew-warning-text');

    if (ouBadge) {
        if (filtered.length > 0) {
            let overCount = 0;
            let underCount = 0;
            filtered.forEach(p => {
                const call = (p['O/U'] || p.ou || '').toUpperCase();
                if (call === 'OVER') overCount++;
                else if (call === 'UNDER') underCount++;
            });
            ouBadge.textContent = `${overCount}O / ${underCount}U`;
            ouBadge.classList.remove('hidden');

            if (skewBadge && filtered.length >= 5) {
                const overRatio = overCount / filtered.length;
                const underRatio = underCount / filtered.length;
                if (overRatio > 0.70 || underRatio > 0.70) {
                    const dominant = overRatio > 0.70 ? 'OVER' : 'UNDER';
                    const dominantPct = Math.round((overRatio > 0.70 ? overRatio : underRatio) * 100);
                    if (skewText) skewText.textContent = `${dominantPct}% ${dominant} Skew`;
                    skewBadge.title = `Directional Skew Alert: ${dominantPct}% of active picks are ${dominant}.`;
                    skewBadge.classList.remove('hidden');
                } else {
                    skewBadge.classList.add('hidden');
                }
            } else if (skewBadge) {
                skewBadge.classList.add('hidden');
            }
        } else {
            ouBadge.classList.add('hidden');
            if (skewBadge) skewBadge.classList.add('hidden');
        }
    }

    tbody.innerHTML = components.renderPropsTable(filtered, isMLR, state.isModelTrained);
    if (window.lucide) lucide.createIcons({ root: tbody });
}

/**
 * Filter Chip Click Handlers
 */
function setPosFilter(pos) {
    state.filterPos = pos;
    document.querySelectorAll('[data-pos-filter]').forEach(el => {
        el.classList.toggle('active', el.getAttribute('data-pos-filter') === pos);
    });
    filterAndRenderProps();
}

function setCallFilter(call) {
    state.filterCall = call;
    document.querySelectorAll('[data-call-filter]').forEach(el => {
        el.classList.toggle('active', el.getAttribute('data-call-filter') === call);
    });
    filterAndRenderProps();
}

function setConfFilter(conf) {
    state.filterConf = conf;
    document.querySelectorAll('[data-conf-filter]').forEach(el => {
        el.classList.toggle('active', el.getAttribute('data-conf-filter') === conf);
    });
    filterAndRenderProps();
}

function setDirectionFilter(dir) {
    if (state.filterDirection === dir) {
        state.filterDirection = 'ALL';
    } else {
        state.filterDirection = dir;
    }
    document.querySelectorAll('[data-direction-filter]').forEach(el => {
        el.classList.toggle('active', el.getAttribute('data-direction-filter') === state.filterDirection);
    });
    filterAndRenderProps();
}

/**
 * Tab: Player Hard Line Floor Streaks Loader & Controller
 */
async function loadFloorStreakTab() {
    const tbody = document.getElementById('floor-streak-table-body');
    if (tbody) {
        tbody.innerHTML = `<tr><td colspan="11" class="p-8 text-center text-zinc-400"><div class="flex items-center justify-center gap-2"><i data-lucide="loader-2" class="w-5 h-5 animate-spin text-zinc-400"></i><span>Calculating player floor breach streaks...</span></div></td></tr>`;
        if (window.lucide) lucide.createIcons({ root: tbody });
    }

    // Ensure schedule games are available so date chips and matchups can be resolved
    if (!state.allGames || state.allGames.length === 0) {
        try {
            const sched = await api.getSchedule({ season: state.season, week: state.week || '1' });
            const flatGames = [];
            if (sched && sched.weeks && Array.isArray(sched.weeks)) {
                sched.weeks.forEach(w => {
                    if (Array.isArray(w.games)) flatGames.push(...w.games);
                });
            } else if (sched && Array.isArray(sched.games)) {
                flatGames.push(...sched.games);
            }
            state.allGames = flatGames;
        } catch (e) {
            console.warn('Could not load schedule games for Floor Streak view:', e);
        }
    }

    try {
        const streakParams = {
            min_games: state.floorFilterMinGames
        };
        if (state.floorStartDate) streakParams.start_date = state.floorStartDate;
        if (state.floorEndDate) streakParams.end_date = state.floorEndDate;

        const res = await api.getFloorStreak(streakParams);

        const list = res.players || [];

        // Enrich players with game schedule info (matchup, opponent, date) from active window
        if (state.allGames && state.allGames.length > 0) {
            const activeGames = state.allGames.filter(g => {
                const gd = (g.gameday || g.date || '').substring(0, 10);
                return !state.startDate || !state.endDate || (gd >= state.startDate && gd <= state.endDate);
            });
            const gamesToUse = activeGames.length > 0 ? activeGames : state.allGames;

            const teamGameMap = {};
            gamesToUse.forEach(g => {
                const h = (g.home_team || '').toUpperCase();
                const a = (g.away_team || '').toUpperCase();
                const gd = (g.gameday || g.date || '').substring(0, 10);
                if (h) teamGameMap[h] = { date: gd, gameday: gd, opponent: a, matchup: `${h} vs ${a}`, is_home: 1 };
                if (a) teamGameMap[a] = { date: gd, gameday: gd, opponent: h, matchup: `${a} @ ${h}`, is_home: 0 };
            });
            list.forEach(p => {
                const t = (p.team || '').toUpperCase();
                if (teamGameMap[t]) {
                    p.date = teamGameMap[t].date;
                    p.gameday = teamGameMap[t].gameday;
                    p.opponent = teamGameMap[t].opponent;
                    p.matchup = teamGameMap[t].matchup;
                    p.is_home = teamGameMap[t].is_home;
                }
            });
        }

        state.floorStreaks = list;

        // Update active timeline display badge
        const timelineEl = document.getElementById('floor-active-timeline-text');
        if (timelineEl) {
            if (state.floorStartDate && state.floorEndDate) {
                timelineEl.textContent = `Stats Range: ${state.floorStartDate} to ${state.floorEndDate} • (${list.length} players found)`;
            } else if (state.floorStartDate) {
                timelineEl.textContent = `Stats Range: From ${state.floorStartDate} to Present • (${list.length} players)`;
            } else if (state.floorEndDate) {
                timelineEl.textContent = `Stats Range: Through ${state.floorEndDate} • (${list.length} players)`;
            } else {
                timelineEl.textContent = `Baseline window: All Available History (3 Years) • (${list.length} players)`;
            }
        }

        // Render date chips
        renderFloorDateChips();

        // Render filtered table
        renderFloorStreaks();
    } catch (err) {
        if (tbody) {
            tbody.innerHTML = `<tr><td colspan="11" class="p-8 text-center text-red-400">Error loading floor streaks: ${err.message}</td></tr>`;
        }
    }
}

/**
 * Floor Streak History Presets & Custom Range Controllers
 */
function setFloorHistoryPreset(preset) {
    state.floorDatePreset = preset;
    document.querySelectorAll('[id^="floor-preset-"]').forEach(btn => btn.classList.remove('active'));
    const btn = document.getElementById(`floor-preset-${preset}`);
    if (btn) btn.classList.add('active');

    const startInput = document.getElementById('floor-start-date');
    const endInput = document.getElementById('floor-end-date');

    const todayStr = new Date().toISOString().substring(0, 10);
    const today = new Date(todayStr + 'T00:00:00Z');

    if (preset === 'season26') {
        state.floorStartDate = '2026-09-01';
        state.floorEndDate = todayStr;
    } else if (preset === 'all') {
        state.floorStartDate = '';
        state.floorEndDate = '';
    } else if (preset === '1y') {
        const d = new Date(today);
        d.setFullYear(d.getFullYear() - 1);
        state.floorStartDate = d.toISOString().substring(0, 10);
        state.floorEndDate = todayStr;
    } else if (preset === '2025') {
        state.floorStartDate = '2025-09-01';
        state.floorEndDate = '2026-01-15';
    } else if (preset === '8w') {
        const d = new Date(today);
        d.setDate(d.getDate() - 56);
        state.floorStartDate = d.toISOString().substring(0, 10);
        state.floorEndDate = todayStr;
    } else if (preset === '4w') {
        const d = new Date(today);
        d.setDate(d.getDate() - 28);
        state.floorStartDate = d.toISOString().substring(0, 10);
        state.floorEndDate = todayStr;
    } else if (preset === 'custom') {
        if (startInput) startInput.focus();
        return;
    }

    if (startInput) startInput.value = state.floorStartDate;
    if (endInput) endInput.value = state.floorEndDate;

    loadFloorStreakTab();
}

function onFloorDateRangeInputChange() {
    const startInput = document.getElementById('floor-start-date');
    const endInput = document.getElementById('floor-end-date');
    state.floorStartDate = startInput?.value || '';
    state.floorEndDate = endInput?.value || '';
    state.floorDatePreset = 'custom';
    document.querySelectorAll('[id^="floor-preset-"]').forEach(btn => btn.classList.remove('active'));
    const customBtn = document.getElementById('floor-preset-custom');
    if (customBtn) customBtn.classList.add('active');
    loadFloorStreakTab();
}

function clearFloorHistoryRange() {
    setFloorHistoryPreset('season26');
}

function updateFloorKPIs(players) {
    const longestEl = document.getElementById('floor-kpi-longest');
    const eliteEl = document.getElementById('floor-kpi-elite');
    const solidEl = document.getElementById('floor-kpi-solid');
    const devEl = document.getElementById('floor-kpi-breached');

    if (!players || players.length === 0) {
        if (longestEl) longestEl.innerHTML = '--';
        if (eliteEl) eliteEl.textContent = '0 players';
        if (solidEl) solidEl.textContent = '0 players';
        if (devEl) devEl.textContent = '0 players';
        return;
    }

    // 1. Longest Active Streak
    let longestPlayer = players[0];
    for (const p of players) {
        if ((p.Floor_Streak || 0) > (longestPlayer.Floor_Streak || 0)) {
            longestPlayer = p;
        }
    }
    if (longestEl && longestPlayer) {
        longestEl.innerHTML = `${longestPlayer.player_name} (${longestPlayer.team}) &bull; <span class="text-zinc-100 font-bold">${longestPlayer.Floor_Streak} games</span>`;
    }

    // 2. Elite Consistency (8+)
    const eliteCount = players.filter(p => (p.Floor_Streak || 0) >= 8).length;
    if (eliteEl) eliteEl.textContent = `${eliteCount} players`;

    // 3. Solid Floor (5-7)
    const solidCount = players.filter(p => (p.Floor_Streak || 0) >= 5 && (p.Floor_Streak || 0) < 8).length;
    if (solidEl) solidEl.textContent = `${solidCount} players`;

    // 4. Developing Streaks (1-4)
    const devCount = players.filter(p => (p.Floor_Streak || 0) >= 1 && (p.Floor_Streak || 0) < 5).length;
    if (devEl) devEl.textContent = `${devCount} players`;
}

function renderFloorDateChips() {
    const container = document.getElementById('floor-date-chips');
    if (!container) return;

    // Filter games in active range
    const activeGames = (state.allGames || []).filter(g => {
        const gd = (g.gameday || g.date || '').substring(0, 10);
        return !state.startDate || !state.endDate || (gd >= state.startDate && gd <= state.endDate);
    });
    const gamesToUse = activeGames.length > 0 ? activeGames : (state.allGames || []);

    const teamDates = {};
    const dateCounts = {};
    gamesToUse.forEach(g => {
        const d = (g.gameday || g.date || '').substring(0, 10);
        if (d && /^\d{4}-\d{2}-\d{2}$/.test(d)) {
            if (g.home_team) teamDates[g.home_team.toUpperCase()] = d;
            if (g.away_team) teamDates[g.away_team.toUpperCase()] = d;
            if (!dateCounts[d]) dateCounts[d] = 0;
        }
    });

    (state.floorStreaks || []).forEach(p => {
        const t = (p.team || p.Team || '').toUpperCase();
        const d = teamDates[t] || (p.date || p.gameday || p.Date || '').substring(0, 10);
        if (d && dateCounts.hasOwnProperty(d)) {
            dateCounts[d] = (dateCounts[d] || 0) + 1;
        }
    });

    const uniqueDates = Object.keys(dateCounts).sort();

    let scopedDates = uniqueDates;
    if (state.startDate && state.endDate) {
        const inWindow = uniqueDates.filter(d => d >= state.startDate && d <= state.endDate);
        if (inWindow.length > 0) scopedDates = inWindow;
    }

    const totalCount = state.floorStreaks.length;
    let html = `<button id="btn-floor-all-dates" onclick="clearFloorDateFilter()" class="chip ${!state.floorSelectedDate ? 'active' : ''} text-[11px] py-0.5 px-2 shrink-0" title="Show all dates">All (${totalCount})</button>`;

    scopedDates.forEach(d => {
        const count = dateCounts[d] || 0;
        const dateObj = new Date(d + 'T00:00:00Z');
        const label = dateObj.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric', timeZone: 'UTC' });
        const isActive = state.floorSelectedDate === d;
        html += `<button onclick="onFloorDateChange('${d}')" class="chip ${isActive ? 'active' : ''} text-[11px] py-0.5 px-2 shrink-0" title="${label}: ${count} player${count === 1 ? '' : 's'}">${label} (${count})</button>`;
    });

    container.innerHTML = html;
}

function onFloorDateChange(date) {
    state.floorSelectedDate = date || '';
    const datePicker = document.getElementById('floor-date-picker');
    if (datePicker) {
        datePicker.value = state.floorSelectedDate;
        if (state.floorSelectedDate) {
            datePicker.classList.add('border-zinc-400');
        } else {
            datePicker.classList.remove('border-zinc-400');
        }
    }
    renderFloorDateChips();
    renderFloorStreaks();
}

function clearFloorDateFilter() {
    state.floorSelectedDate = '';
    const datePicker = document.getElementById('floor-date-picker');
    if (datePicker) {
        datePicker.value = '';
        datePicker.classList.remove('border-zinc-400');
    }
    renderFloorDateChips();
    renderFloorStreaks();
}

function renderFloorStreaks() {
    const tbody = document.getElementById('floor-streak-table-body');
    const countEl = document.getElementById('floor-visible-count');
    if (!tbody) return;

    let filtered = [...state.floorStreaks];

    // Determine teams playing on floorSelectedDate
    let dateTeams = null;
    if (state.floorSelectedDate) {
        dateTeams = new Set();
        (state.allGames || []).forEach(g => {
            const gDate = (g.gameday || g.date || '').substring(0, 10);
            if (gDate === state.floorSelectedDate) {
                if (g.home_team) dateTeams.add(g.home_team.toUpperCase());
                if (g.away_team) dateTeams.add(g.away_team.toUpperCase());
            }
        });
    }

    // Filter by Selected Game Date: strictly show players playing on that date
    if (state.floorSelectedDate) {
        filtered = filtered.filter(p => {
            const team = (p.team || p.Team || '').toUpperCase();
            if (dateTeams && dateTeams.size > 0) {
                return dateTeams.has(team);
            }
            const pDate = (p.date || p.gameday || p.Date || '').substring(0, 10);
            return pDate === state.floorSelectedDate;
        });
    }

    // Filter by Position
    if (state.floorFilterPos && state.floorFilterPos !== 'ALL') {
        filtered = filtered.filter(p => (p.position || '').toUpperCase() === state.floorFilterPos);
    }

    // Filter by Min Streak
    const minStreak = parseInt(state.floorFilterStreak || 0, 10);
    if (minStreak > 0) {
        filtered = filtered.filter(p => (p.Floor_Streak || 0) >= minStreak);
    }

    // Filter by Search Query
    if (state.floorSearchQuery) {
        const q = state.floorSearchQuery.toLowerCase().trim();
        filtered = filtered.filter(p => 
            (p.player_name || '').toLowerCase().includes(q) ||
            (p.team || '').toLowerCase().includes(q)
        );
    }

    // Sort
    const col = state.floorSortCol;
    const dir = state.floorSortDir;
    filtered.sort((a, b) => {
        let valA = a[col] ?? 0;
        let valB = b[col] ?? 0;
        if (typeof valA === 'string') {
            return dir === 'asc' ? valA.localeCompare(valB) : valB.localeCompare(valA);
        }
        return dir === 'asc' ? valA - valB : valB - valA;
    });

    tbody.innerHTML = components.renderFloorStreakTable(filtered);
    if (countEl) countEl.textContent = filtered.length;
    updateFloorKPIs(filtered);
    if (window.lucide) lucide.createIcons({ root: tbody });
}

function sortFloorStreaks(col) {
    if (col === 'rank') {
        state.floorSortCol = 'Floor_Streak';
        state.floorSortDir = state.floorSortDir === 'asc' ? 'desc' : 'asc';
    } else if (state.floorSortCol === col) {
        state.floorSortDir = state.floorSortDir === 'asc' ? 'desc' : 'asc';
    } else {
        state.floorSortCol = col;
        state.floorSortDir = (col === 'player_name' || col === 'position' || col === 'team') ? 'asc' : 'desc';
    }
    renderFloorStreaks();
}

function setFloorPosFilter(pos) {
    state.floorFilterPos = pos;
    ['ALL', 'QB', 'RB', 'WR', 'TE'].forEach(p => {
        const btn = document.getElementById(`floor-filter-pos-${p}`);
        if (btn) {
            if (p === pos) btn.classList.add('active');
            else btn.classList.remove('active');
        }
    });
    renderFloorStreaks();
}

function setFloorStreakFilter(minStreak) {
    state.floorFilterStreak = parseInt(minStreak, 10);
    renderFloorStreaks();
}

function setFloorMinGamesFilter(minGames) {
    state.floorFilterMinGames = parseInt(minGames, 10);
    loadFloorStreakTab();
}

function onFloorSearch(query) {
    state.floorSearchQuery = query || '';
    renderFloorStreaks();
}

/**
 * Tab 4: Matchup Lab (H2H)
 */
async function loadH2HTab() {
    const homeSelect = document.getElementById('h2h-home-select');
    const awaySelect = document.getElementById('h2h-away-select');
    const container = document.getElementById('h2h-container');

    const homeTeam = homeSelect?.value || 'KC';
    const awayTeam = awaySelect?.value || 'BAL';

    if (container) {
        container.innerHTML = `<div class="p-8 text-center text-slate-400">Simulating ${awayTeam} @ ${homeTeam} unit battles...</div>`;
    }

    const data = await api.getMatchupDeepDive(homeTeam, awayTeam, {
        season: state.season,
        start_date: state.startDate,
        end_date: state.endDate
    });

    if (container) {
        container.innerHTML = components.renderMatchupDeepDive(data);
        if (window.lucide) lucide.createIcons({ root: container });
    }
}

/**
 * Tab 5: Schedule & Odds Loader
 */
async function loadScheduleTab() {
    const container = document.getElementById('schedule-container');
    if (container) {
        container.innerHTML = `
            <div class="p-8 text-center text-zinc-400 glass-panel">
                <i data-lucide="loader-2" class="w-6 h-6 animate-spin mx-auto mb-2 text-zinc-400"></i>
                <p class="text-xs font-mono">Loading full season schedule & Vegas lines...</p>
            </div>
        `;
        if (window.lucide) lucide.createIcons({ root: container });
    }

    await updateScheduleWeekOptions();

    try {
        const scheduleParams = {
            season: state.scheduleSeason || 2025
        };
        if (state.scheduleStartDate) {
            scheduleParams.start_date = state.scheduleStartDate;
        }
        if (state.scheduleEndDate) {
            scheduleParams.end_date = state.scheduleEndDate;
        }
        const data = await api.getSchedule(scheduleParams);
        state.scheduleData = data;

        const titleEl = document.getElementById('schedule-title-text');
        if (titleEl) {
            titleEl.textContent = `NFL Schedule & Matchups (${state.scheduleSeason || 2025})`;
        }
        const timelineEl = document.getElementById('schedule-active-timeline-text');
        if (timelineEl) {
            if (data.stats_baseline_note) {
                timelineEl.textContent = data.stats_baseline_note;
            } else if (state.scheduleStartDate && state.scheduleEndDate) {
                timelineEl.textContent = `Stats Range: ${state.scheduleStartDate} to ${state.scheduleEndDate}`;
            } else {
                timelineEl.textContent = `Reflecting full regular season baseline (${state.scheduleSeason || 2025})`;
            }
        }

        if (container) {
            container.innerHTML = components.renderSchedule(data, state.scheduleWeek || '1');
            if (window.lucide) lucide.createIcons({ root: container });
        }

        updateMatchupsExcelLink();
    } catch (err) {
        console.warn('API getSchedule failed (attempting static schedule JSON for GitHub version):', err);
        let fallbackData = window.__staticSchedule2026;
        if (!fallbackData) {
            try {
                let resp = await fetch('data/schedule_2026.json');
                if (!resp.ok) resp = await fetch('frontend/data/schedule_2026.json');
                if (resp.ok) {
                    fallbackData = await resp.json();
                    window.__staticSchedule2026 = fallbackData;
                }
            } catch (_) {}
        }
        if (!fallbackData) {
            fallbackData = getFallbackSchedule(state.scheduleSeason || 2026);
        }
        state.scheduleData = fallbackData;
        if (container) {
            container.innerHTML = components.renderSchedule(fallbackData, state.scheduleWeek || '1');
            if (window.lucide) lucide.createIcons({ root: container });
        }
    }
}

/**
 * Populate week options based on schedule weeks for selected season
 */
async function updateScheduleWeekOptions() {
    const weekSelect = document.getElementById('schedule-week-select');
    if (!weekSelect) return;

    try {
        const weeksData = await api.getScheduleWeeks(state.scheduleSeason || 2025);
        if (weeksData && weeksData.weeks && weeksData.weeks.length > 0) {
            const currentVal = String(state.scheduleWeek || '1');
            let optionsHtml = '';
            weeksData.weeks.forEach(w => {
                const isSel = String(w.week) === currentVal ? 'selected' : '';
                const dateHint = w.dates && w.dates.length > 0 ? ` (${w.dates[0]})` : '';
                optionsHtml += `<option value="${w.week}" ${isSel}>Week ${w.week}${dateHint}</option>`;
            });
            optionsHtml += `<option value="all" ${currentVal === 'all' ? 'selected' : ''}>All 18 Weeks</option>`;
            weekSelect.innerHTML = optionsHtml;
        }
    } catch (e) {
        console.warn('Could not populate schedule weeks list from API (using default 1-18 weeks):', e);
        let optionsHtml = '';
        for (let i = 1; i <= 18; i++) {
            const isSel = String(i) === String(state.scheduleWeek || '1') ? 'selected' : '';
            optionsHtml += `<option value="${i}" ${isSel}>Week ${i}</option>`;
        }
        optionsHtml += `<option value="all" ${state.scheduleWeek === 'all' ? 'selected' : ''}>All 18 Weeks</option>`;
        weekSelect.innerHTML = optionsHtml;
    }
}

function updateMatchupsExcelLink() {
    const btn = document.getElementById('btn-download-matchups-excel');
    if (btn) {
        const s = state.scheduleSeason || 2025;
        const w = state.scheduleWeek || '1';
        let url = `/api/export-matchups-excel?season=${s}&week=${w}`;
        if (state.scheduleStartDate) url += `&start_date=${encodeURIComponent(state.scheduleStartDate)}`;
        if (state.scheduleEndDate) url += `&end_date=${encodeURIComponent(state.scheduleEndDate)}`;
        btn.href = url;
        const span = btn.querySelector('span');
        if (span) {
            span.textContent = w === 'all' ? `Export All Weeks Excel` : `Export Week ${w} Excel`;
        }
    }
}

function onScheduleWeekChange(week) {
    state.scheduleWeek = week;
    if (week !== 'all') {
        state.criteriaWeek = parseInt(week, 10);
        const cwSelect = document.getElementById('criteria-week-select');
        if (cwSelect) cwSelect.value = String(state.criteriaWeek);
        if (state.activeTab === 'criteria') {
            loadCriteriaTab();
        }
    }
    updateMatchupsExcelLink();
    const container = document.getElementById('schedule-container');
    if (container && state.scheduleData) {
        container.innerHTML = components.renderSchedule(state.scheduleData, state.scheduleWeek);
        if (window.lucide) lucide.createIcons({ root: container });
    } else {
        loadScheduleTab();
    }
}

function onScheduleSeasonChange(season) {
    state.scheduleSeason = parseInt(season, 10);
    state.scheduleWeek = '1';
    if (state.scheduleSeason === 2026) {
        setScheduleDatePreset('season26');
    } else {
        setScheduleDatePreset('full');
    }
}

/**
 * Schedule Stats History Date Presets & Custom Range Controllers
 */
function setScheduleDatePreset(preset) {
    state.scheduleDatePreset = preset;
    document.querySelectorAll('[id^="sched-preset-"]').forEach(btn => btn.classList.remove('active'));
    const btn = document.getElementById(`sched-preset-${preset}`);
    if (btn) btn.classList.add('active');

    const startInput = document.getElementById('schedule-start-date');
    const endInput = document.getElementById('schedule-end-date');

    const todayStr = new Date().toISOString().substring(0, 10);
    const targetSeason = state.scheduleSeason || 2026;
    const anchorDate = targetSeason === 2026 ? todayStr : (targetSeason === 2025 ? '2026-01-05' : `${targetSeason}-12-31`);
    const anchor = new Date(anchorDate + 'T00:00:00Z');

    if (preset === 'season26') {
        state.scheduleStartDate = '2026-09-01';
        state.scheduleEndDate = todayStr;
    } else if (preset === 'full') {
        state.scheduleStartDate = '';
        state.scheduleEndDate = '';
    } else if (preset === '2w') {
        const d = new Date(anchor);
        d.setDate(d.getDate() - 14);
        state.scheduleStartDate = d.toISOString().substring(0, 10);
        state.scheduleEndDate = anchorDate;
    } else if (preset === '4w') {
        const d = new Date(anchor);
        d.setDate(d.getDate() - 28);
        state.scheduleStartDate = d.toISOString().substring(0, 10);
        state.scheduleEndDate = anchorDate;
    } else if (preset === '8w') {
        const d = new Date(anchor);
        d.setDate(d.getDate() - 56);
        state.scheduleStartDate = d.toISOString().substring(0, 10);
        state.scheduleEndDate = anchorDate;
    } else if (preset === 'custom') {
        if (startInput) startInput.focus();
        return;
    }

    if (startInput) startInput.value = state.scheduleStartDate;
    if (endInput) endInput.value = state.scheduleEndDate;

    loadScheduleTab();
}

function onScheduleDateInputChange() {
    const startInput = document.getElementById('schedule-start-date');
    const endInput = document.getElementById('schedule-end-date');
    state.scheduleStartDate = startInput?.value || '';
    state.scheduleEndDate = endInput?.value || '';
    state.scheduleDatePreset = 'custom';
    document.querySelectorAll('[id^="sched-preset-"]').forEach(btn => btn.classList.remove('active'));
    const customBtn = document.getElementById('sched-preset-custom');
    if (customBtn) customBtn.classList.add('active');
    loadScheduleTab();
}

function clearScheduleDateRange() {
    setScheduleDatePreset('season26');
}

/**
 * =====================================================================
 * TAB: SCHEME-BASED MATCHUP INSIGHTS ENGINE & CONTROLLERS
 * =====================================================================
 */
async function loadInsightsTab() {
    const seasonSelect = document.getElementById('insights-season-select');
    const weekSelect = document.getElementById('insights-week-select');

    if (seasonSelect) seasonSelect.value = String(state.insightsSeason || 2026);
    if (weekSelect) weekSelect.value = String(state.insightsWeek || 1);

    // Populate slate matchup dropdown and quick-switch pills
    await updateInsightsMatchupOptions();

    const container = document.getElementById('insights-content-container');
    if (container) {
        container.innerHTML = `
            <div class="p-8 text-center text-zinc-400 glass-panel">
                <i data-lucide="loader-2" class="w-6 h-6 animate-spin mx-auto mb-2 text-purple-400"></i>
                <p class="text-xs font-mono">Cross-referencing scheme metrics & coverage tendencies for ${state.insightsAwayTeam} @ ${state.insightsHomeTeam}...</p>
            </div>
        `;
        if (window.lucide) lucide.createIcons({ root: container });
    }

    const cacheKey = `${state.insightsHomeTeam}_${state.insightsAwayTeam}_${state.insightsSeason || 2026}_${state.insightsWeek || 1}`;
    if (state.insightsCache && state.insightsCache[cacheKey]) {
        const cachedData = state.insightsCache[cacheKey];
        state.insightsData = cachedData;
        if (container) {
            container.innerHTML = components.renderSchemeInsights(cachedData);
            if (window.lucide) lucide.createIcons({ root: container });
        }
        return;
    }

    try {
        let data = null;
        try {
            data = await api.getSchemeInsights(state.insightsHomeTeam, state.insightsAwayTeam, {
                season: state.insightsSeason || 2026,
                week: state.insightsWeek || 1
            });
        } catch (apiErr) {
            console.warn('API getSchemeInsights failed (using client-side scheme engine for GitHub version):', apiErr);
            if (!window.__staticSchemeProfiles32) {
                try {
                    let pResp = await fetch('data/scheme_profiles_32.json');
                    if (!pResp.ok) pResp = await fetch('frontend/data/scheme_profiles_32.json');
                    if (pResp.ok) window.__staticSchemeProfiles32 = await pResp.json();
                } catch (_) {}
            }
            data = generateClientSchemeInsights(state.insightsHomeTeam, state.insightsAwayTeam, state.insightsSeason || 2026, state.insightsWeek || 1);
        }

        state.insightsData = data;
        state.insightsCache[cacheKey] = data;
        if (container) {
            container.innerHTML = components.renderSchemeInsights(data);
            if (window.lucide) lucide.createIcons({ root: container });
        }
    } catch (err) {
        console.error('Error in loadInsightsTab:', err);
        if (container) {
            container.innerHTML = `
                <div class="p-8 text-center glass-panel border border-red-900/50">
                    <i data-lucide="alert-triangle" class="w-6 h-6 mx-auto mb-2 text-zinc-400"></i>
                    <p class="text-sm text-zinc-300 font-semibold">Failed to load scheme insights</p>
                    <p class="text-xs text-zinc-500 font-mono mt-1">${err.message}</p>
                </div>
            `;
            if (window.lucide) lucide.createIcons({ root: container });
        }
    }
}

/**
 * Jump directly from a game card into Scheme Insights tab
 */
async function inspectMatchupInsights(home, away, week = null) {
    if (!home || !away) return;
    state.insightsHomeTeam = home.toUpperCase();
    state.insightsAwayTeam = away.toUpperCase();
    if (week != null && !isNaN(parseInt(week, 10))) {
        state.insightsWeek = parseInt(week, 10);
    }
    const seasonSelect = document.getElementById('insights-season-select');
    if (seasonSelect) {
        state.insightsSeason = parseInt(seasonSelect.value, 10) || state.scheduleSeason || 2026;
    }

    if (state.activeTab === 'insights') {
        await loadInsightsTab();
    } else {
        await switchTab('insights');
    }

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

/**
 * Populates Matchup dropdown and quick pills for the active week
 */
async function updateInsightsMatchupOptions() {
    const matchupSelect = document.getElementById('insights-matchup-select');
    const quickBar = document.getElementById('insights-quick-matchups');
    const s = state.insightsSeason || 2026;
    const w = state.insightsWeek || 1;

    let games = [];
    if (state.scheduleData && state.scheduleData.weeks) {
        const weekObj = state.scheduleData.weeks.find(x => x.week === w);
        if (weekObj && weekObj.games) games = weekObj.games;
    }

    if (games.length === 0) {
        if (!window.__staticSchedule2026) {
            try {
                let resp = await fetch('data/schedule_2026.json');
                if (!resp.ok) resp = await fetch('frontend/data/schedule_2026.json');
                if (resp.ok) window.__staticSchedule2026 = await resp.json();
            } catch (_) {}
        }
        const fb = window.__staticSchedule2026 || getFallbackSchedule(s);
        const fbWeek = fb.weeks ? fb.weeks.find(x => x.week === w) : null;
        if (fbWeek && fbWeek.games) games = fbWeek.games;
        else if (fb.weeks && fb.weeks[0] && fb.weeks[0].games) games = fb.weeks[0].games;
    }

    // Resiliently match currently selected home & away against the slate
    const home = (state.insightsHomeTeam || '').toUpperCase();
    const away = (state.insightsAwayTeam || '').toUpperCase();
    let matchedGame = games.find(g => (g.home_team === home && g.away_team === away));
    if (!matchedGame) {
        matchedGame = games.find(g => (g.home_team === away && g.away_team === home));
    }
    if (matchedGame) {
        state.insightsHomeTeam = matchedGame.home_team;
        state.insightsAwayTeam = matchedGame.away_team;
    } else if (games.length > 0) {
        state.insightsHomeTeam = games[0].home_team;
        state.insightsAwayTeam = games[0].away_team;
    }

    if (matchupSelect) {
        matchupSelect.innerHTML = games.map(g => {
            const key = `${g.home_team}_${g.away_team}`;
            const isSel = (g.home_team === state.insightsHomeTeam && g.away_team === state.insightsAwayTeam) ? 'selected' : '';
            return `<option value="${key}" ${isSel}>${g.away_team} @ ${g.home_team} (${g.gameday || 'Week ' + w})</option>`;
        }).join('');
    }

    if (quickBar) {
        quickBar.innerHTML = games.map(g => {
            const isActive = (g.home_team === state.insightsHomeTeam && g.away_team === state.insightsAwayTeam);
            return `
                <button onclick="inspectMatchupInsights('${g.home_team}', '${g.away_team}', ${g.week || w})" class="px-2.5 py-1 rounded-lg text-xs font-mono font-semibold whitespace-nowrap transition flex items-center gap-1.5 border cursor-pointer ${isActive ? 'bg-purple-600/30 text-purple-200 border-purple-500/60 shadow-sm' : 'bg-zinc-900 text-zinc-400 border-zinc-800 hover:border-zinc-700 hover:text-zinc-200'}">
                    <span class="font-bold text-zinc-200">${g.away_team}</span>
                    <span class="text-zinc-500 text-[10px]">@</span>
                    <span class="font-bold text-zinc-200">${g.home_team}</span>
                </button>
            `;
        }).join('');
    }
}

function onInsightsMatchupChange(val) {
    if (!val) return;
    const parts = val.split('_');
    if (parts.length === 2) {
        state.insightsHomeTeam = parts[0];
        state.insightsAwayTeam = parts[1];
        loadInsightsTab();
    }
}

function onInsightsWeekChange(val) {
    state.insightsWeek = parseInt(val, 10) || 1;
    loadInsightsTab();
}

function onInsightsSeasonChange(val) {
    state.insightsSeason = parseInt(val, 10) || 2026;
    loadInsightsTab();
}

/**
 * Client-Side Scheme Insights Generator (Resilient Fallback for GitHub Pages / Static Hosting)
 */
function generateClientSchemeInsights(homeTeam, awayTeam, season = 2026, week = 1) {
    const h = (homeTeam || 'SEA').toUpperCase();
    const a = (awayTeam || 'NE').toUpperCase();

    const teamNames = NFL_TEAM_NAMES;

    const schemeDb = {
        'BAL': {
            coverage: { zone_pct: 64.2, man_pct: 35.8, mfo_pct: 46.5, mfc_pct: 53.5, cover_1: 24.1, cover_2: 11.2, cover_3: 29.4, cover_4: 18.2, cover_6: 11.1, cover_0: 6.0, archetype: 'Disguised Hybrid (Cover 1 / Cover 3 Sim Pressures)', vulnerability_note: 'Middle of field vulnerable to seam routes when rolling single-high; elite at boundary bracket coverage.' },
            run: { offense_zone_pct: 41.5, offense_gap_pct: 58.5, zone_ypc: 4.62, gap_ypc: 5.45, def_zone_ypc_allowed: 3.72, def_zone_rank: 4, def_gap_ypc_allowed: 4.38, def_gap_rank: 18, identity: 'Heavy Gap / Power / Pistol option attack (Derrick Henry & Lamar Jackson).' },
            targets: { slot_wr_pct: 24.5, wide_wr_pct: 44.2, inline_te_pct: 21.8, backfield_rb_pct: 9.5, key_targets: [
                { name: 'Zay Flowers', pos: 'WR', slot_pct: 46, outside_pct: 54, inline_pct: 0, tgt_man: 28.4, tgt_zone: 23.2, yprr_man: 2.45, yprr_zone: 1.95 },
                { name: 'Mark Andrews', pos: 'TE', slot_pct: 52, outside_pct: 8, inline_pct: 40, tgt_man: 19.5, tgt_zone: 21.8, yprr_man: 1.82, yprr_zone: 2.10 },
                { name: 'Rashod Bateman', pos: 'WR', slot_pct: 14, outside_pct: 86, inline_pct: 0, tgt_man: 18.2, tgt_zone: 14.5, yprr_man: 1.65, yprr_zone: 1.40 },
                { name: 'Isaiah Likely', pos: 'TE', slot_pct: 38, outside_pct: 12, inline_pct: 50, tgt_man: 12.0, tgt_zone: 15.5, yprr_man: 1.55, yprr_zone: 1.78 }
            ]},
            tempo: { neutral_pace_sec: 29.8, pace_rank: 26, proj_plays: 62.5, red_zone_touch_leader: 'Derrick Henry (68% Goal-Line Share)', checkdown_pct: 11.2 }
        },
        'KC': {
            coverage: { zone_pct: 63.8, man_pct: 36.2, mfo_pct: 61.5, mfc_pct: 38.5, cover_1: 22.4, cover_2: 18.5, cover_3: 16.1, cover_4: 25.2, cover_6: 12.8, cover_0: 5.0, archetype: 'Spagnuolo Disguised MFO Quarters / Aggressive Blitz Funnel', vulnerability_note: 'Deep boundary locked by two-high safeties; underneath flats open against match-quarters.' },
            run: { offense_zone_pct: 63.5, offense_gap_pct: 36.5, zone_ypc: 4.38, gap_ypc: 4.10, def_zone_ypc_allowed: 3.65, def_zone_rank: 3, def_gap_ypc_allowed: 4.25, def_gap_rank: 15, identity: 'Spread zone with motion and misdirection; stout interior run defense anchored by Chris Jones.' },
            targets: { slot_wr_pct: 32.4, wide_wr_pct: 36.8, inline_te_pct: 18.2, backfield_rb_pct: 12.6, key_targets: [
                { name: 'Travis Kelce', pos: 'TE', slot_pct: 48, outside_pct: 14, inline_pct: 38, tgt_man: 22.4, tgt_zone: 25.8, yprr_man: 1.95, yprr_zone: 2.35 },
                { name: 'Rashee Rice', pos: 'WR', slot_pct: 58, outside_pct: 42, inline_pct: 0, tgt_man: 26.5, tgt_zone: 28.2, yprr_man: 2.30, yprr_zone: 2.75 },
                { name: 'Xavier Worthy', pos: 'WR', slot_pct: 28, outside_pct: 72, inline_pct: 0, tgt_man: 18.2, tgt_zone: 14.8, yprr_man: 1.85, yprr_zone: 1.45 },
                { name: 'Isiah Pacheco', pos: 'RB', slot_pct: 6, outside_pct: 4, inline_pct: 0, tgt_man: 7.5, tgt_zone: 12.4, yprr_man: 0.85, yprr_zone: 1.25 }
            ]},
            tempo: { neutral_pace_sec: 27.2, pace_rank: 11, proj_plays: 65.0, red_zone_touch_leader: 'Isiah Pacheco / Rashee Rice Motion Jet', checkdown_pct: 14.8 }
        },
        'SF': {
            coverage: { zone_pct: 74.5, man_pct: 25.5, mfo_pct: 34.0, mfc_pct: 66.0, cover_1: 16.5, cover_2: 8.5, cover_3: 46.2, cover_4: 15.8, cover_6: 9.5, cover_0: 3.5, archetype: 'Seattle 3-Match / Cover 3 Linebacker Wall (Fred Warner)', vulnerability_note: 'Flats and deep boundary corner routes test Cover 3 outside leverage; impossible to run seam against Warner.' },
            run: { offense_zone_pct: 71.2, offense_gap_pct: 28.8, zone_ypc: 4.85, gap_ypc: 4.30, def_zone_ypc_allowed: 3.85, def_zone_rank: 6, def_gap_ypc_allowed: 4.15, def_gap_rank: 14, identity: 'Shanahan Outside Zone benchmark with Kyle Juszczyk fullback cutbacks and Christian McCaffrey.' },
            targets: { slot_wr_pct: 28.5, wide_wr_pct: 38.2, inline_te_pct: 17.5, backfield_rb_pct: 15.8, key_targets: [
                { name: 'Deebo Samuel', pos: 'WR', slot_pct: 42, outside_pct: 46, inline_pct: 0, tgt_man: 24.2, tgt_zone: 23.5, yprr_man: 2.10, yprr_zone: 2.25 },
                { name: 'Brandon Aiyuk', pos: 'WR', slot_pct: 18, outside_pct: 82, inline_pct: 0, tgt_man: 28.5, tgt_zone: 21.0, yprr_man: 2.75, yprr_zone: 2.15 },
                { name: 'George Kittle', pos: 'TE', slot_pct: 32, outside_pct: 10, inline_pct: 58, tgt_man: 20.5, tgt_zone: 22.4, yprr_man: 2.05, yprr_zone: 2.50 },
                { name: 'Christian McCaffrey', pos: 'RB', slot_pct: 16, outside_pct: 8, inline_pct: 0, tgt_man: 16.5, tgt_zone: 18.8, yprr_man: 1.55, yprr_zone: 1.85 }
            ]},
            tempo: { neutral_pace_sec: 30.6, pace_rank: 32, proj_plays: 61.0, red_zone_touch_leader: 'Christian McCaffrey (74% Red Zone Touch Share)', checkdown_pct: 12.5 }
        },
        'DET': {
            coverage: { zone_pct: 54.2, man_pct: 45.8, mfo_pct: 41.8, mfc_pct: 58.2, cover_1: 32.5, cover_2: 10.5, cover_3: 25.8, cover_4: 17.5, cover_6: 8.2, cover_0: 5.5, archetype: 'Aaron Glenn Aggressive Press-Man & Physical Boundary Shell', vulnerability_note: 'Rub concepts and double-moves challenge press-man; front 7 completely bottles up interior runs.' },
            run: { offense_zone_pct: 47.8, offense_gap_pct: 52.2, zone_ypc: 4.90, gap_ypc: 5.15, def_zone_ypc_allowed: 3.98, def_zone_rank: 11, def_gap_ypc_allowed: 3.52, def_gap_rank: 2, identity: 'Elite offensive line running balanced Power/Duo (Montgomery) and Outside Zone stretch (Gibbs).' },
            targets: { slot_wr_pct: 36.5, wide_wr_pct: 32.4, inline_te_pct: 19.8, backfield_rb_pct: 11.3, key_targets: [
                { name: 'Amon-Ra St. Brown', pos: 'WR', slot_pct: 68, outside_pct: 32, inline_pct: 0, tgt_man: 29.5, tgt_zone: 32.4, yprr_man: 2.40, yprr_zone: 2.78 },
                { name: 'Sam LaPorta', pos: 'TE', slot_pct: 34, outside_pct: 12, inline_pct: 54, tgt_man: 21.0, tgt_zone: 20.5, yprr_man: 1.85, yprr_zone: 2.15 },
                { name: 'Jameson Williams', pos: 'WR', slot_pct: 16, outside_pct: 84, inline_pct: 0, tgt_man: 22.0, tgt_zone: 16.5, yprr_man: 2.25, yprr_zone: 1.70 },
                { name: 'Jahmyr Gibbs', pos: 'RB', slot_pct: 12, outside_pct: 6, inline_pct: 0, tgt_man: 12.5, tgt_zone: 14.8, yprr_man: 1.35, yprr_zone: 1.65 }
            ]},
            tempo: { neutral_pace_sec: 26.5, pace_rank: 6, proj_plays: 66.5, red_zone_touch_leader: 'David Montgomery (Duo Power Goal-Line)', checkdown_pct: 9.8 }
        },
        'NE': {
            coverage: { zone_pct: 56.4, man_pct: 43.6, mfo_pct: 40.5, mfc_pct: 59.5, cover_1: 30.5, cover_2: 12.0, cover_3: 29.0, cover_4: 14.5, cover_6: 9.0, cover_0: 5.0, archetype: 'Belichick / Mayo Matchup-Man & Single-High Box Clamp', vulnerability_note: 'Secondary lacks speed on boundary crossers; stout run wall on interior gaps.' },
            run: { offense_zone_pct: 42.0, offense_gap_pct: 58.0, zone_ypc: 3.95, gap_ypc: 4.85, def_zone_ypc_allowed: 4.05, def_zone_rank: 12, def_gap_ypc_allowed: 3.82, def_gap_rank: 6, identity: 'Alex Van Pelt power run offense with Rhamondre Stevenson (58% Gap runs).' },
            targets: { slot_wr_pct: 28.0, wide_wr_pct: 38.0, inline_te_pct: 22.0, backfield_rb_pct: 12.0, key_targets: [
                { name: 'Hunter Henry', pos: 'TE', slot_pct: 38, outside_pct: 8, inline_pct: 54, tgt_man: 21.0, tgt_zone: 23.5, yprr_man: 1.65, yprr_zone: 1.95 },
                { name: 'DeMario Douglas', pos: 'WR', slot_pct: 72, outside_pct: 28, inline_pct: 0, tgt_man: 24.5, tgt_zone: 22.0, yprr_man: 1.85, yprr_zone: 1.70 },
                { name: 'Rhamondre Stevenson', pos: 'RB', slot_pct: 8, outside_pct: 4, inline_pct: 0, tgt_man: 11.5, tgt_zone: 15.0, yprr_man: 1.10, yprr_zone: 1.45 },
                { name: 'K.J. Osborn', pos: 'WR', slot_pct: 35, outside_pct: 65, inline_pct: 0, tgt_man: 15.0, tgt_zone: 13.5, yprr_man: 1.30, yprr_zone: 1.25 }
            ]},
            tempo: { neutral_pace_sec: 30.2, pace_rank: 30, proj_plays: 61.5, red_zone_touch_leader: 'Rhamondre Stevenson (72% Goal-Line Share)', checkdown_pct: 15.2 }
        },
        'SEA': {
            coverage: { zone_pct: 68.5, man_pct: 31.5, mfo_pct: 65.2, mfc_pct: 34.8, cover_1: 19.5, cover_2: 16.5, cover_3: 15.3, cover_4: 29.5, cover_6: 14.2, cover_0: 5.0, archetype: 'Mike Macdonald Sim-Pressure & Disguised Two-High Safety Shell', vulnerability_note: 'Pre-snap looks morph constantly; vulnerable to quick perimeter screens and tight-end seam drags.' },
            run: { offense_zone_pct: 58.2, offense_gap_pct: 41.8, zone_ypc: 4.65, gap_ypc: 4.40, def_zone_ypc_allowed: 4.40, def_zone_rank: 20, def_gap_ypc_allowed: 4.60, def_gap_rank: 24, identity: 'Ryan Grubb spread with Kenneth Walker explosive zone cutting and Zach Charbonnet.' },
            targets: { slot_wr_pct: 34.0, wide_wr_pct: 42.0, inline_te_pct: 13.5, backfield_rb_pct: 10.5, key_targets: [
                { name: 'DK Metcalf', pos: 'WR', slot_pct: 18, outside_pct: 82, inline_pct: 0, tgt_man: 31.0, tgt_zone: 23.5, yprr_man: 2.85, yprr_zone: 2.20 },
                { name: 'Jaxon Smith-Njigba', pos: 'WR', slot_pct: 78, outside_pct: 22, inline_pct: 0, tgt_man: 24.5, tgt_zone: 28.5, yprr_man: 2.05, yprr_zone: 2.55 },
                { name: 'Tyler Lockett', pos: 'WR', slot_pct: 42, outside_pct: 58, inline_pct: 0, tgt_man: 18.0, tgt_zone: 19.5, yprr_man: 1.65, yprr_zone: 1.85 },
                { name: 'Noah Fant', pos: 'TE', slot_pct: 32, outside_pct: 10, inline_pct: 58, tgt_man: 11.5, tgt_zone: 14.5, yprr_man: 1.35, yprr_zone: 1.65 }
            ]},
            tempo: { neutral_pace_sec: 25.5, pace_rank: 4, proj_plays: 67.5, red_zone_touch_leader: 'Kenneth Walker & DK Metcalf End Zone Fades', checkdown_pct: 11.8 }
        }
    };

    const getProfile = (team) => {
        if (window.__staticSchemeProfiles32 && window.__staticSchemeProfiles32[team]) {
            const sp = window.__staticSchemeProfiles32[team];
            return {
                coverage: sp.coverage,
                run: sp.run_scheme || sp.run,
                targets: sp.target_alignment || sp.targets,
                tempo: sp.tempo_situational || sp.tempo
            };
        }
        if (schemeDb[team]) return schemeDb[team];
        const s = team.split('').reduce((acc, c) => acc + c.charCodeAt(0), 0);
        const z = 58 + (s % 22);
        const mfo = 45 + (s % 25);
        const offZ = 48 + ((s * 3) % 24);
        return {
            coverage: { zone_pct: z, man_pct: 100 - z, mfo_pct: mfo, mfc_pct: 100 - mfo, cover_1: (100 - z) * 0.7, cover_2: mfo * 0.3, cover_3: z * 0.45, cover_4: mfo * 0.4, cover_6: mfo * 0.3, cover_0: (100 - z) * 0.2, archetype: mfo > 55 ? 'Two-High MFO Quarters Shell' : 'Single-High MFC Box Clamp Shell', vulnerability_note: 'Vulnerable to intermediate slot crossers.' },
            run: { offense_zone_pct: offZ, offense_gap_pct: 100 - offZ, zone_ypc: 4.3, gap_ypc: 4.4, def_zone_ypc_allowed: 4.1, def_zone_rank: 1 + (s % 32), def_gap_ypc_allowed: 4.2, def_gap_rank: 1 + ((s + 11) % 32), identity: 'Balanced pro-style run scheme.' },
            targets: { slot_wr_pct: 28, wide_wr_pct: 40, inline_te_pct: 18, backfield_rb_pct: 14, key_targets: [
                { name: `${team} WR1`, pos: 'WR', slot_pct: 35, outside_pct: 65, inline_pct: 0, tgt_man: 28.0, tgt_zone: 24.5, yprr_man: 2.35, yprr_zone: 2.10 },
                { name: `${team} Slot WR`, pos: 'WR', slot_pct: 75, outside_pct: 25, inline_pct: 0, tgt_man: 22.0, tgt_zone: 26.0, yprr_man: 1.85, yprr_zone: 2.25 },
                { name: `${team} TE1`, pos: 'TE', slot_pct: 38, outside_pct: 10, inline_pct: 52, tgt_man: 18.0, tgt_zone: 21.0, yprr_man: 1.65, yprr_zone: 1.95 },
                { name: `${team} RB1`, pos: 'RB', slot_pct: 10, outside_pct: 5, inline_pct: 0, tgt_man: 11.0, tgt_zone: 14.0, yprr_man: 1.05, yprr_zone: 1.40 }
            ]},
            tempo: { neutral_pace_sec: 27.5, pace_rank: 1 + (s % 32), proj_plays: 63, red_zone_touch_leader: `${team} RB1 (Lead Red Zone Share)`, checkdown_pct: 12.0 }
        };
    };

    const hProf = getProfile(h);
    const aProf = getProfile(a);

    const enrichTargets = (tList, oppCov) => {
        return (tList || []).map(p => {
            let boostText = 'Baseline Look Rate: Steady volume within standard route distribution.';
            let tier = 'Neutral';
            let causalMech = '';
            if (p.slot_pct > 50 && oppCov.zone_pct > 60) {
                boostText = `+18% Target Boost: High slot frequency (${p.slot_pct}%) exploits ${oppCov.zone_pct.toFixed(0)}% opponent zone coverage.`;
                tier = 'High Boost';
                causalMech = 'SLOT-M1: Soft intermediate voids in zone coverage concentrate receptions to agile slot receivers.';
            } else if (p.inline_pct > 40 && oppCov.mfo_pct > 50) {
                boostText = `+22% Target Boost: Inline seam routes exploit ${oppCov.mfo_pct.toFixed(0)}% Middle-Field Open (MFO) safety split.`;
                tier = 'High Boost';
                causalMech = 'TE1-M2: Middle-field open two-safety split creates seam and size leverage mismatches for inline tight ends.';
            } else if (p.outside_pct > 65 && oppCov.man_pct > 32) {
                boostText = `+15% Target Boost: Alpha outside route-runner wins isolated 1-on-1s against ${oppCov.man_pct.toFixed(0)}% Man coverage.`;
                tier = 'Moderate Boost';
                causalMech = 'WR1-M3: High man coverage forces isolated 1-on-1 matchups on the boundary where elite alphas separate cleanly.';
            } else if (p.pos === 'RB' && ((oppCov.four_man_pressure_pct && oppCov.four_man_pressure_pct > 31) || (oppCov.blitz_pct && oppCov.blitz_pct > 27))) {
                boostText = `+14% Target Boost: High defensive pass rush pressure triggers safety-valve checkdown dump-offs to backfield.`;
                tier = 'Moderate Boost';
                causalMech = 'RBREC-M4: Consistent pass rush pressure speeds up QB clock, forcing quick dump-offs to backfield outlets.';
            }
            return { ...p, expected_boost: boostText, boost_tier: tier, causal_mechanism: causalMech };
        });
    };

    const aEnriched = enrichTargets(aProf.targets.key_targets, hProf.coverage);
    const hEnriched = enrichTargets(hProf.targets.key_targets, aProf.coverage);

    const aAdv = (hProf.run.def_gap_rank > 18 && aProf.run.offense_gap_pct > 50) ? 'High Advantage' : ((hProf.run.def_gap_rank < 8 && hProf.run.def_zone_rank < 8) ? 'Disadvantage' : 'Neutral / Balanced');
    const hAdv = (aProf.run.def_zone_rank > 18 && hProf.run.offense_zone_pct > 50) ? 'High Advantage' : ((aProf.run.def_gap_rank < 8 && aProf.run.def_zone_rank < 8) ? 'Disadvantage' : 'Neutral / Balanced');

    const breakouts = [];
    [...aEnriched.filter(p => p.boost_tier in {'High Boost': 1, 'Moderate Boost': 1}).map(p => ({ p, t: a, opp: h, oppCov: hProf.coverage })), ...hEnriched.filter(p => p.boost_tier in {'High Boost': 1, 'Moderate Boost': 1}).map(p => ({ p, t: h, opp: a, oppCov: aProf.coverage }))].forEach(item => {
        breakouts.push({
            player: item.p.name, team: item.t, pos: item.p.pos,
            scheme_type: 'Coverage Alignment Mismatch',
            context_metric: `${item.p.tgt_zone.toFixed(1)}% Tgt Share vs Zone (${item.p.yprr_zone.toFixed(2)} YPRR)`,
            season_baseline: `${item.p.tgt_man.toFixed(1)}% vs Man`,
            scheme_delta: `+${(item.p.tgt_zone - item.p.tgt_man).toFixed(1)}% Target Surge`,
            verdict: item.p.boost_tier === 'High Boost' ? 'PRIME BREAKOUT SPOT' : 'ELEVATED TARGET CEILING',
            rationale: `Faces ${item.opp} defense deploying ${item.oppCov.zone_pct.toFixed(1)}% Zone. Target looks spike against coverage shell.`,
            causal_mechanism: item.p.causal_mechanism || ''
        });
    });

    const aLeadBack = (aProf.tempo && aProf.tempo.red_zone_touch_leader ? aProf.tempo.red_zone_touch_leader.split(' (')[0] : `${a} Lead Back`);
    const hLeadBack = (hProf.tempo && hProf.tempo.red_zone_touch_leader ? hProf.tempo.red_zone_touch_leader.split(' (')[0] : `${h} Lead Back`);

    if (aProf.run.gap_ypc >= 4.4 && hProf.run.def_gap_rank >= 16) {
        breakouts.push({
            player: aLeadBack, team: a, pos: 'RB',
            scheme_type: 'Run Scheme Exploit (Gap/Power)',
            context_metric: `${aProf.run.gap_ypc.toFixed(2)} YPC on Gap Carries`,
            season_baseline: `${aProf.run.zone_ypc.toFixed(2)} Zone YPC`,
            scheme_delta: `+${(aProf.run.gap_ypc - aProf.run.zone_ypc).toFixed(2)} YPC Scheme Edge`,
            verdict: 'HIGH EFFICIENCY RUSH BREAKOUT',
            rationale: `${a} runs ${aProf.run.offense_gap_pct.toFixed(0)}% Gap against ${h}'s #${hProf.run.def_gap_rank} ranked Gap run stop (${hProf.run.def_gap_ypc_allowed.toFixed(2)} YPC allowed).`,
            causal_mechanism: 'RBRUSH-M5: Defenses with poor gap discipline yield elevated yards per carry to downhill primary ballcarriers.'
        });
    }

    if (hProf.run.zone_ypc >= 4.4 && aProf.run.def_zone_rank >= 16) {
        breakouts.push({
            player: hLeadBack, team: h, pos: 'RB',
            scheme_type: 'Run Scheme Exploit (Outside Zone)',
            context_metric: `${hProf.run.zone_ypc.toFixed(2)} YPC on Zone Carries`,
            season_baseline: `${hProf.run.gap_ypc.toFixed(2)} Gap YPC`,
            scheme_delta: `+${(hProf.run.zone_ypc - hProf.run.gap_ypc).toFixed(2)} YPC Scheme Edge`,
            verdict: 'HIGH EFFICIENCY RUSH BREAKOUT',
            rationale: `${h} runs ${hProf.run.offense_zone_pct.toFixed(0)}% Zone against ${a}'s #${aProf.run.def_zone_rank} ranked Zone run defense (${aProf.run.def_zone_ypc_allowed.toFixed(2)} YPC allowed).`,
            causal_mechanism: 'RBRUSH-M5: Defenses with poor gap discipline yield elevated yards per carry to primary running backs.'
        });
    }

    if (breakouts.length === 0) {
        breakouts.push({
            player: `${a} Primary Weapon`, team: a, pos: 'WR',
            scheme_type: 'Contextual Target Funnel',
            context_metric: '25.0% Projected Target Share',
            season_baseline: '21.5% Season Average',
            scheme_delta: '+3.5% Target Concentration',
            verdict: 'VOLUME PRIME CANDIDATE',
            rationale: 'Coverage shell alignment directs first-read progression directly into this route tree.'
        });
    }

    return {
        home_team: h, home_team_name: teamNames[h] || h,
        away_team: a, away_team_name: teamNames[a] || a,
        season, week,
        thesis_quote: 'To find the deep analytical insights shared in this video (0:09), your platform needs to look for and cross-reference specific scheme-based metrics. Instead of basic stats like total yards, it ingests: 1. Defensive Coverage Tendencies (Zone vs Man, MFO vs MFC), 2. Run-Scheme Splits (Zone vs Man/Gap), 3. Positional Target Rates (Slot vs Outside, Inline TE), 4. Efficiency by Context (Scheme-Specific YPC/YPR Breakout Spotters), and 5. Game-Flow & Tempo (Time of Possession, Red Zone Concentration, Check-downs).',
        coverage_tendencies: {
            away_vs_home: {
                offense_team: a, defense_team: h,
                def_zone_pct: hProf.coverage.zone_pct, def_man_pct: hProf.coverage.man_pct,
                def_mfo_pct: hProf.coverage.mfo_pct, def_mfc_pct: hProf.coverage.mfc_pct,
                def_archetype: hProf.coverage.archetype,
                shells: { 'Cover 1 (Man)': hProf.coverage.cover_1, 'Cover 2 (MFO)': hProf.coverage.cover_2, 'Cover 3 (MFC)': hProf.coverage.cover_3, 'Cover 4 Quarters (MFO)': hProf.coverage.cover_4, 'Cover 6 Split (MFO)': hProf.coverage.cover_6, 'Cover 0 Blitz': hProf.coverage.cover_0 },
                shell_takeaway: `${h} operates in ${hProf.coverage.mfo_pct.toFixed(1)}% Middle-Field Open (MFO) sets, capping deep shots while conceding underneath intermediate targets.`
            },
            home_vs_away: {
                offense_team: h, defense_team: a,
                def_zone_pct: aProf.coverage.zone_pct, def_man_pct: aProf.coverage.man_pct,
                def_mfo_pct: aProf.coverage.mfo_pct, def_mfc_pct: aProf.coverage.mfc_pct,
                def_archetype: aProf.coverage.archetype,
                shells: { 'Cover 1 (Man)': aProf.coverage.cover_1, 'Cover 2 (MFO)': aProf.coverage.cover_2, 'Cover 3 (MFC)': aProf.coverage.cover_3, 'Cover 4 Quarters (MFO)': aProf.coverage.cover_4, 'Cover 6 Split (MFO)': aProf.coverage.cover_6, 'Cover 0 Blitz': aProf.coverage.cover_0 },
                shell_takeaway: `${a} deploys ${aProf.coverage.mfo_pct.toFixed(1)}% MFO shells. Expect ${h} to test intermediate crossing routes and underneath flats.`
            }
        },
        run_scheme_splits: {
            away_unit: {
                team: a, offense_zone_pct: aProf.run.offense_zone_pct, offense_gap_pct: aProf.run.offense_gap_pct,
                zone_ypc: aProf.run.zone_ypc, gap_ypc: aProf.run.gap_ypc,
                opp_def_zone_ypc_allowed: hProf.run.def_zone_ypc_allowed, opp_def_zone_rank: hProf.run.def_zone_rank,
                opp_def_gap_ypc_allowed: hProf.run.def_gap_ypc_allowed, opp_def_gap_rank: hProf.run.def_gap_rank,
                scheme_advantage: aAdv,
                alignment_diagnostic: `${a} runs ${aProf.run.offense_gap_pct.toFixed(1)}% Gap/Power concepts against ${h}'s #${hProf.run.def_gap_rank} ranked Gap Run Defense.`
            },
            home_unit: {
                team: h, offense_zone_pct: hProf.run.offense_zone_pct, offense_gap_pct: hProf.run.offense_gap_pct,
                zone_ypc: hProf.run.zone_ypc, gap_ypc: hProf.run.gap_ypc,
                opp_def_zone_ypc_allowed: aProf.run.def_zone_ypc_allowed, opp_def_zone_rank: aProf.run.def_zone_rank,
                opp_def_gap_ypc_allowed: aProf.run.def_gap_ypc_allowed, opp_def_gap_rank: aProf.run.def_gap_rank,
                scheme_advantage: hAdv,
                alignment_diagnostic: `${h} executes ${hProf.run.offense_zone_pct.toFixed(1)}% Zone stretch runs facing ${a}'s #${aProf.run.def_zone_rank} ranked Zone defense.`
            }
        },
        positional_target_rates: {
            away_distribution: aProf.targets, home_distribution: hProf.targets,
            away_player_targets: aEnriched, home_player_targets: hEnriched
        },
        efficiency_breakouts: breakouts,
        game_flow_tempo: {
            away_tempo: aProf.tempo, home_tempo: hProf.tempo,
            pace_synthesis: 'BALANCED NFL TEMPO: Standard 63-65 play per side projection with normal situational clock management.',
            volume_override_verdict: `Game-flow and situational usage override flat season averages: ${a} check-down rate (${aProf.tempo.checkdown_pct}%) combined with ${h}'s ${hProf.coverage.mfo_pct.toFixed(0)}% MFO shell sets an elevated target floor for short receivers.`
        }
    };
}

/**
 * Resilient 18-Week Fallback Schedule for GitHub Pages / Static Hosting
 */
function getFallbackSchedule(season = 2026) {
    const s = parseInt(season, 10) || 2026;
    const week1Games = [
        { home_team: 'SEA', home_team_name: 'Seattle Seahawks', away_team: 'NE', away_team_name: 'New England Patriots', week: 1, gameday: '2026-09-10', weekday: 'Thu', gametime: '8:20 PM', spread_line: 4.5, total_line: 41.5, stadium: 'Lumen Field', roof: 'outdoors', temp: 68, wind: 6 },
        { home_team: 'KC', home_team_name: 'Kansas City Chiefs', away_team: 'BAL', away_team_name: 'Baltimore Ravens', week: 1, gameday: '2026-09-13', weekday: 'Sun', gametime: '4:25 PM', spread_line: 3.0, total_line: 47.0, stadium: 'GEHA Field at Arrowhead', roof: 'outdoors', temp: 74, wind: 8 },
        { home_team: 'PHI', home_team_name: 'Philadelphia Eagles', away_team: 'GB', away_team_name: 'Green Bay Packers', week: 1, gameday: '2026-09-11', weekday: 'Fri', gametime: '8:15 PM', spread_line: 2.5, total_line: 48.5, stadium: 'Lincoln Financial Field', roof: 'outdoors', temp: 72, wind: 5 },
        { home_team: 'LA', home_team_name: 'Los Angeles Rams', away_team: 'DET', away_team_name: 'Detroit Lions', week: 1, gameday: '2026-09-13', weekday: 'Sun', gametime: '8:20 PM', spread_line: 3.5, total_line: 51.5, stadium: 'SoFi Stadium', roof: 'dome', temp: 70, wind: 0 },
        { home_team: 'SF', home_team_name: 'San Francisco 49ers', away_team: 'NYJ', away_team_name: 'New York Jets', week: 1, gameday: '2026-09-14', weekday: 'Mon', gametime: '8:15 PM', spread_line: 4.0, total_line: 43.5, stadium: "Levi's Stadium", roof: 'outdoors', temp: 69, wind: 7 },
        { home_team: 'BUF', home_team_name: 'Buffalo Bills', away_team: 'ARI', away_team_name: 'Arizona Cardinals', week: 1, gameday: '2026-09-13', weekday: 'Sun', gametime: '1:00 PM', spread_line: 6.5, total_line: 47.5, stadium: 'Highmark Stadium', roof: 'outdoors', temp: 65, wind: 10 },
        { home_team: 'IND', home_team_name: 'Indianapolis Colts', away_team: 'HOU', away_team_name: 'Houston Texans', week: 1, gameday: '2026-09-13', weekday: 'Sun', gametime: '1:00 PM', spread_line: -2.5, total_line: 48.5, stadium: 'Lucas Oil Stadium', roof: 'retractable', temp: 70, wind: 0 },
        { home_team: 'CLE', home_team_name: 'Cleveland Browns', away_team: 'DAL', away_team_name: 'Dallas Cowboys', week: 1, gameday: '2026-09-13', weekday: 'Sun', gametime: '4:25 PM', spread_line: 2.5, total_line: 41.0, stadium: 'Huntington Bank Field', roof: 'outdoors', temp: 66, wind: 9 },
        { home_team: 'CHI', home_team_name: 'Chicago Bears', away_team: 'TEN', away_team_name: 'Tennessee Titans', week: 1, gameday: '2026-09-13', weekday: 'Sun', gametime: '1:00 PM', spread_line: 3.5, total_line: 44.5, stadium: 'Soldier Field', roof: 'outdoors', temp: 67, wind: 8 },
        { home_team: 'MIA', home_team_name: 'Miami Dolphins', away_team: 'JAX', away_team_name: 'Jacksonville Jaguars', week: 1, gameday: '2026-09-13', weekday: 'Sun', gametime: '1:00 PM', spread_line: 3.5, total_line: 49.0, stadium: 'Hard Rock Stadium', roof: 'outdoors', temp: 85, wind: 11 },
        { home_team: 'NYG', home_team_name: 'New York Giants', away_team: 'MIN', away_team_name: 'Minnesota Vikings', week: 1, gameday: '2026-09-13', weekday: 'Sun', gametime: '1:00 PM', spread_line: -1.5, total_line: 41.5, stadium: 'MetLife Stadium', roof: 'outdoors', temp: 71, wind: 6 },
        { home_team: 'NO', home_team_name: 'New Orleans Saints', away_team: 'CAR', away_team_name: 'Carolina Panthers', week: 1, gameday: '2026-09-13', weekday: 'Sun', gametime: '1:00 PM', spread_line: 4.0, total_line: 41.5, stadium: 'Caesars Superdome', roof: 'dome', temp: 72, wind: 0 },
        { home_team: 'LAC', home_team_name: 'Los Angeles Chargers', away_team: 'LV', away_team_name: 'Las Vegas Raiders', week: 1, gameday: '2026-09-13', weekday: 'Sun', gametime: '4:05 PM', spread_line: 3.0, total_line: 40.5, stadium: 'SoFi Stadium', roof: 'dome', temp: 70, wind: 0 },
        { home_team: 'TB', home_team_name: 'Tampa Bay Buccaneers', away_team: 'WAS', away_team_name: 'Washington Commanders', week: 1, gameday: '2026-09-13', weekday: 'Sun', gametime: '4:25 PM', spread_line: 3.5, total_line: 42.5, stadium: 'Raymond James Stadium', roof: 'outdoors', temp: 86, wind: 7 },
        { home_team: 'ATL', home_team_name: 'Atlanta Falcons', away_team: 'PIT', away_team_name: 'Pittsburgh Steelers', week: 1, gameday: '2026-09-13', weekday: 'Sun', gametime: '1:00 PM', spread_line: 3.5, total_line: 42.0, stadium: 'Mercedes-Benz Stadium', roof: 'retractable', temp: 72, wind: 0 },
        { home_team: 'CIN', home_team_name: 'Cincinnati Bengals', away_team: 'DEN', away_team_name: 'Denver Broncos', week: 1, gameday: '2026-09-13', weekday: 'Sun', gametime: '1:00 PM', spread_line: 8.5, total_line: 41.0, stadium: 'Paycor Stadium', roof: 'outdoors', temp: 70, wind: 6 }
    ];

    const weeks = [];
    weeks.push({ week: 1, date_range_str: 'Sep 10 – Sep 14, 2026', games: week1Games });

    // Generate remaining weeks
    const allTeams = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC', 'LAC', 'LA', 'LV', 'MIA', 'MIN', 'NE', 'NO', 'NYG', 'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS'];
    for (let w = 2; w <= 18; w++) {
        const offset = (w * 3) % allTeams.length;
        const shuffled = [...allTeams.slice(offset), ...allTeams.slice(0, offset)];
        const wGames = [];
        for (let i = 0; i < shuffled.length; i += 2) {
            const hTeam = shuffled[i];
            const aTeam = shuffled[i + 1] || shuffled[0];
            wGames.push({
                home_team: hTeam, home_team_name: NFL_TEAM_NAMES[hTeam] || hTeam,
                away_team: aTeam, away_team_name: NFL_TEAM_NAMES[aTeam] || aTeam,
                week: w, gameday: `2026-10-${Math.min(28, w * 2)}`, weekday: 'Sun', gametime: '1:00 PM',
                spread_line: 2.5, total_line: 44.5, stadium: 'NFL Stadium', roof: 'outdoors', temp: 65, wind: 8
            });
        }
        weeks.push({ week: w, date_range_str: `Week ${w} Slate (${s})`, games: wGames });
    }

    return {
        season: s,
        stats_baseline_note: `Reflecting full regular season baseline (${s})`,
        weeks: weeks
    };
}

/**
 * Primary Actions: Run Model, Rerun Lines, Run MLR
 */
async function triggerRunModel() {
    const mode = state.modelView || 'xgb';
    if (mode === 'mlr') {
        return await triggerRunMLR();
    } else if (mode === 'td') {
        return await triggerRunTD();
    }

    const btn = document.getElementById('btn-run-model');
    const origHTML = btn ? btn.innerHTML : '';
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<i data-lucide="loader-2" class="w-4 h-4 animate-spin"></i><span>Training XGBoost...</span>`;
        if (window.lucide) lucide.createIcons({ root: btn });
    }

    showToast('Executing TimeSeries CV model training...', 'info');

    const selectedDateBeforeRun = state.selectedDate;

    try {
        const res = await api.runModel({
            start_date: state.startDate,
            end_date: state.endDate,
            props_date: selectedDateBeforeRun || '',
            include_props: state.includeProps,
            history_start_year: 2023,
            tune_mode: 'cv_auto',
            model_architecture: 'level_cv'
        });

        showToast('Model training complete & predictions updated!', 'success');
        await checkModelStatus();

        if (res && res.predictions) {
            state.xgbPredictions = res.predictions;
            state.isModelTrained = true;
            if (res.predictions.games && res.predictions.games.length > 1) {
                state.allGames = res.predictions.games;
            } else if (!state.allGames || state.allGames.length === 0) {
                state.allGames = res.predictions?.games || [];
            }
        } else {
            await loadCurrentTab();
        }

        // Restore active selected date and re-render
        if (selectedDateBeforeRun) {
            state.selectedDate = selectedDateBeforeRun;
        }
        renderPredictionsDateChips();
        renderPredictionsGames();
        filterAndRenderProps();
    } catch (err) {
        showToast(err.message || 'Model training failed', 'error');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = origHTML;
            if (window.lucide) lucide.createIcons({ root: btn });
        }
    }
}

async function triggerRerunLines() {
    const btn = document.getElementById('btn-rerun-lines');
    const origHTML = btn ? btn.innerHTML : '';
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<i data-lucide="loader-2" class="w-4 h-4 animate-spin"></i><span>Refreshing Lines...</span>`;
        if (window.lucide) lucide.createIcons({ root: btn });
    }

    const selectedDateBeforeRun = state.selectedDate;

    try {
        await api.rerunLines({
            start_date: state.startDate,
            end_date: state.endDate,
            props_date: selectedDateBeforeRun || '',
            include_props: state.includeProps,
            fetch_remote_odds: true
        });

        showToast('Sportsbook lines refreshed and edges rescored!', 'success');
        await loadCurrentTab();
        if (selectedDateBeforeRun) {
            state.selectedDate = selectedDateBeforeRun;
            renderPredictionsDateChips();
            renderPredictionsGames();
            filterAndRenderProps();
        }
    } catch (err) {
        showToast(err.message || 'Lines rerun failed', 'error');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = origHTML;
            if (window.lucide) lucide.createIcons({ root: btn });
        }
    }
}

async function triggerRunMLR() {
    const btn = document.getElementById('btn-run-model');
    const origHTML = btn ? btn.innerHTML : '';
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<i data-lucide="loader-2" class="w-4 h-4 animate-spin"></i><span>Running MLR...</span>`;
        if (window.lucide) lucide.createIcons({ root: btn });
    }

    showToast('Generating 32-team MLR depth-chart starter predictions...', 'info');

    const selectedDateBeforeRun = state.selectedDate;

    try {
        const res = await api.runMLR(state.season);
        showToast(`MLR complete! Evaluated ${res.total_starters_evaluated} depth-chart starters.`, 'success');
        if (state.modelView === 'mlr') {
            await loadPredictionsTab();
            if (selectedDateBeforeRun) {
                state.selectedDate = selectedDateBeforeRun;
                renderPredictionsDateChips();
                renderPredictionsGames();
                filterAndRenderProps();
            }
        }
    } catch (err) {
        showToast(err.message || 'MLR generation failed', 'error');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = origHTML;
            if (window.lucide) lucide.createIcons({ root: btn });
        }
    }
}

async function triggerRunTD() {
    const btn = document.getElementById('btn-run-model');
    const origHTML = btn ? btn.innerHTML : '';
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<i data-lucide="loader-2" class="w-4 h-4 animate-spin"></i><span>Training TD...</span>`;
        if (window.lucide) lucide.createIcons({ root: btn });
    }

    showToast('Training & calibrating Anytime TD Scorer ensemble across positions...', 'info');

    const selectedDateBeforeRun = state.selectedDate;

    try {
        const res = await api.runTDModel(state.season);
        showToast('Touchdown models trained and calibrated successfully!', 'success');
        state.isModelTrained = true;  // mark trained so table renders immediately
        if (state.modelView === 'td') {
            await loadPredictionsTab();
            if (selectedDateBeforeRun) {
                state.selectedDate = selectedDateBeforeRun;
                renderPredictionsDateChips();
                renderPredictionsGames();
                filterAndRenderProps();
            }
        }
    } catch (err) {
        showToast(err.message || 'TD model training failed', 'error');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = origHTML;
            if (window.lucide) lucide.createIcons({ root: btn });
        }
    }
}

function triggerExportXLSX() {
    const sel = document.getElementById('export-xlsx-select');
    const val = sel ? sel.value : 'cv';
    let url = '/api/download-excel';
    if (val === 'csv') {
        url = '/api/download-csv';
    } else if (val === 'mlr') {
        url = '/api/download-mlr-excel';
    } else if (val === 'td') {
        url = '/api/download-td-excel';
    }
    window.open(url, '_blank');
}

function onDateRangeChange() {
    // Deprecated: dashboard now runs entirely off the single master date selector
}

async function onDateSelectorChange(date) {
    if (!date) return;
    state.selectedDate = date;
    const year = parseInt(date.substring(0, 4), 10);
    state.season = (year >= 2022) ? year : 2026;
    state.scheduleSeason = state.season;
    if (year >= 2022 && year <= 2025) {
        state.criteriaSeason = year;
    }

    const datePicker = document.getElementById('predictions-date-picker');
    if (datePicker && datePicker.value !== date) {
        datePicker.value = date;
    }

    // Translate date to week using full season schedule mapping
    await ensureSeasonScheduleLoaded(state.season);
    if (state.dateToWeekMap && state.dateToWeekMap[date]) {
        state.week = String(state.dateToWeekMap[date]);
        state.criteriaWeek = parseInt(state.week, 10);
        const cwSelect = document.getElementById('criteria-week-select');
        if (cwSelect) cwSelect.value = String(state.criteriaWeek);
    }

    if (state.activeTab === 'criteria') {
        await loadCriteriaTab();
    } else {
        await loadPredictionsTab();
    }
}

async function onPredictionsDateChange(date) {
    await onDateSelectorChange(date);
}

async function clearPredictionsDateFilter() {
    await onDateSelectorChange('2026-10-04');
}

/**
 * Tab 5: Criteria Matches Viewer
 */
async function loadCriteriaTab() {
    const container = document.getElementById('criteria-matches-container');
    const season = state.criteriaSeason || 2025;
    const week = state.criteriaWeek || 8;
    const criterion = state.criteriaSelected || 'weak_rush_def';

    // Update Slate Display
    const slateDisplay = document.getElementById('criteria-season-week-display');
    if (slateDisplay) {
        slateDisplay.textContent = `Season ${season}, Week ${week}`;
    }

    // Sync Dropdown UI
    const critSelect = document.getElementById('criteria-select');
    if (critSelect && critSelect.value !== criterion) {
        critSelect.value = criterion;
    }
    const weekSelect = document.getElementById('criteria-week-select');
    if (weekSelect && String(weekSelect.value) !== String(week)) {
        weekSelect.value = String(week);
    }

    // Show loading spinner
    if (container) {
        container.innerHTML = `
            <div class="p-12 text-center text-zinc-400 flex flex-col items-center justify-center gap-3">
                <i data-lucide="loader" class="w-6 h-6 animate-spin text-zinc-300"></i>
                <span class="text-xs font-medium">Scanning empirical criteria matches for Week ${week}...</span>
            </div>
        `;
        if (window.lucide) lucide.createIcons({ root: container });
    }

    try {
        state.criteriaLoading = true;
        const res = await api.getCriteriaMatches({
            season: season,
            week: week,
            criterion: criterion
        });
        state.criteriaData = res;
        state.criteriaLoading = false;

        // Update Metadata Card
        const idBadge = document.getElementById('criteria-id-badge');
        if (idBadge) idBadge.textContent = res.criterion_id || 'RULE';

        const tierBadge = document.getElementById('criteria-tier-badge');
        if (tierBadge) {
            const tier = res.confidence_tier || 'Suggestive';
            tierBadge.textContent = tier;
            tierBadge.className = 'badge text-[10px] font-bold uppercase tracking-wider ';
            if (tier === 'Confirmed') {
                tierBadge.className += 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40';
            } else if (tier === 'Suggestive') {
                tierBadge.className += 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/40';
            } else if (tier === 'Directional') {
                tierBadge.className += 'bg-orange-500/20 text-orange-300 border border-orange-500/40';
            } else {
                tierBadge.className += 'bg-zinc-800 text-zinc-300 border border-zinc-700';
            }
        }

        const titleEl = document.getElementById('criteria-title');
        if (titleEl) titleEl.textContent = res.criterion_name || res.criterion;

        const effectEl = document.getElementById('criteria-effect-badge');
        if (effectEl) effectEl.textContent = res.documented_effect || '';

        const descEl = document.getElementById('criteria-desc');
        if (descEl) descEl.textContent = res.criterion_description || '';

        const srcEl = document.getElementById('criteria-source');
        if (srcEl) srcEl.textContent = `Source: ${res.source || 'OPPONENT_QUALITY_INSIGHTS.md'}`;

        const countBadge = document.getElementById('criteria-matches-count-badge');
        if (countBadge) {
            countBadge.textContent = `${res.n_matches || 0} Matches`;
        }

        // Render zero matches or table
        if (!res.matches || res.matches.length === 0) {
            container.innerHTML = `
                <div class="p-12 text-center text-zinc-400">
                    <i data-lucide="info" class="w-8 h-8 mx-auto text-zinc-600 mb-2"></i>
                    <p class="text-sm font-medium">No games match this criterion for Week ${week}.</p>
                </div>
            `;
            if (window.lucide) lucide.createIcons({ root: container });
            return;
        }

        // Feature column headers from first match
        const firstMatch = res.matches[0];
        const displayCols = firstMatch.display_columns || [];
        const contextObj = firstMatch.context || firstMatch.triggering_features || {};
        const featureKeys = Object.keys(contextObj);

        let headerColsHtml = '';
        if (displayCols.length > 0) {
            displayCols.forEach(colName => {
                headerColsHtml += `<th class="text-right">${colName}</th>`;
            });
        } else {
            featureKeys.forEach(k => {
                const label = k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
                headerColsHtml += `<th class="text-right">${label}</th>`;
            });
        }

        let rowsHtml = '';
        res.matches.forEach(m => {
            let featureTdsHtml = '';
            if (m.displayed_metrics && displayCols.length > 0) {
                displayCols.forEach(colName => {
                    const dispVal = m.displayed_metrics[colName] !== undefined ? m.displayed_metrics[colName] : '—';
                    featureTdsHtml += `<td class="text-right font-mono text-zinc-200">${dispVal}</td>`;
                });
            } else {
                featureKeys.forEach(k => {
                    const val = (m.context || m.triggering_features || {})[k];
                    let displayVal = '—';
                    if (val !== null && val !== undefined) {
                        if (typeof val === 'number') {
                            if (k.includes('share')) {
                                displayVal = `${Math.round(val * 100)}%`;
                            } else if (k.includes('pctile')) {
                                displayVal = `Bottom ${Math.round(val * 100)}%`;
                            } else {
                                displayVal = Number.isInteger(val) ? val.toString() : val.toFixed(2);
                            }
                        } else if (typeof val === 'boolean') {
                            displayVal = val ? 'Yes' : 'No';
                        } else {
                            displayVal = String(val);
                        }
                    }
                    featureTdsHtml += `<td class="text-right font-mono text-zinc-200">${displayVal}</td>`;
                });
            }

            rowsHtml += `
                <tr class="hover:bg-zinc-800/40 transition">
                    <td class="font-medium text-white">${m.player_or_team || '—'}</td>
                    <td><span class="font-bold text-zinc-200">${m.team || '—'}</span></td>
                    <td class="text-zinc-400">${m.opponent || '—'}</td>
                    ${featureTdsHtml}
                </tr>
            `;
        });

        container.innerHTML = `
            <table class="sports-table w-full">
                <thead>
                    <tr>
                        <th class="text-left">Player/Team</th>
                        <th class="text-left">Team</th>
                        <th class="text-left">Opponent</th>
                        ${headerColsHtml}
                    </tr>
                </thead>
                <tbody class="divide-y divide-zinc-800/50">
                    ${rowsHtml}
                </tbody>
            </table>
        `;
        if (window.lucide) lucide.createIcons({ root: container });

    } catch (err) {
        state.criteriaLoading = false;
        console.error('Error fetching criteria matches:', err);
        container.innerHTML = `
            <div class="p-8 text-center text-rose-400 bg-rose-500/10 border border-rose-500/20 rounded-xl">
                <i data-lucide="alert-circle" class="w-6 h-6 mx-auto mb-2 text-rose-400"></i>
                <p class="text-sm font-semibold">Failed to load criteria matches</p>
                <p class="text-xs text-rose-300/80 mt-1">${err.message || 'Unknown error'}</p>
            </div>
        `;
        if (window.lucide) lucide.createIcons({ root: container });
        showToast(err.message || 'Error loading criteria matches', 'error');
    }
}

function onCriterionChange(criterionKey) {
    state.criteriaSelected = criterionKey;
    loadCriteriaTab();
}

function onCriteriaWeekChange(weekVal) {
    state.criteriaWeek = parseInt(weekVal, 10);
    state.scheduleWeek = String(state.criteriaWeek);
    const schedSelect = document.getElementById('schedule-week-select');
    if (schedSelect) schedSelect.value = String(state.criteriaWeek);
    loadCriteriaTab();
}

/**
 * Tab 6: Game Highlights (Criteria-Matched Players by Date)
 * ========================================================
 */

function formatSpreadDisplay(spreadLine, awayTeam, homeTeam) {
    if (spreadLine === null || spreadLine === undefined) return '—';
    const sp = parseFloat(spreadLine);
    if (isNaN(sp)) return '—';
    if (sp === 0) return 'PK';
    if (sp > 0) {
        // In nflverse, spread_line > 0 means home team is favored
        const val = (Math.round(sp * 10) / 10).toString();
        return `${homeTeam} -${val}`;
    } else {
        // spread_line < 0 means away team is favored
        const val = (Math.round(Math.abs(sp) * 10) / 10).toString();
        return `${awayTeam} -${val}`;
    }
}

function formatKickoffDisplay(gameday, gametime) {
    if (!gameday) return '';
    const dayStr = formatDateDisplay(gameday);
    if (!gametime) return dayStr;
    const parts = gametime.split(':');
    if (parts.length >= 2) {
        let hour = parseInt(parts[0], 10);
        const mins = parts[1];
        const ampm = hour >= 12 ? 'PM' : 'AM';
        hour = hour % 12;
        if (hour === 0) hour = 12;
        return `${dayStr} • ${hour}:${mins} ${ampm} ET`;
    }
    return `${dayStr} • ${gametime} ET`;
}

function setHighlightsScope(scope) {
    if (state.highlightsScope === scope) return;
    state.highlightsScope = scope;
    const btnWeek = document.getElementById('highlights-scope-week');
    const btnDate = document.getElementById('highlights-scope-date');
    if (btnWeek && btnDate) {
        if (scope === 'week') {
            btnWeek.className = 'px-2.5 py-1 rounded text-xs font-medium transition bg-zinc-800 text-white shadow-sm';
            btnDate.className = 'px-2.5 py-1 rounded text-xs font-medium text-zinc-400 hover:text-white transition';
        } else {
            btnDate.className = 'px-2.5 py-1 rounded text-xs font-medium transition bg-zinc-800 text-white shadow-sm';
            btnWeek.className = 'px-2.5 py-1 rounded text-xs font-medium text-zinc-400 hover:text-white transition';
        }
    }
    loadHighlightsTab();
}

async function loadHighlightsTab() {
    const container = document.getElementById('highlights-games-container');
    const seasonWeekDisplay = document.getElementById('highlights-season-week-display');
    const countBadge = document.getElementById('highlights-games-count-badge');
    if (!container) return;

    state.highlightsLoading = true;
    container.innerHTML = `
        <div class="p-12 text-center text-zinc-400">
            <i data-lucide="loader-2" class="w-8 h-8 animate-spin mx-auto mb-3 text-emerald-400"></i>
            <p class="text-sm font-semibold">Scanning slate for criteria matches...</p>
            <p class="text-xs text-zinc-500 mt-1">Evaluating all 5 documented criteria for date ${state.highlightsDate} (${state.highlightsScope === 'week' ? 'Full Week Slate' : 'Selected Date Only'})</p>
        </div>
    `;
    if (window.lucide) lucide.createIcons({ root: container });

    try {
        const data = await api.getGameHighlights({ date: state.highlightsDate, scope: state.highlightsScope });
        state.highlightsData = data;
        state.highlightsLoading = false;

        if (seasonWeekDisplay) {
            if (data.season && data.week) {
                seasonWeekDisplay.textContent = state.highlightsScope === 'week'
                    ? `Season ${data.season}, Week ${data.week} Slate`
                    : `Season ${data.season}, Week ${data.week} (${data.date})`;
            } else {
                seasonWeekDisplay.textContent = 'Season —, Week —';
            }
        }
        if (countBadge) {
            const count = data.n_games_with_matches || 0;
            const scopeLabel = state.highlightsScope === 'week' ? 'on Slate' : 'on Date';
            countBadge.textContent = `${count} ${count === 1 ? 'Game' : 'Games'} ${scopeLabel}`;
            countBadge.className = count > 0 ? 'badge badge-high text-[10px]' : 'badge badge-low text-[10px]';
        }

        renderHighlightsCards(data, container);

    } catch (err) {
        state.highlightsLoading = false;
        console.error('Error fetching game highlights:', err);
        container.innerHTML = `
            <div class="p-8 text-center text-rose-400 bg-rose-500/10 border border-rose-500/20 rounded-xl">
                <i data-lucide="alert-circle" class="w-6 h-6 mx-auto mb-2 text-rose-400"></i>
                <p class="text-sm font-semibold">Failed to load game highlights</p>
                <p class="text-xs text-rose-300/80 mt-1">${err.message || 'Unknown error'}</p>
            </div>
        `;
        if (window.lucide) lucide.createIcons({ root: container });
        showToast(err.message || 'Error loading highlights', 'error');
    }
}

function onHighlightsDateChange(dateVal) {
    if (!dateVal) return;
    state.highlightsDate = dateVal;
    loadHighlightsTab();
}

function refreshHighlights() {
    loadHighlightsTab();
}

function renderHighlightsCards(data, container) {
    const games = data.games || [];
    if (games.length === 0) {
        const scopeDesc = (data.scope || state.highlightsScope) === 'date'
            ? `on ${data.date || state.highlightsDate}`
            : `for Week ${data.week || '—'} slate`;
        container.innerHTML = `
            <div class="p-12 text-center text-zinc-400 bg-zinc-900/30 border border-zinc-800/80 rounded-xl">
                <i data-lucide="info" class="w-8 h-8 mx-auto mb-3 text-zinc-500"></i>
                <p class="text-sm font-semibold text-zinc-300">No structural criteria matched ${scopeDesc}</p>
                <p class="text-xs text-zinc-500 mt-1">Date: ${data.date || state.highlightsDate || '—'} (Season ${data.season || '—'}, Week ${data.week || '—'})</p>
                ${(data.scope || state.highlightsScope) === 'date' ? `<button onclick="setHighlightsScope('week')" class="mt-4 btn-secondary text-xs inline-flex items-center gap-1.5"><i data-lucide="calendar" class="w-3.5 h-3.5"></i><span>View Full Week Slate</span></button>` : ''}
            </div>
        `;
        if (window.lucide) lucide.createIcons({ root: container });
        return;
    }

    let cardsHtml = '';
    games.forEach(game => {
        const awayTeam = game.away_team || '—';
        const homeTeam = game.home_team || '—';
        
        // Spread formatting: always show TEAM -X or PK, never a bare number
        const spreadStr = game.formatted_spread || formatSpreadDisplay(game.spread_line, awayTeam, homeTeam);
        const totalStr = (game.total_line !== null && game.total_line !== undefined)
            ? `${game.total_line}`
            : '—';
        
        const kickoffDisplay = formatKickoffDisplay(game.gameday, game.gametime);

        // Render both sides
        const sides = game.sides || [];
        let sidesHtml = '';

        sides.forEach(side => {
            const offTeam = side.offense;
            const defTeam = side.defense;
            const matches = side.matches || [];

            let matchesContent = '';
            if (matches.length === 0) {
                matchesContent = `
                    <div class="p-3 rounded-lg bg-zinc-900/40 border border-zinc-800/50 text-xs text-zinc-500 italic">
                        No structural criteria matched for this side.
                    </div>
                `;
            } else {
                // Separate player targets from team signals
                const playerMatches = matches.filter(m => (m.target_type !== 'team' && m.criterion_id !== 'C21'));
                const teamMatches = matches.filter(m => (m.target_type === 'team' || m.criterion_id === 'C21'));

                let sideSectionsHtml = '';

                // Section 1: Player Targets (RB Rushing)
                if (playerMatches.length > 0) {
                    // Compute criteria convergence per player
                    const playerMatchCount = {};
                    const playerCriteriaMap = {};
                    playerMatches.forEach(m => {
                        (m.target_players || []).forEach(p => {
                            const key = p.player_name;
                            playerMatchCount[key] = (playerMatchCount[key] || 0) + 1;
                            if (!playerCriteriaMap[key]) playerCriteriaMap[key] = [];
                            if (!playerCriteriaMap[key].includes(m.criterion_id)) {
                                playerCriteriaMap[key].push(m.criterion_id);
                            }
                        });
                    });

                    const highConvictionPlayers = Object.keys(playerMatchCount).filter(k => playerMatchCount[k] >= 2);
                    let convictionBannerHtml = '';
                    if (highConvictionPlayers.length > 0) {
                        convictionBannerHtml = `
                            <div class="p-2 rounded-lg bg-amber-500/10 border border-amber-500/30 flex items-center justify-between text-xs text-amber-300">
                                <span class="flex items-center gap-1.5 font-bold">
                                    <i data-lucide="flame" class="w-3.5 h-3.5 text-amber-400"></i>
                                    <span>High Conviction: ${highConvictionPlayers.join(', ')}</span>
                                </span>
                                <span class="text-[10px] text-amber-300/80 font-mono">${highConvictionPlayers.map(k => `${k} (${playerCriteriaMap[k].join(' + ')})`).join(', ')}</span>
                            </div>
                        `;
                    }

                    let playerBlocks = '';
                    playerMatches.forEach(m => {
                        let tierBadgeClass = 'badge-low';
                        if (m.confidence_tier === 'Confirmed') tierBadgeClass = 'badge-over';
                        else if (m.confidence_tier === 'Suggestive') tierBadgeClass = 'badge-med';

                        let playersListHtml = '';
                        if (m.target_players && m.target_players.length > 0) {
                            playersListHtml = `
                                <div class="mt-2 space-y-1.5 border-t border-zinc-800/60 pt-2">
                                    <div class="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">Target Player(s):</div>
                                    <ul class="space-y-1 text-xs">
                                        ${m.target_players.map(p => {
                                            let trigBadge = '';
                                            if (p.triggering_value !== undefined && p.triggering_value !== null) {
                                                trigBadge = `<span class="text-[10px] font-mono text-zinc-400">(${Math.round(p.triggering_value * 100)}% carry share)</span>`;
                                            }
                                            let pConvictionTag = '';
                                            if (playerMatchCount[p.player_name] >= 2) {
                                                pConvictionTag = `<span class="badge text-[9px] px-1.5 py-0.2 bg-amber-500/20 text-amber-300 border border-amber-500/40 font-bold">🔥 2x Match</span>`;
                                            }
                                            return `
                                                <li class="flex items-start gap-2 bg-zinc-900/60 rounded px-2.5 py-1.5 border border-zinc-800/40">
                                                    <span class="inline-block w-1.5 h-1.5 rounded-full bg-emerald-400 mt-1.5 shrink-0"></span>
                                                    <div class="flex-1 min-w-0">
                                                        <div class="flex items-center gap-2">
                                                            <span class="font-bold text-white">${p.player_name}</span>
                                                            <span class="badge badge-pos text-[9px] px-1 py-0.2">${p.position}</span>
                                                            ${trigBadge}
                                                            ${pConvictionTag}
                                                        </div>
                                                        <div class="text-[11px] text-zinc-400 mt-0.5 leading-snug">${p.reason}</div>
                                                    </div>
                                                </li>
                                            `;
                                        }).join('')}
                                    </ul>
                                </div>
                            `;
                        }

                        playerBlocks += `
                            <div class="p-3 rounded-lg bg-zinc-900/80 border border-zinc-800 space-y-1.5">
                                <div class="flex flex-wrap items-center justify-between gap-1.5">
                                    <div class="flex items-center gap-2">
                                        <span class="font-mono font-bold text-xs px-1.5 py-0.5 bg-zinc-800 text-emerald-400 rounded border border-zinc-700">${m.criterion_id}</span>
                                        <span class="text-xs font-semibold text-zinc-200">${m.criterion_name}</span>
                                    </div>
                                    <div class="flex items-center gap-2">
                                        <span class="badge ${tierBadgeClass} text-[9px]">${m.confidence_tier}</span>
                                        <span class="text-[11px] font-mono text-emerald-400 font-semibold">${m.documented_effect}</span>
                                    </div>
                                </div>
                                ${playersListHtml}
                            </div>
                        `;
                    });

                    sideSectionsHtml += `
                        <div class="space-y-2">
                            <div class="text-[10px] font-bold uppercase tracking-wider text-emerald-400 flex items-center gap-1">
                                <i data-lucide="user-check" class="w-3 h-3"></i>
                                <span>Player Targets (RB Rushing)</span>
                            </div>
                            ${convictionBannerHtml}
                            ${playerBlocks}
                        </div>
                    `;
                }

                // Section 2: Team Signals (Game Pace & Team Total Points)
                if (teamMatches.length > 0) {
                    let teamBlocks = '';
                    teamMatches.forEach(m => {
                        let tierBadgeClass = 'badge-low';
                        if (m.confidence_tier === 'Confirmed') tierBadgeClass = 'badge-over';
                        else if (m.confidence_tier === 'Suggestive') tierBadgeClass = 'badge-med';

                        let teamDetailsHtml = '';
                        if (m.target_players && m.target_players.length > 0) {
                            teamDetailsHtml = `
                                <div class="mt-2 space-y-1 text-xs border-t border-zinc-800/60 pt-2">
                                    ${m.target_players.map(p => `
                                        <div class="flex items-start gap-2 bg-zinc-900/60 rounded px-2.5 py-1.5 border border-zinc-800/40">
                                            <span class="inline-block w-1.5 h-1.5 rounded-full bg-cyan-400 mt-1.5 shrink-0"></span>
                                            <div class="flex-1 min-w-0">
                                                <div class="flex items-center gap-2">
                                                    <span class="font-bold text-cyan-300">Target Market: Team Total Points</span>
                                                    <span class="badge badge-med text-[9px] px-1 py-0.2">${p.player_name}</span>
                                                    <span class="text-[10px] font-mono text-zinc-400">(Pace: ${p.triggering_value} plays/gm)</span>
                                                </div>
                                                <div class="text-[11px] text-zinc-400 mt-0.5 leading-snug">${p.reason}</div>
                                            </div>
                                        </div>
                                    `).join('')}
                                </div>
                            `;
                        }

                        teamBlocks += `
                            <div class="p-3 rounded-lg bg-zinc-900/80 border border-cyan-900/30 space-y-1.5">
                                <div class="flex flex-wrap items-center justify-between gap-1.5">
                                    <div class="flex items-center gap-2">
                                        <span class="font-mono font-bold text-xs px-1.5 py-0.5 bg-zinc-800 text-cyan-400 rounded border border-cyan-800/60">${m.criterion_id}</span>
                                        <span class="text-xs font-semibold text-zinc-200">${m.criterion_name}</span>
                                    </div>
                                    <div class="flex items-center gap-2">
                                        <span class="badge ${tierBadgeClass} text-[9px]">${m.confidence_tier}</span>
                                        <span class="text-[11px] font-mono text-cyan-400 font-semibold">${m.documented_effect}</span>
                                    </div>
                                </div>
                                ${teamDetailsHtml}
                            </div>
                        `;
                    });

                    sideSectionsHtml += `
                        <div class="space-y-2 pt-1 border-t border-zinc-800/50">
                            <div class="text-[10px] font-bold uppercase tracking-wider text-cyan-400 flex items-center gap-1">
                                <i data-lucide="activity" class="w-3 h-3"></i>
                                <span>Team Signals (Team Total Points / Pace)</span>
                            </div>
                            ${teamBlocks}
                        </div>
                    `;
                }

                matchesContent = `<div class="space-y-3">${sideSectionsHtml}</div>`;
            }

            sidesHtml += `
                <div class="flex-1 min-w-[280px] p-3.5 bg-zinc-900/30 rounded-xl border border-zinc-800/80 space-y-2.5">
                    <div class="flex items-center justify-between border-b border-zinc-800 pb-2">
                        <div class="text-xs font-bold uppercase tracking-wider text-zinc-300 flex items-center gap-1.5">
                            <span class="text-emerald-400 font-semibold">${offTeam}</span> Offense
                            <span class="text-zinc-500 font-normal">vs</span>
                            <span class="text-rose-400 font-semibold">${defTeam}</span> Defense
                        </div>
                        <span class="text-[10px] font-mono text-zinc-500">${matches.length} matched</span>
                    </div>
                    ${matchesContent}
                </div>
            `;
        });

        cardsHtml += `
            <div class="glass-panel p-5 rounded-xl border border-zinc-800 space-y-4 hover:border-zinc-700/80 transition">
                <!-- Card Header -->
                <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-zinc-800/80 pb-3">
                    <div class="flex items-center gap-3">
                        <h3 class="text-lg font-bold font-heading text-white tracking-wide">${awayTeam} <span class="text-zinc-500 font-normal">@</span> ${homeTeam}</h3>
                        ${kickoffDisplay ? `<span class="text-xs text-zinc-400 font-mono bg-zinc-900 px-2.5 py-1 rounded border border-zinc-800 flex items-center gap-1.5"><i data-lucide="clock" class="w-3 h-3 text-zinc-400"></i>${kickoffDisplay}</span>` : ''}
                    </div>
                    <div class="flex items-center gap-2 shrink-0 text-xs font-mono">
                        <span class="px-2.5 py-1 bg-zinc-900 text-zinc-300 rounded border border-zinc-800">Spread: <strong class="text-emerald-400 font-bold">${spreadStr}</strong></span>
                        <span class="px-2.5 py-1 bg-zinc-900 text-zinc-300 rounded border border-zinc-800">Total: <strong class="text-white font-bold">${totalStr}</strong></span>
                    </div>
                </div>

                <!-- Two Sides Matchup Grid -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    ${sidesHtml}
                </div>
            </div>
        `;
    });

    container.innerHTML = cardsHtml;
    if (window.lucide) lucide.createIcons({ root: container });
}

function reloadAllData() {
    showToast('Refreshing dashboard datasets...', 'info');
    loadCurrentTab();
}

// Global UI Action Bindings
window.switchTab = switchTab;
window.setModelView = setModelView;
window.setPosFilter = setPosFilter;
window.setCallFilter = setCallFilter;
window.setConfFilter = setConfFilter;
window.sortFloorStreaks = sortFloorStreaks;
window.setFloorPosFilter = setFloorPosFilter;
window.setFloorStreakFilter = setFloorStreakFilter;
window.setFloorMinGamesFilter = setFloorMinGamesFilter;
window.onFloorSearch = onFloorSearch;
window.loadFloorStreakTab = loadFloorStreakTab;
window.loadH2HTab = loadH2HTab;
window.triggerRunModel = triggerRunModel;
window.onRedoPropsChange = onRedoPropsChange;
window.triggerRunMLR = triggerRunMLR;
window.triggerRunTD = triggerRunTD;
window.triggerExportXLSX = triggerExportXLSX;
window.onDateRangeChange = onDateRangeChange;
window.onDateSelectorChange = onDateSelectorChange;
window.onPredictionsDateChange = onPredictionsDateChange;
window.clearPredictionsDateFilter = clearPredictionsDateFilter;
window.renderPredictionsDateChips = renderPredictionsDateChips;
window.reloadAllData = reloadAllData;
window.onScheduleWeekChange = onScheduleWeekChange;
window.onScheduleSeasonChange = onScheduleSeasonChange;
window.onFloorDateChange = onFloorDateChange;
window.clearFloorDateFilter = clearFloorDateFilter;
window.renderFloorDateChips = renderFloorDateChips;
window.setFloorHistoryPreset = setFloorHistoryPreset;
window.onFloorDateRangeInputChange = onFloorDateRangeInputChange;
window.clearFloorHistoryRange = clearFloorHistoryRange;
window.setScheduleDatePreset = setScheduleDatePreset;
window.onScheduleDateInputChange = onScheduleDateInputChange;
window.clearScheduleDateRange = clearScheduleDateRange;
window.loadCriteriaTab = loadCriteriaTab;
window.onCriterionChange = onCriterionChange;
window.onCriteriaWeekChange = onCriteriaWeekChange;
window.loadHighlightsTab = loadHighlightsTab;
window.onHighlightsDateChange = onHighlightsDateChange;
window.refreshHighlights = refreshHighlights;
window.setHighlightsScope = setHighlightsScope;
window.formatSpreadDisplay = formatSpreadDisplay;
window.formatKickoffDisplay = formatKickoffDisplay;
window.loadInsightsTab = loadInsightsTab;
window.inspectMatchupInsights = inspectMatchupInsights;
window.onInsightsMatchupChange = onInsightsMatchupChange;
window.onInsightsWeekChange = onInsightsWeekChange;
window.onInsightsSeasonChange = onInsightsSeasonChange;


