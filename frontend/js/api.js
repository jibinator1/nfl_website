/**
 * NFL Prediction & Analytics Hub - API Service Layer
 * ==================================================
 * Clean, type-safe API client with unified query serialization and toast alerts.
 */

function toQuery(params = {}) {
    const q = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) {
        if (v !== undefined && v !== null && v !== '') q.set(k, v);
    }
    const s = q.toString();
    return s ? `?${s}` : '';
}

async function request(endpoint, options = {}) {
    try {
        const response = await fetch(endpoint, {
            headers: { 'Content-Type': 'application/json', ...options.headers },
            ...options
        });
        if (!response.ok) {
            let errorMsg = `Server error (${response.status})`;
            try {
                const errData = await response.json();
                if (errData?.detail) {
                    errorMsg = typeof errData.detail === 'string' ? errData.detail : JSON.stringify(errData.detail);
                }
            } catch (_) {}
            throw new Error(errorMsg);
        }
        return await response.json();
    } catch (err) {
        console.error(`API Error on ${endpoint}:`, err);
        throw err;
    }
}

/**
 * Toast Notification System
 */
function showToast(message, type = 'info', duration = 3500) {
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    const iconName = type === 'success' ? 'check-circle-2' : type === 'error' ? 'alert-triangle' : 'info';
    toast.innerHTML = `<i data-lucide="${iconName}" class="w-4 h-4 shrink-0"></i><span>${message}</span>`;
    container.appendChild(toast);
    if (window.lucide) lucide.createIcons({ root: toast });

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(30px)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

const api = {
    getModelStatus: () => request('/api/model-status'),
    runModel: (payload) => request('/api/run-model', { method: 'POST', body: JSON.stringify(payload) }),
    rerunLines: (payload) => request('/api/rerun-lines', { method: 'POST', body: JSON.stringify(payload) }),
    getPredictions: (params) => request(`/api/predictions${toQuery(params)}`),
    getMLRPredictions: (params) => request(`/api/mlr-predictions${toQuery(params)}`),
    runMLR: (season = 2025) => request('/api/run-mlr', { method: 'POST', body: JSON.stringify({ season }) }),
    getTDPredictions: (params) => request(`/api/td-predictions${toQuery(params)}`),
    runTDModel: (season = 2025) => request('/api/run-td-model', { method: 'POST', body: JSON.stringify({ season }) }),
    getTeamStats: (params) => request(`/api/team-stats${toQuery(params)}`),
    getTeamHighlights: (params) => request(`/api/team-highlights${toQuery(params)}`),
    getMatchupDeepDive: (homeTeam, awayTeam, params = {}) => request(`/api/matchup-deepdive${toQuery({ home_team: homeTeam, away_team: awayTeam, ...params })}`),
    getSchedule: (params) => request(`/api/full-schedule${toQuery(params)}`),
    getScheduleWeeks: (season = 2025) => request(`/api/schedule-weeks?season=${season}`),
    getFloorStreak: (params = {}) => request(`/api/floor-streak${toQuery(params)}`),
    getCriteriaMatches: (params = {}) => request(`/api/criteria-matches${toQuery(params)}`),
    getGameHighlights: (params = {}) => request(`/api/game-highlights${toQuery(params)}`),
    getSchemeInsights: (homeTeam, awayTeam, params = {}) => request(`/api/scheme-insights${toQuery({ home_team: homeTeam, away_team: awayTeam, ...params })}`),
};

window.api = api;
window.showToast = showToast;
