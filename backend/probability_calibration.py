"""
Post-hoc recalibration of model win probabilities against empirical outcomes.
Fit on historical backtest data, applied at inference time.
"""
import json
import os
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.isotonic import IsotonicRegression

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CALIB_PATHS = [
    os.path.join(ROOT_DIR, "data", "probability_calibration.json"),
    os.path.join(ROOT_DIR, "api", "data", "probability_calibration.json")
]
_CALIB_PATH = _CALIB_PATHS[0]
_CALIBRATORS: Dict[str, IsotonicRegression] = {}  # stat_focus -> IsotonicRegression


def fit_calibration(records_by_stat: Dict[str, List[Tuple[float, int]]]) -> Dict[str, IsotonicRegression]:
    """
    records_by_stat: dict stat_focus -> list of (p_model, outcome) tuples
    outcome: 1 if prop hit, 0 otherwise
    """
    global _CALIBRATORS
    fitted = {}
    for stat, records in records_by_stat.items():
        if len(records) < 50:
            continue  # not enough data to fit
        X = np.array([r[0] for r in records], dtype=float)
        y = np.array([r[1] for r in records], dtype=float)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.40, y_max=0.68)
        iso.fit(X, y)
        fitted[stat] = iso
        _CALIBRATORS[stat] = iso

        knots_count = len(getattr(iso, 'X_thresholds_', []))
        y_min_val = float(iso.y_thresholds_.min()) if knots_count > 0 else 0.40
        y_max_val = float(iso.y_thresholds_.max()) if knots_count > 0 else 0.68
        print(f"[calib] {stat}: fitted on {len(records)} records, "
              f"{knots_count} knots, range [{y_min_val:.3f}, {y_max_val:.3f}]")

    calib_dict = {}
    for stat, iso in _CALIBRATORS.items():
        x_th = getattr(iso, 'X_thresholds_', np.array([])).tolist()
        y_th = getattr(iso, 'y_thresholds_', np.array([])).tolist()
        calib_dict[stat] = {
            "x": x_th,
            "y": y_th
        }

    for p in _CALIB_PATHS:
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(calib_dict, f, indent=2)
            print(f"[calib] Saved calibration to {p}")
        except Exception as e:
            print(f"[calib] Warning saving {p}: {e}")

    return _CALIBRATORS


def load_calibration() -> Dict[str, IsotonicRegression]:
    global _CALIBRATORS
    if _CALIBRATORS:
        return _CALIBRATORS

    found_path = None
    for p in _CALIB_PATHS:
        if os.path.exists(p):
            found_path = p
            break

    if not found_path:
        return {}

    try:
        with open(found_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for stat, knots in raw.items():
            if knots.get("x") and knots.get("y"):
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.40, y_max=0.68)
                iso.fit(np.array(knots["x"], dtype=float), np.array(knots["y"], dtype=float))
                _CALIBRATORS[stat] = iso
    except Exception as e:
        print(f"[calib] Error loading calibration from {found_path}: {e}")

    return _CALIBRATORS


def recalibrate(p_model: float, stat_focus: str) -> float:
    """Apply isotonic calibration to a raw model probability."""
    cals = load_calibration()
    iso = cals.get(stat_focus)
    if iso is None:
        return float(p_model)  # no calibration available, pass through
    try:
        pred = float(iso.predict(np.array([p_model]))[0])
        return max(0.40, min(0.68, pred))
    except Exception:
        return float(p_model)
