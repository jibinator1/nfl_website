"""
Post-hoc recalibration of model win probabilities against empirical outcomes.
Fit on historical backtest data, applied at inference time using lightweight numpy interpolation.
"""
import json
import os
import numpy as np
from typing import Dict, Any, List, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CALIB_PATHS = [
    os.path.join(ROOT_DIR, "data", "probability_calibration.json"),
    os.path.join(ROOT_DIR, "api", "data", "probability_calibration.json")
]
_CALIB_PATH = _CALIB_PATHS[0]
_CALIBRATORS: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}  # stat_focus -> (x_knots, y_knots)


def fit_calibration(records_by_stat: Dict[str, List[Tuple[float, int]]]) -> Dict[str, Any]:
    """
    Fits isotonic regression and serializes knots to JSON.
    Uses scikit-learn lazily (only required when refitting offline during backtesting).
    """
    from sklearn.isotonic import IsotonicRegression

    global _CALIBRATORS
    calib_dict = {}
    fitted_models = {}

    for stat, records in records_by_stat.items():
        if len(records) < 50:
            continue  # not enough data to fit
        X = np.array([r[0] for r in records], dtype=float)
        y = np.array([r[1] for r in records], dtype=float)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.40, y_max=0.68)
        iso.fit(X, y)
        fitted_models[stat] = iso

        x_th = getattr(iso, 'X_thresholds_', np.array([])).tolist()
        y_th = getattr(iso, 'y_thresholds_', np.array([])).tolist()

        calib_dict[stat] = {
            "x": x_th,
            "y": y_th
        }
        _CALIBRATORS[stat] = (np.array(x_th, dtype=float), np.array(y_th, dtype=float))

        knots_count = len(x_th)
        y_min_val = float(min(y_th)) if y_th else 0.40
        y_max_val = float(max(y_th)) if y_th else 0.68
        print(f"[calib] {stat}: fitted on {len(records)} records, "
              f"{knots_count} knots, range [{y_min_val:.3f}, {y_max_val:.3f}]")

    for p in _CALIB_PATHS:
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(calib_dict, f, indent=2)
            print(f"[calib] Saved calibration to {p}")
        except Exception as e:
            print(f"[calib] Warning saving {p}: {e}")

    return fitted_models


def load_calibration() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Loads calibration knots from disk. Pure JSON/numpy, zero heavy ML dependencies."""
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
                _CALIBRATORS[stat] = (
                    np.array(knots["x"], dtype=float),
                    np.array(knots["y"], dtype=float)
                )
    except Exception as e:
        print(f"[calib] Error loading calibration from {found_path}: {e}")

    return _CALIBRATORS


def recalibrate(p_model: float, stat_focus: str) -> float:
    """
    Applies isotonic calibration using piecewise linear/step interpolation with out-of-bounds clipping.
    Mathematically identical to scikit-learn's IsotonicRegression.predict([p_model])[0].
    """
    cals = load_calibration()
    knots = cals.get(stat_focus)
    if not knots or len(knots[0]) == 0 or len(knots[1]) == 0:
        return float(p_model)

    x_knots, y_knots = knots
    try:
        val = float(np.interp(p_model, x_knots, y_knots, left=y_knots[0], right=y_knots[-1]))
        return max(0.40, min(0.68, val))
    except Exception:
        return float(p_model)
