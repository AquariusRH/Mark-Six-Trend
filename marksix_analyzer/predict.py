"""Prediction helpers using lightweight scikit-learn models.

Adds method "ai_feature_xgboost" (named per request; uses RandomForest as lightweight "XGBoost-like" alternative).

The main entrypoint is predict_once(draws, method, **kwargs) which returns a sorted List[int].
"""
from __future__ import annotations

from typing import List, Tuple
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .analysis import frequency
from .models import Draw


def _draws_to_dataframe(draws: List[Draw]) -> pd.DataFrame:
    """Convert list[Draw] (oldest-first) into a DataFrame with macro features.

    Columns: draw_id, draw_date, total_sum, odd_count, big_count
    """
    rows = []
    for d in draws:
        nums = list(d.numbers)
        total = sum(nums)
        odd = sum(1 for n in nums if n % 2 == 1)
        big = sum(1 for n in nums if 25 <= n <= 49)
        rows.append({"draw_id": d.draw_id, "draw_date": d.draw_date, "total_sum": total, "odd_count": odd, "big_count": big})
    df = pd.DataFrame(rows)
    return df


def _build_supervised(df: pd.DataFrame, lags: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build a lagged supervised dataset. X contains lag features, y contains current targets.

    Example features: total_sum_lag1, odd_count_lag1, big_count_lag1, ...
    """
    df = df.copy().reset_index(drop=True)
    for lag in range(1, lags + 1):
        df[f"total_sum_lag{lag}"] = df["total_sum"].shift(lag)
        df[f"odd_count_lag{lag}"] = df["odd_count"].shift(lag)
        df[f"big_count_lag{lag}"] = df["big_count"].shift(lag)

    # Drop rows that don't have full lag history
    supervised = df.dropna().reset_index(drop=True)
    if supervised.empty:
        return pd.DataFrame(), pd.DataFrame()

    feature_cols = [c for c in supervised.columns if c.endswith(tuple(f"lag{n}" for n in range(1, lags + 1)))]
    X = supervised[feature_cols].astype(float)
    y = supervised[["total_sum", "odd_count", "big_count"]].astype(int)
    return X, y


def _train_models(X: pd.DataFrame, y: pd.DataFrame):
    """Train three lightweight models:
    - RandomForestRegressor for total_sum
    - RandomForestClassifier for odd_count
    - RandomForestClassifier for big_count
    """
    models = {}
    if X.shape[0] < 5:
        # Not enough data to train robust models; return None to signal fallback
        return None

    # Regressor for sum
    reg = RandomForestRegressor(n_estimators=100, random_state=0)
    reg.fit(X, y["total_sum"])  # type: ignore[arg-type]
    models["total_sum"] = reg

    # Classifiers for odd_count and big_count
    clf_odd = RandomForestClassifier(n_estimators=100, random_state=0)
    clf_odd.fit(X, y["odd_count"])  # type: ignore[arg-type]
    models["odd_count"] = clf_odd

    clf_big = RandomForestClassifier(n_estimators=100, random_state=0)
    clf_big.fit(X, y["big_count"])  # type: ignore[arg-type]
    models["big_count"] = clf_big

    return models


def _predict_next(models, last_rows: pd.DataFrame) -> Tuple[float, int, int]:
    """Use trained models to predict next total_sum (float), odd_count (int), big_count (int).

    last_rows should contain exactly one row of lag features representing the latest history.
    """
    if models is None:
        # Signal not trained
        raise ValueError("Models not trained")

    X_pred = last_rows.values.reshape(1, -1).astype(float)
    total_pred = float(models["total_sum"].predict(X_pred)[0])
    odd_pred = int(models["odd_count"].predict(X_pred)[0])
    big_pred = int(models["big_count"].predict(X_pred)[0])
    return total_pred, odd_pred, big_pred


def _select_best_combo(top_numbers: List[int], freq_map: dict, target_sum: float, target_odd: int, target_big: int, sum_tol: int = 15) -> List[int] | None:
    """Examine all 6-number combinations from top_numbers and pick the one best matching targets.

    Preference order:
    1) sum within tolerance, exact odd and big match, highest combined frequency (tie-breaker: closest sum)
    2) If none found, relax to best overall by frequency and sum closeness.
    """
    best = None
    best_score = None

    combos = combinations(top_numbers, 6)
    for combo in combos:
        combo_list = list(combo)
        s = sum(combo_list)
        odd = sum(1 for n in combo_list if n % 2 == 1)
        big = sum(1 for n in combo_list if 25 <= n <= 49)
        freq_score = sum(freq_map.get(n, 0) for n in combo_list)
        sum_diff = abs(s - target_sum)

        within_tol = sum_diff <= sum_tol
        exact_match = within_tol and (odd == target_odd) and (big == target_big)

        # Score tuple: primary -> exact_match (1/0), secondary -> freq_score, tertiary -> -sum_diff (closer better)
        score = (1 if exact_match else 0, freq_score, -sum_diff)

        if best is None or score > best_score:
            best = combo_list
            best_score = score

    return sorted(best) if best is not None else None


def predict_once(draws: List[Draw], method: str = "ai_feature_xgboost", **kwargs) -> List[int]:
    """Generate a single 6-number prediction using the requested method.

    Supported method: "ai_feature_xgboost" which:
      - engineers macro features from history (sum, odd count, big count)
      - trains RandomForest models to predict next-draw macro-features
      - selects a 6-number combination from the top-15 frequent numbers to match predicted macros

    Returns a sorted List[int].
    """
    if method != "ai_feature_xgboost":
        raise ValueError(f"Unsupported method: {method}")

    # Basic validation and defaults
    if not draws:
        raise ValueError("draws list is empty")

    # FEATURE ENGINEERING
    df = _draws_to_dataframe(draws)  # oldest-first expected
    lags = int(kwargs.get("lags", 3))
    X, y = _build_supervised(df, lags=lags)

    # MODEL TRAINING
    models = _train_models(X, y)

    # If not enough data to train, fallback to simple heuristics (use averages)
    if models is None or X.empty:
        # Fallback targets: mean sum and mode odd/big
        target_sum = float(df["total_sum"].iloc[-1]) if len(df) < 3 else float(df["total_sum"].mean())
        target_odd = int(df["odd_count"].mode().iat[0]) if not df["odd_count"].mode().empty else int(df["odd_count"].iloc[-1])
        target_big = int(df["big_count"].mode().iat[0]) if not df["big_count"].mode().empty else int(df["big_count"].iloc[-1])
    else:
        # Build the single-row lag features from the latest draws
        latest = df.iloc[-lags:]
        # If there are exactly lags rows, create the lag row by reversing and placing as lag1..lagN
        lag_row = {}
        # lag1 should be the most recent previous draw (last row)
        for i, row in enumerate(latest.iloc[::-1].itertuples(index=False), start=1):
            lag_row[f"total_sum_lag{i}"] = getattr(row, "total_sum")
            lag_row[f"odd_count_lag{i}"] = getattr(row, "odd_count")
            lag_row[f"big_count_lag{i}"] = getattr(row, "big_count")
        # Ensure ordering of columns matches X
        last_df = pd.DataFrame([lag_row])
        # Reindex columns to match training X if necessary
        if not X.empty:
            last_df = last_df.reindex(columns=X.columns)
        target_sum, target_odd, target_big = _predict_next(models, last_df.iloc[0:1])
        # Round discrete predictions
        target_odd = int(round(target_odd))
        target_big = int(round(target_big))

    # NUMBER FILTERING & SELECTION
    freq_map = frequency(draws, include_extra=False)
    # Top 15 most frequent numbers
    top_numbers = sorted(freq_map.keys(), key=lambda n: (-freq_map[n], n))[:15]

    # Try to select best combo within tolerance ±15
    sum_tol = int(kwargs.get("sum_tolerance", 15))
    best = _select_best_combo(top_numbers, freq_map, target_sum, target_odd, target_big, sum_tol)

    if best is None:
        # Relax: consider top 20 and pick best by loose criteria
        top_numbers = sorted(freq_map.keys(), key=lambda n: (-freq_map[n], n))[:20]
        best = _select_best_combo(top_numbers, freq_map, target_sum, target_odd, target_big, sum_tol)

    if best is None:
        # As a last resort, pick the top-6 frequent numbers
        best = sorted(top_numbers[:6])

    # Ensure 6 numbers and sorted
    best = sorted(list(best))[:6]
    return best


__all__ = ["predict_once"]
