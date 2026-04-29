from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

try:
    from api.modeling import DATASET_PATH, build_training_frame
except ModuleNotFoundError:
    from modeling import DATASET_PATH, build_training_frame  # type: ignore

ARTIFACT_DIR = Path(__file__).resolve().parent / "model_artifacts"
ONE_FIELD_ARTIFACT_PATH = ARTIFACT_DIR / "one_field_gwa_predictor.joblib"
RANDOM_STATE = 42


@dataclass
class OneFieldTrainingResult:
    model: Any
    calibrator: IsotonicRegression
    interval_radius: float
    metrics: dict[str, float | str]
    train_rows: int
    test_rows: int


def _percentage_to_gwa(percentage: float) -> float:
    # 1.00 is highest, 3.00 is lowest passing, 5.00 is failing.
    # Map [75,100] -> [3.00,1.00] and [0,75) -> (5.00,3.00].
    if pd.isna(percentage):
        return float("nan")
    clipped = max(0.0, min(100.0, float(percentage)))
    if clipped >= 75.0:
        return 1.0 + (((100.0 - clipped) / 25.0) * 2.0)
    return 3.0 + (((75.0 - clipped) / 75.0) * 2.0)


def _to_one_field_frame(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    frame = build_training_frame(dataset_path)
    result = pd.DataFrame(
        {
            "Serial No": frame["Serial No"].astype(str),
            "current_gwa": frame["current_average"].apply(_percentage_to_gwa),
            "next_gwa": frame["target_average"].apply(_percentage_to_gwa),
        }
    )
    result = result.dropna(subset=["current_gwa", "next_gwa"]).copy()
    return result


def _score(y_true: pd.Series, y_pred: Any) -> dict[str, float]:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_one_field_model(dataset_path: Path = DATASET_PATH) -> OneFieldTrainingResult:
    frame = _to_one_field_frame(dataset_path)
    x = frame[["current_gwa"]]
    y = frame["next_gwa"]
    groups = frame["Serial No"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_index, test_index = next(splitter.split(x, y, groups=groups))

    x_train = x.iloc[train_index]
    x_test = x.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]
    groups_train = groups.iloc[train_index]

    # Split train into fit/calibration for post-hoc calibration.
    calib_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    fit_index, calib_index = next(calib_splitter.split(x_train, y_train, groups=groups_train))
    x_fit = x_train.iloc[fit_index]
    y_fit = y_train.iloc[fit_index]
    x_calib = x_train.iloc[calib_index]
    y_calib = y_train.iloc[calib_index]

    candidates: list[tuple[str, Any]] = [
        ("linear", LinearRegression()),
        ("poly2_linear", Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)), ("lr", LinearRegression())])),
        (
            "knn",
            KNeighborsRegressor(n_neighbors=9, weights="distance"),
        ),
        (
            "random_forest",
            RandomForestRegressor(
                n_estimators=500,
                max_depth=8,
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "extra_trees",
            ExtraTreesRegressor(
                n_estimators=700,
                max_depth=10,
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "gradient_boosting",
            GradientBoostingRegressor(
                n_estimators=400,
                learning_rate=0.03,
                max_depth=2,
                random_state=RANDOM_STATE,
            ),
        ),
    ]

    best_name = ""
    best_model: Any | None = None
    best_calibrator: IsotonicRegression | None = None
    best_metrics: dict[str, float] | None = None

    for name, candidate in candidates:
        candidate.fit(x_fit, y_fit)
        calib_pred = candidate.predict(x_calib)
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(calib_pred, y_calib)

        predicted = calibrator.predict(candidate.predict(x_test))
        metrics = _score(y_test, predicted)
        if best_metrics is None or metrics["rmse"] < best_metrics["rmse"]:
            best_name = name
            best_model = candidate
            best_calibrator = calibrator
            best_metrics = metrics

    if best_model is None or best_calibrator is None or best_metrics is None:
        raise RuntimeError("Failed to train one-field model.")

    # Conformal-style interval: use calibration residual quantile around calibrated prediction.
    calib_center = best_calibrator.predict(best_model.predict(x_calib))
    calib_abs_residual = (y_calib - calib_center).abs()
    interval_radius = float(calib_abs_residual.quantile(0.8))

    test_center = best_calibrator.predict(best_model.predict(x_test))
    interval_coverage = float(
        ((y_test >= (test_center - interval_radius)) & (y_test <= (test_center + interval_radius))).mean()
    )
    interval_width = float(interval_radius * 2.0)

    best_metrics["selected_model"] = best_name
    best_metrics["interval_coverage_80"] = interval_coverage
    best_metrics["interval_avg_width"] = interval_width
    return OneFieldTrainingResult(
        model=best_model,
        calibrator=best_calibrator,
        interval_radius=interval_radius,
        metrics=best_metrics,
        train_rows=len(train_index),
        test_rows=len(test_index),
    )


def save_one_field_artifact(
    result: OneFieldTrainingResult, artifact_path: Path = ONE_FIELD_ARTIFACT_PATH
) -> Path:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": result.model,
            "calibrator": result.calibrator,
            "interval_radius": result.interval_radius,
            "metrics": result.metrics,
            "train_rows": result.train_rows,
            "test_rows": result.test_rows,
        },
        artifact_path,
    )
    return artifact_path


def load_or_train_one_field_artifact(
    artifact_path: Path = ONE_FIELD_ARTIFACT_PATH,
    dataset_path: Path = DATASET_PATH,
) -> dict[str, Any]:
    if artifact_path.exists():
        return joblib.load(artifact_path)

    result = train_one_field_model(dataset_path)
    save_one_field_artifact(result, artifact_path)
    return {
        "model": result.model,
        "calibrator": result.calibrator,
        "interval_radius": result.interval_radius,
        "metrics": result.metrics,
        "train_rows": result.train_rows,
        "test_rows": result.test_rows,
    }
