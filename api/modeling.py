from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATASET_PATH = Path(r"C:\Users\Admin\Downloads\StudentGradesDataset.csv")
ARTIFACT_DIR = Path(__file__).resolve().parent / "model_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "gwa_predictor.joblib"

GRADE_COLUMNS = [
    "Religion",
    "Kurdish",
    "Arabic",
    "English",
    "Math",
    "Physic",
    "Chemistry",
    "Biology",
    "Computer",
    "History",
    "Geography",
    "Economics",
    "Sociology",
    "Science",
    "Social",
]
NUMERIC_FEATURES = ["Class", *GRADE_COLUMNS]
CATEGORICAL_FEATURES = ["Years"]
ALL_FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
MIN_GRADES_REQUIRED = 4


@dataclass
class TrainingResult:
    model: Pipeline
    metrics: dict[str, float]
    feature_names: list[str]
    train_rows: int
    test_rows: int


def _read_dataset(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    frame = pd.read_csv(dataset_path)
    frame = frame.rename(columns=lambda value: value.strip())
    frame["Serial No"] = frame["Serial No"].astype(str).str.strip()
    frame["Years"] = frame["Years"].astype(str).str.strip()

    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def _row_average(frame: pd.DataFrame) -> pd.Series:
    return frame[GRADE_COLUMNS].mean(axis=1, skipna=True)


def build_training_frame(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    frame = _read_dataset(dataset_path)
    frame["current_average"] = _row_average(frame)
    frame["grade_count"] = frame[GRADE_COLUMNS].notna().sum(axis=1)

    sortable = frame.sort_values(["Serial No", "Class", "Years"]).copy()
    sortable["target_average"] = sortable.groupby("Serial No")["current_average"].shift(-1)
    sortable["next_class"] = sortable.groupby("Serial No")["Class"].shift(-1)

    trainable = sortable[
        sortable["target_average"].notna()
        & sortable["Class"].notna()
        & (sortable["grade_count"] >= MIN_GRADES_REQUIRED)
        & sortable["next_class"].notna()
        & (sortable["next_class"] > sortable["Class"])
    ].copy()

    trainable["Years"] = trainable["Years"].fillna("unknown")
    return trainable


def _make_pipeline(model: Any) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def _evaluate_predictions(actual: pd.Series, predicted: Any) -> dict[str, float]:
    rmse = mean_squared_error(actual, predicted) ** 0.5
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(rmse),
        "r2": float(r2_score(actual, predicted)),
    }


def train_model(dataset_path: Path = DATASET_PATH) -> TrainingResult:
    frame = build_training_frame(dataset_path)
    feature_frame = frame[ALL_FEATURES]
    target = frame["target_average"]
    groups = frame["Serial No"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_index, test_index = next(splitter.split(feature_frame, target, groups=groups))

    x_train = feature_frame.iloc[train_index]
    x_test = feature_frame.iloc[test_index]
    y_train = target.iloc[train_index]
    y_test = target.iloc[test_index]

    candidates = [
        (
            "random_forest",
            _make_pipeline(
                RandomForestRegressor(
                    n_estimators=90,
                    max_depth=16,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                )
            ),
        ),
        (
            "extra_trees",
            _make_pipeline(
                ExtraTreesRegressor(
                    n_estimators=120,
                    max_depth=16,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                )
            ),
        ),
    ]

    best_name = ""
    best_model: Pipeline | None = None
    best_metrics: dict[str, float] | None = None

    for candidate_name, candidate in candidates:
        candidate.fit(x_train, y_train)
        predicted = candidate.predict(x_test)
        metrics = _evaluate_predictions(y_test, predicted)
        if best_metrics is None or metrics["mae"] < best_metrics["mae"]:
            best_name = candidate_name
            best_model = candidate
            best_metrics = metrics

    assert best_model is not None
    assert best_metrics is not None

    best_metrics["selected_model"] = best_name

    return TrainingResult(
        model=best_model,
        metrics=best_metrics,
        feature_names=ALL_FEATURES,
        train_rows=len(train_index),
        test_rows=len(test_index),
    )


def save_artifact(result: TrainingResult, artifact_path: Path = ARTIFACT_PATH) -> Path:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": result.model,
            "metrics": result.metrics,
            "feature_names": result.feature_names,
            "train_rows": result.train_rows,
            "test_rows": result.test_rows,
        },
        artifact_path,
    )
    return artifact_path


def load_or_train_artifact(
    artifact_path: Path = ARTIFACT_PATH,
    dataset_path: Path = DATASET_PATH,
) -> dict[str, Any]:
    if artifact_path.exists():
        return joblib.load(artifact_path)

    result = train_model(dataset_path)
    save_artifact(result, artifact_path)
    return {
        "model": result.model,
        "metrics": result.metrics,
        "feature_names": result.feature_names,
        "train_rows": result.train_rows,
        "test_rows": result.test_rows,
    }
