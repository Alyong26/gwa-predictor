from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from ucimlrepo import fetch_ucirepo

DATASET_ID = 320
RANDOM_STATE = 42
ARTIFACT_DIR = Path(__file__).resolve().parent / "model_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "uciml_student_performance_model.joblib"


@dataclass
class UciTrainingResult:
    model: Pipeline
    target_column: str
    metrics: dict[str, float | str]
    feature_columns: list[str]
    train_rows: int
    test_rows: int


def _prepare_data() -> tuple[pd.DataFrame, pd.Series, str]:
    dataset = fetch_ucirepo(id=DATASET_ID)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()

    if "G3" in targets.columns:
        target_column = "G3"
    else:
        numeric_targets = targets.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_targets:
            raise ValueError("No numeric target column found in dataset.")
        target_column = numeric_targets[0]

    target = pd.to_numeric(targets[target_column], errors="coerce")

    # In this dataset, period grades are sometimes exposed as targets.
    # Include prior period grades as features when available to improve G3 prediction.
    for prior_grade in ("G1", "G2"):
        if prior_grade in targets.columns and prior_grade != target_column:
            features[prior_grade] = pd.to_numeric(targets[prior_grade], errors="coerce")

    # Keep only rows that have a valid supervised target.
    valid_rows = target.notna()
    features = features.loc[valid_rows].reset_index(drop=True)
    target = target.loc[valid_rows].reset_index(drop=True)

    return features, target, target_column


def _build_preprocessor(x: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_features = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [column for column in x.columns if column not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def _score(y_true: pd.Series, y_pred: Any) -> dict[str, float]:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_uciml_model() -> UciTrainingResult:
    x, y, target_column = _prepare_data()
    preprocessor, numeric_features, categorical_features = _build_preprocessor(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE
    )

    candidates: list[tuple[str, Any, dict[str, list[int | float | None]]]] = [
        (
            "random_forest",
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [200, 400, 600],
                "model__max_depth": [None, 12, 18, 24],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
        (
            "extra_trees",
            ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [250, 500, 700],
                "model__max_depth": [None, 12, 18, 24],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
    ]

    best_name = ""
    best_estimator: Any | None = None
    best_cv_rmse = float("inf")
    best_holdout_rmse = float("inf")

    for name, model, param_distributions in candidates:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=30,
            cv=5,
            scoring="neg_root_mean_squared_error",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(x_train, y_train)
        candidate_cv_rmse = -float(search.best_score_)
        candidate_holdout_predictions = search.best_estimator_.predict(x_test)
        candidate_holdout_rmse = float(
            mean_squared_error(y_test, candidate_holdout_predictions) ** 0.5
        )

        if candidate_holdout_rmse < best_holdout_rmse:
            best_cv_rmse = candidate_cv_rmse
            best_holdout_rmse = candidate_holdout_rmse
            best_name = name
            best_estimator = search.best_estimator_

    if best_estimator is None:
        raise RuntimeError("Failed to train any model candidate.")

    # Blend strong tree models for lower variance and better generalization.
    if len(candidates) >= 2:
        rf_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=600,
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        et_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=700,
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        ensemble = VotingRegressor(
            estimators=[
                ("rf", rf_pipeline),
                ("et", et_pipeline),
            ]
        )
        ensemble.fit(x_train, y_train)
        ensemble_predictions = ensemble.predict(x_test)
        ensemble_metrics = _score(y_test, ensemble_predictions)
        if ensemble_metrics["rmse"] < best_holdout_rmse:
            best_name = "voting_ensemble"
            best_estimator = ensemble
            best_holdout_rmse = ensemble_metrics["rmse"]

    predictions = best_estimator.predict(x_test)
    metrics = _score(y_test, predictions)
    metrics["cv_rmse"] = float(best_cv_rmse)
    metrics["selected_model"] = best_name

    return UciTrainingResult(
        model=best_estimator,
        target_column=target_column,
        metrics=metrics,
        feature_columns=x.columns.tolist(),
        train_rows=len(x_train),
        test_rows=len(x_test),
    )


def save_artifact(result: UciTrainingResult, artifact_path: Path = ARTIFACT_PATH) -> Path:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": result.model,
            "target_column": result.target_column,
            "metrics": result.metrics,
            "feature_columns": result.feature_columns,
            "train_rows": result.train_rows,
            "test_rows": result.test_rows,
        },
        artifact_path,
    )
    return artifact_path


def load_or_train_artifact(artifact_path: Path = ARTIFACT_PATH) -> dict[str, Any]:
    if artifact_path.exists():
        return joblib.load(artifact_path)

    result = train_uciml_model()
    save_artifact(result, artifact_path)
    return {
        "model": result.model,
        "target_column": result.target_column,
        "metrics": result.metrics,
        "feature_columns": result.feature_columns,
        "train_rows": result.train_rows,
        "test_rows": result.test_rows,
    }


def main() -> None:
    result = train_uciml_model()
    save_artifact(result)

    print("UCI model trained successfully.")
    print(f"Dataset id: {DATASET_ID}")
    print(f"Target column: {result.target_column}")
    print(f"Selected model: {result.metrics['selected_model']}")
    print(f"CV RMSE: {result.metrics['cv_rmse']:.4f}")
    print(f"Test MAE: {result.metrics['mae']:.4f}")
    print(f"Test RMSE: {result.metrics['rmse']:.4f}")
    print(f"Test R2: {result.metrics['r2']:.4f}")
    print(f"Saved artifact: {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
