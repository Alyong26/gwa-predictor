from __future__ import annotations

try:
    from api.one_field_model import (
        ONE_FIELD_ARTIFACT_PATH,
        save_one_field_artifact,
        train_one_field_model,
    )
except ModuleNotFoundError:
    from one_field_model import (  # type: ignore
        ONE_FIELD_ARTIFACT_PATH,
        save_one_field_artifact,
        train_one_field_model,
    )


def main() -> None:
    result = train_one_field_model()
    save_one_field_artifact(result, ONE_FIELD_ARTIFACT_PATH)
    print("One-field model trained successfully.")
    print(f"Saved artifact: {ONE_FIELD_ARTIFACT_PATH}")
    print(f"Training rows: {result.train_rows}")
    print(f"Test rows: {result.test_rows}")
    print(f"Selected model: {result.metrics['selected_model']}")
    print(f"MAE: {result.metrics['mae']:.4f}")
    print(f"RMSE: {result.metrics['rmse']:.4f}")
    print(f"R2: {result.metrics['r2']:.4f}")
    print(f"80% interval coverage: {result.metrics['interval_coverage_80']:.4f}")
    print(f"80% interval avg width: {result.metrics['interval_avg_width']:.4f}")


if __name__ == "__main__":
    main()
