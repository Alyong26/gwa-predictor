from api.modeling import ARTIFACT_PATH, DATASET_PATH, save_artifact, train_model


def main() -> None:
    result = train_model(DATASET_PATH)
    save_artifact(result, ARTIFACT_PATH)

    print("Model trained successfully.")
    print(f"Saved artifact: {ARTIFACT_PATH}")
    print(f"Training rows: {result.train_rows}")
    print(f"Test rows: {result.test_rows}")
    print(f"Selected model: {result.metrics['selected_model']}")
    print(f"MAE: {result.metrics['mae']:.4f}")
    print(f"RMSE: {result.metrics['rmse']:.4f}")
    print(f"R2: {result.metrics['r2']:.4f}")


if __name__ == "__main__":
    main()
