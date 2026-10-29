"""Command-line entry point for the complete training workflow."""
from modeling import train_and_evaluate


def main():
    result = train_and_evaluate()
    print("\nCross-validation comparison (lower MAE is better):")
    print(result["comparison"].to_string(index=False))
    print("\nFinal held-out test results:")
    for key, value in result["metrics"].items():
        print(f"{key}: {value}")
    print("\nSaved model: models/best_model.joblib")
    print("Saved metadata: models/model_metadata.json")
    print("Saved evaluation outputs in reports/")


if __name__ == "__main__":
    main()
