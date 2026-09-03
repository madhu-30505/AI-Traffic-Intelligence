import subprocess
import sys
from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_step(step_number, description, script_name):
    """Run one project step and stop if it fails."""

    print("\n" + "=" * 70)
    print(f"STEP {step_number}: {description}")
    print("=" * 70)

    script_path = PROJECT_ROOT / "src" / script_name

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:
        print(f"\n❌ Step {step_number} failed.")
        print(f"Script: {script_name}")
        sys.exit(result.returncode)

    print(f"\n✅ Step {step_number} completed successfully.")


def main():

    print("\n")
    print("=" * 70)
    print("🚦 AI TRAFFIC INTELLIGENCE - COMPLETE PIPELINE")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    # ---------------------------------------------------------
    # 1. Generate detailed historical traffic data
    # ---------------------------------------------------------
    run_step(
        1,
        "Generate detailed traffic history",
        "traffic_logger.py"
    )

    # ---------------------------------------------------------
    # 2. Convert frame-level data to 1-second dataset
    # ---------------------------------------------------------
    run_step(
        2,
        "Create 1-second traffic dataset",
        "create_1sec_dataset.py"
    )

    # ---------------------------------------------------------
    # 3. Prepare ML forecasting features
    # ---------------------------------------------------------
    run_step(
        3,
        "Prepare 1-second ML features",
        "prepare_1sec_ml_features.py"
    )

    # ---------------------------------------------------------
    # 4. Train forecasting models
    # ---------------------------------------------------------
    run_step(
        4,
        "Train traffic forecasting models",
        "train_1sec_forecasting_models.py"
    )

    # ---------------------------------------------------------
    # 5. Evaluate and select best model
    # ---------------------------------------------------------
    run_step(
        5,
        "Evaluate models and select best model",
        "evaluate_traffic_model.py"
    )

    # ---------------------------------------------------------
    # 6. Predict future traffic
    # ---------------------------------------------------------
    run_step(
        6,
        "Predict future traffic congestion",
        "predict_future_traffic.py"
    )

    # ---------------------------------------------------------
    # 7. Generate traffic alerts
    # ---------------------------------------------------------
    run_step(
        7,
        "Generate traffic alerts",
        "traffic_alerts.py"
    )

    # ---------------------------------------------------------
    # Pipeline completed
    # ---------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("🎉 COMPLETE PIPELINE FINISHED SUCCESSFULLY")
    print("=" * 70)

    print("\nGenerated outputs:")
    print("  ✓ outputs/traffic_history_detailed.csv")
    print("  ✓ outputs/traffic_1sec_dataset.csv")
    print("  ✓ outputs/traffic_1sec_ml_features.csv")
    print("  ✓ models/traffic_forecasting_model.pkl")
    print("  ✓ outputs/model_metadata.csv")
    print("  ✓ outputs/future_traffic_predictions.csv")
    print("  ✓ outputs/traffic_alerts.csv")

    print("\nNext step:")
    print("  streamlit run src/dashboard.py")

    print("\n🚦 AI Traffic Intelligence pipeline is ready!")


if __name__ == "__main__":
    main()
