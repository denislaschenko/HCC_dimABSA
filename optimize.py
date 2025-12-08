import optuna
import sys
import os


project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from train_subtask1 import main as run_training_trial
    from src.subtask_1 import config
    from src.shared import utils
    from src.subtask_1.progress_visualization.generate_results_plot import generate_plot
except ImportError as e:
    print(
        f"Fehler: Notwendige Skripte konnten nicht importiert werden. Stellen Sie sicher, dass __init__.py-Dateien vorhanden sind.")
    print(f"Import-Fehler: {e}")
    sys.exit(1)


def objective(trial: optuna.trial.Trial) -> float:
    override_params = {
        "lr": trial.suggest_float("lr", 6e-6, 3e-5, log=True),
        "batch_size": 4,
        "dropout": trial.suggest_float("dropout", 0.08, 0.16),

        "model_name": "roberta-base",
        "epochs": config.EPOCHS,
        "patience": config.PATIENCE,

        "version_id": f"optuna-trial-{trial.number}"
    }

    try:
        rmse_score = run_training_trial(override_config=override_params)
    except Exception as e:
        print(f"--- TRIAL {trial.number} FAILED ---")
        print(f"Fehler während des Trainingslaufs: {e}")
        raise optuna.exceptions.TrialPruned()

    return rmse_score


if __name__ == "__main__":
    print("Starting Optuna Hyperparameter Optimization...")

    sampler = optuna.samplers.TPESampler(
        seed=42,
        multivariate=True,
        group=True
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=0,
        interval_steps=1
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner
    )

    try:
        study.optimize(objective, n_trials=50)
    except KeyboardInterrupt:
        print("\nOptimierung durch Benutzer abgebrochen.")

    print("\n--- Optimization Finished ---")

    if study.best_trial:
        print(f"Beste RMSE (niedrigster): {study.best_value:.4f}")
        print("Beste Parameter gefunden:")
        print(study.best_params)
    else:
        print("Keine erfolgreichen Trials beendet.")

    print("\nUpdating final Plot and README with all trial results...")
    try:
        generate_plot()
        print("Plotting complete.")
    except Exception as e:
        print(f"Error running plot script: {e}")
