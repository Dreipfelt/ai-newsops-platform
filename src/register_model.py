"""
src/register_model.py
MLflow Model Registry — versioning et rollback
AI NewsOps Platform · AIA Bloc 4

Usage :
  # Enregistrer le modèle actuel
  python src/register_model.py --action register

  # Lister les versions
  python src/register_model.py --action list

  # Promouvoir en Production
  python src/register_model.py --action promote --version 1

  # Rollback vers une version précédente
  python src/register_model.py --action rollback --version 1

  # Comparer deux versions
  python src/register_model.py --action compare
"""

import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
MODELS_DIR = Path("models/distilbert")
BEST_MODEL_DIR = MODELS_DIR / "best_model"
METRICS_FILE = MODELS_DIR / "training_metrics.json"
BACKUP_DIR = MODELS_DIR / "versions"
MODEL_NAME = "news-classifier-distilbert"
MLFLOW_URI = "sqlite:///mlflow.db"

mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────


def load_current_metrics() -> dict:
    """Charge les métriques du modèle actuel."""
    if not METRICS_FILE.exists():
        return {}
    with open(METRICS_FILE) as f:
        return json.load(f)


def get_or_create_experiment(name: str) -> str:
    """Récupère ou crée une expérience MLflow."""
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        log.info(f"Expérience créée : {name} (id={exp_id})")
        return exp_id
    return exp.experiment_id


# ─────────────────────────────────────────────────────────────
# ACTIONS
# ─────────────────────────────────────────────────────────────


def register_model():
    """Enregistre le modèle actuel dans le MLflow Model Registry."""
    metrics = load_current_metrics()

    if not BEST_MODEL_DIR.exists():
        log.error(f"Modèle non trouvé : {BEST_MODEL_DIR}")
        return

    log.info("Enregistrement du modèle dans MLflow Registry...")
    mlflow.set_experiment("news-classifier-registry")

    with mlflow.start_run(
        run_name=f"register-{datetime.now().strftime('%Y%m%d-%H%M')}"
    ) as run:
        # Logger les métriques
        if metrics:
            mlflow.log_metrics(
                {
                    "test_f1_macro": metrics.get("test_f1_macro", 0),
                    "test_accuracy": metrics.get("test_accuracy", 0),
                    "best_val_f1": metrics.get("best_val_f1", 0),
                    "baseline_f1": metrics.get("baseline_f1", 0),
                    "delta_f1": metrics.get("delta_f1", 0),
                }
            )
            mlflow.log_params(
                {
                    "model_name": metrics.get("model", "distilbert-base-uncased"),
                    "epochs_run": metrics.get("epochs_run", 0),
                    "num_labels": len(metrics.get("class_names", [])),
                    "fast_mode": str(metrics.get("fast_mode", False)),
                    "registered_at": datetime.now().isoformat(),
                }
            )

        # Logger les artifacts
        mlflow.log_artifact(str(BEST_MODEL_DIR / "config.json"))
        mlflow.log_artifact(str(METRICS_FILE))

        # Enregistrer dans le Model Registry
        model_uri = f"runs:/{run.info.run_id}/model"

        try:
            registered = mlflow.register_model(
                model_uri=model_uri,
                name=MODEL_NAME,
            )
            version = registered.version
            log.info(f"✅ Modèle enregistré : {MODEL_NAME} v{version}")

            # Sauvegarder une copie locale versionnée
            BACKUP_DIR.mkdir(exist_ok=True)
            backup_path = BACKUP_DIR / f"v{version}_{datetime.now().strftime('%Y%m%d')}"
            if not backup_path.exists():
                shutil.copytree(BEST_MODEL_DIR, backup_path)
                log.info(f"   Backup local : {backup_path}")

            # Sauvegarder le mapping version → run_id
            version_map_file = BACKUP_DIR / "version_map.json"
            version_map = {}
            if version_map_file.exists():
                with open(version_map_file) as f:
                    version_map = json.load(f)
            version_map[str(version)] = {
                "run_id": run.info.run_id,
                "backup_path": str(backup_path),
                "registered_at": datetime.now().isoformat(),
                "metrics": {
                    "test_f1_macro": metrics.get("test_f1_macro"),
                    "test_accuracy": metrics.get("test_accuracy"),
                },
            }
            with open(version_map_file, "w") as f:
                json.dump(version_map, f, indent=2)

            print("\n✅ Modèle enregistré dans MLflow Model Registry")
            print(f"   Nom    : {MODEL_NAME}")
            print(f"   Version: {version}")
            print(f"   Run ID : {run.info.run_id}")
            if metrics:
                print(f"   F1     : {metrics.get('test_f1_macro', 'N/A')}")
            return version

        except Exception as e:
            log.error(f"Enregistrement MLflow échoué : {e}")
            log.info(
                "Conseil : MLflow Model Registry nécessite un backend SQL (sqlite:///mlflow.db)"
            )


def list_versions():
    """Liste toutes les versions du modèle dans le Registry."""
    print(f"\n── Versions de '{MODEL_NAME}' dans MLflow Registry ──────")

    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if not versions:
            print("  Aucune version enregistrée.")
            print("  → Lancer : python src/register_model.py --action register")
            return

        for v in sorted(versions, key=lambda x: int(x.version)):
            print(f"\n  Version {v.version}")
            print(f"    Stage   : {v.current_stage}")
            print(f"    Run ID  : {v.run_id}")
            print(f"    Créée   : {v.creation_timestamp}")
            # Charger les métriques du run
            try:
                run = client.get_run(v.run_id)
                f1 = run.data.metrics.get("test_f1_macro", "N/A")
                print(f"    F1 macro: {f1}")
            except Exception:
                pass

    except Exception as e:
        log.error(f"Erreur listing : {e}")

    # Afficher aussi le version_map local
    version_map_file = BACKUP_DIR / "version_map.json"
    if version_map_file.exists():
        print(f"\n── Backups locaux ({BACKUP_DIR}) ─────────────────────")
        with open(version_map_file) as f:
            version_map = json.load(f)
        for ver, info in version_map.items():
            metrics = info.get("metrics", {})
            print(
                f"  v{ver} | F1={metrics.get('test_f1_macro', 'N/A')} | {info.get('registered_at', '')[:10]}"
            )


def promote_model(version: int):
    """Promeut une version en Production."""
    try:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=str(version),
            stage="Production",
            archive_existing_versions=True,  # Archive les autres versions Production
        )
        log.info(f"✅ Version {version} promue en Production")
        print(f"\n✅ {MODEL_NAME} v{version} → Production")

    except Exception as e:
        log.error(f"Promotion échouée : {e}")


def rollback_model(version: int):
    """
    Rollback vers une version précédente.
    1. Restaure les fichiers du modèle depuis le backup local
    2. Met à jour le stage MLflow
    """
    print(f"\n🔄 Rollback vers version {version}...")

    # Chercher dans le version_map local
    version_map_file = BACKUP_DIR / "version_map.json"
    if not version_map_file.exists():
        log.error(
            "Aucun backup local trouvé. Le rollback nécessite des backups préalables."
        )
        log.error(
            "→ Enregistrer d'abord : python src/register_model.py --action register"
        )
        return

    with open(version_map_file) as f:
        version_map = json.load(f)

    if str(version) not in version_map:
        log.error(f"Version {version} non trouvée dans les backups locaux.")
        log.info(f"Versions disponibles : {list(version_map.keys())}")
        return

    backup_info = version_map[str(version)]
    backup_path = Path(backup_info["backup_path"])

    if not backup_path.exists():
        log.error(f"Backup introuvable : {backup_path}")
        return

    # Sauvegarder la version actuelle avant rollback
    current_backup = (
        MODELS_DIR / f"pre_rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    if BEST_MODEL_DIR.exists():
        shutil.copytree(BEST_MODEL_DIR, current_backup)
        log.info(f"Version actuelle sauvegardée : {current_backup}")

    # Restaurer la version cible
    if BEST_MODEL_DIR.exists():
        shutil.rmtree(BEST_MODEL_DIR)
    shutil.copytree(backup_path, BEST_MODEL_DIR)

    log.info(f"✅ Modèle restauré depuis {backup_path}")

    # Mettre à jour MLflow
    try:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=str(version),
            stage="Production",
            archive_existing_versions=True,
        )
        log.info(f"✅ MLflow Registry mis à jour — v{version} → Production")
    except Exception as e:
        log.warning(f"MLflow update échoué (non bloquant) : {e}")

    metrics = backup_info.get("metrics", {})
    print("\n✅ Rollback réussi !")
    print(f"   Version restaurée : {version}")
    print(f"   F1 macro          : {metrics.get('test_f1_macro', 'N/A')}")
    print(f"   Backup pré-rollback : {current_backup}")
    print("\n⚠️  Redémarrer l'API pour charger le modèle restauré :")
    print("   fuser -k 8000/tcp && uvicorn src.api.main:app --host 0.0.0.0 --port 8000")


def compare_versions():
    """Compare les métriques de toutes les versions disponibles."""
    print("\n── Comparaison des versions ────────────────────────────")

    version_map_file = BACKUP_DIR / "version_map.json"
    if not version_map_file.exists():
        print("  Aucune version enregistrée.")
        return

    with open(version_map_file) as f:
        version_map = json.load(f)

    print(f"  {'Version':<10} {'F1 macro':<12} {'Accuracy':<12} {'Date'}")
    print(f"  {'-'*50}")

    best_f1 = 0
    best_version = None

    for ver, info in sorted(version_map.items(), key=lambda x: int(x[0])):
        metrics = info.get("metrics", {})
        f1 = metrics.get("test_f1_macro", 0) or 0
        acc = metrics.get("test_accuracy", 0) or 0
        date = info.get("registered_at", "")[:10]
        marker = " ← BEST" if f1 > best_f1 else ""
        if f1 > best_f1:
            best_f1 = f1
            best_version = ver
        print(f"  v{ver:<9} {f1:<12.4f} {acc:<12.4f} {date}{marker}")

    print(f"\n  Meilleure version : v{best_version} (F1={best_f1:.4f})")
    print(
        f"\n  Pour promouvoir : python src/register_model.py --action promote --version {best_version}"
    )


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────


def main(args):
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    if args.action == "register":
        register_model()
    elif args.action == "list":
        list_versions()
    elif args.action == "promote":
        if not args.version:
            log.error("--version requis pour promote")
            return
        promote_model(args.version)
    elif args.action == "rollback":
        if not args.version:
            log.error("--version requis pour rollback")
            return
        rollback_model(args.version)
    elif args.action == "compare":
        compare_versions()
    else:
        log.error(f"Action inconnue : {args.action}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MLflow Model Registry — versioning et rollback"
    )
    parser.add_argument(
        "--action",
        choices=["register", "list", "promote", "rollback", "compare"],
        default="register",
        help="Action à effectuer",
    )
    parser.add_argument(
        "--version",
        type=int,
        help="Numéro de version (pour promote/rollback)",
    )
    main(parser.parse_args())
