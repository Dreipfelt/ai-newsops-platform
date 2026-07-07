#!/usr/bin/env bash
#
# scripts/verify_all.sh
# Vérification exhaustive de toutes les briques du projet avant soutenance.
# Usage : bash scripts/verify_all.sh
#
# Ce script ne modifie rien — il vérifie uniquement et rapporte un statut
# PASS/FAIL/WARN pour chaque composant, avec un résumé final.

set -uo pipefail

PASS=0
FAIL=0
WARN=0

green()  { echo -e "\033[32m$1\033[0m"; }
red()    { echo -e "\033[31m$1\033[0m"; }
yellow() { echo -e "\033[33m$1\033[0m"; }

check() {
  local name="$1"
  local cmd="$2"
  echo -n "  [ ] ${name}... "
  if eval "$cmd" > /tmp/verify_out.log 2>&1; then
    green "✅ PASS"
    PASS=$((PASS+1))
  else
    red "❌ FAIL"
    echo "      → $(tail -3 /tmp/verify_out.log | tr '\n' ' ')"
    FAIL=$((FAIL+1))
  fi
}

warn_check() {
  local name="$1"
  local cmd="$2"
  echo -n "  [ ] ${name}... "
  if eval "$cmd" > /tmp/verify_out.log 2>&1; then
    green "✅ PASS"
    PASS=$((PASS+1))
  else
    yellow "⚠️  WARN (non bloquant)"
    WARN=$((WARN+1))
  fi
}

echo "═══════════════════════════════════════════════════════════"
echo "  VÉRIFICATION COMPLÈTE — AI NewsOps Platform"
echo "═══════════════════════════════════════════════════════════"

echo -e "\n── 1. Repo & Git ─────────────────────────────────────────"
check "Pas de modifications non commitées"    "test -z \"\$(git status --porcelain)\""
check "Branche à jour avec origin/main"        "git fetch --quiet && git diff --quiet HEAD origin/main"
check "Dernier CI en succès (nécessite gh)"    "gh run list --limit 1 --json conclusion -q '.[0].conclusion' | grep -q success"

echo -e "\n── 2. Qualité du code ────────────────────────────────────"
check "Ruff — zéro erreur de lint"             "ruff check src/"
check "Black — formatage conforme"             "black --check src/"
check "Tests pytest — 34 tests passent"        "pytest tests/ -q"

echo -e "\n── 3. Docker Compose — 8 services ────────────────────────"
check "docker-compose config valide"           "docker-compose config --quiet"
check "Tous les containers Up"                 "test \$(docker compose --profile airflow ps --status running -q | wc -l) -ge 7"

echo -e "\n── 4. API FastAPI ────────────────────────────────────────"
check "GET /health retourne 200"               "curl -sf http://localhost:8000/health"
check "GET / retourne 200"                     "curl -sf http://localhost:8000/"
check "POST /predict fonctionne"               "curl -sf -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{\"headline\":\"Test article about politics\"}'"
check "POST /predict/batch fonctionne"         "curl -sf -X POST http://localhost:8000/predict/batch -H 'Content-Type: application/json' -d '{\"articles\":[{\"headline\":\"Test\"}]}'"
check "GET /metrics retourne du Prometheus"    "curl -sf http://localhost:8000/metrics | grep -q http_requests_total"
check "model_f1_score n'est PAS zéro"          "curl -sf http://localhost:8000/metrics | grep model_f1_score | tail -1 | grep -qv ' 0$'"

echo -e "\n── 5. Monitoring routes ──────────────────────────────────"
check "GET /monitoring/health retourne 200"    "curl -sf http://localhost:8000/monitoring/health"
check "GET /monitoring/drift retourne 200"     "curl -sf http://localhost:8000/monitoring/drift"

echo -e "\n── 6. Prometheus ─────────────────────────────────────────"
check "Prometheus healthy"                     "curl -sf http://localhost:9090/-/healthy"
check "Target newsops-api scrapée (up=1)"      "curl -sf 'http://localhost:9090/api/v1/query?query=up{job=\"newsops-api\"}' | grep -q '\"1\"'"

echo -e "\n── 7. Grafana ────────────────────────────────────────────"
check "Grafana healthy"                        "curl -sf http://localhost:3000/api/health"
check "Datasource UID = prometheus"            "curl -sf -u admin:admin http://localhost:3000/api/datasources | grep -q '\"uid\":\"prometheus\"'"
check "Dashboard provisionné existe"           "curl -sf -u admin:admin http://localhost:3000/api/dashboards/uid/ai-newsops-monitoring"

echo -e "\n── 8. MLflow ─────────────────────────────────────────────"
check "MLflow UI accessible"                   "curl -sf http://localhost:5000"
warn_check "Au moins 1 modèle enregistré"      "python src/register_model.py --action list | grep -q Version"

echo -e "\n── 9. Streamlit Dashboard ────────────────────────────────"
check "Streamlit accessible"                   "curl -sf http://localhost:8501"

echo -e "\n── 10. Airflow ───────────────────────────────────────────"
check "Airflow webserver accessible"           "curl -sf http://localhost:8080/health"
warn_check "DAG newsops_retraining_pipeline existe" "docker exec newsops-airflow-webserver airflow dags list 2>/dev/null | grep -q news_classifier_retraining"

echo -e "\n── 11. Données & modèle ──────────────────────────────────"
check "train.parquet existe"                   "test -f data/processed/train.parquet"
check "test.parquet existe"                    "test -f data/processed/test.parquet"
check "label_mapping.json — 13 classes"        "test \$(python3 -c 'import json; print(len(json.load(open(\"data/processed/label_mapping.json\"))))') -eq 13"
check "best_model/ contient model.safetensors" "test -f models/distilbert/best_model/model.safetensors"
check "training_metrics.json — F1 cohérent"    "python3 -c 'import json; m=json.load(open(\"models/distilbert/training_metrics.json\")); exit(0 if m[\"test_f1_macro\"] > 0.5 else 1)'"

echo -e "\n── 12. DVC ───────────────────────────────────────────────"
warn_check "DVC status propre"                 "dvc status"

echo -e "\n── 13. Documentation ─────────────────────────────────────"
check "README.md existe et non vide"           "test -s README.md"
check "docs/API.md existe"                     "test -s docs/API.md"
check "Postman collection existe"              "test -s docs/newsops_postman_collection.json"

echo -e "\n═══════════════════════════════════════════════════════════"
echo "  RÉSUMÉ"
echo "═══════════════════════════════════════════════════════════"
green "  PASS : ${PASS}"
yellow "  WARN : ${WARN}"
red   "  FAIL : ${FAIL}"
echo "═══════════════════════════════════════════════════════════"

if [ "$FAIL" -gt 0 ]; then
  red "\n⚠️  Des composants sont en échec — à corriger avant la soutenance."
  exit 1
else
  green "\n✅ Tous les composants critiques sont opérationnels."
  exit 0
fi
