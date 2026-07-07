#!/usr/bin/env bash
#
# scripts/generate_traffic.sh
# Génère du trafic de test vers l'API /predict — utile pour peupler
# les dashboards Grafana/Prometheus avant une démo ou une capture d'écran.
#
# Usage :
#   bash scripts/generate_traffic.sh            # 5 requêtes, 1s d'intervalle
#   bash scripts/generate_traffic.sh 20 0.5      # 20 requêtes, 0.5s d'intervalle
#   API_URL=http://localhost:8000 bash scripts/generate_traffic.sh

set -euo pipefail

N=${1:-5}
DELAY=${2:-1}
API_URL=${API_URL:-http://localhost:8000}

HEADLINES=(
  "Senate votes on new climate legislation"
  "Lakers beat Warriors in overtime thriller"
  "Apple announces new iPhone with AI features"
  "Scientists discover new exoplanet in habitable zone"
  "Federal Reserve raises interest rates by 25 basis points"
  "New study links coffee consumption to longevity"
  "Local chef wins national cooking competition"
  "Stock markets rally on strong earnings reports"
)

echo "Génération de ${N} requêtes vers ${API_URL}/predict (intervalle ${DELAY}s)..."

for i in $(seq 1 "$N"); do
  idx=$(( (i - 1) % ${#HEADLINES[@]} ))
  headline="${HEADLINES[$idx]} (test #${i})"

  curl -s -X POST "${API_URL}/predict" \
    -H "Content-Type: application/json" \
    -d "{\"headline\": \"${headline}\"}" \
    -o /dev/null -w "  [%{http_code}] %{time_total}s — ${headline}\n"

  sleep "$DELAY"
done

echo "✅ Terminé — vérifie le dashboard Grafana : http://localhost:3000/d/ai-newsops-monitoring"
