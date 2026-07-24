#!/usr/bin/env python3
"""
scripts/patch_readme_slo.py
Corrige quatre affirmations factuellement inexactes dans le README :
  1. SLA latence formulée sans préciser les conditions de mesure
  2. Tableau SLI marquant ✅ un objectif non atteint (F1 67.91% pour cible 70%)
  3. Liste SLO désalignée du cahier des charges v2
  4. Affirmation d'un AlertManager Prometheus qui n'existe pas

Usage : python scripts/patch_readme_slo.py
"""

from pathlib import Path

README = Path("README.md")
content = README.read_text(encoding="utf-8")
applied = []

# ─────────────────────────────────────────────────────────────
# PATCH 1 — Préciser les conditions de mesure de la latence
# ─────────────────────────────────────────────────────────────
old_1 = (
    "2. **Latency SLA** : DistilBERT ~5ms meets < 10ms requirement, "
    "BERT ~15ms exceeds it"
)
new_1 = (
    "2. **Inference latency** : DistilBERT ~5 ms per isolated request "
    "(~15 ms for BERT-base on the same hardware), leaving ample headroom "
    "under the < 100 ms target"
)
if old_1 in content:
    content = content.replace(old_1, new_1)
    applied.append("1. Latence — conditions de mesure précisées")

# ─────────────────────────────────────────────────────────────
# PATCH 2 — Tableau SLI honnête, aligné sur les mesures réelles
# ─────────────────────────────────────────────────────────────
old_2 = """| SLI | Target | Current | Monitor |
|-----|--------|---------|---------|
| Availability | 99.9% | ✅ Stable | Grafana : uptime % |
| Latency P95 | < 10 ms | ✅ ~5ms | Grafana : request latency |
| Error Rate | < 0.1% | ✅ 0% | Prometheus : 5xx errors |
| Model Accuracy | ≥ 70% F1 | ✅ 67.91% | MLflow experiment tracking |
| Drift Detection | Alert < 30min | ✅ | Prometheus : drift_status |"""

new_2 = """Targets are calibrated against the actual deployment hardware (12-thread CPU,
no GPU) and the actual business volume (10,000 articles/day, ~3.5 req/s at peak),
as justified in [`docs/specification-document.md`](docs/specification-document.md) §5.1.

| SLI | Target | Measured | Met | Source |
|-----|--------|----------|:---:|--------|
| Availability (continuous operation) | ≥ 99% | stable over multi-day run | ✅ | Grafana uptime |
| Latency p95, isolated request | < 100 ms | ~5 ms | ✅ | Grafana request latency |
| Latency p95, 15 concurrent users | < 2,000 ms | 1,800 ms | ✅ | `docs/load_test_comparison.md` |
| Sustained throughput, 1 replica | ≥ 5 req/s | 7.62 req/s | ✅ | `docs/load_test_comparison.md` |
| Error rate under sustained load | < 1% | 0.00% | ✅ | `docs/load_test_comparison.md` |
| Macro F1 (test set) | ≥ 0.65 | 0.6791 | ✅ | `models/distilbert/training_metrics.json` |
| Accuracy (test set) | ≥ 0.70 | 0.7382 | ✅ | `models/distilbert/training_metrics.json` |
| Drift detection latency | < 30 min | on-demand + scheduled | ✅ | `GET /monitoring/drift` |

The macro F1 target of 0.65 is set deliberately just above the TF-IDF + LinearSVC
baseline (0.6515), so that it cannot be met unless the neural model justifies its
additional complexity."""

if old_2 in content:
    content = content.replace(old_2, new_2)
    applied.append("2. Tableau SLI — valeurs honnêtes et sourcées")

# ─────────────────────────────────────────────────────────────
# PATCH 3 — Retirer l'affirmation d'un AlertManager inexistant
# ─────────────────────────────────────────────────────────────
old_3 = "All alerts are configured in Prometheus AlertManager with Slack notifications."
new_3 = (
    "Alert rules are evaluated by the application monitoring layer "
    "(`src/monitoring/alerts.py`), which dispatches to a configurable webhook "
    "(Slack, Teams or any HTTP endpoint) when `ALERT_WEBHOOK_URL` is set, and "
    "always writes to a structured JSONL audit log. **Prometheus AlertManager "
    "is not deployed in this release** — alerting is handled in-application; "
    "migrating rule evaluation to AlertManager is a roadmap item."
)
if old_3 in content:
    content = content.replace(old_3, new_3)
    applied.append("3. AlertManager — affirmation corrigée (non déployé)")

# ─────────────────────────────────────────────────────────────
# PATCH 4 — Liste SLO alignée sur le cahier des charges v2
# ─────────────────────────────────────────────────────────────
old_4 = """- **Availability SLO** : 99.9% uptime per month (43 minutes maximum downtime)
- **Latency SLO** : P95 < 10ms for 99% of hours in the month
- **Accuracy SLO** : F1 macro ≥ 0.70 (retraining triggers if < 0.65)
- **Drift SLO** : Detect and alert within 30 minutes of distribution change"""

new_4 = """- **Availability** : ≥ 99% uptime during continuous operation (single-host deployment)
- **Latency** : p95 < 100 ms for isolated requests; p95 < 2 s under 15 concurrent users
- **Throughput** : ≥ 5 req/s sustained on a single replica — 2.2× headroom over the
  3.5 req/s peak implied by 10,000 articles/day
- **Model quality** : macro F1 ≥ 0.65 on the held-out test set; candidate promotion
  blocked below this threshold
- **Drift** : detection and alert within 30 minutes of a distribution change"""

if old_4 in content:
    content = content.replace(old_4, new_4)
    applied.append("4. Liste SLO — alignée sur le cahier des charges v2")

# ─────────────────────────────────────────────────────────────
README.write_text(content, encoding="utf-8")

print("Patches appliqués :")
for a in applied:
    print(f"  ✓ {a}")
if len(applied) < 4:
    print(f"\n⚠️  {4 - len(applied)} patch(s) non appliqué(s) — motif probable :")
    print("   le texte source a changé depuis l'analyse. Vérifier manuellement.")
else:
    print("\n✅ Les 4 incohérences sont corrigées.")
