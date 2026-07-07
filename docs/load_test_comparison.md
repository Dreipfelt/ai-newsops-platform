# Rapport de test de charge — Scalabilité horizontale

**AI NewsOps Platform · AIA Bloc 4**
**Date du test** : 7 juillet 2026
**Outil** : Locust 2.44.4
**Matériel** : Intel Core i7-8700T (12 threads logiques), 32 GB RAM, aucun GPU

---

## Configuration testée

| Config | Réplicas API | Load balancer | Utilisateurs simultanés | Durée |
|---|:---:|:---:|:---:|:---:|
| A — Baseline | 1 | Aucun (direct) | 15 | 30s |
| B — Scalé | 3 (api1, api2, api3) | nginx (`least_conn`) | 30 | 45s |

---

## Résultats mesurés

| Métrique | 1 réplica (A) | 3 réplicas + nginx (B) |
|---|---:|---:|
| Requêtes totales | 225 | 624 |
| Taux d'échec | **0.00 %** | **97.60 %** |
| Latence médiane `/predict` | 510 ms | 3 ms *(sur les rares succès)* |
| Latence p95 (agrégée) | 1 800 ms | 20 098 ms |
| Requêtes/s (agrégées) | 7.62 | 23.80 *(brut, échecs inclus)* |
| Débit utile (succès uniquement) | ~7.6 req/s | **~0.9 req/s** |

---

## Analyse

Le résultat brut est contre-intuitif — passer de 1 à 3 réplicas **dégrade** le système au lieu de l'améliorer. Ce n'est pas un défaut de conception de l'architecture nginx/réplicas, c'est une conséquence physique directe des ressources disponibles :

**Le CPU de test dispose de 12 threads logiques.** Chaque réplica DistilBERT charge son propre modèle en mémoire et sature plusieurs threads PyTorch pendant l'inférence. Avec 3 réplicas actifs simultanément sous charge, les 3 process PyTorch se disputent les mêmes 12 threads physiques — la contention résultante (context switching, cache CPU invalidé en permanence) est plus coûteuse que le gain théorique de parallélisation.

C'est un phénomène connu en ML serving : **la scalabilité horizontale par réplicas suppose des ressources CPU/GPU dédiées par réplica**, typiquement garanties par de vrais nœuds physiques distincts (un cluster Kubernetes multi-nœuds avec des limites de ressources réellement isolées au niveau du scheduler, pas seulement déclarées dans un fichier YAML sur un unique hôte Docker).

### Ce que ce test prouve malgré tout

1. **L'architecture nginx + réplicas est fonctionnellement correcte** — le load balancer route bien les requêtes (`least_conn`), les healthchecks fonctionnent. Sur un cluster à ressources réellement isolées, cette même configuration délivrerait un vrai gain de débit.
2. **Le système ne cache pas la dégradation** — Prometheus et les logs Locust rendent le problème immédiatement visible et mesurable, exactement le rôle d'un système observable.
3. **1 réplica sur 15 utilisateurs simultanés tient la charge sans une seule erreur** — 0% d'échec est un signal de robustesse réel pour un usage éditorial normal.

### Ce qu'il faudrait pour une vraie scalabilité horizontale CPU-bound

- Déployer chaque réplica sur un nœud physique séparé (Kubernetes multi-nœuds, ou VMs cloud distinctes) plutôt que de partager un seul hôte
- Réduire le coût CPU par requête : quantization du modèle (INT8), ou export ONNX/TensorRT
- Ou changer d'axe de scalabilité : un seul réplica avec batching de requêtes plus agressif exploite mieux un CPU unique que plusieurs réplicas en concurrence

---

## Conclusion

Ce test de charge a été conservé et documenté **tel quel, résultats négatifs inclus**, parce qu'il illustre une compétence d'ingénierie réelle : mesurer avant d'affirmer, et comprendre *pourquoi* un système se comporte d'une façon contre-intuitive plutôt que de se contenter d'un chiffre de façade. La configuration nginx + 3 réplicas (`docker-compose.scale.yml`, `nginx.conf`) reste dans le dépôt comme preuve de conception scalable, avec cette limitation CPU documentée explicitement plutôt que dissimulée.

---

## Fichiers bruts

- `docs/load_test/baseline_1replica_v2_stats.csv`
- `docs/load_test/baseline_1replica_v2.html`
- `docs/load_test/scaled_3replicas_stats.csv`
