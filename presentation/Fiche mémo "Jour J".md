# AI NewsOps Platform — Fiche mémo "Jour J"

> **Objectif :** démontrer la maîtrise d'une plateforme MLOps complète, pas seulement d'un modèle de Machine Learning.

---

# 🚀 Avant de partir

- [ ] Ordinateur chargé
- [ ] Chargeur
- [ ] Souris
- [ ] Adaptateur HDMI / USB-C
- [ ] Connexion Internet (si nécessaire)
- [ ] Présentation (.pptx + PDF)
- [ ] Projet Git à jour (`git status` propre)
- [ ] Plan B (captures d'écran disponibles)

---

# 🚀 Vérification de l'environnement

```bash
docker compose up -d
docker ps
```

Tous les conteneurs doivent être **Up**.

---

# 🚀 Vérification des services

## API

```bash
curl http://localhost:8000/health
```

Attendu :

```json
{
  "status": "ok"
}
```

---

## Monitoring

```bash
curl http://localhost:8000/monitoring/drift
```

Attendu :

```json
{
  "status": "ok"
}
```

---

## Métriques Prometheus

```bash
curl http://localhost:8000/metrics | head
```

---

## Prometheus

```
http://localhost:9090
```

Vérifier :

```
up
```

---

## Grafana

```
http://localhost:3000
```

Dashboard chargé.

---

## MLflow

```
http://localhost:5000
```

Les runs sont visibles.

---

## Swagger

```
http://localhost:8000/docs
```

Tous les endpoints sont disponibles.

---

# 🎤 Déroulé de la soutenance

## 1. Contexte

Pourquoi ce projet ?

> Construire une plateforme MLOps de classification d'articles de presse.

---

## 2. Besoin métier

Automatiser la catégorisation.

Aider les journalistes.

Préparer un futur système RAG.

---

## 3. Architecture

Expliquer le pipeline :

```
Data

↓

Préprocessing

↓

Training

↓

MLflow

↓

FastAPI

↓

Prometheus

↓

Grafana

↓

Evidently
```

---

## 4. Machine Learning

Expliquer :

- baseline
- DistilBERT
- MLflow

---

## 5. Industrialisation

Montrer :

- Docker
- API
- CI
- versionnement
- monitoring

---

## 6. Démonstration

Ordre :

1. Swagger

2. Prediction

3. MLflow

4. Prometheus

5. Grafana

6. Drift

---

## 7. Conclusion

Message final :

> Ce projet démontre la mise en œuvre d'une plateforme MLOps complète couvrant le cycle de vie d'un modèle de NLP, depuis les données jusqu'à la supervision en production.

---

# 📌 Les messages clés

Le jury doit retenir :

✅ Ce n'est pas un notebook.

✅ Ce n'est pas uniquement un modèle.

✅ C'est une plateforme.

---

# 💬 Les phrases fortes

Ne pas dire :

> J'ai utilisé Docker.

Dire :

> J'ai conteneurisé l'application afin de garantir la reproductibilité de l'environnement.

---

Ne pas dire :

> J'ai MLflow.

Dire :

> Toutes les expérimentations sont historisées afin d'assurer la traçabilité des modèles.

---

Ne pas dire :

> J'ai Grafana.

Dire :

> La plateforme est observable en temps réel grâce à Prometheus et Grafana.

---

Ne pas dire :

> J'ai Evidently.

Dire :

> Le système surveille automatiquement les dérives statistiques des données afin d'anticiper une baisse des performances du modèle.

---

# ⚠️ À éviter

❌ Ouvrir VS Code

❌ Montrer le code

❌ Faire défiler des fichiers

❌ Chercher une commande

❌ Dire :

> "Normalement ça marche..."

---

# 🛟 Plan B

Si Grafana est indisponible :

→ Montrer Prometheus.

---

Si MLflow est indisponible :

→ Montrer les artefacts et expliquer le versionnement.

---

Si Internet est indisponible :

→ Tout fonctionne localement avec Docker.

---

# 🎯 Questions probables

- Pourquoi DistilBERT ?
- Pourquoi FastAPI ?
- Pourquoi Docker ?
- Pourquoi MLflow ?
- Pourquoi Prometheus ?
- Pourquoi Grafana ?
- Pourquoi Evidently ?
- Pourquoi une baseline ?
- Comment gérer le drift ?
- Comment passer à l'échelle ?

---

# ⏱️ Répartition du temps

| Partie | Durée |
|---------|-------|
| Introduction | 1 min |
| Problème métier | 1 min |
| Architecture | 1 min 30 |
| Pipeline ML | 1 min |
| Industrialisation | 1 min |
| Monitoring | 1 min |
| Résultats | 1 min |
| Roadmap | 1 min |
| Conclusion | 30 s |

**Total : ~9 min 30**

---

# 🎯 Message final

> Ce projet ne démontre pas seulement ma capacité à entraîner un modèle de Machine Learning. Il montre ma capacité à concevoir une plateforme MLOps complète, reproductible, observable et évolutive, en appliquant des pratiques proches de celles utilisées en environnement de production.
