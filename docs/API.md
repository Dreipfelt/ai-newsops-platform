# AI NewsOps Platform — API Documentation

**Version** : 1.0.0  
**Base URL** : `http://localhost:8000`  
**Format** : JSON  
**Auth** : Aucune (déploiement local/interne)

---

## Vue d'ensemble

L'API AI NewsOps classifie automatiquement des articles de presse dans **13 catégories** grâce à un modèle DistilBERT fine-tuné sur le HuffPost News Archive (208k articles, 2012-2022).

### Catégories disponibles

| ID | Catégorie | Description |
|---|---|---|
| 0 | `arts_culture` | Arts, culture, spectacles |
| 1 | `business` | Business, finance, économie |
| 2 | `crime` | Faits divers, justice |
| 3 | `entertainment` | Divertissement, célébrités |
| 4 | `family_education` | Famille, parentalité, éducation |
| 5 | `health_wellness` | Santé, bien-être, nutrition |
| 6 | `international` | International, religion |
| 7 | `lifestyle` | Style de vie, voyage, food |
| 8 | `media` | Médias, voix minoritaires |
| 9 | `other` | Mariage, divorce, divers |
| 10 | `politics` | Politique, actualités US |
| 11 | `sports` | Sports |
| 12 | `tech_science` | Tech, science, environnement |

---

## Endpoints

### `GET /`
Point d'entrée — informations générales sur l'API.

**Réponse**
```json
{
  "name": "AI NewsOps Platform API",
  "version": "1.0.0",
  "status": "running",
  "docs": "/docs",
  "health": "/health",
  "model_loaded": true,
  "device": "cpu"
}
```

---

### `GET /health`
Vérifie que l'API et le modèle sont opérationnels.

**Réponse 200**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu",
  "tokenizer_loaded": true,
  "monitor_initialized": true
}
```

**Codes de retour**
| Code | Description |
|---|---|
| 200 | API et modèle opérationnels |
| 503 | Modèle non chargé |

---

### `POST /predict`
Classifie un article dans l'une des 13 catégories.

**Body**
```json
{
  "headline": "string (5-500 chars, obligatoire)",
  "short_description": "string (max 1000 chars, optionnel)"
}
```

**Exemple de requête**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "headline": "Senate votes on new climate legislation",
    "short_description": "Democrats and Republicans reach compromise on green energy."
  }'
```

**Réponse 200**
```json
{
  "category": "politics",
  "confidence": 0.9957,
  "top3": [
    ["politics", 0.9957],
    ["tech_science", 0.0013],
    ["media", 0.0008]
  ],
  "model_version": "1.0.0"
}
```

**Codes de retour**
| Code | Description |
|---|---|
| 200 | Prédiction réussie |
| 422 | Validation échouée (headline trop court/long) |
| 503 | Modèle non chargé |
| 500 | Erreur interne |

---

### `POST /predict/batch`
Classifie jusqu'à **100 articles** en une seule requête.

**Body**
```json
{
  "articles": [
    {"headline": "Article 1 headline"},
    {"headline": "Article 2 headline", "short_description": "Description..."}
  ]
}
```

**Exemple**
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "articles": [
      {"headline": "Lakers beat Warriors in overtime"},
      {"headline": "Apple announces new MacBook Pro with M4 chip"},
      {"headline": "Senate passes infrastructure bill"}
    ]
  }'
```

**Réponse 200**
```json
{
  "predictions": [
    {"category": "sports",      "confidence": 0.976, "top3": [...], "model_version": "1.0.0"},
    {"category": "tech_science","confidence": 0.963, "top3": [...], "model_version": "1.0.0"},
    {"category": "politics",    "confidence": 0.991, "top3": [...], "model_version": "1.0.0"}
  ]
}
```

---

### `GET /metrics`
Retourne les métriques Prometheus (format texte scrape-ready).

```bash
curl http://localhost:8000/metrics
```

**Réponse** (texte Prometheus)
```
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="POST",endpoint="/predict",status="200"} 42.0

# HELP http_request_duration_seconds HTTP request latency
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{method="POST",endpoint="/predict",le="0.1"} 38.0

# HELP model_f1_score Model F1 score
# TYPE model_f1_score gauge
model_f1_score 0.6805

# HELP model_drift_score Current model drift score
# TYPE model_drift_score gauge
model_drift_score 0.0
```

---

### `GET /monitoring/health`
Statut complet du système de monitoring.

**Réponse 200**
```json
{
  "status": "healthy",
  "reference_data": true,
  "drift_logs_count": 5,
  "summary": {
    "total": 5,
    "recent_drifts": 0,
    "last_check": {"timestamp": "...", "drift_score": 0.0},
    "last_drift": null
  },
  "evidently_available": true
}
```

---

### `GET /monitoring/drift`
Retourne le dernier résultat de détection de drift.

```bash
curl http://localhost:8000/monitoring/drift
```

**Réponse 200**
```json
{
  "timestamp": "2026-07-03T11:24:25",
  "n_drifted": 0,
  "n_features": 3,
  "drift_share": 0.0,
  "alert_level": "ok",
  "features": {
    "text_length": {"test": "kolmogorov-smirnov", "p_value": 0.95, "drift_detected": false},
    "word_count":  {"test": "kolmogorov-smirnov", "p_value": 0.87, "drift_detected": false},
    "category_distribution": {"test": "chi-squared", "p_value": 0.72, "drift_detected": false}
  }
}
```

---

### `GET /monitoring/drift/run`
Lance une détection de drift en temps réel.

```bash
curl http://localhost:8000/monitoring/drift/run
```

---

### `GET /monitoring/drift/logs`
Historique des détections de drift.

```bash
curl "http://localhost:8000/monitoring/drift/logs?limit=10"
```

**Paramètres**
| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `limit` | int | 10 | Nombre de logs à retourner |

---

## Codes d'erreur globaux

| Code | Description |
|---|---|
| 200 | Succès |
| 422 | Validation Pydantic échouée (body invalide) |
| 500 | Erreur serveur interne |
| 503 | Service indisponible (modèle non chargé) |

---

## Exemples complets

### Python
```python
import requests

BASE_URL = "http://localhost:8000"

# Prédiction simple
response = requests.post(f"{BASE_URL}/predict", json={
    "headline": "SpaceX successfully launches Starship to orbit",
    "short_description": "Elon Musk's company completes full orbital test flight."
})
print(response.json())
# {"category": "tech_science", "confidence": 0.94, ...}

# Batch
articles = [
    {"headline": "Fed raises interest rates by 25 basis points"},
    {"headline": "NFL playoffs: Chiefs defeat Ravens in thriller"},
    {"headline": "New study links coffee consumption to longevity"},
]
response = requests.post(f"{BASE_URL}/predict/batch", json={"articles": articles})
for pred in response.json()["predictions"]:
    print(f"{pred['category']:20} {pred['confidence']:.3f}")
```

### JavaScript
```javascript
const predict = async (headline, description = '') => {
  const res = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({headline, short_description: description})
  });
  return res.json();
};

predict('Apple launches Vision Pro 2', 'New mixed reality headset with improved battery.')
  .then(r => console.log(r.category, r.confidence));
```

---

## Performance

| Métrique | Valeur |
|---|---|
| Latence p50 | ~75ms |
| Latence p95 | ~120ms |
| Throughput | ~13 req/s (CPU) |
| Modèle | DistilBERT 66M params |
| Dataset entraînement | 208k articles |
| F1 macro test | 0.68 |
| Accuracy test | 0.74 |

---

## Documentation interactive

Swagger UI disponible sur : **`http://localhost:8000/docs`**  
ReDoc disponible sur : **`http://localhost:8000/redoc`**
