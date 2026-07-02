# Guide d'utilisation de l'API NewsOps

## Endpoints disponibles

| Endpoint              | Méthode   | Description                   |
|-----------------------|-----------|-------------------------------|
| `/health`             | GET       | Vérifier le statut du service |
| `/predict`            | POST      | Classifier un article         |
| `/predict/batch`      | POST      | Classifier plusieurs articles |
| `/monitoring/drift`   | GET       | Vérifier la dérive du modèle  |
| `/metrics`            | GET       | Métriques Prometheus          |

## Exemple d'utilisation

### Prédiction simple

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "headline": "Congress passes new climate bill",
    "short_description": "The Senate voted 51-49 in favor"
  }'