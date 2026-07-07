"""
scripts/locustfile.py
Test de charge — AI NewsOps Platform
AIA Bloc 4 — MLOps Pipeline

Mesure le comportement réel du système sous charge : throughput,
latence p50/p95/p99, taux d'erreur, avec 1 vs 3 réplicas API.

Usage :
  pip install locust

  # Mode interactif (UI web sur http://localhost:8089)
  locust -f scripts/locustfile.py --host http://localhost:8000

  # Mode headless (résultats directs en console, pour un rapport reproductible)
  locust -f scripts/locustfile.py --host http://localhost:8000 \
      --headless -u 50 -r 5 -t 60s --csv=reports/load_test
"""

import random

from locust import HttpUser, task, between


HEADLINES = [
    "Senate votes on new climate legislation",
    "Lakers beat Warriors in overtime thriller",
    "Apple announces new iPhone with AI features",
    "Scientists discover new exoplanet in habitable zone",
    "Federal Reserve raises interest rates by 25 basis points",
    "New study links coffee consumption to longevity",
    "Local chef wins national cooking competition",
    "Stock markets rally on strong earnings reports",
    "New exhibition opens at the modern art museum",
    "Championship game ends in dramatic finish",
    "Tech company unveils breakthrough AI chip",
    "Wedding industry sees record growth this year",
]

DESCRIPTIONS = [
    "Experts weigh in on the implications for the coming year.",
    "The decision comes after months of negotiation.",
    "Analysts say this could reshape the industry landscape.",
    "",  # Certaines requêtes sans description, comme en usage réel
]


class NewsClassifierUser(HttpUser):
    """
    Simule un client réel de l'API de classification.
    Temps d'attente entre requêtes : 0.5 à 2s, comme un usage éditorial normal.
    """
    wait_time = between(0.5, 2.0)

    @task(10)
    def predict_single(self):
        """Prédiction simple — le cas d'usage le plus fréquent (poids 10)."""
        payload = {"headline": random.choice(HEADLINES)}
        if random.random() > 0.3:
            payload["short_description"] = random.choice(DESCRIPTIONS)

        with self.client.post(
            "/predict", json=payload, catch_response=True, name="/predict"
        ) as response:
            if response.status_code != 200:
                response.failure(f"Status {response.status_code}")
            else:
                data = response.json()
                if "category" not in data:
                    response.failure("Réponse sans champ 'category'")

    @task(2)
    def predict_batch(self):
        """Prédiction batch — usage moins fréquent mais plus coûteux (poids 2)."""
        n = random.randint(2, 10)
        articles = [{"headline": random.choice(HEADLINES)} for _ in range(n)]

        with self.client.post(
            "/predict/batch", json={"articles": articles},
            catch_response=True, name="/predict/batch",
        ) as response:
            if response.status_code != 200:
                response.failure(f"Status {response.status_code}")

    @task(3)
    def health_check(self):
        """Health check — simule les probes de monitoring externes (poids 3)."""
        self.client.get("/health", name="/health")

    @task(1)
    def metrics_scrape(self):
        """Simule un scrape Prometheus (poids 1, peu fréquent côté client)."""
        self.client.get("/metrics", name="/metrics")
