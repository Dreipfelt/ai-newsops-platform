# GUIDE DE DÉMONSTRATION — AI NewsOps Platform
Durée : 3 min 30 à 4 min

## Les objectifs

À la fin de la démonstration, le jury doit pouvoir répondre "oui" aux questions suivantes :

✔ Le modèle fonctionne.

✔ Il est exposé par une API.

✔ Les expérimentations sont historisées.

✔ Le système est monitoré.

✔ Les dérives sont surveillées.

✔ L'architecture est industrialisable.

Tu n'as besoin de démontrer rien d'autre.

# Avant la soutenance (checklist)

Lance tous les services 30 minutes avant.

docker compose up -d

Vérifie :

docker ps

Tous les conteneurs doivent être Up.

Swagger
http://localhost:8000/docs
MLflow
http://localhost:5000
Prometheus
http://localhost:9090
Grafana
http://localhost:3000
Streamlit
http://localhost:8501
Vérifie le monitoring
curl http://localhost:8000/metrics
Vérifie le drift
curl http://localhost:8000/monitoring/drift

Tout doit répondre avant la soutenance.

Démonstration

## Étape 1
30 secondes

Tu commences directement par Swagger.

Tu dis :

Je vais commencer par montrer que le modèle est réellement déployé.

Ouvre

/docs

Le jury voit immédiatement :

FastAPI

OpenAPI

Endpoints

Tu inspires confiance.

## Étape 2
40 secondes

Tu fais une prédiction.

Tu prends un article.

Tu cliques sur Execute.

Le modèle renvoie :

politics

0.97

Tu dis :

Ici le modèle répond via une API REST exactement comme il le ferait dans un environnement de production.

Ne parle plus.

Passe à la suite.

## Étape 3
30 secondes

MLflow.

Tu ouvres :

localhost:5000

Tu montres :

les runs
les paramètres
les métriques
les artefacts

Tu dis :

Chaque entraînement est historisé afin de pouvoir comparer plusieurs versions d'un modèle et revenir facilement à une version précédente.

Fin.

## Étape 4
40 secondes

Prometheus.

localhost:9090

Tu montres

up

Puis

model_drift_score

Tu expliques :

Prometheus collecte automatiquement les métriques exposées par l'API.

Ne montre rien d'autre.

## Étape 5
45 secondes

Grafana.

Tu montres :

Dashboard.

Tu expliques :

Grafana permet d'avoir une vision temps réel de l'état de la plateforme.

Montre :

requêtes
temps de réponse
drift
santé

Tu ne fais pas défiler.

## Étape 6
30 secondes

Monitoring.

curl

/monitoring/drift

Résultat.

status

ok

Tu dis :

Evidently compare les données actuelles aux données de référence afin de détecter une éventuelle dérive.

Fin.

## Étape 7
20 secondes

Tu reviens sur le PowerPoint.

Dernière slide.

Conclusion.

Ce qu'il ne faut JAMAIS faire

Ne jamais :

ouvrir VS Code.

Ne jamais :

montrer le code.

Ne jamais :

faire du scrolling.

Ne jamais :

chercher un fichier.

Ne jamais :

taper une commande pendant 30 secondes.

Ne jamais :

dire

"normalement ça marche..."

Ne jamais :

ouvrir Docker Desktop.

Le jury s'en moque.

## Plan B

Supposons.

Grafana plante.

Tu dis.

Les métriques continuent d'être collectées par Prometheus.

Grafana n'est qu'une couche de visualisation.

Le jury est rassuré.

Supposons.

MLflow plante.

Tu montres :

mlruns/

et tu expliques.

Supposons.

Internet coupe.

Tout fonctionne en local.

C'est un énorme avantage.

Astuce

La meilleure démo n'est pas celle où tu montres le plus de choses.

C'est celle où :

le jury ne voit jamais de temps mort.

Déroulé final
30 s

Swagger

↓

40 s

Prediction

↓

30 s

MLflow

↓

40 s

Prometheus

↓

45 s

Grafana

↓

30 s

Drift

↓

20 s

Conclusion

Total

≈ 3 min 55

Les 5 phrases qui impressionnent un jury

Au lieu de dire :

"J'ai utilisé Docker."

Dis :

### "J'ai privilégié une architecture conteneurisée afin de garantir la reproductibilité de l'environnement."

Au lieu de dire :

"J'ai MLflow."

Dis :

### "Chaque expérimentation est historisée afin d'assurer la traçabilité des modèles."

Au lieu de dire :

"J'ai Grafana."

Dis :

### "La plateforme est observable en temps réel grâce à une chaîne Prometheus–Grafana."

Au lieu de dire :

"J'ai Evidently."

Dis :

### "Le système est capable d'identifier une dérive statistique des données avant qu'elle n'impacte significativement la qualité des prédictions."

Au lieu de dire :

"J'ai une API."

Dis :

### "Le modèle est servi via une API REST documentée automatiquement et directement exploitable par une application cliente."

Mon retour de "Lead"

En regardant tout ce que tu as construit depuis plusieurs semaines, je pense que tu n'as pas un projet de classification d'articles.

Tu as construit une plateforme MLOps.

Et c'est exactement cela qu'il faut démontrer.

Pendant la démo, ne cherche pas à prouver que ton modèle est intelligent. Le jury verra surtout qu'il est déployé, traçable, monitoré et maintenable. C'est cette maturité qui différencie un projet de Data Science d'une plateforme prête pour un contexte professionnel.