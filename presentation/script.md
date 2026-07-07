# SCRIPT COMPLET (≈10 minutes)

# Slide 1 — AI NewsOps Platform (≈1 min)
Ce que l'on voit

Architecture générale.

Ce que tu dis

Bonjour à toutes et à tous.

Aujourd'hui je vais vous présenter AI NewsOps Platform, une plateforme MLOps dédiée à la classification automatique d'articles de presse.

L'objectif du projet n'était pas uniquement de développer un modèle de Machine Learning performant, mais surtout de construire une plateforme capable d'être industrialisée.

J'ai donc travaillé sur l'ensemble du cycle de vie d'un modèle IA :

préparation des données,
entraînement,
versionnement,
déploiement,
monitoring,
détection de dérive,
et automatisation.

Autrement dit, j'ai cherché à reproduire les pratiques que l'on retrouve aujourd'hui dans les équipes MLOps.

Transition

Avant de parler technique, revenons d'abord au besoin métier.

# Slide 2 — Le problème métier (≈1 min)
Ce que tu dis

Chaque jour, les rédactions produisent et consultent des milliers d'articles.

Les journalistes doivent ensuite :

classer les contenus,
retrouver rapidement une information,
détecter les sujets similaires,
suivre l'actualité.

Toutes ces tâches sont chronophages.

L'idée est donc d'utiliser l'IA afin d'automatiser cette première étape de classification.

Le modèle n'a pas vocation à remplacer le journaliste.

Il sert à accélérer son travail.

Transition

Pour répondre à ce besoin, j'ai conçu une architecture modulaire.

# Slide 3 — Architecture (≈1 min 20)

Tu montres le schéma.

Tu racontes.

Les données sont d'abord collectées puis nettoyées.

Ensuite elles passent dans le pipeline de préparation.

Le modèle DistilBERT est entraîné.

Les expériences sont historisées dans MLflow.

Le modèle est ensuite exposé via une API FastAPI.

Toutes les métriques sont collectées par Prometheus.

Grafana permet de superviser le système.

Enfin Evidently surveille les dérives de données.

Chaque composant est indépendant.

Cette modularité facilite :

les tests,
le déploiement,
les évolutions futures.
Transition

Intéressons-nous maintenant au cœur du projet : le Machine Learning.

# Slide 4 — Pipeline Machine Learning (≈1 min)

Pourquoi DistilBERT ?

Parce qu'il offre un excellent compromis.

Je suis parti d'une baseline classique TF-IDF.

Cela m'a permis d'avoir un point de comparaison.

J'ai ensuite entraîné DistilBERT.

Toutes les expérimentations sont historisées dans MLflow.

Je peux ainsi comparer les performances, conserver les modèles et reproduire facilement un entraînement.

Transition

Mais un modèle performant ne suffit pas.

# Slide 5 — Industrialisation (≈1 min)

La principale difficulté d'un projet IA n'est pas l'entraînement.

C'est sa mise en production.

J'ai donc ajouté :

Docker afin d'assurer la reproductibilité.

FastAPI pour exposer le modèle.

GitHub Actions afin d'automatiser les tests.

MLflow pour le versionnement.

DVC pour la gestion des données.

L'objectif est que n'importe quel développeur puisse reproduire l'environnement.

Transition

Une fois en production, il faut également surveiller le système.

# Slide 6 — Monitoring (≈1 min)

L'observabilité est essentielle.

L'API expose ses métriques.

Prometheus les collecte.

Grafana les visualise.

J'ai également intégré Evidently.

Il permet de détecter une dérive des données.

Ainsi, si les données évoluent au fil du temps, le système est capable d'alerter qu'un réentraînement devient nécessaire.

Transition

Voyons maintenant les résultats obtenus.

# Slide 7 — Résultats (≈1 min)

Le pipeline fonctionne de bout en bout.

Les données sont préparées.

Le modèle est entraîné.

Les expériences sont historisées.

L'API répond.

Le monitoring est opérationnel.

Le projet comprend également :

une API documentée,
une interface Streamlit,
un monitoring complet.

Le résultat n'est donc pas seulement un modèle de Machine Learning.

C'est une plateforme IA complète.

Transition

Ce projet peut naturellement évoluer.

# Slide 8 — Roadmap (≈50 s)

L'architecture a été pensée afin d'être évolutive.

Les prochaines étapes seraient :

intégrer une base vectorielle comme Qdrant,
ajouter une couche RAG,
automatiser totalement le réentraînement,
déployer sur Kubernetes.

Aucune de ces évolutions ne nécessite de refondre l'architecture actuelle.

Transition

Pour conclure.

# Slide 9 — Conclusion (≈1 min)

Aujourd'hui, beaucoup de projets IA s'arrêtent au notebook.

J'ai souhaité aller plus loin.

Ce projet couvre tout le cycle de vie d'un modèle :

préparation,
entraînement,
déploiement,
supervision,
monitoring,
amélioration continue.

Au-delà de la performance du modèle, c'est cette vision globale de l'industrialisation qui constitue selon moi la principale valeur du projet.

Je vous remercie de votre attention.

Durée
Slide	Temps
Intro	1:00
Métier	1:00
Architecture	1:20
ML	1:00
Industrialisation	1:00
Monitoring	1:00
Résultats	1:00
Roadmap	0:50
Conclusion	0:50

Total : ≈ 9 min 55 s