# SCRIPT INTÉGRAL – AI NewsOps Platform

Durée : 9 min 45 s

# SLIDE 1 — Construire une plateforme MLOps de classification d'articles

(≈ 1 min)

Bonjour à toutes et à tous.

Aujourd'hui, je vais vous présenter AI NewsOps Platform, une plateforme MLOps de classification automatique d'articles de presse.

Lorsque j'ai commencé ce projet, je me suis fixé un objectif volontairement plus ambitieux que simplement entraîner un modèle de Machine Learning.

Je voulais construire une plateforme complète, capable de couvrir tout le cycle de vie d'un modèle d'intelligence artificielle.

Autrement dit, je ne voulais pas seulement répondre à la question :

"Quel est le meilleur modèle ?"

mais également :

"Comment déployer ce modèle, le superviser, détecter lorsqu'il se dégrade et le faire évoluer dans le temps ?"

C'est cette vision MLOps qui a guidé toutes les décisions d'architecture du projet.

## Transition

Avant de parler technique, revenons quelques instants sur le besoin métier auquel cette plateforme répond.

# SLIDE 2 — Le problème métier

(≈ 1 min)

Les médias et les rédactions produisent aujourd'hui un volume considérable d'informations.

Trier, classer et retrouver rapidement un article devient rapidement une tâche chronophage.

L'objectif du projet est donc de proposer une première couche d'intelligence capable de classifier automatiquement un article dès son arrivée.

Cette classification facilite ensuite plusieurs usages :

la recherche d'information,
l'organisation éditoriale,
l'analyse statistique,
et, à terme, l'alimentation d'un assistant conversationnel basé sur le RAG.

Il est important de préciser que le système n'a pas vocation à remplacer le journaliste.

Son rôle est de l'assister en automatisant les tâches répétitives afin qu'il puisse consacrer davantage de temps à la création de contenu.

## Transition

Pour répondre à ce besoin, j'ai conçu une architecture modulaire composée de plusieurs services indépendants.

# SLIDE 3 — Architecture générale

(≈ 1 min 20)

Cette slide représente l'architecture globale de la plateforme.

Le pipeline commence par la préparation des données.

Les articles sont nettoyés, normalisés puis répartis en jeux d'entraînement, de validation et de test.

Le modèle DistilBERT est ensuite entraîné.

Toutes les expériences sont automatiquement enregistrées dans MLflow afin de conserver les paramètres, les métriques et les artefacts de chaque entraînement.

Une fois validé, le modèle est servi via une API FastAPI documentée automatiquement avec OpenAPI.

Enfin, la partie exploitation repose sur Prometheus, Grafana et Evidently afin de superviser la plateforme et de détecter les éventuelles dérives de données.

Cette séparation des responsabilités permet de faire évoluer chaque composant indépendamment sans remettre en cause l'ensemble de l'architecture.

## Transition

Regardons maintenant le cœur de la plateforme : le pipeline de Machine Learning.

# SLIDE 4 — Pipeline Machine Learning

(≈ 1 min)

Avant d'entraîner DistilBERT, j'ai commencé par construire une baseline classique basée sur TF-IDF et un classifieur linéaire.

Cette étape est importante car elle fournit un point de comparaison objectif.

J'ai ensuite entraîné DistilBERT, qui offre un excellent compromis entre performances et coût de calcul.

L'ensemble des expérimentations est historisé dans MLflow.

Cela permet de comparer facilement plusieurs modèles, de reproduire un entraînement et de revenir à une version antérieure si nécessaire.

Cette approche est essentielle lorsqu'un projet passe d'un prototype à une utilisation en équipe.

## Transition

Un modèle performant ne suffit cependant pas à construire une plateforme exploitable.

# SLIDE 5 — Industrialisation

(≈ 1 min)

L'un des objectifs majeurs de ce projet était sa reproductibilité.

Pour cela, l'environnement est entièrement conteneurisé avec Docker.

Les données sont versionnées.

Les modèles sont historisés.

Les tests sont exécutés automatiquement via GitHub Actions.

L'API est documentée et chaque service possède une responsabilité clairement définie.

Cette organisation permet à un nouveau développeur de reconstruire rapidement l'environnement et limite fortement les effets liés aux différences de configuration locale.

## Transition

Une fois le système déployé, une nouvelle problématique apparaît : comment savoir qu'il continue de fonctionner correctement ?

# SLIDE 6 — Monitoring et observabilité

(≈ 1 min 20)

La supervision est un aspect souvent négligé dans les projets IA.

Pourtant, un modèle peut se dégrader progressivement alors que l'infrastructure continue de fonctionner parfaitement.

L'API expose donc des métriques compatibles Prometheus.

Ces métriques sont collectées automatiquement puis visualisées dans Grafana.

J'ai également intégré Evidently afin d'analyser les données entrantes et détecter d'éventuelles dérives statistiques.

L'objectif est de pouvoir identifier le moment où un réentraînement devient nécessaire avant que la qualité des prédictions ne diminue significativement.

Cette approche permet de passer d'un modèle statique à un système capable d'évoluer dans le temps.

## Transition

Regardons maintenant ce que cette architecture permet d'obtenir.

# SLIDE 7 — Résultats

(≈ 1 min)

Au-delà des performances du modèle, je souhaitais surtout obtenir une plateforme cohérente et industrialisable.

Aujourd'hui, le pipeline couvre l'ensemble du cycle de vie :

les données sont préparées,

les modèles sont entraînés,

les expérimentations sont historisées,

l'API est déployée,

les métriques sont supervisées,

et la dérive des données est surveillée.

Ce projet ne se limite donc plus à un notebook de Data Science.

Il constitue une première version d'une véritable plateforme MLOps.

## Transition

Cette architecture a également été pensée pour évoluer.

# SLIDE 8 — Perspectives

(≈ 50 s)

Plusieurs évolutions sont déjà identifiées.

La première consiste à intégrer une base vectorielle afin d'ajouter une recherche sémantique.

Cette évolution permettra ensuite de construire une architecture RAG capable de répondre à des questions en langage naturel à partir du corpus d'articles.

À plus long terme, le déploiement sur Kubernetes et l'automatisation complète du réentraînement permettront d'augmenter la résilience et la scalabilité de la plateforme.

L'intérêt de l'architecture actuelle est qu'elle a été conçue pour accueillir ces évolutions sans remise en cause majeure.

## Transition

Je vais conclure en revenant sur ce que j'ai cherché à démontrer avec ce projet.

# SLIDE 9 — Conclusion

(≈ 1 min)

Ce projet m'a permis de dépasser le cadre d'un simple exercice de Machine Learning.

Il m'a conduit à réfléchir à l'ensemble du cycle de vie d'un modèle, depuis la préparation des données jusqu'à son exploitation en production.

J'ai ainsi abordé des problématiques de reproductibilité, de supervision, de monitoring, d'automatisation et d'architecture logicielle.

Si je devais résumer ce projet en une seule phrase, je dirais qu'il ne s'agit pas seulement d'un modèle de classification, mais d'une plateforme MLOps pensée pour être maintenable, observable et évolutive.

Je vous remercie de votre attention et je serai ravi de répondre à vos questions.

---

Trois conseils qui feront une vraie différence le jour J
Ne parle jamais des outils avant le problème. Le jury doit comprendre pourquoi chaque technologie est présente. Tu n'as pas "mis Grafana parce que c'était au programme", tu l'as utilisé pour répondre à un besoin d'observabilité.
Ne récite pas les métriques. Cite uniquement les chiffres que tu peux justifier (taille du dataset, nombre de classes, principales performances, couverture de tests si elle est à jour). Si on te demande des détails, tu pourras ouvrir MLflow ou Grafana pendant la démo.
Conclue toujours sur la vision d'ensemble. Ce qui distingue ton projet est moins le choix de DistilBERT que la cohérence de la plateforme MLOps : préparation des données, entraînement, versionnement, déploiement, supervision et amélioration continue. C'est ce message que le jury doit retenir.