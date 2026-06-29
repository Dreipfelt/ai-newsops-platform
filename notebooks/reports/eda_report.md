# EDA Report — News Category Dataset
*Projet AI NewsOps Platform · AIA Bloc 4*

## Dataset
| Métrique       | Valeur                                                       |
|----------------|--------------------------------------------------------------|
| Source         | HuffPost News Archive (Kaggle)                               |
| Articles bruts | 209,527                                                      |
| Colonnes       | link, headline, category, short_description, authors, date   |
| Période        | 2012-01-28 → 2022-09-23                                      |
| Mémoire        | 132.1 MB                                                     |

## Qualité des données
- Doublons (headline + description) : **489**
- Valeurs manquantes : **aucune NaN critique** (descriptions vides → chaîne vide)
- Outliers longueur (IQR) : **6,564** (3.1%)
- Headlines ambiguës (même texte, catégorie différente) : **78**

## Classes
- Catégories originales : **42**
- Super-catégories retenues : **12**
- Ratio déséquilibre avant : **35x**
- Ratio déséquilibre après fusion : **12.7x**

## Features textuelles
| Feature                       | Médiane   | Moyenne   | Max   |
|-------------------------------|-----------|-----------|-------|
| Longueur headline (chars)     | 60        | 58        | 320   |
| Longueur description (chars)  | 120       | 114       | 1472  |
| Mots totaux                   | 28        | 29        | 245   |
| Tokens approx. DistilBERT     | 36        | 38        | 318   |

## Dérive temporelle
- Test χ² indépendance catégorie~année : χ²=106,062.0, **p=0.00e+00**
- Conclusion : Dérive significative → monitoring drift indispensable en production

## Décisions de preprocessing
1. **Fusion 42→15 catégories** : réduction du ratio déséquilibre de 35x à 12.7x
2. **Texte = headline [SEP] description** : exploite le token [SEP] natif de BERT
3. **max_length=128** : couvre >95% des textes, réduit la mémoire GPU par rapport à 512
4. **Split 70/15/15 stratifié** : garantit la représentation de chaque classe
5. **Versioning DVC** : reproductibilité totale, rollback possible

## Figures générées
- `01_category_distribution.png` — Distribution des 42 catégories
- `02_category_fusion_before_after.png` — Impact de la fusion
- `03_temporal_analysis.png` — Volume par année + heatmap
- `04_text_length_analysis.png` — Distributions longueurs texte
- `05_tfidf_heatmap.png` — Termes discriminants par catégorie
- `06_temporal_drift.png` — Évolution proportions dans le temps
- `07_correlation_analysis.png` — Corrélations features numériques
