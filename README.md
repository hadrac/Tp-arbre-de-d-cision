# TP — Détection de fraude bancaire par arbres de décision

> **Contexte :** Ce projet a été réalisé dans le cadre d'un TP de Data Mining. L'objectif est de prédire si une transaction bancaire est frauduleuse ou légitime à partir de ses caractéristiques, en utilisant un algorithme d'arbre de décision (CART).

---

## Dataset utilisé

**Fichier :** `Fraud_Detection_Dataset.csv`

## Structure du projet

```
├── Fraud_Detection_Dataset.csv          
├── TP_Arbre_de_decision.ipynb          
└── README.md                            
```

---

##  Étapes réalisées

### Phase 0 — Imports

Chargement des librairies nécessaires : `numpy`, `pandas`, `matplotlib`, `seaborn`,et `scikit-learn` .

---

### Phase 1 — Business Understanding

**Question métier :** Peut-on prédire, à partir des caractéristiques d'une transaction bancaire (montant, type, appareil, localisation, historique utilisateur), si cette transaction est frauduleuse, afin d'alerter automatiquement la banque en temps réel ?

**Traduction ML :**
- Type de problème : Classification binaire
- Variable cible Y : `Fraudulent`

---

### Phase 2 — Data Understanding

**2.1 Chargement :** 51 000 lignes × 12 colonnes chargées avec `pd.read_csv`.

**2.2 Exploration initiale :** Vérification des types, aperçu des premières lignes, identification des variables catégorielles et numériques.

**2.3 Distribution de la variable cible :**
- Légitime (0) : 48 499 transactions — 95,10 %
- Fraude (1) : 2 501 transactions — 4,90 %


**2.4 Valeurs manquantes :** Environ 5 % de valeurs manquantes détectées sur plusieurs colonnes (`Transaction_Amount`, `Time_of_Transaction`, `Device_Used`, `Location`, `Payment_Method`).

**2.5 Statistiques descriptives :** Les variables V numériques affichent des distributions variées. `Transaction_Amount` présente des valeurs extrêmes côté fraude.

**2.6 Analyse catégorielle :** Visualisation du taux de fraude par modalité sur `Transaction_Type`, `Device_Used`, `Payment_Method`. 

**2.7 Variables numériques vs fraude :** Boxplots comparatifs — `Transaction_Amount`, `Time_of_Transaction` et `Account_Age` présentent des distributions légèrement différentes entre les deux classes.

**2.8 Corrélations :** La matrice de corrélation révèle que `Previous_Fraudulent_Transactions` et `Transaction_Amount` sont les variables numériques les plus corrélées à la cible `Fraudulent`.

---

### Phase 3 — Data Preparation

**3.1 Suppression des colonnes non prédictives :** `Transaction_ID` et `User_ID` ont été supprimés car ce sont de simples identifiants sans valeur prédictive pour le modèle.

**3.2 Traitement des valeurs manquantes :**
- Variables numériques (`Transaction_Amount`, `Time_of_Transaction`) → imputation par la **médiane** (robuste aux valeurs extrêmes)
- Variables catégorielles (`Device_Used`, `Location`, `Payment_Method`) → imputation par le **mode** (valeur la plus fréquente)

**3.3 Encodage des variables catégorielles :**  
Utilisation du **One-Hot Encoding** (`pd.get_dummies`) sur `Transaction_Type`, `Device_Used`, `Location`, `Payment_Method`.


**3.4 Séparation X / y :** X contient toutes les colonnes après encodage sauf `Fraudulent`. y contient la colonne cible.

**3.5 Split train / test (80/20) :**
- `stratify=y` utilisé pour conserver les proportions de classes dans chaque split
- `random_state=42` pour assurer le mélange dans les splits
- Train : 40 800 lignes | Test : 10 200 lignes

---

### Phase 4 — Modeling

Trois modèles ont été entraînés, tous avec `class_weight='balanced'` pour compenser le déséquilibre des classes.

| Modèle | Critère | max_depth |
|--------|---------|-----------|
| Modèle 1 | Gini | 3 |
| Modèle 2 | Entropie | 3 |
| Modèle 3 | Gini | Aucune limite |

**Exploration de max_depth (1 à 15) :** Le recall fraude sur le test est maximal à **max_depth = 10** (recall ≈ 0.58).

---

### Phase 5 — Evaluation

#### Comparaison des trois modèles

| Modèle | Accuracy Train | Accuracy Test | Écart |
|--------|---------------|--------------|-------|
| Gini (depth=3) | 0.6498 | 0.6404 | 0.0094 |
| Entropie (depth=3) | 0.5641 | 0.5588 | 0.0053 |
| Sans limite | **1.0000** | 0.9122 | **0.0878** |

**Analyse du surapprentissage :** Le modèle sans contrainte présente un écart de 0.0878 entre train et test — il mémorise les données d'entraînement (accuracy parfaite de 1.0) mais généralise mal. Les modèles avec `max_depth=3` affichent des écarts quasi nuls, signe d'une bonne généralisation.



#### Métriques détaillées — Modèle retenu (Gini, depth=3)

| Classe | Precision | Recall | F1-score |
|--------|-----------|--------|----------|
| Légitime (0) | 0.95 | 0.66 | 0.78 |
| Fraude (1) | 0.04 | 0.29 | 0.07 |

#### Matrice de confusion

|  | Prédit Légitime | Prédit Fraude |
|--|----------------|---------------|
| **Réel Légitime** | 6 387 (TN) | 3 311 (FP) |
| **Réel Fraude** | 357 (FN)  | 145 (TP) |

**Interprétation :**
- **357 fraudes non détectées** (faux négatifs) → montant moyen par fraude manquée : 2 785 € → **~961 000 € de pertes non couvertes**
- **3 311 transactions légitimes bloquées à tort** (faux positifs) → impact négatif sur l'expérience client

#### Courbe ROC — AUC

| Modèle | AUC |
|--------|-----|
| Gini (depth=3) | 0.4661 |
| Entropie (depth=3) | 0.4857 |


#### Gini vs Entropie

| Critère | Recall fraude | F1 fraude | AUC |
|---------|--------------|-----------|-----|
| Gini | 0.29 | 0.07 | 0.47 |
| Entropie | 0.40 | 0.08 | 0.49 |

L'Entropie détecte légèrement plus de fraudes (recall 0.40 vs 0.29) mais au prix d'une accuracy globale plus faible. Pour un cas de fraude bancaire où le recall est prioritaire.

#### Importance des variables (modèle Gini)

| Variable | Importance |
|----------|-----------|
| `Transaction_Amount` | 29 % |
| `Location_Houston` | 24 % |
| `Time_of_Transaction` | 20 % |
| `Account_Age` | 17 % |
| `Location_Chicago` | 9 % |
| Autres | 0 % |


---




## Conclusion

Le modèle arbre de décision (Gini, `max_depth=3`) **n'est pas encore prêt pour un déploiement en production** :
- Recall fraude de seulement 29 % — 71 % des fraudes passent inaperçues
- AUC de 0.47 — performances proches du hasard
- ~961 000 € de fraudes non détectées sur le jeu de test

### Recommandations
- Ajuster le seuil de décision (threshold) en dessous de 0.5 pour favoriser la détection des fraudes.
- Tester des algorithmes plus puissants : **Random Forest**, **XGBoost**, **LightGBM**
- Mettre en place une **validation croisée** pour une évaluation plus robuste

---

*TP réalisé avec Python 3 · scikit-learn · pandas · seaborn · matplotlib*
