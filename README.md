# Activity Monitoring & Classification System

Prédiction du churn client bancaire grâce à XGBoost optimisé avec feature engineering et calibration du seuil pour maximiser le F1-score.

## 📂 Description

Ce projet suit une approche progressive pour améliorer les performances d’un modèle de classification sur un dataset bancaire :  

1. Analyse exploratoire des données (EDA) pour détecter doublons, valeurs manquantes, outliers et déséquilibre des classes.  
2. Préparation des données : nettoyage, encodage des variables catégorielles, création de nouvelles features (`AgeGroup`, `LowProducts`, `Age_Products`).  
3. Entraînement et évaluation de modèles : RandomForest et XGBoost, avec optimisation des hyperparamètres et calibration du seuil pour maximiser le F1-score sur la classe minoritaire.  

Le modèle final XGBoost atteint un **F1-score de 0.766** sur le jeu de validation après optimisation.

## 🔍 Structure du projet

```

.
├── notebooks/                # Notebooks d’analyse et d’entraînement
├── data/                     # Jeux de données (train/test)
├── scripts/                  # Scripts Python pour feature engineering et entraînement
├── README.md                 # Documentation du projet
└── requirements.txt          # Dépendances Python

````

## 🛠️ Installation

1. Cloner le dépôt :  
```bash
git clone https://github.com/AidanAcartis/taskMonitor.git
cd taskMonitor
````

2. Créer un environnement conda et installer les dépendances :

```bash
conda create -n churn_env python=3.12
conda activate churn_env
pip install -r requirements.txt
```

3. Lancer les notebooks ou scripts pour reproduire les analyses et l’entraînement du modèle.

## 📊 Méthodologie

### 1. Analyse des données

* Vérification de la qualité des données (doublons, NaN, erreurs de labels, outliers)
* Étude des corrélations et du pouvoir prédictif des variables
* Analyse du déséquilibre des classes

### 2. Préparation des données

* Suppression des colonnes peu informatives (`Satisfaction Score`, `RowNumber`, `Surname`, `Point Earned`)
* Encodage des variables catégorielles avec OneHotEncoder
* Création de nouvelles features :

  * `AgeGroup` : tranche d’âge
  * `LowProducts` : flag pour clients ayant un seul produit
  * `Age_Products` : interaction âge × nombre de produits

### 3. Modélisation

* RandomForestClassifier : test avec/sans feature `Complain`, SMOTE/ADASYN
* XGBoost : hyperparameter tuning avec RandomizedSearchCV
* Calibration du seuil pour maximiser le F1-score de la classe minoritaire

## 📈 Résultats

* **RandomForest avec Complain** : F1 ≈ 0.994 (biais dû à la fuite de données)
* **RandomForest sans Complain** : F1 ≈ 0.59
* **RandomForest + SMOTE** : F1 ≈ 0.55
* **XGBoost final** : F1 ≈ 0.766 (avec hyperparameter tuning et feature engineering)

## ⚙️ Technologies

* Python 3.12
* Pandas, NumPy, Matplotlib, Seaborn
* Scikit-learn, Imbalanced-learn
* XGBoost

## 📎 Lien GitHub

[Voir le dépôt sur GitHub](https://github.com/AidanAcartis/taskMonitor)

## 🔗 License

MIT License
