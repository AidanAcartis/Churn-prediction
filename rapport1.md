J’ai suivi une approche progressive afin d’améliorer progressivement les performances du modèle.
Le dernier bloc d’entraînement, situé à la fin du notebook, correspond au modèle final optimisé.

---

## 1- Problem Framing
L’objectif est de prédire **le départ d’un client (churn)** à partir des données bancaires.  
Ce problème a une **valeur business forte** : identifier à l’avance les clients susceptibles de partir permet de cibler des actions de fidélisation.  
Le problème est formulé comme une **classification binaire**, avec pour métrique principale le **F1-score**, car les classes sont déséquilibrées et l’on cherche un bon compromis entre précision et rappel.

---

## 2- Data Collection
Les données proviennent d’un dataset de churn clients (fichier CSV).  
Elles contiennent des caractéristiques démographiques, financières et comportementales des clients.  
Elles sont considérées comme représentatives de la population à modéliser.  
La séparation entre les ensembles **train** et **validation** a été effectuée pour garantir une évaluation juste.

---

## 3- Data Analysis
### 1. Measure data quality
- **a)** Recherche de **doublons**
- **b)** Vérification des **valeurs manquantes (NaN)**
- **c)** Contrôle des **erreurs dans les labels et features**
- **d)** Détection d’**outliers**
- **e)** Étude de la **corrélation et redondance**
- **f)** Évaluation du **pouvoir prédictif (Mutual Information)**
- **g)** Analyse de la **balance des classes**

Ces analyses ont permis de comprendre la structure du dataset et d’identifier les axes d’amélioration avant l’entraînement.

---

## 4- Data Preparation
### Basic Feature Engineering
Suppression des variables peu informatives :
`['Satisfaction score', 'RowNumber', 'Surname', 'Point Earned']`  
Cette étape vise à éliminer le bruit et réduire la dimensionnalité sans perte de pouvoir prédictif.

### Split Data
Séparation du dataset en ensembles :
- **train** (pour l’apprentissage)
- **validation** (pour l’évaluation finale)

Cela garantit une évaluation indépendante des modèles.

---

## 5️⃣ Model Training and Evaluation
### 1. RandomForestClassifier
Plusieurs variantes ont été testées pour évaluer leur impact :
- **a)** Avec la colonne `Complain`
- **b)** Sans la colonne `Complain`
- **c)** En appliquant **SMOTE** pour équilibrer les classes
- **d)** En appliquant **ADASYN** (autre méthode d’oversampling)

Chaque sous-expérience a été évaluée pour suivre la progression du F1-score.

---

### 2. XGBoost
Un modèle plus avancé a ensuite été introduit.  
Une recherche d’hyperparamètres a été réalisée avec **GridSearch**, permettant d’optimiser la performance sur le jeu d’entraînement.

---

### 3. LightGBM
Pour comparaison, un modèle **LightGBM** a été entraîné en réutilisant les **meilleurs hyperparamètres trouvés avec XGBoost**, afin de tester sa capacité à généraliser plus efficacement.

---

### 4. New Feature Engineering + XGBoost + Hyperparameters
Enfin, un **nouvel ensemble de features dérivées** a été créé :
- `AgeGroup` (catégorisation des âges)
- `LowProducts` (flag pour les clients avec peu de produits)
- `Age_Products` (interaction entre âge et nombre de produits)

Ces nouvelles variables ont ensuite été intégrées dans un **modèle XGBoost optimisé** avec les meilleurs hyperparamètres.  
Un **seuil de probabilité optimisé (threshold = 0.56)** a été appliqué pour maximiser le F1-score sur la classe minoritaire.

Cette dernière étape représente la version **finale et la plus performante** du pipeline, fruit d’une évolution itérative et méthodique.

---

## 6️⃣ Error Analysis
Une **matrice de confusion** et des **métriques de performance détaillées** (accuracy, precision, recall, F1, ROC-AUC) ont permis d’analyser les erreurs résiduelles du modèle.  
Cette analyse aide à comprendre les cas mal prédits et à orienter d’éventuelles améliorations futures.

---

## 7️⃣ Presentation
Ce notebook a été conçu pour **montrer la progression étape par étape** vers la solution finale :  
du prétraitement initial à l’ingénierie avancée de variables, chaque amélioration est justifiée et mesurée, illustrant le **cycle complet de développement d’un modèle de Machine Learning**.
