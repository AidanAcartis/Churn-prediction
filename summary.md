J’ai suivi une approche progressive afin d’améliorer progressivement les performances du modèle.
Le dernier bloc d’entraînement, situé à la fin du notebook, correspond au modèle final optimisé.

## 1- Problem Framing

## 2- Data Collection

## 3- Data Analysis
### 1. Measure data quality
- **a)** Recherche de **doublons**
- **b)** Vérification des **valeurs manquantes (NaN)**
- **c)** Contrôle des **erreurs dans les labels et features**
- **d)** Détection d’**outliers**
- **e)** Étude de la **corrélation et redondance**
- **f)** Évaluation du **pouvoir prédictif (Mutual Information)**
- **g)** Analyse de la **balance des classes**

## 4- Data Preparation
### a) Basic Feature Engineering
### b) Split Data

## 5- Model Training and Evaluation
### 1. RandomForestClassifier
Plusieurs variantes ont été testées pour évaluer leur impact :
- **a)** Avec la colonne `Complain`
- **b)** Sans la colonne `Complain`
- **c)** En appliquant **SMOTE** pour équilibrer les classes
- **d)** En appliquant **ADASYN** (autre méthode d’oversampling)


### 2. XGBoost

### 3. LightGBM

### 4. New Feature Engineering + XGBoost + Hyperparameters

## 6- Error Analysis

## 7- Presentation
