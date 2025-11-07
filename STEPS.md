# Data COllection:
## 1- Measure data quality:
* a) Duplicates
* b) Missing values and Nan
* c) Check labels and features errors
* d) Outliers
* e) Correlation and redundacy
* f) Predictive power
* g) balance des classes

## 2- Basic feature engineering:
- Drop column wich are less predictive ['Satisfaction score', 'RowNumber', 'Surname', 'Point Earned']

## 3- Split data:

# Training:
## 1- Model RandomForestClassifier
### a) With the column 'Complain'
### b) Training without 'Complain'
### c) With SMOTE
### d) With ADASYN

## 2- Model XGBoost
### a) Use hyperparameters with GridSearch

## 3- LightGBM (with the hyperparameters found lastly in XGBoost) 

## 4- New feature engineering (add new column) + XGBoost + Hyperparameters:
