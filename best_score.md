Best parameters found: {'subsample': 0.9, 'reg_lambda': 1, 'reg_alpha': 0.3, 'n_estimators': 300, 'min_child_weight': 1, 'max_depth': 7, 'learning_rate': 0.03, 'gamma': 0, 'colsample_bytree': 0.9}
Best F1-score (CV): 0.5797720470469689
Train F1-score: 0.8167895632444696
Best threshold for train: 0.5599999999999997

Best threshold for minority class: 0.56
F1-score corresponding: 0.8006

=== Évaluation sur le jeu de validation avec XGBoost ===
Accuracy : 0.9125
F1-score : 0.8005698005698005
Precision: 0.7533512064343163
Recall   : 0.8541033434650456
ROC-AUC  : 0.9660009709225438

=== Évaluation sur le jeu de validation avec XGBoost ===
Accuracy : 0.9125
F1-score : 0.8005698005698005
Precision: 0.7533512064343163
Recall   : 0.8541033434650456
ROC-AUC  : 0.9660009709225438


  bst.update(dtrain, iteration=i, fobj=obj)
Best parameters found: {'subsample': 0.8, 'reg_lambda': 1, 'reg_alpha': 0.1, 'n_estimators': 300, 'min_child_weight': 3, 'max_depth': 5, 'learning_rate': 0.03, 'gamma': 0, 'colsample_bytree': 1.0}
Best F1-score (CV): 0.5778310304867869