# ============================
# === Imports nécessaires ===
# ============================
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, 
    recall_score, confusion_matrix, ConfusionMatrixDisplay, make_scorer
)

# ============================
# === 1️⃣ Création des features ===
# ============================
X_train_nocomp['AgeGroup'] = pd.cut(
    X_train_nocomp['Age'],
    bins=[18, 30, 40, 50, 100],
    labels=['Young','Stable','AtRisk','Senior']
)

X_val_nocomp['AgeGroup'] = pd.cut(
    X_val_nocomp['Age'],
    bins=[18, 30, 40, 50, 100],
    labels=['Young','Stable','AtRisk','Senior']
)

X_train_nocomp['LowProducts'] = (X_train_nocomp['NumOfProducts'] == 1).astype(int)
X_val_nocomp['LowProducts'] = (X_val_nocomp['NumOfProducts'] == 1).astype(int)

X_train_nocomp['Age_Products'] = X_train_nocomp['Age'] * X_train_nocomp['NumOfProducts']
X_val_nocomp['Age_Products'] = X_val_nocomp['Age'] * X_val_nocomp['NumOfProducts']

print("✅ Nouvelles colonnes ajoutées :", 
      set(X_train_nocomp.columns) & {'Age_Products', 'AgeGroup', 'LowProducts'})

# ============================
# === 2️⃣ Sélection des features finales ===
# ============================
selected_features = [
    'Age_Products', 'Age', 'NumOfProducts', 'AgeGroup', 'LowProducts',
    'IsActiveMember', 'Balance', 'CreditScore', 'Tenure', 'EstimatedSalary'
]

# Encoder la colonne catégorielle
le = LabelEncoder()
X_train_nocomp['AgeGroup'] = le.fit_transform(X_train_nocomp['AgeGroup'].astype(str))
X_val_nocomp['AgeGroup'] = le.transform(X_val_nocomp['AgeGroup'].astype(str))

X_train_final = X_train_nocomp[selected_features].copy()
X_val_final   = X_val_nocomp[selected_features].copy()
y_train_final = y_train.copy()
y_val_final   = y_val.copy()

print("✅ Dimensions finales :", X_train_final.shape, X_val_final.shape)
print("✅ Colonnes finales :", X_train_final.columns.tolist())

# ============================
# === 3️⃣ XGBoost - Modèle ===
# ============================
ratio = y_train_final.value_counts()[0] / y_train_final.value_counts()[1]

xgb_model = xgb.XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=ratio
)

# Hyperparamètres finaux testés
param_dist = {
    'n_estimators': [300, 400, 500],
    'max_depth': [5, 7, 9],
    'learning_rate': [0.03, 0.05, 0.07],
    'min_child_weight': [1, 3],
    'gamma': [0, 0.1, 0.3],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 0.3],
    'reg_lambda': [1, 1.5, 2]
}

f1_scorer = make_scorer(f1_score, pos_label=1)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================
# === 4️⃣ RandomizedSearchCV ===
# ============================
from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=50,
    scoring=f1_scorer,
    cv=cv,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train_final, y_train_final)

print("Best parameters found:", random_search.best_params_)
print("Best F1-score (CV):", random_search.best_score_)

# ============================
# === 5️⃣ Optimisation du seuil sur le train ===
# ============================
best_xgb = random_search.best_estimator_
y_train_proba = best_xgb.predict_proba(X_train_final)[:, 1]

thresholds = np.arange(0.1, 0.9, 0.01)
best_f1_train = 0
best_threshold_train = 0.5
for t in thresholds:
    y_train_pred_t = (y_train_proba >= t).astype(int)
    f1 = f1_score(y_train_final, y_train_pred_t)
    if f1 > best_f1_train:
        best_f1_train = f1
        best_threshold_train = t

print("Train F1-score:", best_f1_train)
print("Best threshold for train:", best_threshold_train)

# ============================
# === 6️⃣ Prédictions sur la validation ===
# ============================
y_val_proba = best_xgb.predict_proba(X_val_final)[:, 1]

best_f1 = 0
best_threshold = 0.5
for t in thresholds:
    y_val_pred_t = (y_val_proba >= t).astype(int)
    f1 = f1_score(y_val_final, y_val_pred_t)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

y_val_pred_opt = (y_val_proba >= best_threshold).astype(int)

print(f"\nBest threshold for minority class: {best_threshold:.2f}")
print(f"F1-score corresponding: {best_f1:.4f}")

# ============================
# === 7️⃣ Évaluation finale ===
# ============================
print("\n=== Évaluation sur le jeu de validation avec XGBoost ===")
print("Accuracy :", accuracy_score(y_val_final, y_val_pred_opt))
print("F1-score :", f1_score(y_val_final, y_val_pred_opt))
print("Precision:", precision_score(y_val_final, y_val_pred_opt))
print("Recall   :", recall_score(y_val_final, y_val_pred_opt))
print("ROC-AUC  :", roc_auc_score(y_val_final, y_val_proba))

# Matrice de confusion
cm = confusion_matrix(y_val_final, y_val_pred_opt)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()
