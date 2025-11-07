import matplotlib.pyplot as plt
import numpy as np

metrics = list(scoring.keys())
train_scores = [cv_results['train_'+m].mean() for m in metrics]
val_scores = [cv_results['test_'+m].mean() for m in metrics]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8,5))
plt.bar(x - width/2, train_scores, width, label='Train', color='skyblue')
plt.bar(x + width/2, val_scores, width, label='Validation', color='lightgreen')
plt.xticks(x, [m.upper() for m in metrics])
plt.ylabel('Score moyen (5 folds)')
plt.title('Résultats de la Cross-Validation - RandomForest (sans Complain)')
plt.legend()
plt.show()
