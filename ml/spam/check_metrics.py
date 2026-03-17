import joblib
import json

metrics = joblib.load('training_metrics.pkl')
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
print("Recall (Fraud):", metrics['report']['1']['recall'])
print("Precision (Fraud):", metrics['report']['1']['precision'])
print("F1-Score (Fraud):", metrics['report']['1']['f1-score'])
print("Confusion Matrix:")
print(metrics['confusion_matrix'])
