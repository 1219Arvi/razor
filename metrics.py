import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report

preds = pd.read_csv("data/outputs/test_predictions.csv")
y_true = preds["y_true"]
probs = preds["log_reg_prob"]

DECISION_THRESHOLD = 0.16

y_pred_deployed = (probs >= DECISION_THRESHOLD).astype(int)

cm = confusion_matrix(y_true, y_pred_deployed)
tn, fp, fn, tp = cm.ravel()

print(f"=== Confusion Matrix (threshold = {DECISION_THRESHOLD}, the DEPLOYED threshold) ===")
print(f"                  Predicted: Safe   Predicted: Risky")
print(f"Actually Safe:    {tn:>13}   {fp:>15}")
print(f"Actually Returned:{fn:>13}   {tp:>15}")

print(f"\nTrue Positives (caught real returns):   {tp}")
print(f"False Positives (false alarms):         {fp}")
print(f"False Negatives (missed real returns):  {fn}")
print(f"True Negatives (correctly let through):  {tn}")

print(f"\nPrecision: {precision_score(y_true, y_pred_deployed):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred_deployed):.4f}")
print(f"F1 Score:  {f1_score(y_true, y_pred_deployed):.4f}")

print("\nFull classification report:")
print(classification_report(y_true, y_pred_deployed, target_names=["Not Returned", "Returned"]))

print("\n=== Comparison: default (0.5) vs deployed (0.16) threshold ===")
y_pred_default = (probs >= 0.5).astype(int)
comparison = pd.DataFrame({
    "Metric": ["Precision", "Recall", "F1"],
    "Threshold 0.5 (rejected)": [
        precision_score(y_true, y_pred_default),
        recall_score(y_true, y_pred_default),
        f1_score(y_true, y_pred_default),
    ],
    "Threshold 0.16 (deployed)": [
        precision_score(y_true, y_pred_deployed),
        recall_score(y_true, y_pred_deployed),
        f1_score(y_true, y_pred_deployed),
    ],
})
print(comparison.to_string(index=False))