import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

from agent_logic import DECISION_THRESHOLD

TEST_PREDICTIONS_CSV = "data/outputs/test_predictions.csv"


def print_metrics(predictions_csv: str = TEST_PREDICTIONS_CSV, threshold: float = DECISION_THRESHOLD) -> pd.DataFrame:
    """
    Prints the confusion matrix / precision / recall / F1 at the deployed
    threshold, plus a comparison against the naive 0.5 threshold. Returns
    the comparison table.
    """
    preds = pd.read_csv(predictions_csv)
    y_true = preds["y_true"]
    probs = preds["log_reg_prob"]

    y_pred_deployed = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_deployed)
    tn, fp, fn, tp = cm.ravel()

    print(f"=== Confusion Matrix (threshold = {threshold}, the DEPLOYED threshold) ===")
    print(f"                   Predicted: Safe   Predicted: Risky")
    print(f"Actually Safe:     {tn:>13}   {fp:>15}")
    print(f"Actually Returned: {fn:>13}   {tp:>15}")

    print(f"\nTrue Positives (caught real returns):  {tp}")
    print(f"False Positives (false alarms):        {fp}")
    print(f"False Negatives (missed real returns): {fn}")
    print(f"True Negatives (correctly let through): {tn}")

    print(f"\nPrecision: {precision_score(y_true, y_pred_deployed):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred_deployed):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred_deployed):.4f}")

    print("\nFull classification report:")
    print(classification_report(y_true, y_pred_deployed, target_names=["Not Returned", "Returned"]))

    print(f"\n=== Comparison: default (0.5) vs deployed ({threshold}) threshold ===")
    y_pred_default = (probs >= 0.5).astype(int)
    comparison = pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1"],
        "Threshold 0.5 (rejected)": [
            precision_score(y_true, y_pred_default),
            recall_score(y_true, y_pred_default),
            f1_score(y_true, y_pred_default),
        ],
        f"Threshold {threshold} (deployed)": [
            precision_score(y_true, y_pred_deployed),
            recall_score(y_true, y_pred_deployed),
            f1_score(y_true, y_pred_deployed),
        ],
    })
    print(comparison.to_string(index=False))
    return comparison


if __name__ == "__main__":
    print_metrics()