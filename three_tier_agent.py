"""
Three-tier extension: fixes t1 (the two-tier threshold) and searches t2 to
add a 'hold_for_verification' middle tier, using its own cost function.
Independent analysis - doesn't consume two_tier_agent's outputs, just the
same test_predictions.csv.
"""

import os

import numpy as np
import pandas as pd

import agent_logic
import train_pipeline

THREE_TIER_SWEEP_CSV = "data/outputs/three_tier_cost_search.csv"


def run_three_tier_search(
    predictions_csv: str = train_pipeline.TEST_PREDICTIONS_CSV,
    x_test_csv: str = train_pipeline.X_TEST_CSV,
    t1: float = agent_logic.DECISION_THRESHOLD,
    t2_candidates=np.arange(0.20, 0.90, 0.02),
    save: bool = True,
) -> tuple[pd.Series, pd.DataFrame]:
    preds = pd.read_csv(predictions_csv)
    X_test = pd.read_csv(x_test_csv)
    preds["order_value"] = X_test["order_value"].values

    y_true = preds["y_true"]
    probs = preds["log_reg_prob"]
    order_value = preds["order_value"]

    results = []
    for t2 in t2_candidates:
        if t2 <= t1:
            continue
        actions = pd.Series([agent_logic.three_tier_action(p, t1, t2) for p in probs])

        approve_mask = (actions == "auto_approve").values
        flag_mask = (actions == "flag_for_review").values
        hold_mask = (actions == "hold_for_verification").values

        missed_returns_mask = approve_mask & (y_true.values == 1)
        total_fn_cost = agent_logic.fn_cost(order_value[missed_returns_mask]).sum()
        total_flag_cost = flag_mask.sum() * agent_logic.FLAG_COST
        total_hold_cost = agent_logic.hold_cost(order_value[hold_mask]).sum()

        total_cost = total_fn_cost + total_flag_cost + total_hold_cost

        results.append({
            "t1": t1, "t2": round(t2, 2),
            "n_approved": int(approve_mask.sum()), "n_flagged": int(flag_mask.sum()), "n_held": int(hold_mask.sum()),
            "fn_cost": round(total_fn_cost, 2),
            "flag_cost": total_flag_cost,
            "hold_cost": round(total_hold_cost, 2),
            "total_cost": round(total_cost, 2),
        })

    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df["total_cost"].idxmin()]

    print("=== Three-tier cost search - top 10 ===")
    print(results_df.sort_values("total_cost").head(10).to_string(index=False))
    print("\n=== Best (T1, T2) combination ===")
    print(best.to_string())

    if save:
        os.makedirs(os.path.dirname(THREE_TIER_SWEEP_CSV), exist_ok=True)
        results_df.to_csv(THREE_TIER_SWEEP_CSV, index=False)
        print(f"\nSaved: {THREE_TIER_SWEEP_CSV}")

    return best, results_df


if __name__ == "__main__":
    run_three_tier_search()