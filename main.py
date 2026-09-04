"""
main.py - end-to-end orchestrator for the return-risk pipeline.

Stages, in order:
  1. data      -> data_generation.generate_dataset() + save_dataset()
  2. train     -> train_pipeline.run_full_training() (persists the model)
  3. metrics   -> metrics.print_metrics() at the deployed threshold
  4. two-tier  -> two_tier_agent.run_two_tier_agent() (decisions + explanations + audit log)
  5. three-tier-> three_tier_agent.run_three_tier_search() (optional extension analysis)
  6. razorpay  -> creates real test-mode orders + audits them (optional, needs .env credentials)

Each stage is skipped if its output already exists, unless you force it
with a flag. This means you can run `python main.py` repeatedly while
iterating on one stage without re-running everything upstream.

Examples
--------
    python main.py                          # run whatever hasn't been run yet
    python main.py --retrain                # force retrain even if a model exists
    python main.py --regenerate-data         # force regenerate the synthetic dataset
    python main.py --skip-three-tier         # skip the three-tier exploration
    python main.py --run-razorpay            # also hit the real Razorpay test-mode API
"""

import argparse
import os

import data_generation
import metrics
import three_tier_agent
import train_pipeline
import two_tier_agent


def stage_data(force: bool = False) -> None:
    if not force and os.path.exists(train_pipeline.RAW_MODEL_CSV):
        print(f"[data] {train_pipeline.RAW_MODEL_CSV} already exists - skipping (use --regenerate-data to force).")
        return
    print("[data] Generating synthetic dataset...")
    df_full, df_model = data_generation.generate_dataset()
    data_generation.save_dataset(df_full, df_model)


def stage_train(force: bool = False):
    if not force and os.path.exists(train_pipeline.MODEL_PATH):
        print(f"[train] {train_pipeline.MODEL_PATH} already exists - skipping (use --retrain to force).")
        return None
    print("[train] Training model...")
    return train_pipeline.run_full_training()


def stage_metrics() -> None:
    print("[metrics] Evaluating at the deployed threshold...")
    metrics.print_metrics()


def stage_two_tier():
    print("[two-tier] Assigning actions + explanations...")
    return two_tier_agent.run_two_tier_agent()


def stage_three_tier():
    print("[three-tier] Searching for a cost-optimal hold-tier threshold...")
    return three_tier_agent.run_three_tier_search()


def stage_razorpay(n_orders: int = 25) -> None:
    # Imported lazily: this subpackage needs `razorpay` + a valid .env,
    # neither of which the rest of the pipeline depends on.
    from razorpay_test_mode import audit as razorpay_audit
    from razorpay_test_mode import orders as razorpay_orders

    print(f"[razorpay] Creating {n_orders} real test-mode orders...")
    razorpay_orders.create_test_orders(n_orders=n_orders)
    print("[razorpay] Auditing them with the trained model...")
    razorpay_audit.run_audit()


def main():
    parser = argparse.ArgumentParser(description="Run the return-risk pipeline end to end.")
    parser.add_argument("--regenerate-data", action="store_true", help="Force regenerate the synthetic dataset.")
    parser.add_argument("--retrain", action="store_true", help="Force retrain the model even if one is saved.")
    parser.add_argument("--skip-three-tier", action="store_true", help="Skip the three-tier extension analysis.")
    parser.add_argument("--run-razorpay", action="store_true", help="Also create + audit real Razorpay test-mode orders (needs .env credentials).")
    parser.add_argument("--razorpay-orders", type=int, default=25, help="Number of Razorpay test-mode orders to create.")
    args = parser.parse_args()

    stage_data(force=args.regenerate_data)
    stage_train(force=args.retrain)
    stage_metrics()
    stage_two_tier()

    if not args.skip_three_tier:
        stage_three_tier()

    if args.run_razorpay:
        stage_razorpay(n_orders=args.razorpay_orders)

    print("\nDone. Key outputs:")
    print(f"  Model:              {train_pipeline.MODEL_PATH}")
    print(f"  Test predictions:   {train_pipeline.TEST_PREDICTIONS_CSV}")
    print(f"  Threshold sweep:    {train_pipeline.THRESHOLD_SWEEP_CSV}")
    print(f"  Two-tier audit log: {two_tier_agent.AUDIT_LOG_CSV}")
    if not args.skip_three_tier:
        print(f"  Three-tier sweep:   {three_tier_agent.THREE_TIER_SWEEP_CSV}")


if __name__ == "__main__":
    main()