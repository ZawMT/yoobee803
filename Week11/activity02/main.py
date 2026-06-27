import argparse
import importlib.util
import os


def _parse_args():
    parser = argparse.ArgumentParser(description="NZ Wellbeing Forecasting Pipeline")
    parser.add_argument(
        "--no-total-pop", dest="run_total_pop", action="store_false", default=True,
        help="Skip the Total Population forecast step",
    )
    parser.add_argument(
        "--predict-all-models", dest="predict_all_models", default="all",
        metavar="MODELS",
        help='Models for predict_all: "all", "LR;XGB;ARIMA", or "" to skip',
    )
    return parser.parse_args()


def import_step(filename):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(filename, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    args = _parse_args()
    RUN_TOTAL_POP      = args.run_total_pop
    PREDICT_ALL_MODELS = args.predict_all_models

    print("\n*** NZ WELLBEING FORECASTING PIPELINE ***\n")

    # Step 1: Load and preprocess
    preprocessing = import_step("preprocessing.py")
    preprocessing.load_and_preprocess()

    # Step 2: Forecast — Total population only
    df_total = None
    if RUN_TOTAL_POP:
        predict_total = import_step("predict_total_pop.py")
        df_total = predict_total.run()
    else:
        print("  [SKIPPED] Total population forecast (RUN_TOTAL_POP=False)\n")

    # Step 3: Forecast — All demographics
    df_all = None
    if PREDICT_ALL_MODELS.strip():
        predict_all_mod = import_step("predict_all.py")
        df_all = predict_all_mod.run(models=PREDICT_ALL_MODELS)
    else:
        print("  [SKIPPED] All demographics forecast (PREDICT_ALL_MODELS is blank)\n")

    print("\n*** PIPELINE COMPLETE ***")
    if df_total is not None:
        print(f"  Total population series  : {len(df_total)}")
    if df_all is not None:
        print(f"  All demographics series  : {len(df_all)}")
    print(f"  Reports saved in         : reports/")


if __name__ == "__main__":
    main()
