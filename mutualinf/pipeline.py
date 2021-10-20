from datetime import date
from experiment import Experiment
from postprocessor import Postprocessor


def run_experiment(ds_name, model_name, n=500):

    camel_case_ds_name = "".join(word.title() for word in ds_name.split('_'))

    # Create dataset
    print("Building dataset...")
    ds_cls_name = f"{camel_case_ds_name}Dataset"
    exec(f"from {ds_name} import {ds_cls_name}")
    eval(f"{ds_cls_name}(n={n})")

    # Pass data through model
    print("Passing data through model...")
    Experiment(model_name=model_name, ds_name=ds_name)

    # Postprocessing
    print("Postprocessing...")
    date_str = date.today().strftime("%d-%m-%Y")
    in_fname = (f"data/{ds_name}/exp_results_"
                f"{model_name}_{date_str}.pkl")
    out_fname = (f"data/{ds_name}/exp_results_"
                 f"{model_name}_{date_str}_processed.pkl")
    # replace '/' with '-'
    out_fname = out_fname.replace('/', '-')
    Postprocessor(results_fname=in_fname,
                  save_fname=out_fname,
                  matching_strategy="startswith")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("ds_name", type=str,
                        help="Name of dataset (e.g., imdb, example)")
    parser.add_argument("model_name", type=str,
                        help="Name of model (see lmsampler for choices)")
    parser.add_argument("-n", "--n", type=int,
                        help="Number of rows of data to process")
    args = parser.parse_args()

    run_experiment(ds_name=args.ds_name,
                   model_name=args.model_name,
                   n=args.n)
