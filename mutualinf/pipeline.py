from experiment import Experiment


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
