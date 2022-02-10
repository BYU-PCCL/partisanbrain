from datetime import date
from experiment import Experiment
from postprocessor import Postprocessor
from datetime import date
from pdb import set_trace as breakpoint
import os
# import variable API_KEY from environment
# API_KEY = os.environ.get("API_KEY")



def run_experiment(ds_path, model_name, n=500):

    # camel_case_ds_name = "".join(word.title() for word in ds_name.split('_'))

    # TODO - create ds dataframes?
    # # Create dataset
    # print("Building dataset...")
    # ds_cls_name = f"{camel_case_ds_name}Dataset"
    # exec(f"from {ds_name} import {ds_cls_name}")
    # eval(f"{ds_cls_name}(n={n})")

    # Pass data through model
    print("Passing data through model...")
    # Experiment(model_name=model_name, ds_name=ds_name)
    in_fname = ds_path
    date_str = date.today().strftime("%Y-%m-%d")
    # get directory where ds_path is located
    # replace_str
    replace_str = f"_exp_results_{model_name.replace('/', '-')}_{date_str}.pkl"
    # replace .pkl with replace_str
    out_fname = in_fname.replace(".pkl", replace_str)
    Experiment(
        model_name=model_name,
        in_fname=in_fname,
        out_fname=out_fname,
    )

    model_name = model_name.replace('/', '-')

    # Postprocessing
    print("Postprocessing...")
    date_str = date.today().strftime("%d-%m-%Y")
    processed_in = out_fname
    # replace .pkl with _processed.pkl
    processed_out = processed_in.replace('.pkl', '_processed.pkl')

    Postprocessor(results_fname=processed_in,
                  save_fname=processed_out,
                  matching_strategy="startswith")


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("ds_path", type=str,
    #                     help="Path to ds.pkl file")
    # parser.add_argument("model_name", type=str,
    #                     help="Name of model (see lmsampler for choices)")
    # # parser.add_argument("-n", "--n", type=int,
    # #                     help="Number of rows of data to process")
    # args = parser.parse_args()
    import sys
    # sys.argv
    ds_path = sys.argv[1]
    model_name = sys.argv[2]

    # run_experiment(ds_name=args.ds_name,
    #                model_name=args.model_name,
    #                n=args.n)

    # run_experiment(ds_name=args.ds_name,
    #                model_name=args.model_name)

    run_experiment(ds_path=ds_path,
                   model_name=model_name)
