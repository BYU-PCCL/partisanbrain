if __name__ == "__main__":

    from imdb import ImdbDataset
    from experiment import Experiment

    # Run IMDB experiment
    print("Building dataset...")
    ImdbDataset(n=5)

    print("Passing dataset through model...")
    Experiment(model_name="gpt3-ada",
               ds_name="imdb")
