class Experiment:
    """For running experiments with GPT-3 API and saving them"""

    def __init__(self, dataset):
        # Results has form
        # {idx: [(prompt_1, response_1)...(prompt_n, response_n)]}
        self.results = {}

    def run(self):
        """Get results from GPT-3 API"""
        # Iterate over the rows of dataset (iterrows)
        #   For each row make prompt list
        #   Pass prompts to GPT-3
        #   Store result in results
        pass

    def store_results(self):
        """Store results obtained from run method"""
        # Pickle results
        pass
