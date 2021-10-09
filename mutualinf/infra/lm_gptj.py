from lmsampler import LMSampler

from transformers import AutoModelForCausalLM, AutoTokenizer

class LM_GPTJ(LMSampler):
    def __init__(self, model_name):
        super().__init__(model_name)

        # initialize model with model_name
        self.model = None # TODO load docker img file
        self.tokenizer = None

    def send_prompt(self, prompt, n_probs):
        # send prompt to LM_GPTJ -- TODO how to test gptj (too big for colab)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        gen_tokens = self.model.generate(input_ids, do_sample=True, max_length=1,)
        logits = gen_tokens[0][:,-1,:]
        # create dictionary of output
