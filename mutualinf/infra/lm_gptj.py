from lmsampler import LMSampler

from transformers import GPTJForCausalLM, AutoTokenizer

class LM_GPTJ(LMSamplerBaseClass):
    def __init__(self, model_name):
        '''
        Supported model names: 'EleutherAI/gpt-j-6B'.
        '''
        # check if model_name is supported
        if model_name not in ['EleutherAI/gpt-j-6B']:
            raise ValueError('Model name not supported. Supported model names: \'EleutherAI/gpt-j-6B\'.')
        super().__init__(model_name)

        # initialize model with model_name
        # TODO - add GPU support
        self.model = GPTJForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def send_prompt(self, prompt, n_probs):
        # send prompt to LM_GPTJ -- TODO how to test gptj (too big for colab)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        gen_tokens = self.model.generate(input_ids, do_sample=True, max_length=1,)
        logits = gen_tokens[0][:,-1,:]
        # create dictionary of output