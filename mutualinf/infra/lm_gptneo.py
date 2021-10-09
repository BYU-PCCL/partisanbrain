from lmsampler_baseclass import LMSamplerBaseClass

import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import numpy as np

class LM_GPTNEO(LMSamplerBaseClass):
    def __init__(self, model_name):
        super().__init__(model_name)
        '''
        Supported models: 'EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-125M'
        '''
        # check if model name is supported
        if model_name not in ['EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-125M']:
            raise ValueError('Model name not supported. Supported models: EleutherAI/gpt-neo-2.7B, EleutherAI/gpt-neo-1.3B, EleutherAI/gpt-neo-125M')
        # initialize model with model_name
        print(f'Loading {model_name}...')
        # TODO - add GPU support
        self.model = GPTNeoForCausalLM.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        if torch.cuda.is_available():
            # 1.3B and 125M fit on one GPU
            self.device = 'cuda:0'
            # TODO - add parallelization support for biggest model
        else:
            self.device = 'cpu'
        # send to device
        self.model = self.model.to(self.device)
        print(f'Loaded!')

    def send_prompt(self, prompt, n_probs):
        # encode prompt and pass to model
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(inputs)

        # get logits for final word (the prediction) from model output
        logits = output.logits[-1][-1].to('cpu')

        # get 'n_probs' predicted tokens associated with the above logits
        tokens = torch.argsort(logits, descending=True)[:n_probs]

        # decode tokens into text
        preds = self.tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=True)

        # calculate real probabilities associated with each prediction
        logits_probs = torch.nn.functional.softmax(logits, dim=0)
        probs = torch.argsort(logits_probs, descending=True)[:n_probs]

        # create dictionary and map prediction word to log prob
        self.pred_dict = {}
        for i in range(len(preds)):
            self.pred_dict[preds[i]] = np.log(logits_probs[probs[i]].item())

        return self.pred_dict