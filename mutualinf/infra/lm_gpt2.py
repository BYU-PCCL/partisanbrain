from lmsampler import LMSampler

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np

class LM_GPT2(LMSampler):
    def __init__(self, model_name):
        super().__init__(model_name)

        # initialize model with model_name
        self.model = None # TODO load docker img file
        self.tokenizer = None

    def send_prompt(self, prompt, n_probs):
        # encode prompt and pass to model
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model(inputs)

        # get logits for final word (the prediction) from model output
        logits = output.logits[-1][-1]

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
