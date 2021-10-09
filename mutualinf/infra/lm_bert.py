from lmsampler_baseclass import LMSamplerBaseClass

import torch
from transformers import BertTokenizer, BertForMaskedLM
import numpy as np

class LM_BERT(LMSamplerBaseClass):
    def __init__(self, model_name):
        super().__init__(model_name)
        '''
        Supported models: 'bert-base-uncased', ...
        '''
        # initialize model with model_name
        print(f'Loading {model_name}...')
        # TODO - add GPU support
        self.model = BertForMaskedLM.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        print(f'Loaded!')

    def send_prompt(self, prompt, n_probs):
        # add mask to end of prompt and period after mask for accurate predictions
        bert_prompt = prompt + ' ' + tokenizer.mask_token + '.'

        # encode bert_prompt
        input = self.tokenizer.encode_plus(bert_prompt, return_tensors = "pt")

        # store the masked token index
        mask_index = torch.where(input["input_ids"][0] == self.tokenizer.mask_token_id)

        # get the output from the model
        output = self.model(**input)
        logits = output.logits

        # run softmax on logits
        softmax = F.softmax(logits, dim = -1)
        
        # get the softmaxed logits for the masked word
        mask_word = softmax[0, mask_index, :]
        pred_index = mask_index[0].item()

        # get n_probs for masked token
        top_n = torch.topk(mask_word, n_probs, dim = 1)[1][0]

        # create dictionary and map prediction word to log prob
        self.pred_dict = {}
        for token in top_100:
            pred = self.tokenizer.decode([token])
            self.pred_dict[pred] = np.log((softmax[-1][pred_index][token].item()))

        return self.pred_dict

