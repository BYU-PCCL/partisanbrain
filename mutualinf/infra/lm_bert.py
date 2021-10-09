from lmsampler_baseclass import LMSamplerBaseClass

import torch
# from torch import functional as F
from transformers import BertTokenizer, BertForMaskedLM
import numpy as np

class LM_BERT(LMSamplerBaseClass):
    def __init__(self, model_name):
        super().__init__(model_name)
        '''
        Supported models: 'bert-base-uncased', 'bert-base-cased'
        '''
        # check if model_name is supported
        if model_name not in ['bert-base-uncased', 'bert-base-cased']:
            raise ValueError('Model name not supported. Must be one of: bert-base-uncased, bert-base-cased')
        # initialize model with model_name
        print(f'Loading {model_name}...')
        # TODO - add GPU support
        self.model = BertForMaskedLM.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
        # send to device
        self.model = self.model.to(self.device)
        print(f'Loaded!')

    def send_prompt(self, prompt, n_probs):
        # add mask to end of prompt and period after mask for accurate predictions
        bert_prompt = prompt + ' ' + self.tokenizer.mask_token + '.'

        # encode bert_prompt
        input = self.tokenizer.encode_plus(bert_prompt, return_tensors = "pt").to(self.device)

        # store the masked token index
        mask_index = torch.where(input["input_ids"][0] == self.tokenizer.mask_token_id)

        # get the output from the model
        with torch.no_grad():
            output = self.model(**input)
        # logits = output.logits.to('cpu').numpy()

        # # run softmax on logits
        # # softmax = F.softmax(logits, dim = -1)
        # breakpoint()
        # softmax = torch.nn.functional.softmax(logits, dim=0)
        
        # # get the softmaxed logits for the masked word
        # mask_word = softmax[0, mask_index, :]
        # pred_index = mask_index[0].item()

        # # get n_probs for masked token
        # top_n = torch.topk(mask_word, n_probs, dim = 1)[1][0]

        # # create dictionary and map prediction word to log prob
        # self.pred_dict = {}
        # for token in top_100:
        #     pred = self.tokenizer.decode([token])
        #     self.pred_dict[pred] = np.log((softmax[-1][pred_index][token].item()))
        
        # logits = output.logits[-1][-1].to('cpu')
        logits = output.logits[0, mask_index].to('cpu').reshape(-1)


        # get 'n_probs' predicted tokens associated with the above logits
        tokens = torch.argsort(logits, descending=True)[:n_probs]
        
        # decode tokens into text
        preds = self.tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=True)
        # for some reason, bert tokenizer adds a bunch of whitespace. Remove
        preds = [p.replace(' ', '') for p in preds]

        # calculate real probabilities associated with each prediction
        logits_probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = torch.argsort(logits_probs, descending=True)[:n_probs]

        # create dictionary and map prediction word to log prob
        self.pred_dict = {}
        for i in range(len(preds)):
            self.pred_dict[preds[i]] = np.log(logits_probs[probs[i]].item())

        return self.pred_dict

