from lmsampler_baseclass import LMSamplerBaseClass
from lm_utils import get_device_map

import torch
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
        # if torch.cuda.is_available():
        #     self.device = 'cuda:0'
        # else:
        #     self.device = 'cpu'
        # # send to device
        # self.model = self.model.to(self.device)

        # get the number of attention layers
        n_blocks = self.model.config.n_layer
        if torch.cuda.is_available():
            # get all available GPUs
            gpus = np.arange(torch.cuda.device_count())
            self.device = 'cuda:0'
            if len(gpus) > 1:
                device_map = get_device_map(gpus, n_blocks)
                self.model.parallelize(device_map)
            else:
                self.model = self.model.to(self.device)
            print(f'Loaded model on {len(gpus)} GPUs.')
        else:
            self.device = 'cpu'
            print('Loaded model on cpu.')

    def send_prompt(self, prompt, n_probs):
        '''
        For BERT style prompts, you can put the '[MASK]' token in where you would like the model to predict.
        '''
        if '[MASK]' not in prompt:
            # add mask to end of prompt and period after mask for accurate predictions
            bert_prompt = prompt + ' ' + self.tokenizer.mask_token + '.'
        else:
            bert_prompt = prompt

        # encode bert_prompt
        input = self.tokenizer.encode_plus(bert_prompt, return_tensors = "pt").to(self.device)

        # store the masked token index
        mask_index = torch.where(input["input_ids"][0] == self.tokenizer.mask_token_id)

        # get the output from the model
        with torch.no_grad():
            output = self.model(**input)

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
