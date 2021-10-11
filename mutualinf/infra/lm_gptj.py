from lmsampler_baseclass import LMSamplerBaseClass
from lm_utils import get_device_map
import torch
import numpy as np
from pdb import set_trace as breakpoint

from transformers import AutoModelForCausalLM, AutoTokenizer

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
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

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
        else:
            self.device = 'cpu'

    def send_prompt(self, prompt, n_probs):
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(inputs)

        # get logits for final word (the prediction) from model output
        logits = output.logits[-1][-1].to('cpu')

        # get 'n_probs' predicted tokens associated with the above logits
        tokens = torch.argsort(logits, descending=True)[:n_probs]

        # decode tokens into text
        preds = self.tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=True)
        # TODO - better way to do this?
        # Sometimes symbols don't come out great in ascii encoding
        preds = [p.encode('ascii', 'ignore').decode('ascii') for p in preds]

        # calculate real probabilities associated with each prediction
        logits_probs = torch.nn.functional.softmax(logits, dim=0)
        probs = torch.argsort(logits_probs, descending=True)[:n_probs]

        # create dictionary and map prediction word to log prob
        self.pred_dict = {}
        for i in range(len(preds)):
            self.pred_dict[preds[i]] = np.log(logits_probs[probs[i]].item())

        return self.pred_dict
