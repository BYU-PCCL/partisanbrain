from lmsampler_baseclass import LMSamplerBaseClass

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
        # if torch.cuda.is_available():
        #     self.device = 'cuda:0'
        # else:
        #     self.device = 'cpu'
        # TODO - support parallelization. GPT-J doesn't fit on one GPU
        self.device = 'cpu'
        # send to device
        self.model = self.model.to(self.device)

    def send_prompt(self, prompt, n_probs):
        # # send prompt to LM_GPTJ -- TODO how to test gptj (too big for colab)
        # input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        # gen_tokens = self.model.generate(input_ids, do_sample=True, max_length=1,)
        # logits = gen_tokens[0][:,-1,:]
        # create dictionary of output
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
