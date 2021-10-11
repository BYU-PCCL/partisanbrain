from lmsampler_baseclass import LMSamplerBaseClass
from lm_gpt3 import LM_GPT3
from lm_gpt2 import LM_GPT2
from lm_gptj import LM_GPTJ
from lm_gptneo import LM_GPTNEO
from lm_bert import LM_BERT

class LMSampler(LMSamplerBaseClass):
    '''
    Class to wrap all other LMSampler classes. This way, we can instantiate just by passing a model name, and it will initialize the corresponding class.
    '''
    def __init__(self, model_name):
        super().__init__(model_name)
        '''
        Supported models:
            - GPT-3: 'gpt3-ada', 'gpt3-babbage', 'gpt3-curie', 'gpt3-davinci', 'ada', 'babbage', 'curie', 'davinci'
            - GPT-2: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
            - GPT-J: 'EleutherAI/gpt-j-6B'
            - GPT-Neo: 'EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-125M'
            - BERT: 'bert-base-uncased', 'bert-base-cased'
        '''
        if model_name in ['gpt3-ada', 'gpt3-babbage', 'gpt3-curie', 'gpt3-davinci', 'ada', 'babbage', 'curie', 'davinci']:
            self.model = LM_GPT3(model_name)
        elif model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
            self.model = LM_GPT2(model_name)
        elif model_name in ['EleutherAI/gpt-j-6B']:
            self.model = LM_GPTJ(model_name)
        elif model_name in ['EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-125M']:
            self.model = LM_GPTNEO(model_name)
        elif model_name in ['bert-base-uncased', 'bert-base-cased']:
            self.model = LM_BERT(model_name)
        else:
            raise ValueError(f'Model {model_name} not supported.')

    def send_prompt(self, prompt, n_probs=100):
        return self.model.send_prompt(prompt, n_probs)

if __name__ == '__main__':
    # model_name = 'gpt3-ada'
    model_name = 'EleutherAI/gpt-j-6B'
    sampler = LMSampler(model_name)
    print(sampler.send_prompt('The best city in Spain is', 5))
    print(sampler.send_prompt('In 2016, I voted for', 5))