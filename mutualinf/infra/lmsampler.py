from lmsampler_baseclass import LMSamplerBaseClass
from lm_gpt3 import LM_GPT3
from lm_gpt2 import LM_GPT2

class LMSampler(LMSamplerBaseClass):
    '''
    Class to wrap all other LMSampler classes. This way, we can instantiate just by passing a model name, and it will initialize the corresponding class.
    '''
    def __init__(self, model_name):
        super().__init__(model_name)
        '''
        Supported models:
            - GPT3: 'gpt3-ada', 'gpt3-babbage', 'gpt3-curie', 'gpt3-davinci', 'ada', 'babbage', 'curie', 'davinci'
            - GPT2: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
        '''
        if model_name in ['gpt3-ada', 'gpt3-babbage', 'gpt3-curie', 'gpt3-davinci', 'ada', 'babbage', 'curie', 'davinci']:
            self.model = LM_GPT3(model_name)
        elif model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
            self.model = LM_GPT2(model_name)
        else:
            raise ValueError('Model name not supported. Supported models: gpt3, gpt2.')

    def send_prompt(self, prompt, n_probs=100):
        return self.model.send_prompt(prompt, n_probs)

if __name__ == '__main__':
    # model_name = 'gpt3-ada'
    model_name = 'gpt2'
    sampler = LMSampler(model_name)
    print(sampler.send_prompt('The best city in Spain is', 5))