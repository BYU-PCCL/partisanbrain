from lmsampler import LMSampler

def download_models():
    '''
    Download all supported models.
    Supported models:
        - GPT-3: 'gpt3-ada', 'gpt3-babbage', 'gpt3-curie', 'gpt3-davinci', 'ada', 'babbage', 'curie', 'davinci'
        - GPT-2: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
        - GPT-J: 'EleutherAI/gpt-j-6B'
        - GPT-Neo: 'EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-125M'
        - BERT: 'bert-base-uncased', 'bert-base-cased'
    '''
    models = [
        # 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl',
        # 'EleutherAI/gpt-j-6B',
        # 'EleutherAI/gpt-neo-2.7B',
        'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-125M',
        'bert-base-uncased', 'bert-base-cased'
    ]
    for model in models:
        # instantiate model
        try:
            lm = LMSampler(model)
            # make sure it runs
            print(lm.send_prompt('My favorite city in Spain is', 5))
        except:
            print(f'Somethind went wrong!')

if __name__ == '__main__':
    download_models()
