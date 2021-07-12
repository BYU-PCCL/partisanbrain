# from datasets.tasks import TextClassification
from transformers import T5Tokenizer
import torch
import numpy as np
import pandas as pd
import pdb
import json
import os
from torch.utils.data import IterableDataset
from unidecode import unidecode

def extract_words(text, n_ext):
    ''' Extracts a random number of words from the sequence '''
    words = text.split()
    n_words = len(words)
    n_ext = np.minimum( n_words - 1, n_ext )
    if n_ext <= 0:
        return ['']
        # raise Exception('This sequence is too short to extract words from')
    start_ix = np.random.randint(0, n_words - n_ext + 1)
    end_ix = start_ix + n_ext
    words = words[start_ix:end_ix]
    #joined_words = " ".join(words[start_ix:end_ix])
    return words

# base_size = 't5-3b' # or 't5-11b'
base_size = 't5-11b'
# TODO - add this as a param?
tokenizer = T5Tokenizer.from_pretrained(base_size)

def sized_fib_preprocessor(
        txt_instance,
        label='fill: ',
        # TODO -change
        # bins=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        bins=[1, 2, 4, 8, 16, 32, 64, 128],
        ):

    # input_text = str(txt_instance['text'].encode('utf-8'))
    input_text = unidecode(txt_instance['text'])

    #This is a problem, potentially, since 512 is the maximum
    #number of tokens, not of words
    # input_words = extract_words(input_text, 512)
    input_words = extract_words(input_text, 128)
    n_words = len(input_words)

    
    lo = binmin = np.min(bins)
    binmax = np.max(bins)
    hi = np.minimum(binmax, n_words) + 1


    blank_size = np.random.randint(lo, hi)
    bin_delta = np.abs(np.array(bins) - blank_size)
    bin_ = bins[np.argmin(bin_delta)]
    # print(blank_size, bin_)
    blank_start = np.random.randint(0,np.maximum(0, n_words-blank_size) + 1)
    pre_blank = " ".join(input_words[0:blank_start])
    post_blank = " ".join(input_words[blank_start+blank_size:])
    # if error, send input = 'fill: _0_', target = ''
    if n_words == 1 and input_words[0] == '':
        bin_ = 0
    blank = f"_{bin_}_"    
    # We strip to handle cases where blank is at beginning or end.
    input_ = " ".join((pre_blank, blank, post_blank)).strip()
    input_ = "".join((label, input_))
    target = " ".join(input_words[blank_start:blank_start+blank_size])
    # print(input_words, n_words, input_, target,'\n\n')
    #There is a filtering step here in the MTF code where they only keep 
    #examples that have at least one token. I'm not worried about this, prolly.

    tokenized_inputs = tokenizer(
        input_,
        return_tensors="pt",
        max_length = 512,
        # max_length = 64,
        # padding = 'max_length',
        truncation = True,
    )
    input_ids = tokenized_inputs.input_ids
    attention_mask = tokenized_inputs.attention_mask
    target_ids = tokenizer(
        target,
        return_tensors="pt",
        max_length = 512, # TODO - remove?
        # max_length = 64,
        # padding = 'max_length',
        truncation = True,
    ).input_ids

    # print(f'Input length: {len(input_ids[0])}')
    # print(f'Label length: {len(target_ids[0])}')
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': target_ids}


# TODO - enable workers to be more efficient?
class PileDataset(IterableDataset):
    def __init__(self, files):
        super().__init__()
        if not isinstance(files, list):
            files = [files]
        self.files = files

    def __iter__(self):
        for filepath in self.files:
            with open(filepath) as f:
                for id_, row in enumerate(f):
                    # dumped = json.dumps(row, ensure_ascii=False)
                    # data = json.loads(dumped)
                    data = json.loads(row)
                    # text = str(data['text']).encode('utf-8')
                    text = unidecode(data['text'])
                    if len(text.split()) < 3:
                        continue
                    data = sized_fib_preprocessor(data)
                    # data['text'] = text
                    for key in ['input_ids', 'attention_mask', 'labels']:
                          data[key] = data[key].reshape(-1)
                    #     data[key] = data[key].tolist()
                    yield data

# https://github.com/huggingface/transformers/issues/5990
# https://github.com/huggingface/transformers/pull/7858
#     def __len__(self):
#         return 7_000_000
