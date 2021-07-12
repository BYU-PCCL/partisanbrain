from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import torch
import pdb

# path = 'checkpoints/checkpoint-200000'
# path = 'checkpoints/old/checkpoint-100000'
path = 'checkpoints/checkpoint-1000000'
# path = 't5-11b'
base_size = 't5-11b'
if path == 't5-3b' or path == 't5-11b':
    model = T5ForConditionalGeneration.from_pretrained(path, cache_dir='cache_dir')
else:
    model = T5ForConditionalGeneration.from_pretrained(path)

tokenizer = T5Tokenizer.from_pretrained(base_size, cache_dir='cache_dir')

def generate(input_text):
# TODO - map to cuda?
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids

    # out = model.generate(
    #     input_ids,
    #     max_length = 30,
    #     num_beams = 3,
    #     no_repeat_ngram_size = 2,
    #     num_return_sequences = 3,
    #     early_stopping = True
    # )
    # for output in out:
    #     print(tokenizer.decode(output, skip_special_tokens=True))

    # print('Greedy')
    out = model.generate(input_ids)
    print(tokenizer.decode(out[0], skip_special_tokens=True))

    # print('Random x 2')
    for _ in range(2):
        out = model.generate(input_ids, do_sample=True, top_k=0)
        print(tokenizer.decode(out[0], skip_special_tokens=True))

tests = [
    'fill: America was founded in _1_.',
    'fill: America was founded in _32_.',
    'fill: Ask not what your country _4_ what you can do for your country.',
    'fill: I like to eat peanut butter and _1_ sandwiches.',
    'fill: There were two weeks left in the Trump administration when _16_ governing an obscure corner of the tax code.',
    'fill: I like to eat peanut butter and _16_ sandwiches.',
    # 'fill: I like to eat peanut butter and _32_ sandwiches.',
    'fill: Person: I just don\'t get why we need Black Lives Matter, since all lives matter. In fact, I think that whites get discriminated against because of affirmative action. That seems more racist to me.\nInterviewer: What if I were to tell you that _64_?\nPerson: I\'d never looked at it like that, I guess that makes sense. Maybe there is systemic racism against Blacks.',
    'fill: Person: I am a racist.\nInterviewer: What if I were to tell you that _4_?\nPerson: I am no longer a racist.',
    'fill: Person: I am a racist.\nInterviewer: What if I were to tell you that _8_?\nPerson: I am no longer a racist.',
    'fill: Person: I am a racist.\nInterviewer: What if I were to tell you that _16_?\nPerson: I am no longer a racist.',
    'fill:\ndef add(x, y):\n_4_\n    return z',
    'cola sentence: The course is jumping well.',
    # 'summarize: There were two weeks left in the Trump administration when the Treasury Department handed down a set of rules governing an obscure corner of the tax code.',
    'translate English to German: That is good.',
]

for test in tests:
    print(test)
    generate(test)
    print()

pdb.set_trace()
pass
