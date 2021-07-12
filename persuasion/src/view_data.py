from transformers import T5Tokenizer
from PileIterableDataset import PileDataset
from torch.utils.data import DataLoader
import pdb

base_size = 't5-3b'

# train_dataset = PileDataset('data/pile/00mini.jsonl')
train_dataset = PileDataset(
    [
        'data/pile/00mini.jsonl',
        'data/pile/00.jsonl',
        'data/pile/01.jsonl',
        'data/pile/02.jsonl',
        'data/pile/03.jsonl',
        'data/pile/04.jsonl',
        'data/pile/05.jsonl',
        'data/pile/06.jsonl',
        'data/pile/07.jsonl',
        'data/pile/08.jsonl',
        'data/pile/09.jsonl',
        'data/pile/10.jsonl',
    ]
)
# val_dataset = PileDataset('data/pile/00mini.jsonl')

tokenizer = T5Tokenizer.from_pretrained(base_size)

# collator - automatically pads input from datasets
label_pad_token_id = tokenizer.pad_token_id

dataloader = DataLoader(
	train_dataset,
	batch_size = 1,
)

for i, batch in enumerate(dataloader):
    print(i)
    print(tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True))
    print(tokenizer.decode(batch['labels'][0], skip_special_tokens=True))
    pdb.set_trace()
    pass
