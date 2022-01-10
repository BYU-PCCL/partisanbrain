from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    T5Config,
    Trainer,
    Seq2SeqTrainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Adafactor,
)
from PileIterableDataset import PileDataset
from torch.utils.data import DataLoader
import logging
import pdb

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(message)s'
)

# TODO - enable different tokenizer

device = 'cuda'
base_size = 't5-11b'

# train_dataset = PileDataset('data/pile/00mini.jsonl')
train_dataset = PileDataset(
    [
        # 'data/pile/00mini.jsonl',
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

tokenizer = T5Tokenizer.from_pretrained(base_size, cache_dir='cache_dir')

# collator - automatically pads input from datasets
label_pad_token_id = tokenizer.pad_token_id

learning_rate = 1e-5
# learning_rate = 1e-3

batch_size = 1
training_args = TrainingArguments(
    output_dir = 'checkpoints',
    adafactor = True, # T5 paper uses AdaFactor with lr = 1e-3
    learning_rate = learning_rate,
    # learning_rate = 1e-5,
    save_strategy = 'steps',
    # save_steps = 500,
    save_steps = 50_000,
    logging_steps = 5_000,
    # logging_steps = 100,
    # logging_steps = 100,
    max_steps = 2_000_000, #TODO - change. Needed for dataset
    per_device_train_batch_size = batch_size,
    lr_scheduler_type = 'constant',
    # dataloader_num_workers = 8,
)

model = T5ForConditionalGeneration.from_pretrained(base_size, cache_dir='cache_dir')

# device_map = {
#     0: [0, 1, 2, 3, 4, 5],
#     1: [6, 7, 8, 9, 10, 11],
#     2: [12, 13, 14, 15, 16, 17],
#     3: [18, 19, 20, 21, 22, 23],
# } 
device_map = {
    0: [0, 1, 2],
    1: [3, 4, 5],
    2: [6, 7, 8],
    3: [9, 10, 11],
    4: [12, 13, 14],
    5: [15, 16, 17],
    6: [18, 19, 20],
    7: [21, 22, 23],
} 

# parallelize across gpus
model.parallelize(device_map)

class T5Trainer(Seq2SeqTrainer):
    def create_optimizer(self):
        self.optimizer = Adafactor(
            self.model.parameters(),
            lr=learning_rate,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
        )

# trainer = Trainer(
trainer = T5Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    # eval_dataset = val_dataset,
    tokenizer = tokenizer,
    # data_collator = data_collator,
)
# TODO - resume from checkpoint?
# checkpoint = None
checkpoint = 'checkpoints/checkpoint-120000'
train_result = trainer.train(resume_from_checkpoint=checkpoint)
trainer.save_model()

metrics = train_result.metrics
metrics['train_samples'] = len(train_dataset)
trainer.log_metrics('train', metrics)
trainer.save_metrics('train', metrics)
trainer.save_state()
