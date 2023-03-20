import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import requests
from tqdm import tqdm
import logging
import math
logger = logging.getLogger(__name__)
import urllib
import zipfile
from transformers import DataCollatorForLanguageModeling, TrainingArguments, BertTokenizer, BertModel
import math

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

def tokenize_function(examples):
  result = tokenizer(examples["text"])
  if tokenizer.is_fast:
      result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
  return result


# Use batched=True to activate fast multithreading!
tokenized_datasets = raw_dataset.map(
    tokenize_function, batched=True, remove_columns=['sentence1', 'sentence2', 'label', 'idx', 'text']
)

batch_size = 32
# Show the training loss with every epoch
logging_steps = len(tokenized_datasets["train"]) // batch_size

training_args = TrainingArguments(
    output_dir='out',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
)

from transformers import Trainer
model_mlm = BertForMaskedLM.from_pretrained('bert-base-uncased')
out_path = "/saved_models/mlm/"

trainer = Trainer(
    model=model_mlm,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)

trainer.train()

eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")