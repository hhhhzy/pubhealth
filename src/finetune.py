import torch
import pandas as pd
import numpy as np
import gc
from transformers import BertTokenizer, AutoTokenizer, RobertaTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer,BertForSequenceClassification, RobertaForSequenceClassification
from datasets import load_dataset, ClassLabel, Value, load_metric
from .utils import *




def finetune(train_dataset, val_dataset, config):
    # load a pretrained model
    if config['model_name'] == 'bert':
        model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=4)
    elif config['model_name'] == 'roberta':
        model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=4)

    # set training arguments manually if needed, otherwise use the defalut
    training_args = TrainingArguments(
        output_dir='./output',          # output directory
        num_train_epochs=config['num_epochs'],              # total number of training epochs
        per_device_train_batch_size=config['batch_size'],  # batch size per device during training
        per_device_eval_batch_size=config['batch_size'],   # batch size for evaluation
        warmup_steps=config['warmup_steps'],                # number of warmup steps for learning rate scheduler
        learning_rate=config['lr'],               # learning rate
        logging_dir='./logs',            # directory for storing logs
        logging_steps=1000,
        evaluation_strategy='epoch'
    )


    # Create a Trainer object with the model, training arguments, training and test datasets, and evaluation function
    trainer = Trainer(
        model=model,
        args = training_args,
        train_dataset=train_dataset,
        eval_dataset = val_dataset,
        compute_metrics=compute_accuracy)
    
    # clean up gpu cache before training
    gc.collect()
    torch.cuda.empty_cache()

    trainer.train()
    # trainer.train(resume_from_checkpoint=True) # True if already trained, to save time by continuing on a checkpoint
    
    # save model
    model.save_pretrained(config+ '-pubhealth')


    
    


