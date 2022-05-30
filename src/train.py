import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, AutoTokenizer, RobertaTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer,BertForSequenceClassification, RobertaForSequenceClassification
from datasets import load_dataset, ClassLabel, Value, load_metric
import os
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler, ASHAScheduler
from ray.tune.suggest.basic_variant import BasicVariantGenerator
from utils import *


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def train_model(config):
    # load dataset
    Train = load_dataset('health_fact', split='train') 
    Val = load_dataset('health_fact', split='validation') 

    # data preprocess
    train, val = data_preprocess(Train), data_preprocess(Val)

    # embedding
    train_dataset, val_dataset = embedding(train, config), embedding(val, config)

    # load a pretrained model
    if config['model_name'] == 'bert':
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=4)
    elif config['model_name'] == 'roberta':
        model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=4)

    # set training arguments manually if needed, otherwise use the defalut
    training_args = TrainingArguments(
        output_dir='../output',          # output directory
        num_train_epochs=config['num_epochs'],              # total number of training epochs
        per_device_train_batch_size=config['batch_size'],  # batch size per device during training
        per_device_eval_batch_size=config['batch_size'],   # batch size for evaluation
        warmup_steps=config['warmup_steps'],                # number of warmup steps for learning rate scheduler
        learning_rate=config['lr'],               # learning rate
        logging_dir='../logs',            # directory for storing logs
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

    trainer.train() # resume_from_checkpoint=True if already trained, to save time by continuing on a checkpoint
    
    # save evaluation results for hyperparameter tuning
    evaluation = trainer.evaluate()
    tune.report(val_loss = evaluation['eval_loss'], accuracy = evaluation['eval_accuracy'])
    print(f"Validation Accuracy: {evaluation['eval_accuracy']}")
    
    return model, trainer



if __name__ == "__main__":
    # perform hybrid hyperparameter search with raytune
    num_samples = 10  # number of trails we plan to do for each model
    config = {
        'model_name':tune.grid_search(['bert', 'roberta']), 
        'batch_size':tune.choice([16, 32]),
        'lr':tune.choice([5e-3, 5e-4, 5e-5]),
        'num_epochs':tune.choice([3, 4, 5]),
        'warmup_steps':tune.choice([0, 500]),
    }   

    ray.init(ignore_reinit_error=False, include_dashboard=True, dashboard_host='0.0.0.0')
    sched = ASHAScheduler(
            max_t=200,
            grace_period=20,
            reduction_factor=2)
    analysis = tune.run(tune.with_parameters(train_model), config=config, num_samples=num_samples, metric='val_loss', mode='min',\
        scheduler=sched, resources_per_trial={"cpu": 10,"gpu": 1},local_dir="/scratch/zh2095/pubhealth/ray_results")

    best_config = analysis.get_best_config(mode='min')
    print('The best configs are: ',best_config, flush=True)
    ray.shutdown()


    # train the model with the best tuned config and save the model
    best_model, trainer = train_model(best_config)
    save_model(best_model, config)

