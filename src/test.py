import torch
import pandas as pd
import numpy as np
import time
from transformers import BertTokenizer, AutoTokenizer, RobertaTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer,BertForSequenceClassification, RobertaForSequenceClassification
from datasets import load_dataset, ClassLabel, Value, load_metric
from utils import *


def test(best_config):
    # load the test dataset
    Test = load_dataset('health_fact', split='test') 

    # data preprocess and embedding
    test =  data_preprocess(Test)
    test_dataset = embedding(test, best_config)

    # load the fine-tuned model
    model = AutoModelForSequenceClassification.from_pretrained("../roberta-pubhealth", num_labels=4)

    # predict the veracity label with the fine-tuned model
    trainer = Trainer(model=model)

    # measure for model efficiency
    start_time = time.time()
    output = trainer.predict(test_dataset)
    runtime = time.time()-start_time
    pred_second = len(output)*best_config['batch_size']/runtime
    print(' ')
    print('-'*20 + ' Measure for Model Efficiency ' + '-'*20, flush=True)
    print(f'Number of samples predicted per second: {pred_second}')

    # measure for model performance
    predictions = output[0].argmax(axis=1)
    truth = output[1]
    print('-'*20 + ' Measure for Model Performance ' + '-'*20, flush=True)
    for metric_name in ['weighted f1', 'micro f1', 'macro f1']:
        score = compute_metrics(predictions, truth, metric_name)
        print(metric_name + f': {score}', flush=True)


if __name__ == "__main__":
    # copy the best config from the output file of train
    best_config = {
        'model_name': 'roberta', 
        'batch_size': 32,
        'lr': 5e-5,
        'num_epochs': 4,
        'warmup_steps': 0,
    }  
    print('-'*20 + 'Best Config' + '-'*20, flush=True)
    print(best_config, flush=True)
    
    test(best_config)