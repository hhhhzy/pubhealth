import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, AutoTokenizer, RobertaTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer,BertForSequenceClassification, RobertaForSequenceClassification
from datasets import load_dataset, ClassLabel, Value, load_metric


def data_preprocess(data):
    # only keep the useful columns 'explanation' and 'label'
    data = data.remove_columns(['claim', 'claim_id', 'date_published', 'fact_checkers', 'main_text', 'sources', 'subjects'])
    
    # remove the invalid data whose label is '-1'(should be 0,1,2,3)
    data = data.filter(lambda x: x['label'] != -1)
    return data


def embedding(data, config):
    # load the tokenizer pretrained on the selected model
    if config['model_name'] == 'bert':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    elif config['model_name'] == 'roberta':
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    # truncate at length=256 for a balance of time consuming and information coverage 
    data_tokenized = data.map(lambda batch: tokenizer(batch['explanation'], padding='max_length', truncation=True, max_length=256))

    return data_tokenized


def compute_metrics(predictions, truth, metric_name):
    if metric_name == 'weighted f1':
        metric = load_metric('f1')
        score = metric.compute(predictions=predictions, references=truth, average='weighted')
    elif metric_name == 'micro f1':
        metric = load_metric('f1')
        score = metric.compute(predictions=predictions, references=truth, average='micro')
    elif metric_name == 'macro f1':
        metric = load_metric('f1')
        score = metric.compute(predictions=predictions, references=truth, average='macro')

    return score


def compute_accuracy(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric = load_metric("accuracy")
    accuracy = metric.compute(predictions=predictions, references=labels)
    return accuracy


# save the best finetuned model
def save_model(model, config):
    name = config['model_name']+ '-pubhealth'
    model.save_pretrained('../name')
    print('The best fine-tuned model has been saved as: ', name, flush=True)
