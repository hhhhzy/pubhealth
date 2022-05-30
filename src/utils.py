import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, AutoTokenizer, RobertaTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer,BertForSequenceClassification, RobertaForSequenceClassification
from datasets import load_dataset, ClassLabel, Value, load_metric


def data_preprocess(data):
    # only keep the useful columns
    data = data.remove_columns(['claim_id', 'date_published', 'fact_checkers', 'main_text', 'sources', 'subjects'])

    return data


def embedding(data, config):
    # load the tokenizer pretrained on the selected model
    if config['model_name'] == 'bert':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    elif config['model_name'] == 'roberta':
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    # truncate at length=128 for a balance of time consuming and information coverage
    data_tokenized = data.map(lambda batch: tokenizer(batch['explanation'], padding='max_length', truncation=True, max_length=128))

    return data_tokenized


def compute_metrics(eval_pred, metric_name):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    if metric_name == 'accuracy':
        metric = load_metric('accuracy')
        score = metric.compute(predictions=predictions, references=labels)
    elif metric_name == 'f1':
        metric = load_metric('f1')
        score = metric.compute(predictions=predictions, references=labels, average='weighted')

    return score


def compute_accuracy(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric = load_metric("accuracy")
    accuracy = metric.compute(predictions=predictions, references=labels)
    return accuracy

def compute_f1(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric = load_metric("f1")
    accuracy = metric.compute(predictions=predictions, references=labels, average='weighted')
    return accuracy