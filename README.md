# pubhealth

## Usage
### Set up singularity environment for running in NYU HPC
This task is done on the NYU greene cluster, and the following steps describe the environment set up:
1. Clone this repo to greene scratch.

2. Create the two overlay files in the `scripts` folder that contains all the packages needed.

3. In the main directory of pubhealth, run `bash ./scripts/create_base_overlay.sh` and `bash ./scripts/create_package_overlay.sh` to create container overlay.

### File explanation
1. `src/utils.py` predefines some useful functions, such as preprocessing on the dataset, tokenization and embedding, etc.

2. `src/train.py` fine tunes the pretrained language model, and use `raytune` for model selection and hyperparameter tuning. Finally, the model fine-tuned with the best configuration will be saved and uploaded to huggingface for convenience. Run `sbatch src/train.sbatch` is to submit a SLURM job that cast the `train.py` inside the singularity container.

3. `predict.py` loads the fine-tuned model and predict the veracity of claims in the test dataset. Run `sbatch src/prediction.sbatch` will output the predicted veracity and evaluate the model with accuracy and F1 score as well as runtime.

## Workflow
### Task defination
The task is to verify the veracity of claims on the PUBHEALTH dataset. Noticing that `explanation` is the variable that gives the information of the veracity, the task can be seen as extracting feature and sentiment information from the text variable `explanation` and then predict on the veracity of claims. The veracity of claims has 4 classes (true, false, unproven, mixture), so this can be further considered as a sequence classification task, with 'explanation' as the text input, and veracity as target label.

### Methodology
I take advantage of the pretrained language models (PLMs), and use the mature pretrain-finetune paradigm to learn the language model. Among all types of PLMs, masked language models (MLMs) usually have a better performance and are more efficient than other model types(e.g. GPT and seq2seq models like T5) in sequence classification tasks. In this case, I choosed `BERT` and `RoBERTa` that are the most representative MLMs and can be easily imported from the `Huggingface` API. The task can be divided into the following 3 steps:
1. Data preprocessing and text embedding
    - Filter the useful columns
    - Remove the row with label '-1' since the dataset is supposed to have labels `0, 1, 2, 3` that denote `true, false, unproven, mixture`
    - Use the pretrained tokenizer to extract the feature in `explanation`, and the tokenizer is set to truncate at max length 128 for a balance of time consuming and information coverage.
2. Fine tune the PLMs and search for optimal hyperparameters
    - Import the PLMs from huggingface and fine tune on the training split of the dataset.
    - Apply hybrid search for hyperparamter tuning, specifically, grid search on the model type, and random search on the model config including number of epochs, batch size, learning rate and warmup steps. The tuning is done with `ray tune` because it is efficient and the tuning process can be monitored through the dashboard.
    - Retrain the model with the best tuned config, and save the fine-tuned model and upload it to Hugginface with the model name 'roberta-pubhealth' or 'bert-pubhealth.
    - The notebook `finetune_example.ipynb` gives an example of the workflow of finetuning.
3. Prediction and evaluation
    - Load the fine tuned model `roberta-pubhealth` and use it to predict the veracity on the test dataset
    - Evaluate the model performance  with `accuracy` and `F1 score` as this is a classification task
    - Evaluate the model efficiency with the number of sample predicted per second

### Test Results
The tuning results show that RoBERTa performs much better than BERT with an roughly 10% overall performance gain.

- The tuned best config is as following:
    - model: roberta base
    - batch size: 32
    - learning rate: 5e-5
    - number of epochs: 4
    - warmup steps: 500

- `roberta-pubhealth`, the RoBERTa model fine tuned on the PUBHEALTH dataset with above best config achieves the following results on the test dataset:\
    - micro f1(accuracy): 0.7137
    - macro f1: 0.6056
    - weighted f1: 0.7106
    - samples predicted per second: 9.31 

`results/output.txt`, the slurm output file of running the `predict.py`, contains the test results. The fine tuned model `roberta-pubhealth` has been uploaded to `Huggingface` as well  
