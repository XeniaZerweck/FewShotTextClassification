import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from setfit import SetFitModel, Trainer 
from optuna import Trial
from typing import Dict, Any, Union
import warnings
import time
import subprocess

# ========================== Preparations ==========================
warnings.simplefilter('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ========================== Data Import ==========================
full_data_set = pd.read_csv('/Users/xeniamarlenezerweck/Documents/Verzeichnis/Master/Thesis/Daten/numbers_files/data.csv', sep=';') # len(full_data_set) = 3497
full_data_set = full_data_set.dropna(axis=0, how='any') # len(full_data_set) = 3496

# ========================== Splitting Data ==========================
# Split into train and test
X, y = full_data_set['sentences'], full_data_set['label']
full_X_train, X_temp, full_y_train, y_temp = train_test_split(X, y, test_size=0.9, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Calculate class weights
class_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(y), y = y)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# ========================== Definition of Functions ==========================
# Data preprocessing function
def turn_into_dataset(X_train, y_train, X_val, y_val, X_test, y_test):
    train_dataset = Dataset.from_pandas(pd.DataFrame({'text': X_train.reset_index(drop=True), 
                                                    'label': y_train.reset_index(drop=True)}))
    eval_dataset = Dataset.from_pandas(pd.DataFrame({'text': X_val.reset_index(drop=True), 
                                                    'label': y_val.reset_index(drop=True)}))
    test_dataset = Dataset.from_pandas(pd.DataFrame({'text': X_test.reset_index(drop=True), 
                                                    'label': y_test.reset_index(drop=True)}))
    return train_dataset, eval_dataset, test_dataset

# Model initialisation
def model_init(params: Dict[str, Any]) -> SetFitModel:
    params = params or {}
    max_iter = params.get('max_iter', 100)
    model_body = SentenceTransformer('sentence-transformers/all-mpnet-base-v2') # https://huggingface.co/sentence-transformers/all-mpnet-base-v2, https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
    model_head = SVC(max_iter=max_iter, class_weight = class_weights_dict) # https://huggingface.co/docs/setfit/how_to/classification_heads
    model = SetFitModel(model_body=model_body, model_head=model_head)
    return model

# Hyperparameter search space function
def hp_space_for_optuna(trial: Trial) -> Dict[str, Union[float, int, str]]:
    return {
        # for sentence transformer
        'body_learning_rate': trial.suggest_float('body_learning_rate', 1e-5, 1e-3),
        'batch_size': trial.suggest_int('batch_size', 4, 40),
        'num_epochs': trial.suggest_int('num_epochs', 1, 10),
        
        # for classification head
        'head_learning_rate': trial.suggest_float('head_learning_rate', 1e-5, 1e-3),
        'C': trial.suggest_float('C', 1e-5, 1e-0), # squared l2 penalty, see SVC docs
        'kernel': trial.suggest_categorical('kernel', ['linear', 'poly' 'rbf', 'sigmoid'])

        }

# Save datasets as CSV
def save_dataset(dataset, desired_name_of_dataset_file):
    dataset_df = dataset.to_pandas()
    dataset_df.to_csv(f'./{desired_name_of_dataset_file}.csv')

# ========================== Misc Preparation ==========================
# Define sample sizes
sample_sizes = [3, 5, 10, 20, 30, 40, 50]

# For logging
durations = []
best_scores = []

# ========================== Modelling ==========================
for sample_size in sample_sizes:
    # Time counter
    start_time = time.time()

    # Pull balanced training data
    train_df = pd.DataFrame({'sentences': full_X_train, 'label': full_y_train})
    train_df = train_df.groupby('label').apply(lambda x: x.sample(n=sample_size)).reset_index(drop=True)
    if sample_size == 3:
        train_df.to_csv('./train_dataset_SVC.csv')
        subprocess.run(['git', 'add', './train_dataset_SVC.csv'])
    X_train, y_train = train_df['sentences'], train_df['label']
    train_dataset, eval_dataset, test_dataset = turn_into_dataset(X_train, y_train, X_val, y_val, X_test, y_test)

    if sample_size == 3:
        subprocess.run(['git', 'add', './test_dataset_SVC.csv', './eval_dataset_SVC.csv'])

    # Save data for reproducibility
    if sample_size == 3:
        save_dataset(test_dataset, 'test_dataset_SVC')
        save_dataset(eval_dataset, 'eval_dataset_SVC')

    # Initiate trainer
    trainer = Trainer(
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        model_init=model_init,
        metric= 'f1',
        metric_kwargs = {'average': 'macro'}, # should be 'macro'
    )

    # Save best run 
    best_run = trainer.hyperparameter_search(direction='maximize', hp_space=hp_space_for_optuna, n_trials=50)
    trainer.apply_hyperparameters(best_run.hyperparameters, final_model=True)
    trainer.train()
    model_name = f'SVC_model_{sample_size}_samples_per_label'
    trainer.model.save_pretrained(model_name)

    # Push to git
    subprocess.run(['git', 'add', f'{model_name}'])
    commit_message = f'Add {model_name} and datasets for sample size {sample_size}'
    subprocess.run(['git', 'commit', '-m', commit_message])
    
    subprocess.run(['git', 'push'])

    # End time count
    end_time = time.time()
    duration = end_time - start_time
    duration_min = int(duration / 60)
    duration_hour = duration_min / 60

    # Print and log key data for overview over training process
    durations.append(duration_min)
    best_scores.append(best_run.objective)
    print(f'Iteration for {model_name} took {duration_min} minutes or {duration_hour:.2f} hours.')
    print(f'Durations so far: {durations}')
    print(f'Best Scores so far: {best_scores}')
    print(f'Run for {model_name} complete.')
