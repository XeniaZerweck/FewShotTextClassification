# ========================== Packages ==========================
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from setfit import SetFitModel, Trainer 
from optuna import Trial
from typing import Dict, Any, Union
import warnings
import time

# ========================== Preparations ==========================
warnings.simplefilter("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========================== Data Import ==========================
full_data_set = pd.read_csv('./data.csv', sep=';') # len(full_data_set) = 3497
full_data_set = full_data_set.dropna(axis=0, how='any') # len(full_data_set) = 3496

# ========================== Definition of Functions ==========================
# Split into train and test
X, y = full_data_set['sentences'], full_data_set['label']
full_X_train, X_temp, full_y_train, y_temp = train_test_split(X, y, test_size=0.9, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

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
    max_iter = params.get("max_iter", 100)
    #solver = params.get("solver", "liblinear")
    #model_body = SentenceTransformer("sentence-transformers/all-mpnet-base-v2") # https://huggingface.co/sentence-transformers/all-mpnet-base-v2, https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
    #model_head = GradientBoostingClassifier(loss='log_loss') # https://huggingface.co/docs/setfit/how_to/classification_heads
    #model = SetFitModel(model_body=model_body, model_head=model_head)
    model = SetFitModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
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
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'subsample': trial.suggest_float('subsample', 0.0, 1.0),
        'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        }
# Define sample sizes
sample_sizes = [1, 3, 5, 10, 20, 30, 40, 50]
durations = []
best_scores = []

# Save datasets as CSV
def save_dataset(dataset, desired_name_of_dataset_file):
    dataset_df = dataset.to_pandas()
    dataset_df.to_csv(f'./{desired_name_of_dataset_file}.csv')

# ========================== Modelling ==========================
for sample_size in sample_sizes:

    start_time = time.time()

    train_df = pd.DataFrame({'sentences': full_X_train, 'label': full_y_train})
    if sample_size == 3:
        train_df.to_csv('train_dataset_LogisticRegression.csv')
    train_df = train_df.groupby('label').apply(lambda x: x.sample(n=sample_size)).reset_index(drop=True)
    X_train, y_train = train_df['sentences'], train_df['label']
    train_dataset, eval_dataset, test_dataset = turn_into_dataset(X_train, y_train, X_val, y_val, X_test, y_test)

    if sample_size == 3:
        save_dataset(test_dataset, 'test_dataset_LogisticRegression')
        save_dataset(eval_dataset, 'eval_dataset_LogisticRegression')

    len_train_dataset = len(train_dataset)
    print(f'Length of Training Dataset: {len_train_dataset}')

    trainer = Trainer(
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        model_init=model_init,
        metric= 'f1',
        metric_kwargs = {'average': 'macro'}, 
    )

    best_run = trainer.hyperparameter_search(direction="maximize", hp_space=hp_space_for_optuna, n_trials=100)

    trainer.apply_hyperparameters(best_run.hyperparameters, final_model=True)
    trainer.train()

    model_name = f'LogisticRegression_model_{sample_size}_samples_per_label'
    trainer.model.save_pretrained(model_name)
    print(f'Run for {model_name} complete.')

    end_time = time.time()
    duration = end_time - start_time
    duration_min = int(duration / 60)
    duration_hour = duration_min / 60
    
    durations.append(duration_min)
    best_scores.append(best_run.objective)
    
    print(best_scores)
    print(durations)
    print(f'Iteration for {model_name} took {duration_min} minutes or {duration_hour:.2f} hours.')


'''
# ========= RESULTS =========
n_samples = [1, 3, 5, 10, 20, 30, 40, 50]

# === SVC ===
[0.17795678011885466, 0.40030983096485634, 0.46292115964002584,
 0.4127155655123904, 0.5191894168469259, 0.5391068254846626,
 0.5366664831299305, 0.5716833095873302]

 # === GradientBoostingClassifier === 
[0.3106745381703343, 0.41039541009045677, 0.4230810739777218,
0.4674327670126357, 0.48473460756875325, 0.49821409462245564,
0.5272439995389141, 0.5909367814858419]

# === LogisticRegression ===
[0.40029805116420974, 0.4049049337148445, 0.4480098573748999, 
0.40798545224541427, 0.44940168071820824, 0.5388840494877835,
0.5622384186485875, 0.5807185593761462]
'''

# ========================== Push Models to Hub ==========================
model_kinds = ['RandonForest', 'SVC', 'LogisticRegression']

for model_kind in model_kinds:
    for sample_size in sample_sizes:

        model_name = (f'{model_kind}_model_{sample_size}_samples_per_label')
        model = SetFitModel.from_pretrained(f'./{model_name}')

        if model_kind == 'RandonForest':
            model_name = (f'GradientBoosting_model_{sample_size}_samples_per_label')
        else:
            model_name = (f'{model_kind}_model_{sample_size}_samples_per_label')

        print(f'Pushing {model_name} to hub.')
        save_path = f'Catchy1282/{model_name}'
        model.push_to_hub(save_path)

