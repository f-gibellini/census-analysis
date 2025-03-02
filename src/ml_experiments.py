from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from pickle import load,dump
import os
import pandas as pd
import optuna
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from config import *

def evaluate_model(y_true, y_pred):

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    return [f1, precision, recall, str(conf_matrix)]


def get_model_string(model):
    model_name = model.__class__.__name__
    params = model.get_params()
    param_str = ",".join(f"{k}={v}" for k, v in params.items())
    return f"{model_name}({param_str})"

def log_model(exp_name, model_name, fname, metrics, col_names = ['experiment_name', 'model_name', 'f1_score', 'precision', 'recall', 'conf_matrix']):

    #if log file exists, append
    if os.path.isfile(fname):
        existing_logs = pd.read_csv(fname)
        pd.concat([existing_logs,
                  pd.DataFrame([[exp_name, model_name, *metrics]], columns = existing_logs.columns)]).to_csv(fname, index = False)

    #log file doesnt exist yet
    else:
        pd.DataFrame([[exp_name, model_name, *metrics]], columns = col_names).to_csv(fname, index = False)


def objective(trial, train_data, feat_names):
    
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.9),
        'depth': trial.suggest_int('depth', 2, 8),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.01, 1),
        'iterations': trial.suggest_int('iterations', 100, 2500),
    }
    
    model = CatBoostClassifier(
        **params,
        random_seed=32,
        silent = True,
        task_type = 'GPU' if GPU else 'CPU'
    )
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=32)
    f1_scores = cross_val_score(
        model,
        train_data[feat_names],
        train_data[TARGET_NAME],
        scoring='f1',
        cv=cv,
        n_jobs=1
    )
    
    return np.mean(f1_scores)

def run_optuna_study(train_data, feat_names, n_trials=100):
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, train_data, feat_names), n_trials=n_trials) 
    
    print(f"Best F1 Score: {study.best_value}")
    print("Best Parameters:", study.best_params)
    
    return  study.best_params, study.best_value


def save_model_pickle(model, fname):

    with open(fname, 'wb') as f:
        dump(model, f)

def load_model_pickle(fname):

    if os.path.isfile(fname):
        with open(fname, 'rb') as f:
            return load(f)
    else:
        print('File not found')
