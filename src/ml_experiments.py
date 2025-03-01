from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from pickle import load,dump
import os
import pandas as pd

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

def log_model(exp_name, fname, metrics, col_names = ['expriment_name', 'f1-score', 'precision', 'recall', 'conf_matrix']):

    #if log file exists, append
    if os.path.isfile(fname):
        existing_logs = pd.read_csv(fname)
        pd.concat([existing_logs,
                  pd.DataFrame([[exp_name, *metrics]], columns = existing_logs.columns)]).to_csv(fname, index = False)

    #log file doesnt exist yet
    else:
        pd.DataFrame([[exp_name, *metrics]], columns = col_names).to_csv(fname, index = False)



def save_model_pickle(model, fname):

    with open(fname, 'wb') as f:
        dump(model, f)

def load_model_pickle(fname):

    if os.path.isfile(fname):
        with open(fname, 'rb') as f:
            return load(f)
    else:
        print('File not found')
