from pickle import dump, load
from catboost import CatBoostClassifier
from income_model import IncomeModel
from config import *
import os

if __name__ == '__main__':

    model = IncomeModel(model = CatBoostClassifier, 
                    hparams = {**BEST_HP, 
                                'silent' : True,
                                'task_type' : 'GPU' if GPU else 'CPU',
                                 },#'iterations': 100}, for debug purposes
                    metadata_filename = DATA_PATH + METADATA_FNAME,
                    target_name = TARGET_NAME,
                    ignore_features = IGNORE_FEATURES,
                    features_processor = None, 
                    make_calculated_feats = True)

    model.fit_from_file(DATA_PATH + TRAIN_FNAME)

    model.evaluate_from_file(DATA_PATH + TEST_FNAME)

    model.plot_importances(save_path = '../docs/')

    model.save_to_pickle(MODELS_PATH, 'trained_model.pkl')

    # quick check if file dumped works as intended
    # with open(MODELS_PATH + 'trained_model.pkl', 'rb') as f:
    #     model = load(f)