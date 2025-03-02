
from config import *
from features_processor import FeaturesProcessor
from data_reader import read_data
from ml_experiments import evaluate_model
from matplotlib import pyplot as plt
from catboost import CatBoostClassifier
import warnings
from sklearn.metrics import classification_report
import os
from pickle import dump, load

warnings.filterwarnings('ignore')


class IncomeModel():

    def __init__ (self, model, hparams, metadata_filename, target_name = 'income',
                    ignore_features = [],
                    features_processor = None, make_calculated_feats = True):

        self.model = model(**hparams)

        self.model_trained = False
        
        self.metadata_filename = metadata_filename
        self.make_calculated_feats = make_calculated_feats
        self.target_name = target_name
        self.ignore_features = ignore_features

        if features_processor is None:
            self.features_processor = FeaturesProcessor('OHE')
        else:
            self.features_processor = features_processor

        self.feat_names = []


    def make_features_from_file(self, filename):

        df = read_data(filename, self.metadata_filename, drop_cols = self.ignore_features)

        df, _ = self.features_processor.transform_features(df, make_calculated = self.make_calculated_feats)

        return df


    def fit_from_file(self, filename):

        encoded_train_df = self.make_features_from_file(filename)

        self.feat_names = [col for col in encoded_train_df.columns if (col != self.target_name)]
        
        print('Training model on data with shape', encoded_train_df.shape)

        self.fit( encoded_train_df[self.feat_names], encoded_train_df[self.target_name])
  
        return self

    def evaluate_from_file(self, filename):

        encoded_test_df = self.make_features_from_file(filename)

        if len(self.feat_names) < 1:
            self.feat_names = [col for col in encoded_test_df.columns if (col != self.target_name)]

        self.evaluate(encoded_test_df[self.feat_names], encoded_test_df[self.target_name])


    def fit(self, x, y):

        self.model.fit(x, y)
        self.model_trained = True

        print('Model Trained\n')

        return self


    def predict(self, x):

        return self.model.predict(x)

    def evaluate(self, x, y_true):

        #check if model was trained
        if self.model_trained:
            print(f'Evaluating on {len(x)} data points')

            y_pred = self.predict(x)
            print(classification_report(y_pred, y_true))
            print({k:v for k,v in zip(['f1_score', 'precision', 'recall', 'conf_matrix'], evaluate_model(y_pred, y_true))})
            print()
        
        else:
            print('Cannot evaluate a un-trained model, please fit model first')



    def plot_importances(self, max_importances: int = 20, save_path = None):
        
        #check if model was trained
        if self.model_trained:
            model = self.model

            importances = sorted([(name,imp/max(model.feature_importances_)) for name, imp in zip(self.feat_names, model.feature_importances_)], 
                                key = lambda x:-x[1])[:max_importances]

            plt.figure(figsize=(14, 8))
            plt.bar([e[0] for e in importances], [e[1] for e in importances], color = COLORS[0])
            plt.title('Feature Importances')
            plt.xticks(rotation = 90)
            
            #save figure if required
            if not (save_path is None):
                if os.path.isdir(save_path):
                    plt.savefig(save_path + 'feat_importances.png')
                    print(f'\nImportances plot saved to {save_path}/feat_importances.png')

            plt.show()

        else:
            print('\nCannot produce importances of a un-trained model, please fit model first')

    def save_to_pickle(self, path, fname = 'trained_model.pkl'):
        
        #make sure folder exists
        if not os.path.isdir(path):
            os.mkdirs(path)
        
        #write model to file
        with open(path + fname, 'wb') as f:
            dump(self, f)

        print(f'\nModel Saved to Pickle file {path + fname}')