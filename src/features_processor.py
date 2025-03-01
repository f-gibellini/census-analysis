import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from src.constants import *

class FeaturesProcessor():

    def __init__(self, cat_encoding = 'OHE', encoders = None, 
                 age_buckets = [(18,25),(25, 35),(35,45),(45,55),(55, 65),(65,120)],
                 work_weeks_buckets = [(0,10), (10,30), (30,45), (45,52)],
                 investment_buckets = [(0,2000), (2000, 10000), (10000,100000), (100000,10000000)],
                 marital_status_mapping = {
                    'widowed':0, 'divorced':0, 'never married':0, 'separated':0, 'married-spouse absent':0,
                    'married-civilian spouse present':1, 'married-a f spouse present':1 },
                 working_h_per_week = 40
                ):

        self.cat_encoding = cat_encoding
        self.encoders = encoders
        self.age_buckets = age_buckets
        self.work_weeks_buckets = work_weeks_buckets
        self.investment_buckets = investment_buckets
        self.marital_status_mapping = marital_status_mapping
        self.working_h_per_week = working_h_per_week


    def _make_encoder(self, df, col):

        mode = self.cat_encoding
        
        if mode == 'OHE':
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        else:
            encoder = LabelEncoder()
        
        encoder.fit(df[[col]])
        self.encoders[col] = encoder


    
    def make_calculated_features(self, df):

        #discretize age
        for ab in self.age_buckets:
            df[f'age_bucket_{ab[0]}_{ab[1]}'] = df.age.apply(lambda x:1 if (x>=ab[0] and x<ab[1]) else (0)).astype(int)

        for wb in self.work_weeks_buckets:
            df[f'ww_bucket_{wb[0]}_{wb[1]}'] = df.weeks_worked_in_year.apply(lambda x:1 if (x>=wb[0] and x<wb[1]) else (0)).astype(int)

        # calculate overall income from investments
        df['investments_income'] = (df.capital_gains + df.dividends_from_stocks - df.capital_losses).astype(int)

        # calculate discrete buckets for it
        for wb in self.investment_buckets:
            df[f'ww_bucket_{wb[0]}_{wb[1]}'] = df.investments_income.apply(lambda x:1 if (x>=wb[0] and x<wb[1]) else (0)).astype(int)

        # simplify marital status
        df['marital_status_simple'] = df.marital_stat.map(self.marital_status_mapping).astype(int)

        # look for different type of employment and simplify
        df['work_govm'] = df.class_of_worker.fillna('').apply(lambda x:1 if ('government' in x) else(0)).astype(int)
        df['work_pvt'] = df.class_of_worker.fillna('').apply(lambda x:1 if ('private' in x) else(0)).astype(int)
        df['work_self'] = df.class_of_worker.fillna('').apply(lambda x:1 if ('self-employed' in x) else(0)).astype(int)

        # calculate yearly salary 
        df['tot_salary'] = (df.wage_per_hour*df.weeks_worked_in_year*self.working_h_per_week).astype(int)
        
        
        return df
    
    def transform_features(self, df, make_calculated = False):

        mode = self.cat_encoding

        #if want to make calculated features
        if make_calculated:
            df = self.make_calculated_features(df)
        
        if mode not in ['OHE', 'LE']:
            print(f'Unsupported mode {mode} defaulting to One Hot Encoding')
            mode = 'OHE'

        if self.encoders is None:
            print('Encoders not found, creating new encoders')
            self.encoders = {}
        else:
            print('Encoders were found, using existing encoders')
        
        encoded_df = pd.DataFrame()
        
        for col in df.columns:
            #if column is object (not numerical)
            if (df[col].dtype == 'O') and (col != TARGET_NAME):

                if not (col in self.encoders.keys()):
                    self._make_encoder(df, col)
                
                encoder = self.encoders[col]

                encoded_features = encoder.transform(df[[col]])
                if mode == 'OHE':
                    feat_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                else:
                    feat_names = [col]
                    
                encoded_df = pd.concat([encoded_df,
                                        pd.DataFrame(encoded_features, columns=feat_names)], axis=1)
            elif (df[col].dtype != 'O') and (col != TARGET_NAME):
                encoded_df[col] = df[col]
            
            else: #target column
                encoder = LabelEncoder()
        
                encoder.fit(df[[col]])
                self.encoders[col] = encoder 
                encoded_df[col] = encoder.transform(df[[col]])
                
        return encoded_df, self.encoders