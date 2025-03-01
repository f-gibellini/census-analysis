class FeaturesProcessor():

    def __init__(self, cat_encoding = 'OHE', encoders = None):

        self.cat_encoding = cat_encoding
        self.encoders = encoders


    def _make_encoder(self, df, col):

        mode = self.cat_encoding
        
        if mode == 'OHE':
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        else:
            encoder = LabelEncoder()
        
        encoder.fit(df[[col]])
        self.encoders[col] = encoder        
    
    def transform_cat_features(self, df):

        mode = self.cat_encoding
        
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
            if (train_df[col].dtype == 'O') and (col != TARGET_NAME):

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
            elif (train_df[col].dtype != 'O') and (col != TARGET_NAME):
                encoded_df[col] = df[col]
            
            else: #target column
                encoder = LabelEncoder()
        
                encoder.fit(df[[col]])
                self.encoders[col] = encoder 
                encoded_df[col] = encoder.transform(df[[col]])
                
        return encoded_df, self.encoders