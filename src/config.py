DATA_PATH = './data/'  # path where data is stored
MODELS_PATH = './models/' # path where models are stored

TRAIN_FNAME = 'census_income_learn.csv' #train data
TEST_FNAME = 'census_income_test.csv'  #test data
METADATA_FNAME = 'census_income_metadata.txt'  #metadata file
MODELS_LOG_FNAME = 'experiments_log.csv'  #file to log models to

COLORS = ['#4DC9C3', '#221C35', '#FCCD20', '#20C3EF', '#00B257', '#FF7700'] #colors for plots

TARGET_NAME = 'income'  #column to predict
IGNORE_FEATURES = ['instance_weight'] #feature to ignore as of instructions

N_JOBS = 6 #number of parallel jobs for training
GPU = True #whether to use GPU for training

#best hyper parameters found in experiments, rounded
BEST_HP = {'learning_rate': 0.123, 'depth': 4, 'l2_leaf_reg': 0.95, 'iterations': 2480}