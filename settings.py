DATA_DIR = 'Unprocessed Data'
PROCESSED_DIR = 'Processed Data'
DATA_FILE = 'data.csv'

TARGET = 'Total_Comp'
CV_FOLDS = 3

FEATURES = ['Company', 'Location','Job_Title', 'Subspecialty',
            'Role']

CATEGORICAL_IMPUTE = ['Company', 'Job_Title', 'Subspecialty']

CATEGORICAL_ENCODE = ['Company', 'Location', 'Job_Title', 'Subspecialty', 'Role']

NUMERICAL_LOG = ['Total_Comp']