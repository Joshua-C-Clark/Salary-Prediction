DATA_DIR = 'Unprocessed_Data'
PROCESSED_DIR = 'Processed_Data'
DATA_FILE = 'data.csv'

TARGET = 'Total_Comp'

CATEGORICAL_IMPUTE = ['Company', 'Location', 'Job_Title', 'Subspecialty']

CATEGORICAL_ENCODE = ['Company', 'Location', 'Job_Title', 'Subspecialty', 'Role']

THRESHOLD = {'Company': 75, 'Location': 50, 'Job_Title': 100,
            'Subspecialty': 25,
            'Role': 5}

FEATURES = ['Company', 'Location', 'Job_Title', 'Subspecialty', 'Role']