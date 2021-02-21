import pandas as pd
import os 
import settings
from pipeline import Pipeline
import joblib
from load import load

def train():
    data = pd.read_csv(os.path.join(settings.PROCESSED_DIR, settings.DATA_FILE), index_col= 0).reset_index(drop = True)
    pipeline = Pipeline(target =  settings.TARGET,
                    categorical_to_impute = settings.CATEGORICAL_IMPUTE,
                    numerical_log = settings.NUMERICAL_LOG,
                    categorical_encode = settings.CATEGORICAL_ENCODE,
                    features = settings.FEATURES
                    )

    model = pipeline.fit(data)
    joblib.dump(model, 'model.pkl')
    print('model performance')
    pipeline.evaluate_model()

if __name__ == '__main__':
    load()
    train()