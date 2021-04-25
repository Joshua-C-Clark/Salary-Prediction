import pandas as pd
import os 
import settings
from pipe import Pipe
import joblib

def train():
    data = pd.read_csv(os.path.join(settings.PROCESSED_DIR, settings.DATA_FILE), index_col= 0).reset_index(drop = True)
    pipeline = Pipe(target =  settings.TARGET,
                    categorical_encode = settings.CATEGORICAL_ENCODE,
                    threshold = settings.THRESHOLD,
                    features = settings.FEATURES
                    )

    model = pipeline.fit(data)
    # test_item = pd.DataFrame(columns=['Company', 'Location', 
    #                         'Job_Title', 'Subspecialty','Role'])
    # test_item.loc[0] = ['US Navy','San Diego, CA', 'L4', 'Pickle','Other']
    # print('Estimated Salary: ${}'.format(model.predict(test_item.loc[0])))
    joblib.dump(model, 'model.pkl')
    print('model performance')
    pipeline.evaluate_model()

if __name__ == '__main__':
    train()