import os

from pandas.core.arrays.sparse import dtype
import settings
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


class Pipe:

    def __init__(self, target, categorical_encode, threshold,
                features, test_size = 0.2, random_state = 0):
                
        # Data sets
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.threshold = threshold

        self.enc = OneHotEncoder(drop='first', sparse=False)

        self.model = LinearRegression()

        self.threshold_dict = {}

        self.target = target
        self.features = features
        self.categorical_encode = categorical_encode
        self.test_size = test_size
        self.random_state = random_state


    def target_preparation(self, data):
        data = data.copy()
        data = data.dropna(subset= [self.target], axis=0)
        data[self.target] = data[self.target].apply(
            lambda x: str(x).replace('$','').replace(',','')).astype(float)
        return data

    def outlier_removal(self, data):
        data = data.copy()
        Q1 = data[self.target].quantile(0.25)
        Q3 = data[self.target].quantile(0.75)
        IQR = Q3 - Q1
        outlier = Q3 + 1.5 * IQR 
        data = data[data[self.target] <= outlier].reset_index(drop=True)
        return data

    def location(self, data):
        data = data.copy()
        data['Location'] = data['Location'].apply(
            lambda x: str(x).strip().split(',')[0])
        return data

    def find_threshold_filters(self):
        for variable in self.threshold.keys():
            values = self.X[variable].value_counts().index[:self.threshold[variable]]
            self.threshold_dict[variable] = values
        return self

    def threshold_filter(self, data):
        data = data.copy()
        for variable in self.threshold.keys():
            msk = data[variable].isin(self.threshold_dict[variable])            
            data.loc[~msk, variable] = 'Other'
        return data



    def fit(self, data):
        ''' 
        Function that will clean and format feature and target variables
        before training the Linear Regression Model.
        Inpts: Full DataFrame (prepared or unpreparred)
        Outputs: Returns 'self' that encompasses the trained (fit) model.
        '''

        # Initially prepare data by dropping null target variable rows, 
        # removing outliers, formatting target as type int, and selecting
        # features.
        data = self.target_preparation(data)
        data = self.outlier_removal(data)
        self.X = data[self.features]
        self.y = data[self.target]

        # Perform data cleaning on 'Location'
        self.X = self.location(self.X)

        self.find_threshold_filters()

        # Perform data cleaning on threshold features to 
        # map to 'other' if infrequent values
        self.X = self.threshold_filter(self.X)

        # One Hot Encode the categorical features
        self.X = self.enc.fit_transform(self.X)
        
        # Perform train/test split and drop and non-pertinent features.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size= self.test_size,
            random_state= self.random_state
        )

        # Train the model
        self.model.fit(self.X, self.y)

        return self

    def evaluate_model(self):
        '''
        Function that evaluates the trained model based on R_squared and
        RMSE (for regression models). This function allows easier changes to
        the deployed model and evaluating performance.
        '''

        pred = self.model.predict(self.X_train)
        print('Train R2:', r2_score(self.y_train, pred))
        print('Train RMSE:', np.sqrt(mean_squared_error(self.y_train, pred)))

        pred = self.model.predict(self.X_test)
        print('Test R2:', r2_score(self.y_test, pred))
        print('Test RMSE:', np.sqrt(mean_squared_error(self.y_test, pred)))

        pred = self.model.predict(self.X)
        print('Full R2:', r2_score(self.y, pred))
        print('Full RMSE:', np.sqrt(mean_squared_error(self.y, pred)))


    def predict(self, inpt):
        '''
        Takes an array (inpt) of feature columns and returns a
        string of the predicted salary based on previously trained
        model.
        '''
        predictor = pd.DataFrame(columns= settings.FEATURES)
        predictor.loc[0] = inpt
        predictor = self.location(predictor)
        predictor = self.threshold_filter(predictor)
        predictor = self.enc.transform(predictor)
        prediction = self.model.predict(predictor)

        return str(round(prediction[0])).split('.')[0]





        ''' To Do:

        Adjust app.py to only allow viable variable options as selections. Or else cast the inputs to 'other'


        '''


    
