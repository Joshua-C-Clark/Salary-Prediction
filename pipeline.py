import os
import settings
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class Pipeline:
    
    ''' This Pipeline class streamlines and containes all of 
        the categorical and numerical feature imputing and encoding. The model
        training is embedded towards the end, being build with a Random Forest Regressor
        based on notebook performance metric analysis. '''
    
    
    def __init__(self, target, numerical_log, categorical_encode, categorical_to_impute,
                 features, test_size = 0.2, random_state = 0,
                 percentage = 0.0001):
        
        # data sets
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # engineering parameters (to be learnt from data)
        self.frequent_category_dict = {}
        self.encoding_dict = {}
        
        # models
        self.scaler = MinMaxScaler()
        self.model = RandomForestRegressor(n_estimators = 100,
        bootstrap = True, 
        max_depth = 110,
        max_features = 3,
        min_samples_leaf = 3,
        min_samples_split = 12)
        
        # groups of variables to engineer
        self.target = target
        self.numerical_log = numerical_log
        self.categorical_encode = categorical_encode
        self.features = features
        self.categorical_to_impute = categorical_to_impute
        
        # more parameters
        self.test_size = test_size
        self.random_state = random_state
        self.percentage = percentage    

    # ======= functions to learn parameters from train set ============
   
    
    def find_frequent_categories(self):
        ''' find list of frequent categories in categorical variables'''
        
        for variable in self.categorical_encode:
            
            tmp = self.X_train.groupby(variable)[
                    self.target].count() / len(self.X_train)

            self.frequent_category_dict[variable] = tmp[tmp > self.percentage].index
    
        return self        
    
    
    def find_categorical_mappings(self):
        ''' create category to integer mappings for categorical encoding'''
        
        for variable in self.categorical_encode:
            # Groupby the categorical columns. The index becomes the unique values for that given column.
            # The method below aranges the categories by frequency of occurance vice random unique sortings.
            # labels = self.X_train[variable].unique()
            ordered_labels = self.X_train.groupby([
                    variable])[self.target].count().sort_values().index
            # Enumerate over the ordered labels to create an integer map
            # ordinal_labels = {k: i for i, k in enumerate(labels, 0)}

            ordinal_labels = {k: i for i, k in enumerate(ordered_labels, 0)}
            # Store the map in the dictionary for each respective variable
            self.encoding_dict[variable] = ordinal_labels
    
        return self      
    
    
    
    # ======= functions to transform data =================
       
            
    def remove_rare_labels(self, df):
        ''' group infrequent labels in group Other'''
        
        df = df.copy()
        
        for variable in self.categorical_encode:
            
            df[variable] = np.where(
                    df[variable].isin(
                            self.frequent_category_dict[variable]),
                            df[variable], 'Other')
       
        return df    
    
    
    def encode_categorical_variables(self, df):
        
        ''' replace categories by numbers in categorical variables'''

        df = df.copy()
            
        for variable in self.categorical_encode:
            
            df[variable] = df[variable].map(self.encoding_dict[variable])
        
        return df

    def drop_missing_target(self, df):
        '''Drop rows that have missing target variable data'''
        df = df.copy()
        df.dropna(subset=[self.target], inplace = True)
        return df

    def target_to_numerical(self, df):
        df = df.copy()
        df[self.target] = df[self.target].str.replace(',', '').str.replace('$','').astype(float)
        return df    
    
    
    # ====   master function that orchestrates feature engineering =====

    def fit(self, data):
        '''pipeline to learn parameters from data, fit the scaler and model'''

        # Drop rows with missing target values
        data = self.drop_missing_target(data)

        data = self.target_to_numerical(data)
        
        # separate data sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                data, data[self.target],
                test_size = self.test_size,
                random_state = self.random_state)
        
        # impute missing data
        # categorical
        self.X_train[self.categorical_to_impute] = self.X_train[
                self.categorical_to_impute].fillna('Missing')
        
        self.X_test[self.categorical_to_impute] = self.X_test[
                self.categorical_to_impute].fillna('Missing')
 
        
        # transform numerical variables
        self.X_train[self.numerical_log] = np.log(self.X_train[self.numerical_log])
        self.X_test[self.numerical_log] = np.log(self.X_test[self.numerical_log])
        
        
        # find frequent labels
        self.find_frequent_categories()
        
        # remove rare labels
        self.X_train = self.remove_rare_labels(self.X_train)
        self.X_test = self.remove_rare_labels(self.X_test)
        
        # find categorical mappings
        self.find_categorical_mappings()
        
        # encode categorical variables
        self.X_train = self.encode_categorical_variables(self.X_train)
        self.X_test = self.encode_categorical_variables(self.X_test)          
        
        # train scaler
        self.scaler.fit(self.X_train[self.features])
        
        # scale variables
        self.X_train = self.scaler.transform(self.X_train[self.features])
        self.X_test = self.scaler.transform(self.X_test[self.features])
        print(self.X_train.shape, self.X_test.shape)
        
        # train model
        self.model.fit(self.X_train, np.log(self.y_train))
        
        return self    
    
    def evaluate_model(self):
        '''evaluates trained model on train and test sets'''
        
        pred = self.model.predict(self.X_train)
        pred = np.exp(pred)
        print('train r2: {}'.format((r2_score(self.y_train, pred))))
        
        
        pred = self.model.predict(self.X_test)
        pred = np.exp(pred)
        print('test r2: {}'.format((r2_score(self.y_test, pred)))) 
