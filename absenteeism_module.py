# coding: utf-8

# Import all libraries needed
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# Custom Scaler Class
class CustomScaler(BaseEstimator, TransformerMixin): 
    
    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        # Correct the StandardScaler initialization by using keyword arguments
        self.scaler = StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y=None):
        # Only fit on the columns that exist in the DataFrame
        valid_columns = [col for col in self.columns if col in X.columns]
        if valid_columns:
            self.scaler.fit(X[valid_columns], y)
            self.mean_ = np.array(np.mean(X[valid_columns]))
            self.var_ = np.array(np.var(X[valid_columns]))
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        # Only transform on the columns that exist in the DataFrame
        valid_columns = [col for col in self.columns if col in X.columns]
        if valid_columns:
            X_scaled = pd.DataFrame(self.scaler.transform(X[valid_columns]), columns=valid_columns)
        else:
            X_scaled = pd.DataFrame(columns=valid_columns)  # Empty DataFrame if no columns to scale

        X_not_scaled = X.loc[:, ~X.columns.isin(valid_columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


# Create the special class for prediction
class absenteeism_model():
      
    def __init__(self, model_file, scaler_file):
        # Read the 'model' and 'scaler' files which were saved
        with open(model_file, 'rb') as model_file, open(scaler_file, 'rb') as scaler_file:
            self.reg = pickle.load(model_file)
            # Initialize CustomScaler with columns to scale
            columns_to_scale = ['Transportation Expense', 'Distance to Work', 'Age', 
                                'Daily Work Load Average', 'Body Mass Index', 'Education', 
                                'Children', 'Pets']
            self.scaler = CustomScaler(columns=columns_to_scale)
            self.data = None
        
    def load_and_clean_data(self, data_file):
        # Import the data
        df = pd.read_csv(data_file, delimiter=',')
        self.df_with_predictions = df.copy()
        
        # Drop the 'ID' column
        df = df.drop(['ID'], axis=1)
        
        # Add a column with NaN values for 'Absenteeism Time in Hours'
        df['Absenteeism Time in Hours'] = 'NaN'

        # Create dummy columns for the reasons for absence
        reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first=True)
        
        # Split reason columns into 4 types of reasons
        reason_type_1 = reason_columns.loc[:, 1:14].max(axis=1)
        reason_type_2 = reason_columns.loc[:, 15:17].max(axis=1)
        reason_type_3 = reason_columns.loc[:, 18:21].max(axis=1)
        reason_type_4 = reason_columns.loc[:, 22:].max(axis=1)
        
        # Drop 'Reason for Absence' column from df
        df = df.drop(['Reason for Absence'], axis=1)
        
        # Concatenate df and the 4 types of reason for absence
        df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis=1)
        
        # Assign names to the 4 reason type columns
        column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
                        'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',
                        'Pets', 'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']
        df.columns = column_names

        # Re-order the columns
        column_names_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Date', 'Transportation Expense', 
                                  'Distance to Work', 'Age', 'Daily Work Load Average', 'Body Mass Index', 'Education', 
                                  'Children', 'Pets', 'Absenteeism Time in Hours']
        df = df[column_names_reordered]
        
        # Convert 'Date' column to datetime
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

        # Create 'Month Value' and 'Day of the Week' columns
        df['Month Value'] = df['Date'].apply(lambda x: x.month)
        df['Day of the Week'] = df['Date'].apply(lambda x: x.weekday())

        # Drop 'Date' column from df
        df = df.drop(['Date'], axis=1)

        # Re-order columns
        column_names_upd = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month Value', 'Day of the Week',
                            'Transportation Expense', 'Distance to Work', 'Age',
                            'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',
                            'Pets', 'Absenteeism Time in Hours']
        df = df[column_names_upd]

        # Map 'Education' variable to 0 and 1
        df['Education'] = df['Education'].map({1: 0, 2: 1, 3: 1, 4: 1})

        # Replace NaN values
        df = df.fillna(value=0)

        # Drop original absenteeism time column
        df = df.drop(['Absenteeism Time in Hours'], axis=1)
        
        # Drop unnecessary variables
        df = df.drop(['Day of the Week', 'Daily Work Load Average', 'Distance to Work'], axis=1)
        
        # Store the preprocessed data
        self.preprocessed_data = df.copy()
        
        # Fit the scaler and transform the data
        self.scaler.fit(df)
        self.data = self.scaler.transform(df)
    
    # Function to output the probability of a data point to be 1
    def predicted_probability(self):
        if self.data is not None:  
            pred = self.reg.predict_proba(self.data)[:, 1]
            return pred
        
    # Function to output 0 or 1 based on the model's prediction
    def predicted_output_category(self):
        if self.data is not None:
            pred_outputs = self.reg.predict(self.data)
            return pred_outputs
        
    # Function to predict outputs and probabilities and add them to the preprocessed data
    def predicted_outputs(self):
        if self.data is not None:
            self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:, 1]
            self.preprocessed_data['Prediction'] = self.reg.predict(self.data)
            return self.preprocessed_data
