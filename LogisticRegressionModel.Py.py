import pandas as pd
import numpy as np

data_preprocessed = pd.read_csv("Absenteeism_preprocessed.csv")


#Task: We will take all the independent variables like reasons, month, day, 
# transportation expense, distance to work, age,dailyworkload, BMI, 
# education,children and pet and predict absenteeism

#Creating Classes
#We are gong to take the median value of Absenteeism time and use it as a cut off line

#print(data_preprocessed['Absenteeism Time in Hours'].median()) #O/P = 3

#Absend <= 3 hour = Moderately absent else excessively absent

targets = np.where(data_preprocessed['Absenteeism Time in Hours']>data_preprocessed['Absenteeism Time in Hours'].median(),1,0)
#print(targets)

data_preprocessed['Excessive Absenteeism'] = targets
#print(data_preprocessed.head())

data_with_targets = data_preprocessed.drop(['Absenteeism Time in Hours', 'Day of the week', 'Daily Work Load Average', 'Distance to Work'],axis=1)
#print(data_with_targets.head())

unscaled_inputs = data_with_targets.iloc[:,:-1] #all columns except excess absenteeism>>>> Our targets

#To scale the imputs we need to input StandardScaler module

# from sklearn.preprocessing import StandardScaler

# absenteeism_scaler = StandardScaler() #declaring a standard scaler object

# absenteeism_scaler.fit(unscaled_inputs)
# scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)
# #print(scaled_inputs.shape) o/p 700x14

# the custom scaler class 
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        # Fix: Use keyword arguments for StandardScaler
        self.scaler = StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None


    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.array(X[self.columns].mean(axis=0))
        self.var_ = np.array(X[self.columns].var(axis=0))
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]

#columns_to_scale = ['Month Value', 'Day of the week','Transportation Expense', 'Distance to Work', 'Age','Daily Work Load Average', 'Body Mass Index', 'Education', 'Children','Pets']
columns_to_omit = ['Reason 1', 'Reason 2', 'Reason 3', 'Reason 4', 'Education']
columns_to_scale = [x for x in unscaled_inputs.columns.values if x not in columns_to_omit]
absenteeism_scaler = CustomScaler(columns_to_scale)
absenteeism_scaler.fit(unscaled_inputs)
scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(scaled_inputs,targets, train_size=0.8, random_state=20)

### Modelling ###
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

reg = LogisticRegression()
reg.fit(x_train,y_train)
print("The Test accuracy is:",reg.score(x_train,y_train)*100) #76% accuracy

#Manually claculating the score

model_outputs = reg.predict(x_train)
#print(np.sum(model_outputs==y_train)/model_outputs.shape[0]) #Same 76% accuracy

#finding the intercept and coefficients

# print(reg.intercept_)
# print(reg.coef_)

feature_name = unscaled_inputs.columns.values
summary_table = pd.DataFrame(columns=['Feature name'], data=feature_name)
summary_table['Coefficient'] = np.transpose(reg.coef_)

# print(summary_table)
summary_table.index = summary_table.index+1
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
summary_table=summary_table.sort_index()

summary_table['Odds_ratio'] = np.exp(summary_table.Coefficient)
#print(summary_table)
summary_table=summary_table.sort_values('Odds_ratio',ascending=False)
#print(summary_table)
print("The Test accuracy is:", reg.score(x_test,y_test)*100)

#predicted_proba = reg.predict_proba(x_test)
#print(predicted_proba[:,1])

#Pickeling the Reg and StandardScaler Object for reusability 

# import pickle

# with open('model','wb') as file:
#     pickle.dump(reg, file)

# with open('scaler','wb') as file:
#     pickle.dump(StandardScaler,file)


