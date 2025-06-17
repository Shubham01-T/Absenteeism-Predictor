import pandas as pd

raw_csv_data = pd.read_csv("Absenteeism_data.csv")

df = raw_csv_data.copy()
"""Step 1:
Remove Unnecessary Columns
Here the ID column does not help us in understanding w
whether a person will remain absent or not so we will drop that column"""

df=df.drop(['ID'], axis=1)
#print(df.describe())

##########
# STEP 2
reason_columns = pd.get_dummies(df['Reason for Absence'])
# print(reason_columns)
reason_columns['check'] = reason_columns.sum(axis=1)
#print(reason_columns)
"""To verify the uniqueness"""
"""
print("reason_columns['check'].sum(axis=0)) #op 700
print("reason_columns['check].unique()) #o/p = array([1])
"""
reason_columns = reason_columns.drop(['check'],axis=1)

#Step 3 Removing the column 0 to avoid multicolinerity

reason_columns = pd.get_dummies(df['Reason for Absence'],drop_first=True)
#print(reason_columns) 

#Step 4 Classifying various reasons for absence

df = df.drop(['Reason for Absence'], axis=1)

"""Reasons 1-14 are related to some kind of medical condition
Reasons 15-17 are related to pregnancy
reasons 18-21 are related to poisoning 
reasons 22-28 are light reasons eg. dental appointments etc."""

reasone_type_1 = reason_columns.loc[:, 1:14].max(axis=1)
reasone_type_2 = reason_columns.loc[:, 15:17].max(axis=1)
reasone_type_3 = reason_columns.loc[:, 18:21].max(axis=1)
reasone_type_4 = reason_columns.loc[:, 22:28].max(axis=1)

#print(reasone_type_1.max(axis=1))

df = pd.concat([df,reasone_type_1,reasone_type_2,reasone_type_3,reasone_type_4],axis=1)
#print(df)
#print(df.columns.values) to change column names for reasons catagories

column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
 'Daily Work Load Average' ,'Body Mass Index', 'Education', 'Children', 'Pets',
 'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']

df.columns = column_names
#print(df.head())

column_names_reordered=[ 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4','Date', 'Transportation Expense', 'Distance to Work', 'Age',
 'Daily Work Load Average' ,'Body Mass Index', 'Education', 'Children', 'Pets',
 'Absenteeism Time in Hours']

df = df[column_names_reordered]
#print(df.head())

#Creating a checkpoint
df_reason_mod =  df.copy()

#Next step: Analysing date column
#The dates are stored as strings, so we need to convert them to timestamps

df_reason_mod['Date'] = pd.to_datetime(df_reason_mod['Date'], format = '%d/%m/%Y')
#print(df_reason_mod['Date'])

#extracting month values from date

list_months = []

for i in range(df_reason_mod.shape[0]):
    list_months.append(df_reason_mod['Date'][i].month)
#print(len(list_months))

df_reason_mod['Month Value'] = list_months   
#print(df_reason_mod.head())

#Extracting Days of the week 
#Monday to sunday = 0 to 7
#syntex to get day of the week df['Date][i].weekday()

def date_to_weekday(date_value):
    return date_value.weekday()

df_reason_mod['Day of the week'] = df_reason_mod['Date'].apply(date_to_weekday)

#print(df_reason_mod.head())
df_reason_mod = df_reason_mod.drop(['Date'],axis=1)
#print(df_reason_mod.columns.values)

#re-ordering columns
columns_reordered = ['Reason_1', 'Reason_2' ,'Reason_3', 'Reason_4', 'Month Value', 'Day of the week','Transportation Expense',
 'Distance to Work' ,'Age', 'Daily Work Load Average', 'Body Mass Index', 'Education' ,'Children', 'Pets' ,'Absenteeism Time in Hours' ]

df_reason_mod = df_reason_mod[columns_reordered]
#print(df_reason_mod.head())

df_reason_date_mod = df_reason_mod.copy() ######CheckPoint######

"""For education column 1 = Highschool, 2 = Graduate
3 = Postgraduate and 4 = A master of PhD"""

###print(df_reason_date_mod['Education'].value_counts()) O/P suggests that 583 people are from catagory 1 and remaining ~ 110 are from the rest of the 3

df_reason_date_mod['Education'] = df_reason_date_mod['Education'].map({1:0,2:1,3:1,4:1})
#print(df_reason_date_mod['Education'].value_counts())  O/P Education: 0>>583, 1>>117

#Final CheckPoint

df_preprocessed = df_reason_date_mod.copy()
#print(df_preprocessed.head(10))

#Saving the work!

df_preprocessed.to_csv('Absenteeism_preprocessed.csv', index=False)