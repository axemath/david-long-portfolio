# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from statistics import mode
from sklearn.preprocessing import OneHotEncoder

one_hot_encoder = OneHotEncoder()

# get data
path = r'C:\Users\axema\OneDrive\Kentucky Derby'
# path = r'C:\Users\ZenWarrior\OneDrive\Kentucky Derby'
filename = r'derbyData.2022.xlsx'
pastPerformance_import = pd.read_excel((path + '\\' + filename), sheet_name='derbyData')
pastPerformance = pastPerformance_import.copy()
pastPerformance.set_index('name', inplace=True)

field_import = pd.read_excel((path + '\\' + filename), sheet_name='currentField')
field = field_import.copy()
field.set_index('name', inplace=True)
field.drop('finish', axis=1, inplace=True)

# summarize data frame
print('Past Performance Data:\n', pastPerformance, '\n\n', sep='')
print('Past Performance Data Types:\n', pastPerformance.dtypes, '\n\n', sep='')

# drop horses with finish == 0
print('Horses with No Finish Position:\n', pastPerformance.loc[pastPerformance.finish == 0], '\n')
print('Number of Horses to Drop:', len(pastPerformance.loc[pastPerformance.finish == 0]))
pastPerformance.drop(index=pastPerformance.loc[pastPerformance.finish == 0].index, inplace=True)
print('New Size of pastPerformance: ', pastPerformance.shape, '\n\n')

# create target variable and drop 'finish'
y_train = np.array((pastPerformance['finish'] == 1).astype(np.int))
pastPerformance.drop('finish', axis=1, inplace=True)

# summary of missing values
print('Summary of Missing Values:\n', pastPerformance.isnull().sum(), '\n\n', sep='')
print('cd is only available from 2003-2015\n', 'class and stam are only available from 2010-2015\n',
      'These variables will be dropped', sep='')
pastPerformance.drop(columns=['cd', 'class', 'stam'], inplace=True)
print('New Size of pastPerformance: ', pastPerformance.shape, '\n\n')

for col in pastPerformance:
    
    if col not in ['name']:
        
        print('Series Name: ', col)
        print('Series Type: ', pastPerformance[col].dtype)
        
        # if pastPerformance[col].dtype == 'object':
        #     print('Unique Values:\n', pd.unique(pastPerformance[col]), sep='')
        # else:
        #     print('Unique Values:\n', np.sort(pd.unique(pastPerformance[col])), sep='')
            
        print('Number of Missing Values: ', pastPerformance[col].isnull().sum())

        if (pastPerformance[col].dtype == 'int64') & (col not in ['year', 'finish', 'post']):
            colMode = mode(pastPerformance[col])
            print(col, 'Mode: ', colMode)
            if pastPerformance.loc[np.isnan(pastPerformance[col]), ['year', col]].empty == False:
                print('Horses to Impute:\n', pastPerformance.loc[np.isnan(pastPerformance[col]), ['year', col]], '\n')
                imputedIndices = pd.Index(np.isnan(pastPerformance[col]))
                pastPerformance.loc[np.isnan(pastPerformance[col]), [col]] = colMode
                print('After Imputation:\n', pastPerformance.loc[imputedIndices, ['year', col], '\n'])

        elif pastPerformance[col].dtype == 'float64':
            colMean = np.nanmean(pastPerformance[col])
            print('Mean ', col, ':  ', round(colMean, 1), sep='')
            if pastPerformance.loc[np.isnan(pastPerformance[col]), ['year', col]].empty == False:
                print('Horses to Impute:\n', pastPerformance.loc[np.isnan(pastPerformance[col]), ['year', col]], '\n')
                imputedIndices = pd.Index(np.isnan(pastPerformance[col]))
                pastPerformance.loc[np.isnan(pastPerformance[col]), [col]] = colMean
                print('After Imputation:\n', pastPerformance.loc[imputedIndices, ['year', col]], '\n')

        elif pastPerformance[col].dtype == 'object':
            colMode = mode(pastPerformance[col])
            print(col, 'Mode: ', colMode)
            if pastPerformance.loc[pastPerformance[col].isna(), ['year', col]].empty == False:
                print('Horses to Impute:\n', pastPerformance.loc[pastPerformance[col].isna(), ['year', col]], '\n')
                imputedIndices = pd.Index(pastPerformance[col].isna())
                pastPerformance.loc[pastPerformance[col].isna(), [col]] = colMode
                print('After Imputation:\n', pastPerformance.loc[imputedIndices, ['year', col]], '\n')

        print('\n\n')

# scaling
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
X_train_num = standard_scaler.fit_transform(np.array(pastPerformance.drop(['year', 'style'], axis=1)))

# one-hot encoding
print('One-Hot Style:\n', one_hot_encoder.fit_transform(pastPerformance[['style']]).toarray())
print('X_train_num:\n', X_train_num)
X_train = np.concatenate((X_train_num,
                          one_hot_encoder.fit_transform(pastPerformance[['style']]).toarray()),
                          axis=1)
print('X_train Type:\n', type(X_train))
print('X_train Shape:\n', X_train.shape)
print('One-Hot Style Shape:\n', one_hot_encoder.fit_transform(pastPerformance[['style']]).toarray().shape)
print('X_train_num Shape:\n', X_train_num.shape)

# summarize data frame
print('Current Field Data:\n', field, '\n\n', sep='')
print('Current Field Data Types:\n', field.dtypes, '\n\n', sep='')

# summary of missing values
print('Summary of Missing Values:\n', field.isnull().sum(), '\n\n', sep='')
print('cd is only available from 2003-2015\n', 'class and stam are only available from 2010-2015\n',
      'These variables will be dropped', sep='')
field.drop(columns=['cd', 'class', 'stam'], inplace=True)
print('Size of field: ', field.shape, '\n\n')

for col in field:
    
    if col not in ['name']:
        
        print('Series Name: ', col)
        print('Series Type: ', field[col].dtype)
        
        # if field[col].dtype == 'object':
        #     print('Unique Values:\n', pd.unique(field[col]), sep='')
        # else:
        #     print('Unique Values:\n', np.sort(pd.unique(field[col])), sep='')
            
        print('Number of Missing Values: ', field[col].isnull().sum())

        if (field[col].dtype == 'int64') & (col not in ['year', 'finish', 'post']):
            colMode = mode(field[col])
            print(col, 'Mode: ', colMode)
            if field.loc[np.isnan(field[col]), ['year', col]].empty == False:
                print('Horses to Impute:\n', field.loc[np.isnan(field[col]), ['year', col]], '\n')
                imputedIndices = pd.Index(np.isnan(field[col]))
                field.loc[np.isnan(field[col]), [col]] = colMode
                print('After Imputation:\n', field.loc[imputedIndices, ['year', col], '\n'])

        elif field[col].dtype == 'float64':
            colMean = np.nanmean(field[col])
            print('Mean ', col, ':  ', round(colMean, 1), sep='')
            if field.loc[np.isnan(field[col]), ['year', col]].empty == False:
                print('Horses to Impute:\n', field.loc[np.isnan(field[col]), ['year', col]], '\n')
                imputedIndices = pd.Index(np.isnan(field[col]))
                field.loc[np.isnan(field[col]), [col]] = colMean
                print('After Imputation:\n', field.loc[imputedIndices, ['year', col]], '\n')

        elif field[col].dtype == 'object':
            colMode = mode(field[col])
            print(col, 'Mode: ', colMode)
            if field.loc[field[col].isna(), ['year', col]].empty == False:
                print('Horses to Impute:\n', field.loc[field[col].isna(), ['year', col]], '\n')
                imputedIndices = pd.Index(field[col].isna())
                field.loc[field[col].isna(), [col]] = colMode
                print('After Imputation:\n', field.loc[imputedIndices, ['year', col]], '\n')

        print('\n\n')

# scaling
X_test_num = standard_scaler.transform(np.array(field.drop(['year', 'style'], axis=1)))

# one-hot encoding
print('One-Hot Style:\n', one_hot_encoder.transform(field[['style']]).toarray())
print('X_test_num:\n', X_test_num)
X_test = np.concatenate((X_test_num,
                          one_hot_encoder.transform(field[['style']]).toarray()),
                          axis=1)
print('X_train Type:\n', type(X_test))
print('X_train Shape:\n', X_test.shape)
print('One-Hot Style Shape:\n', one_hot_encoder.transform(field[['style']]).toarray().shape)
print('X_test_num Shape:\n', X_test_num.shape)

# Logistic Regression Model
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
probabilities = log_reg.predict_proba(X_test)[:,1]
prob_table = pd.DataFrame({'year':field['year'],
                           'odds':field['odds'],
                           'probability':probabilities})
print('Probabilities:\n', prob_table.sort_values('probability', ascending=False))