import pandas as pd
import numpy as np
import bz2
import pickle
import os

def transformDate(dateVal):
    dateStr = str(dateVal)
    year = str(int(dateStr[0:2]) + 1900)
    month = int(dateStr[2:4])
    # Apparently if a woman, then month has 50 added!
    if month >= 50:
        month -= 50
    day = int(dateStr[4:6])
    #if month in [2, 4, 6, 9, 11] and day > 28:
        #day = (day % 28) + 1
    month = f'{month:02d}'
    day = f'{day:02d}'
    return f"{year}-{month}-{day}"

def populateSex(dateVal):
    dateStr = str(dateVal)
    year = str(int(dateStr[0:2]) + 1900)
    month = int(dateStr[2:4])
    if month > 40:
        return 'F'
    else:
        return 'M'

def doGenericTransformations(df):
    if 'account' in df.columns:
        df['account'].fillna(-1, inplace=True)
        df['account'] = df['account'].astype(int)
        df['account'] = df['account'].astype(str)
        df['account'].replace('-1', np.nan, inplace=True)
    if 'account_to' in df.columns:
        df['account_to'] = df['account_to'].astype(str)
    if 'account_id' in df.columns:
        df['account_id'] = df['account_id'].astype(str)
    if 'client_id' in df.columns:
        df['client_id'] = df['client_id'].astype(str)
    if 'district_id' in df.columns:
        df['district_id'] = df['district_id'].astype(str)
    if 'card_id' in df.columns:
        df['card_id'] = df['card_id'].astype(str)
    if 'disp_id' in df.columns:
        df['disp_id'] = df['disp_id'].astype(str)
    if 'loan_id' in df.columns:
        df['loan_id'] = df['loan_id'].astype(str)
    if 'order_id' in df.columns:
        df['order_id'] = df['order_id'].astype(str)
    if 'trans_id' in df.columns:
        df['trans_id'] = df['trans_id'].astype(str)
    if 'date' in df.columns:
        df['date'] = df['date'].apply(transformDate)
        df['date'] = pd.to_datetime(df['date'])
    return df

def saveDf(fileName, df):
    with bz2.BZ2File(f"{fileName}.pbz2", 'w') as f:
        pickle.dump(df, f)
    df.to_csv(f"{fileName}.csv", index=False)

datasetsDir = 'datasets'
if False:
    fileName = 'district'
    filePath = os.path.join(datasetsDir, fileName)
    df = pd.read_csv(filePath+'_orig.csv', sep=';', low_memory=False)
    df.rename(columns={'A1':'district_id'}, inplace=True)
    df.rename(columns={'A2':'city'}, inplace=True)
    df.rename(columns={'A3':'region'}, inplace=True)
    df.rename(columns={'A4':'population'}, inplace=True)
    df.rename(columns={'A11':'avg_salary'}, inplace=True)
    df.rename(columns={'A14':'entrepreneur_rate'}, inplace=True)
    df['average_unemployment_rate'] = df[['A12', 'A13']].mean(axis=1)
    df['average_crime_rate'] = df[['A15', 'A16']].mean(axis=1) / df['population']
    df = doGenericTransformations(df)
    print(df.dtypes)
    saveDf(filePath, df)
if False:
    fileName = 'trans'
    filePath = os.path.join(datasetsDir, fileName)
    df = pd.read_csv(filePath+'_orig.csv', sep=';', low_memory=False)
    df = doGenericTransformations(df)
    print(df.dtypes)
    saveDf(filePath, df)
if False:
    fileName = 'order'
    filePath = os.path.join(datasetsDir, fileName)
    df = pd.read_csv(filePath+'_orig.csv', sep=';', low_memory=False)
    df = doGenericTransformations(df)
    saveDf(filePath, df)
if True:
    fileName = 'loan'
    filePath = os.path.join(datasetsDir, fileName)
    df = pd.read_csv(filePath+'_orig.csv', sep=';', low_memory=False)
    df['defaulted'] = (df['status'] == 'B') | (df['status'] == 'D')
    df = doGenericTransformations(df)
    df.rename(columns={'date':'loan_date'}, inplace=True)
    saveDf(filePath, df)
if False:
    fileName = 'disp'
    filePath = os.path.join(datasetsDir, fileName)
    df = pd.read_csv(filePath+'_orig.csv', sep=';', low_memory=False)
    df = doGenericTransformations(df)
    saveDf(filePath, df)
if False:
    fileName = 'client'
    filePath = os.path.join(datasetsDir, fileName)
    df = pd.read_csv(filePath+'_orig.csv', sep=';', low_memory=False)
    # Add sex column
    df['sex'] = df['birth_number'].apply(populateSex)
    df['birth_number'] = df['birth_number'].apply(transformDate)
    df['birth_number'] = pd.to_datetime(df['birth_number'])
    df = doGenericTransformations(df)
    print(df.dtypes)
    saveDf(filePath, df)
if False:
    fileName = 'card'
    filePath = os.path.join(datasetsDir, fileName)
    df = pd.read_csv(filePath+'_orig.csv', sep=';', low_memory=False)
    if 'issued' in df.columns:
        df['issued'] = df['issued'].apply(transformDate)
        df['issued'] = pd.to_datetime(df['issued'])
    df = doGenericTransformations(df)
    print(df.dtypes)
    saveDf(filePath, df)
if False:
    fileName = 'account'
    filePath = os.path.join(datasetsDir, fileName)
    df = pd.read_csv(filePath+'_orig.csv', sep=';', low_memory=False)
    df = doGenericTransformations(df)
    saveDf(filePath, df)
