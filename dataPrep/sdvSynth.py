import os
import pandas as pd
import numpy as np
import bz2
import pickle
import requests
import pprint
from sdv.metadata import SingleTableMetadata
import sdv 
from sdv.single_table import CTGANSynthesizer
from sdv.sequential import PARSynthesizer

print(f"SDV version {sdv.__version__}")
dataSetDir = 'datasets'

pp = pprint.PrettyPrinter(indent=4)

def download_and_load(url):
    response = requests.get(url)
    data = bz2.decompress(response.content)
    df = pickle.loads(data)
    return df

def getDf(fileName):
    return pd.read_pickle(fileName, compression='bz2')

def saveDf(fileName, df, dataDir=dataSetDir):
    print(f"Save {fileName}")
    fn_pbz2 = os.path.join(dataDir, fileName + '.pbz2')
    fn_csv = os.path.join(dataDir, fileName + '.csv')
    with bz2.BZ2File(fn_pbz2, 'w') as f:
        pickle.dump(df, f)
    df.to_csv(fn_csv, index=False)

filesToSynthesize = [
    'account_card_clients',
]
seqFilesToSynthesize = [
    'trans_account_card_clients',
    'loan_account_card_clients',
    'order_account_card_clients',
]

for fileNameRoot in seqFilesToSynthesize:
    fileName = os.path.join(dataSetDir, fileNameRoot + '.pbz2')
    print(fileName)
    fileNameCsv = os.path.join(dataSetDir, fileNameRoot + '.ctgan.seq.csv')
    print(fileNameCsv)
    fileNamePbz2 = os.path.join(dataSetDir, fileNameRoot + '.ctgan.seq.pbz2')
    print(fileNamePbz2)
    if os.path.exists(fileNameCsv):
        print(f"Already did {fileName}")
        continue
    df = getDf(fileName)
    columns = list(df.columns)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    metadata.update_column(column_name='account_id', sdtype='id')
    if 'account' in columns:
        metadata.update_column(column_name='account', sdtype='id')
    if 'trans_id' in columns:
        metadata.update_column(column_name='trans_id', sdtype='id')
    metadata.set_sequence_key(column_name='account_id')
    metadata.set_sequence_index(column_name='date')

    metadataDict = metadata.to_dict()
    print(f"\n{fileName}:")
    pp.pprint(metadataDict)
    num_sequences = df['account_id'].nunique()
    print(num_sequences)
    synthesizer = PARSynthesizer(metadata)
    print("Before fit")
    synthesizer.fit(df)
    print("After fit")
    df_syn = synthesizer.sample(num_sequences=num_sequences)
    print("After sample")
    saveDf(fileNameRoot+'.ctgan.seq', df_syn)

for fileNameRoot in filesToSynthesize:
    fileName = os.path.join(dataSetDir, fileNameRoot + '.pbz2')
    fileNameCsv = os.path.join(dataSetDir, fileNameRoot + '.ctgan.csv')
    fileNamePbz2 = os.path.join(dataSetDir, fileNameRoot + '.ctgan.pbz2')
    if os.path.exists(fileNameCsv):
        print(f"Already did {fileName}")
        continue
    df = getDf(fileName)
    columns = list(df.columns)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    if 'account' in columns:
        metadata.update_column(column_name='account', sdtype='id')
    if 'trans_id' in columns:
        metadata.update_column(column_name='trans_id', sdtype='id')
    metadataDict = metadata.to_dict()
    print(f"\n{fileName}:")
    pp.pprint(metadataDict)
    synthesizer = CTGANSynthesizer(metadata)
    synthesizer.fit(df)
    df_syn = synthesizer.sample(num_rows=len(df))
    saveDf(fileNameRoot+'.ctgan', df_syn)
