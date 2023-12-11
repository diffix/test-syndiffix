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

print(f"SDV version {sdv.__version__}")

pp = pprint.PrettyPrinter(indent=4)

def download_and_load(url):
    response = requests.get(url)
    data = bz2.decompress(response.content)
    df = pickle.loads(data)
    return df


def getDf(fileName):
    with bz2.BZ2File(fileName, "rb") as f:
        # load the pickle object
        return pickle.load(f)

def saveDf(fileName, df):
    with bz2.BZ2File(f"{fileName}.pbz2", 'w') as f:
        pickle.dump(df, f)
    df.to_csv(f"{fileName}.csv", index=False)

filesToSynthesize = [
    'account_card_clients',
    'loan_account_card_clients',
    'loan_account_card',
    'loan',
    'order_account_card_clients',
    'order_account_card',
    'order',
    'trans_account_card_clients',
    'trans_account_card',
    'trans',
]

for fileNameRoot in filesToSynthesize:
    fileName = fileNameRoot + '.pbz2'
    fileNameCsv = fileNameRoot + '.ctgan.csv'
    fileNamePbz2 = fileNameRoot + '.ctgan.pbz2'
    if os.path.exists(fileNameCsv):
        print(f"Already did {fileName}")
        continue
    df = getDf(fileName)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    metadataDict = metadata.to_dict()
    if 'account' in metadataDict['columns']:
        metadataDict['columns']['account']['sdtype'] = 'pii'
    print(f"\n{fileName}:")
    pp.pprint(metadataDict)
    synthesizer = CTGANSynthesizer(metadata)
    synthesizer.fit(df)
    df_syn = synthesizer.sample(num_rows=len(df))
    saveDf(fileNameRoot+'.ctgan', df_syn)
