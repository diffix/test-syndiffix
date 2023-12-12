import os
import pandas as pd
import numpy as np
import bz2
import pickle
import requests
import pprint
import matplotlib.pyplot as plt
import seaborn as sns
from syndiffix import Synthesizer

pp = pprint.PrettyPrinter(indent=4)

datasetsPath = 'datasets'
synPath = 'synthesized'

def download_and_load(url):
    response = requests.get(url)
    data = bz2.decompress(response.content)
    df = pickle.loads(data)
    return df

def getDf(fileName, dataDir=datasetsPath):
    fn_pbz2 = os.path.join(dataDir, fileName + '.pbz2')
    fn_csv = os.path.join(dataDir, fileName + '.csv')
    if not os.path.exists(fn_pbz2):
        df = pd.read_csv(fn_csv, low_memory=False)
        saveDf(fileName, df)
    df = pd.read_pickle(fn_pbz2, compression='bz2')
    #with bz2.BZ2File(fn_pbz2, "rb") as f:
        # load the pickle object
        #return pickle.load(f)
    print(f"{fileName} has {len(df)} rows")
    return df

def saveDf(fileName, df, dataDir=datasetsPath):
    fn_pbz2 = os.path.join(dataDir, fileName + '.pbz2')
    fn_csv = os.path.join(dataDir, fileName + '.csv')
    with bz2.BZ2File(fn_pbz2, 'w') as f:
        pickle.dump(df, f)
    df.to_csv(fn_csv, index=False)

def makeSynFileName(fileName, columns):
    for column in columns:
        fileName += f'.{column}'
    return fileName

def do_synthesize(df, columns, pids, fileName):
    # First check to see if we've stored a prior synthesis
    synFileName = makeSynFileName(fileName, columns)
    filePath = os.path.join(synPath, synFileName + '.pbz2')
    if os.path.exists(filePath):
        print(f"Reading {filePath}")
        return getDf(synFileName, dataDir = synPath)
    # Doesn't exist, so let's make it!
    print(f"Synthesizing {filePath} with columns {columns}")
    df_syn = df[columns]
    df_syn = Synthesizer(df_syn, pids=df[pids]).sample()
    with bz2.BZ2File(filePath, 'w') as f:
        pickle.dump(df_syn, f)
    return df_syn

def unify_lengths(*args):
    results = []
    minLen = 1000000000000000
    for df in args:
        minLen = min(minLen, len(df))
    print(f"Select {minLen} random rows from each dataset")
    for df in args:
        results.append(df.sample(n=minLen))
    return results

def trans_amounts_sorted(df, df_mo, fileName):
    ''' Make a graph with the transaction amounts sorted high to low
    '''
    print(list(df.columns))
    print(list(df_mo.columns))
    columns = ['amount']
    pids = ['account_id']
    df_syn = do_synthesize(df, columns, pids, fileName)
    print(f"df_syn has columns {list(df_syn.columns)}")
    df_mo = df_mo[columns]
    df_orig = df[columns]
    df_orig, df_syn, df_mo = unify_lengths(df_orig, df_syn, df_mo)
    df_syn = df_syn.sort_values(by=columns, ascending=False)
    df_syn = df_syn.reset_index(drop=True)
    df_syn.index.name = 'index'
    df_mo = df_mo.sort_values(by=columns, ascending=False)
    df_mo = df_mo.reset_index(drop=True)
    df_mo.index.name = 'index'
    df_orig = df_orig.sort_values(by=columns, ascending=False)
    df_orig = df_orig.reset_index(drop=True)
    df_orig.index.name = 'index'
    synFilePath = os.path.join('plots', makeSynFileName(fileName, columns)+'.png')
    fig, ax = plt.subplots()
    sns.lineplot(data=df_orig, x=df_orig.index.name, y='amount', label='Original', ax=ax, linewidth=4)
    sns.lineplot(data=df_syn, x=df_syn.index.name, y='amount', label='SynDiffix', ax=ax)
    sns.lineplot(data=df_mo, x=df_mo.index.name, y='amount', label='Mostly AI', ax=ax)
    ax.set_yscale('log')
    ax.set_ylim([1,None])
    ax.grid(axis='y')
    ax.tick_params(labelbottom=False, bottom=False)
    ax.legend()
    ax.set(xlabel='Sorted by Amount', ylabel='Transaction Amount')
    plt.savefig(synFilePath)
    plt.close()

def trans_max_balance_sorted(df, df_mo, fileName):
    ''' I want to display the sorted values of the maximum balance
        of each account
    '''
    df = df[['account_id', 'balance']]
    print(f"df has {len(df)} rows, and {df['account_id'].nunique()} distinct account_id")
    df_max_balance = df.groupby('account_id')['balance'].max().reset_index()
    df_max_balance.rename(columns={'balance': 'max_balance'}, inplace=True)
    print(f"df_max_balance has {len(df_max_balance)} rows, and {df_max_balance['account_id'].nunique()} distinct account_id")

    print(f"df_mo has {len(df_mo)} rows, and {df_mo['account_id'].nunique()} distinct account_id")
    df_mo_max_balance = df_mo.groupby('account_id')['balance'].max().reset_index()
    df_mo_max_balance.rename(columns={'balance': 'max_balance'}, inplace=True)
    print(f"df_mo_max_balance has {len(df_mo_max_balance)} rows, and {df_mo_max_balance['account_id'].nunique()} distinct account_id")

    pass

df_trans = getDf('trans_account_card_clients')
df_trans_mo = getDf('trans_account_card_clients.mostly')
if False:
    trans_amounts_sorted(df_trans, df_trans_mo, 'trans_account_card_clients')
if True:
    trans_max_balance_sorted(df_trans, df_trans_mo, 'trans_account_card_clients')