import os
import pandas as pd
import numpy as np
import bz2
import pickle
import requests
import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from syndiffix import Synthesizer

'''
type
VYDAJ     634571        issue or publish
PRIJEM    405083        receiving
VYBER      16666        select, choose
operation
VYBER             434918    select, choose      minus
PREVOD NA UCET    208283    transfer            minus
VKLAD             156743    deposit             plus
PREVOD Z UCTU      65226    withdrawl           minus
VYBER KARTOU        8036    withdrawl by card   minus
frequency
POPLATEK MESICNE      969253    monthly fee
POPLATEK TYDNE         62567    weekly fee
POPLATEK PO OBRATU     24500    fee after turnover
'''

cols = {'orig':'grey', 'ct':'#eb5757', 'mo':'#438fff', 'syn':'#42f7c0'}
wids = {'orig':6, 'ct':2, 'mo':2, 'syn':2}
lebs = {'orig':'Original', 'ct':'CTGAN', 'mo':'Mostly AI', 'syn':'SynDiffix'}

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
    if 'account_id' in list(df.columns):
        print(f"    and {df['account_id'].nunique()} distinct account_id")
    return df

def saveDf(fileName, df, dataDir=datasetsPath):
    fn_pbz2 = os.path.join(dataDir, fileName + '.pbz2')
    fn_csv = os.path.join(dataDir, fileName + '.csv')
    with bz2.BZ2File(fn_pbz2, 'w') as f:
        pickle.dump(df, f)
    df.to_csv(fn_csv, index=False)

def makeSynFileName(fileName, columns, tag=None):
    for column in columns:
        fileName += f'.{column}'
    if tag is not None:
        fileName += f'.{tag}'
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
    if pids:
        df_syn = Synthesizer(df_syn, pids=df[pids]).sample()
    else:
        df_syn = Synthesizer(df_syn).sample()
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

def trans_total_volume_by_month(df, df_mo, df_ct, fileName):
    df['month'] = df['date'].dt.to_period('M')
    df['month'] = df['month'].astype(str)
    print(df[['month','date']].head(20))
    columns = ['amount','month']
    pids = ['account_id']
    df_syn = do_synthesize(df, columns, pids, fileName)
    df_mo['date'] = pd.to_datetime(df_mo['date'])
    df_mo['month'] = df_mo['date'].dt.to_period('M')
    df_mo['month'] = df_mo['month'].astype(str)
    df_mo = df_mo[columns]
    if df_ct is not None:
        df_ct['date'] = pd.to_datetime(df_ct['date'])
        df_ct['month'] = df_ct['date'].dt.to_period('M')
        df_ct['month'] = df_ct['month'].astype(str)
        df_ct = df_ct[columns]
        df_vol_ct = df_ct.groupby('month')['amount'].sum().reset_index()
        df_vol_ct.rename(columns={'amount': 'volume'}, inplace=True)
    df_orig = df[columns]
    df_vol_orig = df_orig.groupby('month')['amount'].sum().reset_index()
    df_vol_orig.rename(columns={'amount': 'volume'}, inplace=True)
    df_vol_syn = df_syn.groupby('month')['amount'].sum().reset_index()
    df_vol_syn.rename(columns={'amount': 'volume'}, inplace=True)
    df_vol_mo = df_mo.groupby('month')['amount'].sum().reset_index()
    df_vol_mo.rename(columns={'amount': 'volume'}, inplace=True)
    fig, ax = plt.subplots()
    def do_plot(df_vol_orig, df_vol_syn, df_vol_mo, df_vol_ct, df_ct, ax):
        sns.lineplot(data=df_vol_orig, x='month', y='volume', label=lebs['orig'], ax=ax, linewidth=wids['orig'], color=cols['orig'])
        sns.lineplot(data=df_vol_syn, x='month', y='volume', label=lebs['syn'], ax=ax, linewidth=wids['syn'], color=cols['syn'])
        sns.lineplot(data=df_vol_mo, x='month', y='volume', label=lebs['mo'], ax=ax, linewidth=wids['mo'], color=cols['mo'])
        if df_ct is not None:
            sns.lineplot(data=df_vol_ct, x='month', y='volume', label=lebs['ct'], ax=ax, linewidth=wids['ct'], color=cols['ct'])
        ax.grid(axis='y')
        #ax.tick_params(labelbottom=False, bottom=False)
        ax.legend()
        ax.set(xlabel='Year-Month', ylabel='Total Transaction Volume')
        xticklabels = ax.get_xticklabels()
        xticklabels = [label.get_text() for label in xticklabels]
        xticks = []
        newlabels = []
        for i in range(len(xticklabels)):
            if xticklabels[i][-2:] == '01':
                xticks.append(i)
                newlabels.append(xticklabels[i])
        ax.set_xticks(xticks)
        ax.set_xticklabels(newlabels)  # Set the x-tick labels
    do_plot(df_vol_orig, df_vol_syn, df_vol_mo, df_vol_ct, df_ct, ax)
    synFilePath = os.path.join('plots', makeSynFileName(fileName, columns)+'.png')
    plt.savefig(synFilePath)
    plt.close()
    fig = plt.figure(figsize=(12, 6))
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid((2, 3), (0, 2))
    ax3 = plt.subplot2grid((2, 3), (1, 2))
    do_plot(df_vol_orig, df_vol_syn, df_vol_mo, df_vol_ct, df_ct, ax1)
    rect = patches.Rectangle((12.5, 20000000), 4, 40000000, linewidth=1, edgecolor='black', facecolor='none', linestyle='dotted')
    ax1.add_patch(rect)
    rect = patches.Rectangle((64.5, 140000000), 3, 80000000, linewidth=1, edgecolor='black', facecolor='none', linestyle='dotted')
    ax1.add_patch(rect)
    sns.lineplot(data=df_vol_orig, x='month', y='volume', label=lebs['orig'], ax=ax2, linewidth=wids['orig'], color=cols['orig'])
    sns.lineplot(data=df_vol_syn, x='month', y='volume', label=lebs['syn'], ax=ax2, linewidth=wids['syn'], color=cols['syn'])
    sns.lineplot(data=df_vol_mo, x='month', y='volume', label=lebs['mo'], ax=ax2, linewidth=wids['mo'], color=cols['mo'])
    sns.lineplot(data=df_vol_ct, x='month', y='volume', label=lebs['ct'], ax=ax2, linewidth=wids['ct'], color=cols['ct'])
    sns.lineplot(data=df_vol_orig, x='month', y='volume', label=lebs['orig'], ax=ax3, linewidth=wids['orig'], color=cols['orig'])
    sns.lineplot(data=df_vol_syn, x='month', y='volume', label=lebs['syn'], ax=ax3, linewidth=wids['syn'], color=cols['syn'])
    sns.lineplot(data=df_vol_mo, x='month', y='volume', label=lebs['mo'], ax=ax3, linewidth=wids['mo'], color=cols['mo'])
    sns.lineplot(data=df_vol_ct, x='month', y='volume', label=lebs['ct'], ax=ax3, linewidth=wids['ct'], color=cols['ct'])
    ax2.set_ylim(bottom=20000000, top=60000000)
    ax2.set_xlim(left=12.5, right=16.5)
    ax2.set(xlabel='', ylabel='')
    ax2.get_legend().remove()
    ax3.set_ylim(bottom=140000000, top=220000000)
    ax3.set_xlim(left=64.5, right=67.5)
    ax3.set(xlabel='', ylabel='')
    ax3.get_legend().remove()
    synFilePath = os.path.join('plots', makeSynFileName(fileName, columns, tag='grid')+'.png')
    plt.savefig(synFilePath)
    plt.close()

def trans_avg_balance_by_cctype(df, df_mo, df_ct, fileName):
    def get_avg_bal_and_cctype(df):
        df = df[df['card_type'].isin(['gold', 'junior', 'classic'])]
        df_avg_bal = df.groupby('account_id')['balance'].mean().reset_index()
        df_avg_bal.rename(columns={'balance': 'avg_balance'}, inplace=True)
        df_cctype = df.groupby('account_id')['card_type'].max().reset_index()
        df_merged = pd.merge(df_avg_bal, df_cctype, on='account_id', how='inner')
        print(f"rows df_avg_bal {len(df_avg_bal)}, df_cctype {len(df_cctype)}, df_merged {len(df_merged)}")
        print(df_merged.head())
        return df_merged
    df_orig = get_avg_bal_and_cctype(df)
    df_mo = get_avg_bal_and_cctype(df_mo)
    if df_ct is not None:
        df_ct = get_avg_bal_and_cctype(df_ct)
        df_ct['method'] = lebs['ct']
    columns = ['avg_balance','card_type']
    pids = ['account_id']
    df_syn = do_synthesize(df_orig, columns, pids, fileName)
    print(f"df_syn has columns {list(df_syn.columns)}")
    df_orig['method'] = lebs['orig']
    df_syn['method'] = lebs['syn']
    df_mo['method'] = lebs['mo']
    if df_ct is not None:
        df_all = pd.concat([df_orig, df_syn, df_ct, df_mo])
    else:
        df_all = pd.concat([df_orig, df_syn, df_mo])

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4.5))
    sns.boxplot(x='card_type', y='avg_balance', hue='method', data=df_all, ax=axs[1])
    axs[1].set_xlabel('Card Type')
    axs[1].set_ylabel('Average Account Balance')
    axs[1].set_ylim(bottom=-20000, top=140000)
    axs[1].grid(axis='y')
    sns.boxplot(x='method', y='avg_balance', hue='card_type', data=df_all, ax=axs[0])
    axs[0].set_xlabel('Synthesis Method')
    axs[0].set_ylabel('Average Account Balance')
    axs[0].set_ylim(bottom=-20000, top=140000)
    axs[0].grid(axis='y')
    plt.tight_layout()
    synFilePath = os.path.join('plots', makeSynFileName(fileName, columns)+'.png')
    plt.savefig(synFilePath)
    plt.close()

def trans_amounts_in_out_sorted(df, df_mo, df_ct, fileName):
    ''' Make a graph with the transaction amounts sorted high to low
    '''
    columns = ['amount','operation']
    pids = ['account_id']
    df_syn = do_synthesize(df, columns, pids, fileName)
    print(f"df_syn has columns {list(df_syn.columns)}")
    df_mo = df_mo[columns]
    if df_ct is not None:
        df_ct = df_ct[columns]
    df_orig = df[columns]
    df_orig, df_mo, df_syn = unify_lengths(df_orig, df_mo, df_syn)
    # We want to divide these by deposits and withdrawls
    # operation VKLAD is deposit, everything else is withdraw
    def dep_and_sort(df):
        df_in = df[df['operation'] == 'VKLAD']
        df_out = df[df['operation'] != 'VKLAD']
        df_in = df_in.sort_values(by=columns, ascending=False)
        df_in = df_in.reset_index(drop=True)
        df_in.index.name = 'index'
        df_out = df_out.sort_values(by=columns, ascending=False)
        df_out = df_out.reset_index(drop=True)
        df_out.index.name = 'index'
        return df_in, df_out
    df_syn_in, df_syn_out = dep_and_sort(df_syn)
    df_mo_in, df_mo_out = dep_and_sort(df_mo)
    if df_ct is not None:
        df_ct_in, df_ct_out = dep_and_sort(df_ct)
    df_orig_in, df_orig_out = dep_and_sort(df_orig)
    synFilePath = os.path.join('plots', makeSynFileName(fileName, columns)+'.png')
    fig, ax = plt.subplots()
    sns.lineplot(data=df_orig_in, x=df_orig_in.index.name, y='amount', label=lebs['orig'], ax=ax, linewidth=wids['orig'], color=cols['orig'])
    sns.lineplot(data=df_syn_in, x=df_syn_in.index.name, y='amount', label=lebs['syn'], ax=ax, linewidth=wids['syn'], color=cols['syn'])
    sns.lineplot(data=df_mo_in, x=df_mo_in.index.name, y='amount', label=lebs['mo'], ax=ax, linewidth=wids['mo'], color=cols['mo'])
    if df_ct is not None:
        sns.lineplot(data=df_ct_in, x=df_ct_in.index.name, y='amount', label=lebs['ct'], ax=ax, linewidth=wids['ct'], color=cols['ct'])
    sns.lineplot(data=df_orig_out, x=df_orig_out.index.name, y='amount',  ax=ax, linewidth=wids['orig'], color=cols['orig'])
    sns.lineplot(data=df_syn_out, x=df_syn_out.index.name, y='amount', ax=ax, linewidth=wids['syn'], color=cols['syn'])
    sns.lineplot(data=df_mo_out, x=df_mo_out.index.name, y='amount', ax=ax, linewidth=wids['mo'], color=cols['mo'])
    if df_ct is not None:
        sns.lineplot(data=df_ct_out, x=df_ct_out.index.name, y='amount', ax=ax, linewidth=wids['ct'], color=cols['ct'])
    ax.set_yscale('log')
    ax.set_ylim([1,None])
    ax.grid(axis='y')
    #ax.tick_params(labelbottom=False, bottom=False)
    ax.legend()
    ax.set(xlabel='Deposits and Withdrawls, Sorted by Amount', ylabel='Deposit/Withdrawal Amount')
    plt.savefig(synFilePath)
    plt.close()

def trans_amounts_sorted(df, df_mo, df_ct, fileName):
    ''' Make a graph with the transaction amounts sorted high to low
    '''
    columns = ['amount']
    pids = ['account_id']
    df_syn = do_synthesize(df, columns, pids, fileName)
    print(f"df_syn has columns {list(df_syn.columns)}")
    df_mo = df_mo[columns]
    df_orig = df[columns]
    if df_ct is not None:
        df_ct = df_ct[columns]
        df_ct = df_ct.sort_values(by=columns, ascending=False)
        df_ct = df_ct.reset_index(drop=True)
        df_ct.index.name = 'index'
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
    sns.lineplot(data=df_orig, x=df_orig.index.name, y='amount', label=lebs['orig'], ax=ax, linewidth=wids['orig'], color=cols['orig'])
    sns.lineplot(data=df_syn, x=df_syn.index.name, y='amount', label=lebs['syn'], ax=ax, linewidth=wids['syn'], color=cols['syn'])
    sns.lineplot(data=df_mo, x=df_mo.index.name, y='amount', label=lebs['mo'], ax=ax, linewidth=wids['mo'], color=cols['mo'])
    if df_ct is not None:
        sns.lineplot(data=df_ct, x=df_ct.index.name, y='amount', label=lebs['ct'], ax=ax, linewidth=wids['ct'], color=cols['ct'])
    ax.set_yscale('log')
    ax.set_ylim([1,None])
    ax.grid(axis='y')
    #ax.tick_params(labelbottom=False, bottom=False)
    ax.legend()
    ax.set(xlabel='Transactions, Sorted by Amount', ylabel='Transaction Amount')
    plt.savefig(synFilePath)
    plt.close()

def trans_max_balance_sorted(df, df_mo, df_ct, fileName):
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

    if df_ct is not None:
        print(f"df_ct has {len(df_ct)} rows, and {df_ct['account_id'].nunique()} distinct account_id")
        df_ct_max_balance = df_ct.groupby('account_id')['balance'].max().reset_index()
        df_ct_max_balance.rename(columns={'balance': 'max_balance'}, inplace=True)
        print(f"df_ct_max_balance has {len(df_ct_max_balance)} rows, and {df_ct_max_balance['account_id'].nunique()} distinct account_id")

    # Synthesize the max_balance data
    columns = ['max_balance']
    pids = None      # Not needed because one row per pid
    df_syn = do_synthesize(df_max_balance, columns, pids, fileName)
    print(f"df_syn has columns {list(df_syn.columns)}")
    df_mo = df_mo_max_balance[columns]
    if df_ct is not None:
        df_ct = df_ct_max_balance[columns]
        df_ct = df_ct.sort_values(by=columns, ascending=False)
        df_ct = df_ct.reset_index(drop=True)
        df_ct.index.name = 'index'
    df_orig = df_max_balance[columns]
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
    sns.lineplot(data=df_orig, x=df_orig.index.name, y='max_balance', label=lebs['orig'], ax=ax, linewidth=wids['orig'], color=cols['orig'])
    sns.lineplot(data=df_syn, x=df_syn.index.name, y='max_balance', label=lebs['syn'], ax=ax, linewidth=wids['syn'], color=cols['syn'])
    sns.lineplot(data=df_mo, x=df_mo.index.name, y='max_balance', label=lebs['mo'], ax=ax, linewidth=wids['mo'], color=cols['mo'])
    if df_ct is not None:
        sns.lineplot(data=df_ct, x=df_ct.index.name, y='max_balance', label=lebs['ct'], ax=ax, linewidth=wids['ct'], color=cols['ct'])
    #ax.set_yscale('log')
    ax.grid(axis='y')
    #ax.tick_params(labelbottom=False, bottom=False)
    ax.legend()
    ax.set(xlabel='Accounts, Sorted by Max Balance', ylabel='Max Account Balance')
    plt.tight_layout()
    plt.savefig(synFilePath)
    plt.close()
    pass

df_trans = getDf('trans_account_card_clients')
if False:   # Just get some info about transactions
    for column in ['type', 'operation', 'bank','frequency']:
        counts = df_trans[column].value_counts()
        print(counts)
    df_temp = df_trans[df_trans['account_id'] == '2378']
    df_temp = df_temp.sort_values(by=['date'])
    print(df_temp[['operation','amount','balance', 'date']].to_string())

#df_trans_mo = getDf('trans_account_card_clients.mostly')
df_trans_mo = getDf('trans_account_card_clients.mostly.seq')
# TODO replace with CTGAN file when I have it
df_trans_ct = getDf('trans_account_card_clients.ctgan')
#df_trans, df_trans_mo = unify_lengths(df_trans, df_trans_mo)
if True:
    trans_amounts_in_out_sorted(df_trans, df_trans_mo, df_trans_ct, 'trans_account_card_clients')
if True:
    trans_amounts_sorted(df_trans, df_trans_mo, df_trans_ct, 'trans_account_card_clients')
if True:
    trans_max_balance_sorted(df_trans, df_trans_mo, None, 'trans_account_card_clients')
if True:
    trans_avg_balance_by_cctype(df_trans, df_trans_mo, df_trans_ct, 'trans_account_card_clients')
if True:
    trans_total_volume_by_month(df_trans, df_trans_mo, df_trans_ct, 'trans_account_card_clients')