import pandas as pd
import numpy as np
import bz2
import pickle
import requests

def download_and_load(url):
    response = requests.get(url)
    data = bz2.decompress(response.content)
    df = pickle.loads(data)
    return df

def saveDf(fileName, df):
    with bz2.BZ2File(f"{fileName}.pbz2", 'w') as f:
        pickle.dump(df, f)
    df.to_csv(f"{fileName}.csv", index=False)

df_disp = download_and_load('http://open-diffix.org/datasets/disp.pbz2')
df_disp.rename(columns={'type': 'disp_type'}, inplace=True)

print(f"Num distinct disp_id in disp: {df_disp['disp_id'].nunique()}")
print(f"Num distinct client_id in disp: {df_disp['client_id'].nunique()}")
print(f"Num distinct account_id in disp: {df_disp['account_id'].nunique()}")
print('''There is one client_id per disp_id. Purpose of disp_id is to
      show if client is OWNER or DISPONENT''')
df_card = download_and_load('http://open-diffix.org/datasets/card.pbz2')
df_card.rename(columns={'type': 'card_type'}, inplace=True)
print(f"Num distinct disp_id in card: {df_card['disp_id'].nunique()}")
print(f"Num distinct card_id in card: {df_card['card_id'].nunique()}")

df_disp_card = pd.merge(df_disp, df_card, on='disp_id', how='left')
print('df_disp_card:')
print(df_disp_card.head(3))
print(f"Num distinct card_id in disp_card: {df_disp_card['card_id'].nunique()}")
print(f"Num distinct client_id in disp_card: {df_disp_card['client_id'].nunique()}")
print(f"Number rows in disp_card where card is not NULL: {df_disp_card['card_id'].count()}")
print(f"Num distinct account_id in disp_card: {df_disp_card['account_id'].nunique()}")
df_account = download_and_load('http://open-diffix.org/datasets/account.pbz2')
df_account.rename(columns={'date': 'account_date'}, inplace=True)

# Make an account_ext (extended) table that includes the credid card information
acct_card = {}
for index, row in df_account.iterrows():
    if row['account_id'] in acct_card:
        print('duplicate account_id')
        quit()
    acct_card[row['account_id']] = {
        'account_id':row['account_id'],
        'district_id':row['district_id'],
        'frequency':row['frequency'],
        'account_date':row['account_date'],
        'card_type':None,
        'card_issued':None,
    }
for index, row in df_disp_card.iterrows():
    if row['account_id'] not in acct_card:
        print("expected account_id!")
        quit()
    if row['card_id'] is not None:
        acct_card[row['account_id']]['card_type'] = row['card_type']
        acct_card[row['account_id']]['card_issued'] = row['issued']

df_acct_card = pd.DataFrame(list(acct_card.values()))
saveDf('account_card', df_acct_card)
print(f"Num distinct account_id in account: {df_account['account_id'].nunique()}")
print(f"Num distinct account_id in account_card: {df_acct_card['account_id'].nunique()}")
print(f"rows df_account {len(df_account)}, df_disp {len(df_disp)}, df_card {len(df_card)}, df_disp_card {len(df_disp_card)}, df_acct_card {len(df_acct_card)}")

# Now let's add the client information and make another table
df_client = download_and_load('http://open-diffix.org/datasets/client.pbz2')
df_disp_client = pd.merge(df_disp, df_client, on='client_id', how='left')
print('df_disp_client:')
print(df_disp_client.head(3))
# First add the OWNERS
for index, row in df_disp_client.iterrows():
    if row['disp_type'] not in ['OWNER', 'DISPONENT']:
        print("bad disp_type")
        quit()
    if row['disp_type'] == 'OWNER':
        acct_card[row['account_id']]['owner_birth'] = row['birth_number']
        acct_card[row['account_id']]['owner_district_id'] = row['district_id']
        acct_card[row['account_id']]['owner_sex'] = row['sex']
        acct_card[row['account_id']]['disponent_birth'] = None
        acct_card[row['account_id']]['disponent_district_id'] = None
        acct_card[row['account_id']]['disponent_sex'] = None
# And then the Disponents
for index, row in df_disp_client.iterrows():
    if row['disp_type'] == 'DISPONENT':
        if 'owner_district_id' not in acct_card[row['account_id']]:
            print("Missing owner_district_id")
            quit()
        acct_card[row['account_id']]['disponent_birth'] = row['birth_number']
        acct_card[row['account_id']]['disponent_district_id'] = row['district_id']
        acct_card[row['account_id']]['disponent_sex'] = row['sex']

df_acct_card_clients = pd.DataFrame(list(acct_card.values()))
print('df_acct_card_clients')
print(df_acct_card_clients.head(3))
saveDf('account_card_clients', df_acct_card_clients)

# Now make all of the event tables
df_trans = download_and_load('http://open-diffix.org/datasets/trans.pbz2')
df_trans_account = pd.merge(df_trans, df_account, on='account_id', how='left')
saveDf('trans_account', df_trans_account)
df_trans_account_card = pd.merge(df_trans, df_acct_card, on='account_id', how='left')
saveDf('trans_account_card', df_trans_account_card)
df_trans_account_card_clients = pd.merge(df_trans, df_acct_card_clients, on='account_id', how='left')
saveDf('trans_account_card_clients', df_trans_account_card_clients)

df_loan = download_and_load('http://open-diffix.org/datasets/loan.pbz2')
df_loan_account = pd.merge(df_loan, df_account, on='account_id', how='left')
saveDf('loan_account', df_loan_account)
df_loan_account_card = pd.merge(df_loan, df_acct_card, on='account_id', how='left')
saveDf('loan_account_card', df_loan_account_card)
df_loan_account_card_clients = pd.merge(df_loan, df_acct_card_clients, on='account_id', how='left')
saveDf('loan_account_card_clients', df_loan_account_card_clients)

df_order = download_and_load('http://open-diffix.org/datasets/order.pbz2')
df_order_account = pd.merge(df_order, df_account, on='account_id', how='left')
saveDf('order_account', df_order_account)
df_order_account_card = pd.merge(df_order, df_acct_card, on='account_id', how='left')
saveDf('order_account_card', df_order_account_card)
df_order_account_card_clients = pd.merge(df_order, df_acct_card_clients, on='account_id', how='left')
saveDf('order_account_card_clients', df_order_account_card_clients)

