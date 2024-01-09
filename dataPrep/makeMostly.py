import pandas as pd
import os

filePath = os.path.join('datasets', 'trans.mostly.seq.csv')
df_trans = pd.read_csv(filePath, low_memory=False)
filePath = os.path.join('datasets', 'account_card_clients.mostly.csv')
df_account = pd.read_csv(filePath, low_memory=False)
df_merged = pd.merge(df_trans, df_account, on='account_id', how='inner')
df_merged.to_csv(os.path.join('datasets','trans_account_card_clients.mostly.seq.csv'), index=False)
quit()


filePath = os.path.join('datasets', 'trans_account_card_clients.csv')
tgt = pd.read_csv(filePath, low_memory=False)
subject_cols = ['account_id', 'district_id', 'frequency',
       'account_date', 'card_type', 'card_issued', 'owner_birth',
       'owner_district_id', 'owner_sex', 'disponent_birth',
       'disponent_district_id', 'disponent_sex']
trans_cols = ['trans_id', 'account_id', 'date', 'type', 'operation', 'amount',
       'balance', 'k_symbol', 'bank', 'account']
s = tgt[subject_cols].drop_duplicates()
t = tgt[trans_cols]
s.to_csv(os.path.join('datasets','mostly.subject.table.csv'), index=False)
t.to_csv(os.path.join('datasets','mostly.linked.trans.table.csv'), index=False)