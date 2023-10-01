import os
import sys
import pandas as pd
import sqlalchemy as sq
import json
import pprint
import combsTables

pp = pprint.PrettyPrinter(indent=4)

expDir = 'exp_combs'
#synMethod = 'sdx_release'
synMethod = 'sdx_test'
databaseName = 'sdx_demo'
addTable = False

pgHost = os.environ['GDA_POSTGRES_HOST']
pgUser = os.environ['GDA_POSTGRES_USER']
pgPass = os.environ['GDA_POSTGRES_PASS']
print(f"Connect to {pgHost} as user {pgUser} and password {pgPass}")
sio = combsTables.sqlIo(pgHost, databaseName, pgUser, pgPass)
sql = f"SELECT * FROM pg_database WHERE datname = '{databaseName}'"
ans = sio.querySql(sql)
print(ans)

cmd = combsTables.combsMetaData(sio)
ct = combsTables.combsTables()

engine = sq.create_engine(f'postgresql://{pgUser}:{pgPass}@{pgHost}:5432/{databaseName}')
#engine = sq.create_engine(f'postgresql+psycopg2://{pgUser}:{pgPass}@{pgHost}:5432/{databaseName}')
#df.to_sql('table_name', engine)

resultsDir = os.path.join(os.environ['AB_RESULTS_DIR'], expDir, 'results', synMethod)
files = [f for f in os.listdir(resultsDir) if os.path.isfile(os.path.join(resultsDir, f))]

for fileName in files:
    if fileName[-5:] != '.json':
        print(f"Bad filename {fileName}")
        quit()
    filePath = os.path.join(resultsDir, fileName)
    with open(filePath, 'r') as f:
        data = json.load(f)
    job = data['colCombsJob']
    if job['tableBase'] == job['tableName']:
        # This is a dataset with all columns
        cmd.putMetaData(job['tableBase'], job['synColumns'])
        allColumns = cmd.getMetaData(job['tableBase'])
        print(allColumns)
        quit()
    dfAnon = pd.DataFrame(data['anonTable'], columns=data['colNames'])
    pass