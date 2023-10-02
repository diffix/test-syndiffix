import os
import sys
import pandas as pd
import json
import pprint
import combsTables

pp = pprint.PrettyPrinter(indent=4)

expDir = 'exp_combs'
synMethod = 'sdx_release'
#synMethod = 'sdx_test'
databaseName = 'sdx_demo'
addTable = False
part = 2
totalParts = 5

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


resultsDir = os.path.join(os.environ['AB_RESULTS_DIR'], expDir, 'results', synMethod)
files = [f for f in os.listdir(resultsDir) if os.path.isfile(os.path.join(resultsDir, f))]

filesPerPart = int(len(files) / totalParts)
filesStart = min(0,(filesPerPart * (part-1)) - 5)
filesEnd = (filesPerPart * part) + 5
print(filesStart, filesEnd)
quit()

for fileName in files[filesStart:filesEnd]:
    if fileName[-5:] != '.json':
        print(f"Bad filename {fileName}")
        quit()
    filePath = os.path.join(resultsDir, fileName)
    with open(filePath, 'r') as f:
        data = json.load(f)
    job = data['colCombsJob']
    if job['tableBase'] == job['tableName']:
        # This is a dataset with all columns
        # save the column metadata
        cmd.putMetaData(job['tableBase'], job['synColumns'])
        # This not really necessary, but just shows that the put worked
        allColumns = cmd.getMetaData(job['tableBase'])
        print(allColumns)
        # make a dataframe from the original data
        dfOrig = pd.DataFrame(data['originalTable'], columns=data['colNames'])
        print("Columns of dfOrig")
        print(list(dfOrig.columns.values))
        print("Length of dfOrig")
        print(dfOrig.shape[0])
        tableName = job['tableBase'] + '_orig_'
        # Check and see if we've already loaded in the table!
        sio.loadDataFrame(dfOrig, tableName)
    # Now do the synthetic data
    dfAnon = pd.DataFrame(data['anonTable'], columns=data['colNames'])
    if job['tableBase'] == job['tableName']:
        tableName = job['tableBase'] + '_syn_'
    else:
        tableName = ct.makeTableFromColumns(list(dfAnon.columns.values), job['tableBase'])
    print(f"tableName = {tableName}")
    sio.loadDataFrame(dfAnon, tableName)