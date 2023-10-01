import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import psycopg2
import json
import pprint
from misc.csvUtils import readCsv

pp = pprint.PrettyPrinter(indent=4)

expDir = 'exp_combs'
synMethod = 'sdx_release'
databaseName = 'sdx_demo'
createDatabase = True

def runSql(cur, sql):
    print(sql)
    try:
        cur.execute(sql)
    except Exception as err:
        print ("cur.execute() error:", err)
        print ("Exception TYPE:", type(err))
    else:
        print("SQL ok")

pgHost = os.environ['GDA_POSTGRES_HOST']
pgUser = os.environ['GDA_POSTGRES_USER']
pgPass = os.environ['GDA_POSTGRES_PASS']
print(f"Connect to {pgHost} as user {pgUser} and password {pgPass}")
connStr = str(
            f"host={pgHost} port={5432} dbname=sdx_demo user={pgUser} password={pgPass}")
conn = psycopg2.connect(connStr)
cur = conn.cursor()
sql = f"SELECT * FROM pg_database WHERE datname = '{databaseName}'"
runSql(cur, sql)
ans = cur.fetchall()
print(ans)
quit()

resultsDir = os.path.join(os.environ['AB_RESULTS_DIR'], expDir, 'results', synMethod)
files = [f for f in os.listdir(resultsDir) if os.path.isfile(os.path.join(resultsDir, f))]

for fileName in files:
    if fileName[-5:] != '.json':
        print(f"Bad filename {fileName}")
        quit()
    filePath = os.path.join(resultsDir, fileName)
    with open(filePath, 'r') as f:
        data = json.load(f)
    pp.pprint(data['colCombsJob'])
    dfAnon = pd.DataFrame(data['anonTable'], columns=data['colNames'])
    pass