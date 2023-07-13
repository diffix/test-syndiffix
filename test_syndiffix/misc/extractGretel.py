import os
import json
import pprint
import sys
import dateutil
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from misc.csvUtils import readCsv

'''
This routine creates results JSON files from (gretel) CSVs
'''

pp = pprint.PrettyPrinter(indent=4)

baseDir = os.path.join(os.environ['AB_RESULTS_DIR'])
csvTrainDir = os.path.join(baseDir, 'csvAb', 'train')
csvTestDir = os.path.join(baseDir, 'csvAb', 'test')
csvAnonDir = os.path.join(baseDir, 'resultsgretel')
resultsDir = os.path.join(baseDir, 'resultsAb', 'gretel')
os.makedirs(resultsDir, exist_ok=True)


def dropNewNaNRows(dfTrain, dfAnon, csvFile):
    """
    Drops rows which have NaNs in columns which didn't have any NaNs in the original data.
    """
    nanColumnsTrain = dfTrain.columns[dfTrain.isna().any()].tolist()
    nanColumnsAnon = dfAnon.columns[dfAnon.isna().any()].tolist()
    nanColumnsGretel = list(set(nanColumnsAnon) - set(nanColumnsTrain))

    if nanColumnsGretel != []:
        print("Different NaN columns in anon table", csvFile, "gretel introduced NaNs in:", nanColumnsGretel)
        nanRows = np.where(dfAnon[nanColumnsGretel].isna().any(axis=1))[0]
        print(f"Dropping {len(nanRows)} rows: {nanRows}, like: {dfAnon.loc[nanRows, :]}")
        dfAnon.drop(nanRows, inplace=True)


files = [f for f in os.listdir(csvTrainDir) if os.path.isfile(os.path.join(csvTrainDir, f))]
# Get needed files
for csvFile in files:
    if csvFile[-4:] != '.csv':
        print(f"{csvFile} is not a csv file")
        continue
    trainPath = os.path.join(csvTrainDir, csvFile)
    dfTrain = readCsv(trainPath)
    testPath = os.path.join(csvTestDir, csvFile)
    dfTest = readCsv(testPath).reindex(dfTrain.columns, axis=1)
    try:
        anonPath = os.path.join(csvAnonDir, csvFile)
        dfAnon = readCsv(anonPath).reindex(dfTrain.columns, axis=1)
    except FileNotFoundError:
        continue

    dropNewNaNRows(dfTrain, dfAnon, csvFile)

    results = {}
    results['colNames'] = list(dfTrain.columns)
    results['elapsedTime'] = None
    results['originalTable'] = dfTrain.values.tolist()
    results['testTable'] = dfTest.values.tolist()
    results['anonTable'] = dfAnon.values.tolist()

    logPath = os.path.join(csvAnonDir, csvFile + '.log.json')
    try:
        with open(logPath) as f:
            logs = json.load(f)

        logTimestamps = [dateutil.parser.parse(l['ts']).timestamp() for l in logs]
        results['elapsedTime'] = logTimestamps[-1] - logTimestamps[0]
    except FileNotFoundError:
        print("Log file for", csvFile, "not found")

    resultsName = csvFile + '.json'
    jsonPath = os.path.join(resultsDir, resultsName)
    print(f"Writing {jsonPath}")
    with open(jsonPath, 'w') as f:
        json.dump(results, f, indent=4)
