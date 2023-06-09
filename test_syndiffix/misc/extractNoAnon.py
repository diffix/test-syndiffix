import os
import json
import zipfile
import pandas as pd
import pprint

'''
This routine creates results files for the case of no anonymization
'''

pp = pprint.PrettyPrinter(indent=4)

baseDir = os.path.join(os.environ['AB_RESULTS_DIR'])
csvTrainDir = os.path.join(baseDir, 'csvAb', 'train')
csvTestDir = os.path.join(baseDir, 'csvAb', 'test')
resultsDir = os.path.join(baseDir, 'resultsAb', 'noAnon')
os.makedirs(resultsDir, exist_ok=True)

# Get list of datasource names
dataSourceNames = {}
files = [f for f in os.listdir(csvTrainDir) if os.path.isfile(os.path.join(csvTrainDir, f))]
# Get needed files
for csvFile in files:
    if csvFile[-4:] != '.csv':
        print(f"{csvFile} is not a csv file")
        continue
    trainPath = os.path.join(csvTrainDir, csvFile)
    dfTrain = pd.read_csv(trainPath, index_col=False, low_memory=False)
    testPath = os.path.join(csvTestDir, csvFile)
    if not os.path.exists(testPath):
        print(f"ERROR: Missing {testPath}!")
        quit()
    dfTest = pd.read_csv(testPath, index_col=False, low_memory=False)
    results = {}
    results['colNames'] = list(dfTrain.columns)
    # Not 0 to avoid log issues
    results['elapsedTime'] = 0.0000001
    results['originalTable'] = dfTrain.values.tolist()
    results['testTable'] = dfTest.values.tolist()
    results['anonTable'] = dfTrain.values.tolist()

    resultsName = csvFile + '.json'
    jsonPath = os.path.join(resultsDir, resultsName)
    print(f"Writing {jsonPath}")
    with open(jsonPath, 'w') as f:
        json.dump(results, f, indent=4)
