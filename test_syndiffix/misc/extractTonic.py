import os
import json
import pprint
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from misc.csvUtils import readCsv

'''
This routine creates results JSON files from (tonic) CSVs
'''

pp = pprint.PrettyPrinter(indent=4)

baseDir = os.path.join(os.environ['AB_RESULTS_DIR'])
csvTrainDir = os.path.join(baseDir, 'csvAb', 'train')
csvTestDir = os.path.join(baseDir, 'csvAb', 'test')
csvAnonDir = os.path.join(baseDir, 'resultstonic')
resultsDir = os.path.join(baseDir, 'resultsAb', 'tonic')
os.makedirs(resultsDir, exist_ok=True)


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
        anonPath = os.path.join(csvAnonDir, csvFile.lower().replace('-', '_'))
        dfAnon = readCsv(anonPath).reindex([c.lower().replace(':', '').replace('-', '_')
                                            for c in dfTrain.columns], axis=1)
    except FileNotFoundError:
        print(csvFile, "anon DataFrame file not found")
        continue
    results = {}
    results['colNames'] = list(dfTrain.columns)
    # Not 0 to avoid log issues
    results['elapsedTime'] = 0.0000001
    results['originalTable'] = dfTrain.values.tolist()
    results['testTable'] = dfTest.values.tolist()
    results['anonTable'] = dfAnon.values.tolist()

    resultsName = csvFile + '.json'
    jsonPath = os.path.join(resultsDir, resultsName)
    print(f"Writing {jsonPath}")
    with open(jsonPath, 'w') as f:
        json.dump(results, f, indent=4)
