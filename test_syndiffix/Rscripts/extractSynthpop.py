import os
import json
import pprint
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from misc.csvUtils import readCsv


pp = pprint.PrettyPrinter(indent=4)

baseDir = os.path.join(os.environ['AB_RESULTS_DIR'])
# This is where the output of synthpop is
synthpopBaseDir = os.path.join(baseDir, 'exp_synthpop', 'synthpop_builds')
# This is where the csv files are
csvInPath = os.path.join(baseDir, 'exp_synthpop', 'csv')
csvTrainPath = os.path.join(csvInPath, 'train')
csvTestPath = os.path.join(csvInPath, 'test')
synthpopInPath = synthpopBaseDir
# This is where we'll put the resulting json file
synthpopJson = os.path.join(synthpopBaseDir, 'synthpopJson')
os.makedirs(synthpopJson, exist_ok=True)

# Get list of datasource names
dataSourceNames = {}
files = [f for f in os.listdir(synthpopInPath) if os.path.isfile(os.path.join(synthpopInPath, f))]
for file in files:
    if file[-4:] == '.csv':
        dataSourceNames[file[:-4]] = True

for fileRoot in dataSourceNames.keys():
    print(f"fileRoot is {fileRoot}")
    results = {}
    csvTrainPath = os.path.join(csvTrainPath, fileRoot)
    print(f"csvTrainPath is {csvTrainPath}")
    dfTrainCsv = readCsv(csvTrainPath)
    results['colNames'] = list(dfTrainCsv.columns)
    csvTestPath = os.path.join(csvTestPath, fileRoot)
    dfTestCsv = readCsv(csvTestPath)

    elapsedPath = os.path.join(synthpopInPath, fileRoot + '.json')
    with open(elapsedPath, 'r') as f:
        elapsedJson = json.load(f)
        elapsedTime = elapsedJson[0]
    results['elapsedTime'] = elapsedTime

    synCsvPath = os.path.join(synthpopInPath, fileRoot + '.csv')
    dfSynCsv = readCsv(synCsvPath)
    # synthpop for some reasons changed ':' to '.' in column names
    renamer = {}
    for cOrig, cAnon in zip(results['colNames'], list(dfSynCsv.columns)):
        renamer[cAnon] = cOrig
    dfSynCsv = dfSynCsv.rename(columns=renamer)

    results['originalTable'] = dfTrainCsv.values.tolist()
    results['testTable'] = dfTestCsv.values.tolist()
    results['anonTable'] = dfSynCsv.values.tolist()
    jsonPath = os.path.join(synthpopJson, fileRoot + '.json')
    print(f"Writing {jsonPath}")
    with open(jsonPath, 'w') as f:
        json.dump(results, f, indent=4)
