import os
import json
import zipfile
import pandas as pd
import pprint

'''
'''

pp = pprint.PrettyPrinter(indent=4)

baseDir = os.path.join(os.environ['AB_RESULTS_DIR'])
synthpopBaseDir = os.path.join(baseDir, 'synthpop')
csvInPath = os.path.join(baseDir, 'csvAb')
synthpopInPath = synthpopBaseDir
synthpopJson = os.path.join(synthpopBaseDir, 'synthpopJson')
os.makedirs(synthpopJson, exist_ok=True)

# Get list of datasource names
dataSourceNames = {}
files = [f for f in os.listdir(synthpopInPath) if os.path.isfile(os.path.join(synthpopInPath, f))]
for file in files:
    if file[-4:] == '.csv':
        dataSourceNames[file[:-4]] = True
    
for fileRoot in dataSourceNames.keys():
    results = {}
    csvPath = os.path.join(csvInPath, fileRoot)
    dfOrigCsv = pd.read_csv(csvPath,index_col=False)
    results['colNames'] = list(dfOrigCsv.columns)

    elapsedPath = os.path.join(synthpopInPath, fileRoot+'.json')
    with open(elapsedPath, 'r') as f:
        elapsedJson = json.load(f)
        elapsedTime = elapsedJson[0]
    results['elapsedTime'] = elapsedTime

    synCsvPath = os.path.join(synthpopInPath, fileRoot+'.csv')
    dfSynCsv = pd.read_csv(synCsvPath,index_col=False)
    # synthpop for some reasons changed ':' to '.' in column names
    renamer = {}
    for cOrig, cAnon in zip(results['colNames'], list(dfSynCsv.columns)):
        renamer[cAnon] = cOrig
    dfSynCsv = dfSynCsv.rename(columns=renamer)

    results['originalTable'] = dfOrigCsv.values.tolist()
    results['anonTable'] = dfSynCsv.values.tolist()
    jsonPath = os.path.join(synthpopJson, fileRoot+'.json')
    print(f"Writing {jsonPath}")
    with open(jsonPath, 'w') as f:
        json.dump(results, f, indent=4)