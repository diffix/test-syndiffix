import os
import json
import zipfile
import pprint
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from misc.csvUtils import readCsv

'''
After running tests at mostly.io, save the CSV and log files into a single directory `mostlyResults` under a directory `mostly`.

This looks for the original csv files into the directory `csvAb/train`. (Alternatively modify the directory names in the code to match the location of files.)

This routine scrubs the appropriate results (column names, elapsedTime, and original and synthetic data), and generates `.json` files that match the same format as the PPoC and abSharp outputs. These `.json` files can then be read in by the measurement software as normal.
'''

pp = pprint.PrettyPrinter(indent=4)

baseDir = os.path.join(os.environ['AB_RESULTS_DIR'])
# mostlyBaseDir = os.path.join(baseDir, 'mostly', 'half1')
mostlyBaseDir = os.path.join(baseDir, 'mostly')
csvInPath = os.path.join(baseDir, 'csvAb', 'train')
csvTestPath = os.path.join(baseDir, 'csvAb', 'test')
mostlyInPath = os.path.join(mostlyBaseDir, 'mostlyResults')
os.makedirs(mostlyInPath, exist_ok=True)
mostlyOutPath = os.path.join(mostlyBaseDir, 'mostlyExtracted')
os.makedirs(mostlyOutPath, exist_ok=True)
mostlyJson = os.path.join(mostlyBaseDir, 'mostlyJson')
os.makedirs(mostlyJson, exist_ok=True)


def getElapsedTime(f, path):
    # print('---------',path, '------------')
    numFound = 0
    elapsed = 0
    lines = f.readlines()
    for line in lines:
        if 'DELIVER' in line:
            continue
        if 'finished' in line:
            # print(line)
            timeTerm = line.split()[-1]
            if timeTerm[-1] != 's':
                print(f"FAIL1, {line}")
                quit()
            try:
                increment = float(timeTerm[:-1])
            except:
                pass
            else:
                elapsed += increment
                numFound += 1
    if numFound != 1:
        pp.pprint(lines)
        print(f"SOMETHING WRONG numFound = {numFound}")
        print(path)
        quit()
    return elapsed


# Get list of datasource names
dataSourceNames = {}
files = [f for f in os.listdir(mostlyInPath) if os.path.isfile(os.path.join(mostlyInPath, f))]
for file in files:
    if file[-9:] == '-logs.zip':
        dataSourceNames[file[:-9]] = {}
# Get needed files
pp.pprint(dataSourceNames)
for file in files:
    print(file)
    for fileRoot in dataSourceNames.keys():
        if 'CSV' in file and file[:-25] == fileRoot:
            dataSourceNames[fileRoot]['csv'] = file
        if 'logs.zip' in file:
            logsBase = fileRoot + '-'
            if logsBase in file:
                dataSourceNames[fileRoot]['log'] = file
# Extract
for fileRoot, paths in dataSourceNames.items():
    pp.pprint(paths)
    toPath = os.path.join(mostlyOutPath, fileRoot)
    os.makedirs(toPath, exist_ok=True)
    csvExtractPath = os.path.join(mostlyInPath, paths['csv'])
    with zipfile.ZipFile(csvExtractPath, 'r') as zip_ref:
        zip_ref.extractall(toPath)
    logExtractPath = os.path.join(mostlyInPath, paths['log'])
    with zipfile.ZipFile(logExtractPath, 'r') as zip_ref:
        zip_ref.extractall(toPath)
for fileRoot in dataSourceNames.keys():
    results = {}
    csvPath = os.path.join(csvInPath, fileRoot + '.csv')
    dfOrigCsv = readCsv(csvPath)
    testPath = os.path.join(csvTestPath, fileRoot + '.csv')
    dfTestCsv = readCsv(testPath)
    results['colNames'] = list(dfOrigCsv.columns)

    elapsedTime = 0
    resultsDir = os.path.join(mostlyOutPath, fileRoot)
    logsDir = os.path.join(resultsDir, f'{fileRoot}-1-GENERATE_SUBJECT')
    logsList = ['analyze.log', 'encode.log', 'generate.log', 'train.log', 'generate-rep-data.log']
    for logFile in logsList:
        logFilePath = os.path.join(logsDir, logFile)
        with open(logFilePath, 'r') as f:
            elapsedTime += getElapsedTime(f, logFilePath)
    results['elapsedTime'] = elapsedTime

    synCsvPath = os.path.join(resultsDir, fileRoot, fileRoot + '.csv')
    dfSynCsv = readCsv(synCsvPath)
    if list(dfSynCsv.columns) != results['colNames']:
        print("FAIL columns")
        print(dfSynCsv.columns)
        print(results['colNames'])
        quit()
    results['originalTable'] = dfOrigCsv.values.tolist()
    results['testTable'] = dfTestCsv.values.tolist()
    results['anonTable'] = dfSynCsv.values.tolist()
    jsonPath = os.path.join(mostlyJson, fileRoot + '.csv.json')
    print(f"Writing {jsonPath}")
    with open(jsonPath, 'w') as f:
        json.dump(results, f, indent=4)
