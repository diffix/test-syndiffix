import os
import sys
import pandas as pd
import json
import pprint
import fire
import shlex
import time
import subprocess
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import testUtils
import sdmetricsPlay
import sdmTools
import csv

''' This is used to run the SDMetrics synthetic data models in SLURM
'''

pp = pprint.PrettyPrinter(indent=4)


def runTest(runModel, metaData, df, colNames, outPath, dataSourceNum, testData):
    print(f"Training dataframe shape {df.shape}")
    print("First row of data:")
    print(df.iloc[0])
    quit()
    if runModel == 'gaussianCopula':
        from sdv.tabular import GaussianCopula
        model = GaussianCopula()
    elif runModel == 'ctGan':
        from sdv.tabular import CTGAN
        model = CTGAN()
    elif runModel == 'fastMl':
        from sdv.lite import TabularPreset
        model = TabularPreset(name='FAST_ML', metadata=metaData)
    elif runModel == 'copulaGan':
        from sdv.tabular import CopulaGAN
        model = CopulaGAN()
    elif runModel == 'tvae':
        from sdv.tabular import TVAE
        model = TVAE()
    tempPath = f"temp.{dataSourceNum}.csv"
    start = time.time()
    tries = 2
    origNumRows = df.shape[0]
    while True:
        try:
            model.fit(df)
        except:
            print(f"model.fit failed. Try cutting num rows in half (from {df.shape[0]})")
            df = df.sample(frac=0.5)
            tries -= 1
        else:
            break
        if tries == 0:
            print("Failed to run model.fit.")
            return
    synData = model.sample(num_rows=origNumRows, output_file_path=tempPath)
    end = time.time()
    if os.path.exists(tempPath):
        os.remove(tempPath)

    print(synData.head())
    outJson = {}
    outJson['elapsedTime'] = end - start
    outJson['colNames'] = colNames
    outJson['originalTable'] = df.values.tolist()
    outJson['anonTable'] = synData.values.tolist()
    outJson['testTable'] = testData
    print(f"Writing output to {outPath}")
    with open(outPath, 'w') as f:
        json.dump(outJson, f, indent=4)


def runAbSharp(tu, dataSourcePath, outPath, abSharpArgs, columns, focusColumn, testData):
    thisDir = os.path.dirname(os.path.abspath(__file__))
    abSharpDir = os.path.join(tu.abSharpDir, 'src', 'SynDiffix.Debug')

    abSharpArgs = shlex.split(abSharpArgs)
    print(f"cmd-line args: {abSharpArgs}")

    abSharp = subprocess.run(
        ['dotnet', 'run', '--configuration', 'Release', dataSourcePath, '--columns', *columns, *abSharpArgs],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=abSharpDir,
    )

    print(abSharp.stderr.decode("utf-8"))

    outJson = json.loads(abSharp.stdout.decode("utf-8"))
    outJson['originalTable'] = outJson.pop('originalRows')
    outJson['anonTable'] = outJson.pop('synRows')
    outJson['testTable'] = testData
    if focusColumn:
        outJson['focusColumn'] = focusColumn

    print(f"Writing output to {outPath}")
    with open(outPath, 'w') as f:
        json.dump(outJson, f, indent=4)


def makeMetadata(df):
    ''' This makes the metadata file expected by sdmetrics '''
    cols = list(df.columns)
    colTypes = []
    for colType in df.dtypes:
        if pd.api.types.is_integer_dtype(colType):
            colTypes.append('integer')
        elif pd.api.types.is_float_dtype(colType) or pd.api.types.is_numeric_dtype(colType):
            colTypes.append('float')
        elif pd.api.types.is_string_dtype(colType) or pd.api.types.is_object_dtype(colType):
            colTypes.append('text')
        else:
            print(f"ERROR: Unknown column data type {colType}")
            a = 1 / 0
            quit()
    metadata = {'sdvMetaData': {
        "METADATA_SPEC_VERSION": "SINGLE_TABLE_V1",
        'columns': {},
        'constraints': [],
    },
        'classifications': {
    },
    }
    gotBinary = False
    gotNumeric = False
    gotCategorical = False
    for colName, colType in zip(cols, colTypes):
        numUnique = df[colName].nunique()
        if numUnique <= 50 or colType == 'text':
            metadata['sdvMetaData']['columns'][colName] = {'type': 'categorical'}
            if numUnique == 2 and not gotBinary:
                gotBinary = True
                metadata['classifications']['binary'] = [colName]
            elif not gotCategorical:
                gotCategorical = True
                metadata['classifications']['categorical'] = [colName]
        elif colType in ['integer', 'float']:
            metadata['sdvMetaData']['columns'][colName] = {'type': 'numerical', 'subtype': colType}
            if not gotNumeric:
                gotNumeric = True
                metadata['classifications']['numeric'] = [colName]
    return metadata


def oneModel(dataDir='csvGeneral', dataSourceNum=0, model='fastMl', suffix='', synResults='synResults', synMeasures='synMeasures', abSharpArgs='', runsDir='runsAb', doMeasures=False, withFocusColumn=False):
    tu = testUtils.testUtilities()
    tu.registerCsvLib(dataDir)
    tu.registerSynResults(synResults)
    if len(abSharpArgs) > 0:
        print(f"abSharpArgs: {abSharpArgs}")
    focusColumn = None
    if not withFocusColumn:
        inFiles = [f for f in os.listdir(
            tu.csvLib) if os.path.isfile(os.path.join(tu.csvLib, f))]
        dataSources = []
        for fileName in inFiles:
            if fileName[-3:] == 'csv':
                dataSources.append(fileName)
        dataSources.sort()
        if dataSourceNum > len(dataSources) - 1:
            print(f"ERROR: There are not enough datasources (dataSourceNum={dataSourceNum})")
            quit()
        sourceFileName = dataSources[dataSourceNum]
        baseFileName = sourceFileName[:-4]
        print(f"Using source file {sourceFileName}")
    else:
        tu.registerRunsDir(runsDir)
        mc = sdmTools.measuresConfig(tu)
        sourceFileName, focusColumn = mc.getFocusFromJobNumber(dataSourceNum)
        if sourceFileName is None:
            print(f"ERROR: Couldn't find focus job")
            quit()
    dataSourcePath = os.path.join(tu.csvLib, sourceFileName)
    if not os.path.exists(dataSourcePath):
        print(f"ERROR: File {dataSourcePath} does not exist")
        quit()
    testDataPath = os.path.join(tu.csvLibTest, sourceFileName)
    if not os.path.exists(testDataPath):
        print(f"ERROR: File {testDataPath} does not exist")
        quit()

    label = model + '_' + suffix if suffix else model
    modelsDir = os.path.join(tu.synResults, label)
    os.makedirs(modelsDir, exist_ok=True)
    if not withFocusColumn:
        outPath = os.path.join(modelsDir, f"{sourceFileName}.json")
    else:
        outPath = os.path.join(modelsDir, f"{sourceFileName}.{focusColumn}.json")
    if os.path.exists(outPath):
        print(f"Result {outPath} already exists, skipping")
        print("oneModel:SUCCESS (skipped)")
        return
    print(f"Model {label} for dataset {dataSourcePath}, focus column {focusColumn}")

    df = pd.read_csv(dataSourcePath, index_col=False, low_memory=False)
    print(f"Training dataframe shape {df.shape}")
    colNames = list(df.columns.values)
    # quick test to make sure that the test and train data match columns
    dfTest = pd.read_csv(testDataPath, index_col=False, low_memory=False)
    if colNames != list(dfTest.columns.values):
        print(f("ERROR: Train column names {colNames} don't match test column names {list(dfTest.columns.values)}"))
        quit()
    # Pull in training data as list, to be stored in results file as is
    testData = dfTest.values.tolist()
    print(f"Columns {colNames}")
    mls = testUtils.mlSupport(tu)
    metaData = makeMetadata(df)
    if model == 'abSharp' or 'syndiffix' in model:
        colTypeSymbols = {'text': 's', 'real': 'r', 'datetime': 't', 'int': 'i'}
        colTypes = tu.getColTypesFromDataframe(df)
        columns = []
        for colName, colType in zip(colNames, colTypes):
            if colType is None:
                # This means that we couldn't assign a type. I'm guessing that 'text' is a robust
                # type that absharp can handle regardless of what the value are...
                colType = 'text'
            columns.append(f"{colName}:{colTypeSymbols[colType]}")
        if withFocusColumn:
            abSharpArgs += f" --clustering-maincolumn '{focusColumn}' "
        runAbSharp(tu, dataSourcePath, outPath, abSharpArgs, columns, focusColumn, testData)
    else:
        runTest(model, metaData['sdvMetaData'], df, colNames, outPath, dataSourceNum, testData)

    if doMeasures is False:
        print("oneModel:SUCCESS")
        return

    mls = testUtils.mlSupport(tu)
    mlClassInfo = mls.makeMlClassInfo(df, sourceFileName)
    tu.registerSynMeasure(synMeasures)
    with open(outPath, 'r') as f:
        results = json.load(f)
    abInfo = {}
    abInfo['label'] = label
    abInfo['columns'] = results['colNames']
    abInfo['mlClassInfo'] = mlClassInfo
    abInfo['elapsedTime'] = results['elapsedTime'] if 'elapsedTime' in results else None
    abInfo['params'] = {'dataSource': sourceFileName,
                        'outputFile': sourceFileName + '.json',
                        }

    sdm = sdmetricsPlay.abSdmetrics(colNames, results['originalTable'],
                                    results['anonTable'], None, None, None,
                                    fileName=baseFileName, dir=os.path.join(tu.synMeasures, label),
                                    abInfo=abInfo,
                                    )
    sdm.runAll()
    sdm.runMl()
    print("oneModel:SUCCESS")


if __name__ == "__main__":
    fire.Fire(oneModel)
