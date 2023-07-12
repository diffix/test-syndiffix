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
from misc.csvUtils import readCsv

''' This is used to run the SDMetrics synthetic data models in SLURM
'''

pp = pprint.PrettyPrinter(indent=4)

def runTest(runModel, metaData, df, colNames, outPath, dataSourceNum, testData):
    print("First row of data:")
    print(df.iloc[0])
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
    print(df.shape)
    outJson['originalTable'] = df.values.tolist()
    outJson['anonTable'] = synData.values.tolist()
    outJson['testTable'] = testData
    print(f"Writing output to {outPath}")
    with open(outPath, 'w') as f:
        json.dump(outJson, f, indent=4)


def runAbSharp(tu, dataSourcePath, outPath, abSharpArgs, columns, focusColumn, testData, featuresJob, extraArgs=[]):
    print(f"running runAbSharp with:\n dataSourcePath {dataSourcePath}\n outPath {outPath}\n abSharpArgs '{abSharpArgs}'\n columns {columns}\n focusColumn {focusColumn}\n featuresJob {featuresJob}\n extraArgs {extraArgs}\n")
    thisDir = os.path.dirname(os.path.abspath(__file__))
    abSharpDir = os.path.join(tu.abSharpDir, 'src', 'SynDiffix.Debug')

    abSharpArgs = shlex.split(abSharpArgs)
    print(f"cmd-line args: {abSharpArgs}")

    abSharp = subprocess.run(
        ['dotnet', 'run', '--configuration', 'Release', dataSourcePath, '--columns', *columns, *abSharpArgs, *extraArgs],
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
    if featuresJob:
        outJson['features'] = featuresJob
    print("featuresJob:")
    pp.pprint(featuresJob)

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

def getTopFeatures(featuresJob, numFeatures):
    return featuresJob['features'][:numFeatures]

def getMlFeaturesByThreshold(featuresJob, featureThreshold):
    pp.pprint(featuresJob)
    k = featuresJob['k']
    if k == 0:
        print("SUCCESS: (skipped because not enough features)")
        quit()
    # We always include the top feature
    features = [featuresJob['features'][0]]
    topScore = featuresJob['cumulativeScore'][k-1]
    print(f"topScore {topScore} for k = {k}")
    if topScore == 0:
        print("SUCCESS: (skipped because zero score)")
        quit()
    for index in range(1,len(featuresJob['features'])):
        thisScore = featuresJob['cumulativeScore'][index]
        if abs(thisScore - topScore) > featureThreshold:
            features.append(featuresJob['features'][index])
        else:
            print(f"Index {index} with score {thisScore} under thresh")
            break
    return features

def getUniFeaturesByThreshold(featuresJob, featureThreshold):
    # We always include the top feature
    features = [featuresJob['features'][0]]
    topScore = featuresJob['scores'][0]
    if topScore == 0:
        # Don't expect this, but you never know
        return features
    for index in range(1,len(featuresJob['features'])):
        if (featuresJob['scores'][index]/topScore) > featureThreshold:
            features.append(featuresJob['features'][index])
        else:
            break
    return features

def transformClusterSpec(columns, clusterSpec):
    for colName, index in zip(clusterSpec['InitialCluster'], range(len(clusterSpec['InitialCluster']))):
        clusterSpec['InitialCluster'][index] = columns.index(colName)
    for derivedCluster in clusterSpec['DerivedClusters']:
        for colName, index in zip(derivedCluster['StitchColumns'], range(len(derivedCluster['StitchColumns']))):
            derivedCluster['StitchColumns'][index] = columns.index(colName)
        for colName, index in zip(derivedCluster['DerivedColumns'], range(len(derivedCluster['DerivedColumns']))):
            derivedCluster['DerivedColumns'][index] = columns.index(colName)


def makeClusterSpec(allColumns, featuresColumns, focusColumn, maxClusterSize, maxClusters, doPatches):
    '''
    {
  "InitialCluster": [4, 5, 7],
  "DerivedClusters": [
    { "StitchOwner": "Shared", "StitchColumns": [7], "DerivedColumns": [1, 3, 9]
    },
    { "StitchOwner": "Left", "StitchColumns": [9, 1], "DerivedColumns": [8, 0]
    },
    { "StitchOwner": "Right", "StitchColumns": [1, 9], "DerivedColumns": [2] },
    { "StitchOwner": "Shared", "StitchColumns": [7, 3], "DerivedColumns": [6] }
  ]
}
    '''
    numClusters = 1
    cSize = maxClusterSize-1
    # Note that if cSize > len(featuresColumns), then initialCluster is only the featuresColumns
    initialCluster = featuresColumns[:cSize]
    initialCluster += [focusColumn]
    usedColumns = initialCluster.copy()
    remainColumns = featuresColumns[cSize:]
    clusterSpec = {'InitialCluster':initialCluster, 'DerivedClusters':[]}
    # Add the clusters
    while len(remainColumns) > 0:
        if numClusters >= maxClusters:
            break
        derivedCols = remainColumns[:cSize]
        usedColumns += derivedCols
        clusterSpec['DerivedClusters'].append({'StitchColumns':[focusColumn],
                                               'DerivedColumns':derivedCols,
                                               'StitchOwner':'Shared'})

        remainColumns = remainColumns[cSize:]
        numClusters += 1
    if doPatches:
        # Add the patch columns
        for column in allColumns:
            if column in featuresColumns or column == focusColumn:
                continue
            clusterSpec['DerivedClusters'].append({'StitchColumns':[focusColumn],
                                                'DerivedColumns':[column],
                                                'StitchOwner':'Left'})
    print("Cluster information:")
    print(f"All columns: {allColumns}")
    print(f"Features columns: {featuresColumns}")
    print(f"Used columns: {usedColumns}")
    print(f"Target column: {focusColumn}")
    pp.pprint(clusterSpec)
    return clusterSpec, numClusters, usedColumns

def oneModel(dataDir='csvGeneral',
             dataSourceNum=None,
             csvFile=None,
             featuresFile=None,
             featuresType=None,
             featuresDir=None,
             model='fastMl',
             suffix='',
             synResults='synResults',
             synMeasures='synMeasures',
             abSharpArgs='',
             runsDir='runsAb',
             doMeasures=False,
             numFeatures=None,
             featureThreshold=None,
             maxClusterSize=3,
             maxClusters=1000,
             doPatches=True,
             force=False):

    ''' There are two ways to run oneModel without features (i.e. for ctGan or syndiffix):
            1. Specify the dataSourceNum
            2. Specify the csvFile
        Likewise, there are two ways to run oneModel with features:
            1. Specify the dataSourceNum, featuresType, and featuresDir
            2. Specify the featuresFile, featuresType, and featuresDir

        An example of the latter is:
            --featuresDir=featuresAb
            --featuresType=ml
            --featuresFile="ml.census.csv.tax filer stat.json"
        
        Build as many clusters of size maxClusterSize as we can until we either reach
        maxClusters or we have put all features (that pass featureThreshold) into clusters.
        If doPatches==False, then we remove all columns that are not in a cluster.
        Otherwise, we add the columns as patches.
    '''
    tu = testUtils.testUtilities()
    tu.registerCsvLib(dataDir)
    tu.registerSynResults(synResults)
    if len(abSharpArgs) > 0:
        print(f"abSharpArgs: {abSharpArgs}")
    focusColumn = None
    featuresJob = None
    if csvFile:
        if featuresType or dataSourceNum:
            print("ERROR: can't specify featuresType or dataSourceNum along with csvFile")
        sourceFileName = csvFile
        if ((featuresDir or featuresType or featuresFile) and
            (not featuresDir or not featuresType or not featuresFile)):
            print("ERROR: if any of featuresDir, featuresType, or featuresFile are specified, then all must be specified")
            quit()
    if dataSourceNum and not featuresType:
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
    if featuresType:
        tu.registerFeaturesDir(featuresDir)
        tu.registerFeaturesType(featuresType)
    if dataSourceNum and featuresType:
        featuresFiles = tu.getSortedFeaturesFiles()
        if dataSourceNum >= len(featuresFiles):
            print(f"ERROR: dataSourceNum too big {dataSourceNum}")
            quit()
        featuresFile = featuresFiles[dataSourceNum]
        if featuresFile[-5:] != '.json':
            print(f"ERROR: features file not json ({featuresFile})")
            quit()
    if featuresType:
        featuresPath = os.path.join(tu.featuresTypeDir, featuresFile)
        with open(featuresPath, 'r') as f:
            featuresJob = json.load(f)
        sourceFileName = featuresJob['csvFile']      #TODO
        focusColumn = featuresJob['targetColumn']
    print(f"Using source file {sourceFileName}")
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
    if not featuresType:
        outPath = os.path.join(modelsDir, f"{sourceFileName}.json")
    else:
        # We do this whether we have featuresType or not. If we do, then we expect
        # the model name to reflect the featureType...
        outPath = os.path.join(modelsDir, f"{sourceFileName}.{focusColumn}.json")
    if not force and os.path.exists(outPath):
        print(f"Result {outPath} already exists, skipping")
        print("oneModel:SUCCESS (skipped)")
        quit()
    print(f"Model {label} for dataset {dataSourcePath}, focus column {focusColumn}")

    df = readCsv(dataSourcePath)
    print(f"Training dataframe shape (before features) {df.shape}")
    colNames = list(df.columns.values)
    dfTest = readCsv(testDataPath)
    madeTempDataSource = False
    clusterSpec = None
    if featuresJob:
        origColNames = colNames.copy()
        print("Original columns")
        print(origColNames)
        if 'fixed' in featuresJob:
            print("Using fixed features")
            featuresColumns = featuresJob['features']
        elif 'kFeatures' in featuresJob:
            if featureThreshold:
                featuresColumns = getMlFeaturesByThreshold(featuresJob, featureThreshold)
            else:
                featuresColumns = featuresJob['kFeatures']
                print(f"There are {len(featuresJob['kFeatures'])} K features")
        else:
            if numFeatures:
                featuresColumns = getTopFeatures(featuresJob, numFeatures)
            if featureThreshold:
                featuresColumns = getUniFeaturesByThreshold(featuresJob, featureThreshold)
        featuresWithoutMax = len(featuresColumns)
        clusterSpec, numClusters, usedColumns = makeClusterSpec(origColNames, featuresColumns, focusColumn, maxClusterSize, maxClusters, doPatches)
        if not doPatches:
            for origCol in origColNames:
                if origCol not in usedColumns:
                    df.drop(origCol, axis=1, inplace=True)
                    dfTest.drop(origCol, axis=1, inplace=True)
            # Now we need to make a csv out of df to later give to abSharp
            print("columns in dfTest")
            print(dfTest.columns)
            print("columns in df")
            print(df.columns)
            colNames = list(df.columns.values)
            print("New columns")
            print(colNames)
            import uuid
            sourceFileName = sourceFileName + '.' + str(uuid.uuid4()) + '.csv'
            tempFilesDir = os.path.join(tu.baseDir, 'tempCsvFiles')
            os.makedirs(tempFilesDir, exist_ok=True)
            dataSourcePath = os.path.join(tempFilesDir, sourceFileName)
            madeTempDataSource = True
            df.to_csv(dataSourcePath, index=False, header=df.columns)
            print(dataSourcePath)
        featuresJob['params'] = {
             'doPatches':doPatches,
            'featureThreshold':featureThreshold,
            'usedFeatures':colNames,
            'featuresWithoutMax':featuresWithoutMax,
            'featureThreshold':featureThreshold,
            'maxClusterSize':maxClusterSize,
            'numClusters':numClusters,
        }
    print(list(dfTest.columns.values))
    # quick test to make sure that the test and train data match columns
    if colNames != list(dfTest.columns.values):
        print(f("ERROR: Train column names {colNames} don't match test column names {list(dfTest.columns.values)}"))
        quit()
    # Pull in training data as list, to be stored in results file as is
    testData = dfTest.values.tolist()
    print(f"Columns {colNames}")
    mls = testUtils.mlSupport(tu)
    metaData = makeMetadata(df)
    if model == 'abSharp' or 'syndiffix' in model or 'sdx_' in model:
        colTypeSymbols = {'text': 's', 'real': 'r', 'datetime': 't', 'int': 'i', 'boolean': 'b'}
        colTypes = tu.getColTypesFromDataframe(df)
        columns = []
        for colName, colType in zip(colNames, colTypes):
            if colType is None:
                # This means that we couldn't assign a type. I'm guessing that 'text' is a robust
                # type that absharp can handle regardless of what the value are...
                colType = 'text'
            columns.append(f"{colName}:{colTypeSymbols[colType]}")
        extraArgs = []
        if clusterSpec:
            transformClusterSpec(list(df.columns.values), clusterSpec)
            print("After transform:")
            pp.pprint(clusterSpec)
            clusterSpecJson = json.dumps(clusterSpec)
            extraArgs = ["--clusters", clusterSpecJson]
        elif featuresJob:
            extraArgs = ["--no-clustering"]
        print("Extra args:")
        pp.pprint(extraArgs)
        runAbSharp(tu, dataSourcePath, outPath, abSharpArgs, columns, focusColumn, testData, featuresJob, extraArgs=extraArgs)
        if madeTempDataSource:
            os.remove(dataSourcePath)
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

    baseFileName = sourceFileName[:-4]
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
