import pandas as pd
import sys
import os
import json


class mlSupport:
    def __init__(self, tu):
        self.tu = tu
        pass

    def getColumnsFromClassInfo(self, mlClassInfo):
        columns = []
        for cols in mlClassInfo.values():
            for column in cols:
                columns.append(column)
        return columns

    def makeMlClassInfo(self, df, dataSource, columns=None):
        # First check to see if there is a metadata file for this dataSource
        mlClassInfo = None
        metadataDir = self.tu.sdvMetaFiles
        if dataSource:
            if '.csv' in dataSource:
                dataSource = dataSource[:-4]
            if os.path.exists(metadataDir):
                metaFiles = [f for f in os.listdir(
                    metadataDir) if os.path.isfile(os.path.join(metadataDir, f))]
                for metaFile in metaFiles:
                    if dataSource not in metaFile:
                        continue
                    if metaFile[-5:] != '.json':
                        continue
                    metadataPath = os.path.join(metadataDir, metaFile)
                    f = open(metadataPath, 'r')
                    metadata = json.load(f)
                    f.close()
                    if 'classifications' in metadata:
                        return metadata['classifications']
        # If we get here, we couldn't find a metafile for the data source,
        # so we derive one as best we can
        if columns is None:
            columns = list(df.columns)
        mlClassInfo = {'binary': [], 'numeric': [], 'categorical': [], 'colInfo': []}
        colTypes = self.getColTypes(df)
        if len(colTypes) == len(columns) + 1 and colTypes[0] == 'integer':
            # The first colType is for an index column, which we want to ignore
            colTypes.pop(0)
        for i in range(len(colTypes)):
            column = columns[i]
            colType = colTypes[i]
            numUnique = df[column].nunique()
            mlClassInfo['colInfo'].append({'column': column, 'colType': colType, 'numDistinct': numUnique})
            if numUnique == 2 and len(mlClassInfo['binary']) == 0:
                mlClassInfo['binary'].append(column)
            elif numUnique > 2 and ((colType == 'text' and numUnique < 20) or numUnique < 10) and len(mlClassInfo['categorical']) == 0:
                mlClassInfo['categorical'].append(column)
            elif colType in ['float', 'integer'] and numUnique >= 10 and len(mlClassInfo['numeric']) == 0:
                mlClassInfo['numeric'].append(column)
        return mlClassInfo

    def getColTypes(self, df):
        colTypes = []
        for colType in df.dtypes:
            if pd.api.types.is_bool_dtype(colType):
                colTypes.append('boolean')
            elif pd.api.types.is_integer_dtype(colType):
                colTypes.append('integer')
            elif pd.api.types.is_float_dtype(colType) or pd.api.types.is_numeric_dtype(colType):
                colTypes.append('float')
            elif pd.api.types.is_string_dtype(colType) or pd.api.types.is_object_dtype(colType):
                colTypes.append('text')
            else:
                print(f"ERROR: Unknown column data type {colType}")
                sys.exit()
        return colTypes

    def makeMetadata(self, rows, colNames=None):
        ''' This makes the metadata file expected by sdmetrics.
        Note that the metadata produced assumes that there is a column named 'aid',
        and accordingly sets that column to the the `primary_key`. This code still
        works if there is no such column, but the caller needs to understand that the
        `primary_key` designation needs to be ignored in that case.

        Note this is the old metadata version.
        '''
        if type(rows) == list:
            df = pd.DataFrame(rows, columns=colNames)
        else:
            df = rows
        cols = list(df.columns)
        colTypes = self.getColTypes(df)
        metadata = {'primary_key': 'aid',
                    'fields': {'aid': {'subtype': 'integer', 'type': 'id'}, },
                    }
        for colName, colType in zip(cols, colTypes):
            if colName == 'aid':
                continue
            if colType in ['integer', 'float']:
                metadata['fields'][colName] = {
                    'type': 'numerical', 'subtype': colType}
            elif colType == 'text':
                metadata['fields'][colName] = {
                    'type': 'categorical'}
        return metadata


class testUtilities:
    def __init__(self):
        self.baseDir = os.environ['AB_RESULTS_DIR']
        self.pythonDir = os.environ['AB_PYTHON_DIR']
        self.abSharpDir = os.environ['AB_SHARP_DIR']
        # These are the default directories and config file locations
        self.expDir = os.path.join(self.baseDir, 'exp_base')
        self.setDirs()

    def registerExpDir(self, expDir):
        self.expDir = os.path.join(self.baseDir, expDir)
        self.setDirs()

    def setDirs(self):
        self.csvLib = os.path.join(self.expDir, 'csv', 'train')
        self.csvLibTest = os.path.join(self.expDir, 'csv', 'test')
        self.synResults = os.path.join(self.expDir, 'results')
        os.makedirs(self.synResults, exist_ok=True)
        self.synMeasures = os.path.join(self.expDir, 'measures')
        os.makedirs(self.synMeasures, exist_ok=True)
        self.tempSynMeasures = os.path.join(self.expDir, 'measures_samples')
        os.makedirs(self.tempSynMeasures, exist_ok=True)
        self.origMlDir = os.path.join(self.expDir, 'origMl')
        os.makedirs(self.origMlDir, exist_ok=True)
        self.tempOrigMlDir = os.path.join(self.expDir, 'origMl_samples')
        os.makedirs(self.tempOrigMlDir, exist_ok=True)
        self.runsDir = os.path.join(self.expDir, 'runs')
        os.makedirs(self.runsDir, exist_ok=True)
        self.synthpopBuildsDir = os.path.join(self.expDir, 'synthpop_builds')
        os.makedirs(self.synthpopBuildsDir, exist_ok=True)
        self.synthpopScriptsDir = os.path.join(self.runsDir, 'synthpop_jobs')
        os.makedirs(self.synthpopScriptsDir, exist_ok=True)
        self.featuresDir = os.path.join(self.expDir, 'features')
        os.makedirs(self.featuresDir, exist_ok=True)
        self.featuresTypeDir = os.path.join(self.featuresDir, 'univariate')
        self.summariesDirCore = os.path.join(self.expDir, 'summaries')
        os.makedirs(self.summariesDirCore, exist_ok=True)
        self.summariesDir = os.path.join(self.expDir, 'summaries')
        os.makedirs(self.summariesDir, exist_ok=True)
        self.sdvMetaFiles = os.path.join(self.expDir, 'sdvMetaFiles')
        self.tableBuildMetadataPath = os.path.join(self.expDir, 'tables.json')
        self.tableBuildMetadata = None
        self.configFilesDir = self.expDir

    def getColTypesFromDataframe(self, df):
        colTypes = []
        from pandas.errors import ParserError
        for c in df.columns[df.dtypes == 'object']:  # don't cnvt num
            try:
                df[c] = pd.to_datetime(df[c])
            except (ParserError, ValueError):  # Can't cnvrt some
                pass  # ...so leave whole column as-is unconverted
        for colType in df.dtypes:
            if pd.api.types.is_bool_dtype(colType):
                colTypes.append('boolean')
            elif pd.api.types.is_integer_dtype(colType):
                colTypes.append('int')
            elif pd.api.types.is_float_dtype(colType) or pd.api.types.is_numeric_dtype(colType):
                colTypes.append('real')
            elif pd.api.types.is_datetime64_any_dtype(colType):
                colTypes.append('datetime')
            elif pd.api.types.is_string_dtype(colType) or pd.api.types.is_object_dtype(colType):
                colTypes.append('text')
            else:
                colTypes.append(None)
        return colTypes

    def getSynOutputFileInfo(self, sourceFileName, colName=None):
        if colName:
            fileName = sourceFileName + '.' + colName + '.' + self.synTestOutputDirName + '.json'
        else:
            fileName = sourceFileName + '.' + self.synTestOutputDirName + '.json'
        filePath = os.path.join(self.synTestOutputDirPath, fileName)
        fileExists = os.path.exists(filePath)
        return fileName, filePath, fileExists

    def getDataFromMeasuresFile(self, inPath):
        data = {}
        explain = {}
        if inPath[-5:] != '.json':
            return None
        with open(inPath, 'r') as f:
            testRes = json.load(f)
        if 'abInfo' not in testRes:
            return None
        abInfo = testRes['abInfo']
        self.updateParams(data, explain, abInfo['params'])
        data['e_tim'] = abInfo['elapsedTime']
        return data

    def updateParams(self, data, explain, params):
        data['for_p'] = params['forest']['pName']
        explain['for_p'] = 'Forest parameters grouping'
        data['for_v'] = params['forest']['version']
        explain['for_v'] = 'Forest algorithm version'
        if params['clustered'] is True:
            data['clu_p'] = params['cluster']['pName']
            data['clu_v'] = params['cluster']['version'] if 'version' in params['cluster'] else 'v1'
            pass
        else:
            data['clu_p'] = 'none'
            data['clu_v'] = 'none'
        explain['clu_p'] = 'Cluster parameters grouping ("no" if no clustering)'
        explain['clu_v'] = 'Cluster algorithm version'
        data['har_v'] = params['harvest']['version']
        explain['har_v'] = 'Harvest algorithm version'
        data['mcr_v'] = params['microdata']['version']
        explain['mcr_v'] = 'Microdata version'
        if 'fileName' in params:
            fileName = params['fileName']
        if 'outputFile' in params:
            fileName = params['outputFile']
        data['d_src'], data['p_str'] = fileName.split('.csv.')
        data['p_str'] = data['p_str'][:-5]
        explain['d_src'] = 'Data source'
        explain['p_str'] = 'String with all AB build information'
        # TODO: uncomment
        # data['cols'] = params['columns']
        # explain['cols'] = 'Column names (in indexed order)'

    def validTestDir(self, dirName):
        return True
        if dirName[2:7] == '.for_' or dirName.startswith(('copulaGan', 'ctGan', 'fastMl', 'tvae', 'gaussianCopula', 'syndiffix', 'mostly', 'synthpop')):
            return True
        return False

    def getSynTestDirs(self):
        return self.getDirs(self.synResults)

    def getSynMeasuresDirs(self):
        return self.getDirs(self.synMeasures)

    def getDirs(self, synDir):
        inDirPaths = []
        inDirs = []
        allObjects = [f for f in os.listdir(synDir)]
        for thing in allObjects:
            thingPath = os.path.join(synDir, thing)
            if not os.path.isdir(thingPath):
                continue
            if not self.validTestDir(thing):
                continue
            inDirs.append(thing)
            inDirPaths.append(thingPath)
        return inDirPaths, inDirs

    def setSynTestOutputDir(self, params):
        import harvest
        import microdata
        import forest
        import clusters
        cs = clusters.clusterSchedule()
        clThing = f"{params['cluster']['pName']}_{cs.version}"
        baf = forest.buildAbForest(None, getVersionOnly=True)
        bh = harvest.bucketHarvest(None, getVersionOnly=True)
        harvestVersion = bh.version
        md = microdata.microdata(None, None, getVersionOnly=True)
        self.synTestOutputDirName = f"py.for_{params['forest']['pName']}_{baf.version}.har_{harvestVersion}.cl_{clThing}.md_{md.version}"
        self.synTestOutputDirPath = os.path.join(self.synResults, self.synTestOutputDirName)
        if not os.path.exists(self.synTestOutputDirPath):
            os.makedirs(self.synTestOutputDirPath, exist_ok=True)

    def getAbDefaultParams(self):
        defaults = {
            'forest': {'sing': 10, 'range': 50, 'lcf': 5, 'dependence_old': 10,
                       'threshSd': 1.0, 'noiseSd': 1.0, 'lcdBound': 2,
                       'pName': 'def',
                       },
            'cluster': {'maxStitchCluster': 1,
                        'maxClusterSize': 3, 'clusterQualityThreshold': 0.2,
                        'patchThreshold': 0.05,
                        'clusterMinColumnExponent': 1.5,
                        'clusterNumColumnsExponent': 2,
                        'singThreshHigh': 0.95, 'singThreshLow': 0.5,
                        'compositePenalty': 0.5,
                        'sampleSize': None, 'pName': 'def',
                        },
        }
        return defaults

    def getDataSources(self, sourceHalf='train'):
        if sourceHalf == 'train':
            csvDir = self.csvLib
        else:
            csvDir = self.csvLibTest
        files = [f for f in os.listdir(csvDir) if os.path.isfile(os.path.join(csvDir, f))]
        return sorted(files)

    def getSynthpopScripts(self):
        files = [f for f in os.listdir(self.synthpopScriptsDir) if os.path.isfile(os.path.join(self.synthpopScriptsDir, f))]
        return sorted(files)

    def getOrigMlFiles(self):
        files = [f for f in os.listdir(self.origMlDir) if os.path.isfile(os.path.join(self.origMlDir, f))]
        return sorted(files)

    def getSortedFeaturesFiles(self):
        files = [f for f in os.listdir(self.featuresTypeDir) if os.path.isfile(os.path.join(self.featuresTypeDir, f))]
        return sorted(files)

    def getResultsPaths(self, synMethod=None):
        allResults = []
        dirs = [f for f in os.listdir(self.synResults) if os.path.isdir(os.path.join(self.synResults, f))]
        for dirName in dirs:
            if synMethod is not None and dirName != synMethod:
                print(f"    skipping {dirName}")
                continue
            dirPath = os.path.join(self.synResults, dirName)
            files = [f for f in os.listdir(dirPath) if os.path.isfile(os.path.join(dirPath, f))]
            for fileName in files:
                if fileName[-5:] != '.json':
                    continue
                # Focus columns have extra stuff in the file name after '.csv', so remove:
                csvPos = fileName.find('.csv')
                csvName = fileName[:csvPos + 4]
                allResults.append({'dirName': dirName, 'fileName': fileName, 'csvName': csvName})
        return allResults

    def getColumnsFromResult(self, result):
        resultPath = os.path.join(self.synResults, result['dirName'], result['fileName'])
        with open(resultPath, 'r') as f:
            resultData = json.load(f)
        return resultData['colNames']

    def registerTableBuildMetadata(self, fileName):
        self.tableBuildMetadataPath = os.path.join(self.baseDir, fileName)

    def readTableBuildMetadata(self):
        with open(self.tableBuildMetadataPath, 'r') as f:
            self.tableBuildMetadata = json.load(f)

    def getOracleClusters(self, table):
        if self.tableBuildMetadata is None:
            self.readTableBuildMetadata()
        clusters = []
        if table in self.tableBuildMetadata:
            for cluster in self.tableBuildMetadata[table]['corrShape']:
                clusters.append(list(range(cluster['beginIdx'], cluster['endIdx'])))
        return clusters

    def registerConfigFilesDir(self, dirName):
        self.configFilesDir = os.path.join(self.baseDir, dirName)

    def registerFeaturesType(self, name):
        self.featuresTypeDir = os.path.join(self.featuresDir, name)
        os.makedirs(self.featuresTypeDir, exist_ok=True)

    def synConfigFileName(self, cnt):
        return f"syn_{cnt:05d}.json"

    def measuresConfigFileName(self, cnt):
        return f"measure_{cnt:05d}.json"
