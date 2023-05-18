import os
import sys
import json
import time
import copy
import sdmetrics
import sdmetrics.single_table
import sdmetrics.reports.single_table
import sdmetrics.reports
import pandas as pd
import cufflinks as cf
import plotly.graph_objects as go
from anonymeter.evaluators import SinglingOutEvaluator
from anonymeter.evaluators import InferenceEvaluator
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import testUtils
import pprint
pp = pprint.PrettyPrinter(indent=4)

class sdmTools:
    def __init__(self, tu):
        self.maxEvalsPerType = 20
        self.maxTrainingSize = 20000
        self.synMethods = [
            'tvae',
            'gaussianCopula',
            'ctGan',
            'copulaGan',
            'syndiffix',
            'syndiffix_ns',
            'syndiffix_focus',
        ]
        self.mlConfig = [
            { 'type':'text', 'low':2, 'high':2,
                    'methods':  ['BinaryAdaBoostClassifier',
                                 'BinaryLogisticRegression',
                                 'BinaryMLPClassifier',
                                 ],
                    'methodType': 'binary',
            },
            { 'type':'text', 'low':3, 'high':20,
                    'methods':  ['MulticlassDecisionTreeClassifier',
                                 'MulticlassMLPClassifier',
                                 ],
                    'methodType': 'category',
            },
            { 'type':'integer', 'low':2, 'high':2,
                    'methods':  ['BinaryAdaBoostClassifier',
                                 'BinaryLogisticRegression',
                                 'BinaryMLPClassifier',
                                 ],
                    'methodType': 'binary',
            },
            { 'type':'integer', 'low':3, 'high':20,
                    'methods':  ['MulticlassDecisionTreeClassifier',
                                 'MulticlassMLPClassifier',
                                 ],
                    'methodType': 'category',
            },
            { 'type':'integer', 'low':50, 'high':1000000000000,
                    'methods':  ['LinearRegression',
                                 'MLPRegressor',
                                 ],
                    'methodType': 'numeric',
            },
            { 'type':'float', 'low':50, 'high':1000000000000,
                    'methods':  ['LinearRegression',
                                 'MLPRegressor',
                                 ],
                    'methodType': 'numeric',
            },
        ]
        self.exec = {
            'BinaryAdaBoostClassifier': sdmetrics.single_table.BinaryAdaBoostClassifier,
            'BinaryLogisticRegression': sdmetrics.single_table.BinaryLogisticRegression,
            'BinaryMLPClassifier': sdmetrics.single_table.BinaryMLPClassifier,
            'MulticlassDecisionTreeClassifier': sdmetrics.single_table.MulticlassDecisionTreeClassifier,
            'MulticlassMLPClassifier': sdmetrics.single_table.MulticlassMLPClassifier,
            'LinearRegression': sdmetrics.single_table.LinearRegression,
            'MLPRegressor': sdmetrics.single_table.MLPRegressor,
        }
        self.kwargs = {
            'BinaryAdaBoostClassifier': False,
            'BinaryLogisticRegression': False,
            'BinaryMLPClassifier': True,
            'MulticlassDecisionTreeClassifier': False,
            'MulticlassMLPClassifier': True,
            'LinearRegression': False,
            'MLPRegressor': True,
        }
        self.tu = tu

    def runPrivMeasureJob(self, privJob, force):
        print(f"runPrivMeasureJob: (force {force})")
        pp.pprint(privJob)
        self.privReport = None
        if 'half1' not in privJob['csvName']:
            print(f"ERROR: 'half1' not in filename {privJob}")
            quit()
        controlFileName = privJob['csvName'].replace('half1.csv', 'half2.csv')
        controlFilePath = os.path.join(self.tu.controlDir, controlFileName)
        resultsPath = os.path.join(self.tu.synResults, privJob['dirName'], privJob['fileName'])
        self.measuresDirPath = os.path.join(self.tu.synMeasures, privJob['dirName'])
        os.makedirs(self.measuresDirPath, exist_ok=True)
        privFileName = privJob['dirName'] + '.' + privJob['csvName'] + '.' + privJob['label'] + '.priv.json'
        self.privMeasuresPath = os.path.join(self.measuresDirPath, privFileName)
        if not force and os.path.exists(self.privMeasuresPath):
            print(f"Measures file already exists, skipping ({self.privMeasuresPath})")
            print("onePrivMeasure: SUCCESS (skipped)")
            quit()
        with open(resultsPath, 'r') as f:
            results = json.load(f)
        if 'originalTable' not in results:
            print(f"ERROR: Missing 'originalTable' in {resultsPath}")
            quit()
        self.dfOrig = pd.DataFrame(results['originalTable'], columns=results['colNames'])
        self.dfAnon = pd.DataFrame(results['anonTable'], columns=results['colNames'])
        self.dfControl = pd.read_csv(controlFilePath, low_memory=False, skipinitialspace=True)
        if privJob['task'] == 'singlingOut':
            if privJob['subtask'] == 'multivariate' and self.dfOrig.shape[1] <= 3:
                # Cannot to a multivariate attack with 3 or fewer columns
                print(f"Too few columns in table, exit")
                return
            evaluator = SinglingOutEvaluator(ori=self.dfOrig, 
                                            syn=self.dfAnon, 
                                            control=self.dfControl,
                                            n_attacks=privJob['numAttacks'])
            try:
                evaluator.evaluate(mode=privJob['subtask'])
            except RuntimeError as ex: 
                print(f"Singling out evaluation failed with {ex}. Please re-run this cell."
                    "For more stable results increase `n_attacks`. Note that this will "
                    "make the evaluation slower.")
                quit()
        elif privJob['task'] == 'inference':
            # evaluate crashes if the number of attacks (guesses) is more than the number of
            # possible values to guess
            numAttacks = min(privJob['numAttacks']+1, self.dfOrig.shape[0], self.dfAnon.shape[0], self.dfControl.shape[0]) - 1
            evaluator = InferenceEvaluator(ori=self.dfOrig, 
                                        syn=self.dfAnon, 
                                        control=self.dfControl,
                                        aux_cols=privJob['auxCols'],
                                        secret=privJob['secret'],
                                        n_attacks=numAttacks)
            evaluator.evaluate(n_jobs=-2)
        self.privReports = self.getPrivReport(evaluator)
        self.privReports['privJob'] = privJob
        self.privReports['outputPath'] = self.privMeasuresPath
        if privJob['task'] == 'singlingOut':
            self.privReports['queries'] = evaluator.queries()
        if self.privReports['privRisk'][0] > 0.3:
            print(f"NOTICE: privRisk = {self.privReports['privRisk']}")
        pp.pprint(self.privReports)
        with open(self.privMeasuresPath, 'w') as f:
            json.dump(self.privReports, f, indent=4)

    def getPrivReport(self, evaluator):
        evalRes = evaluator.results()
        print("Successs rate of main attack:", evalRes.attack_rate)
        print("Successs rate of baseline attack:", evalRes.baseline_rate)
        print("Successs rate of control attack:", evalRes.control_rate)
        privRisk = evalRes.risk()
        print(privRisk)
        return {
            'attack_rate':evalRes.attack_rate,
            'baseline_rate':evalRes.baseline_rate,
            'control_rate':evalRes.control_rate,
            'privRisk': privRisk,
        }

    def runQualityMeasureJob(self, resultsInfo, force):
        self.qualityReport = None
        self.resultsInfo = resultsInfo
        resultsPath = os.path.join(self.tu.synResults, self.resultsInfo['dirName'], self.resultsInfo['fileName'])
        self.qualDirPath = os.path.join(self.tu.synMeasures, self.resultsInfo['dirName'])
        os.makedirs(self.qualDirPath, exist_ok=True)
        qualFileName = self.resultsInfo['dirName'] + '.' + self.resultsInfo['fileName'][:-5] + '.qual.json'
        self.qualPath = os.path.join(self.qualDirPath, qualFileName)
        if not force and os.path.exists(self.qualPath):
            print(f"Measures file already exists, skipping ({self.qualPath})")
            return
        with open(resultsPath, 'r') as f:
            results = json.load(f)
        if 'originalTable' not in results:
            print(f"Missing 'originalTable' in {resultsPath}")
            return
        self.dfOrig = pd.DataFrame(results['originalTable'], columns=results['colNames'])
        self.dfAnon = pd.DataFrame(results['anonTable'], columns=results['colNames'])
        print(f"orig shape {self.dfOrig.shape}, anon shape {self.dfAnon.shape}")
        self.metadata = self._getMetadataFromCsvFile(self.resultsInfo['csvName'])
        print("Metadata:")
        pp.pprint(self.metadata)
        # Quality of text columns with lots of unique values should always be poor, so we
        # remove those from our data
        self._removeIdentifyingColumns()
        self.qualReports = self.getQualityReports()
        self.qualReports['metadata'] = self.metadata
        with open(self.qualPath, 'w') as f:
            json.dump(self.qualReports, f, indent=4)

    def _getPairsScore(self, col1, col2):
        pairs = self.qualReports['pairs']
        for i in range(len(pairs['Column 1'])):
            c1 = pairs['Column 1'][i]
            c2 = pairs['Column 2'][i]
            if (c1 == col1 and c2 == col2) or (c1 == col2 and c2 == col1):
                return pairs['Quality Score'][i]
        return None

    def _removeIdentifyingColumns(self):
        newMetadata = copy.deepcopy(self.metadata)
        for colName, colInfo in newMetadata['columns'].items():
            if colInfo['type'] == 'categorical' and self.dfOrig[colName].nunique() > 250:
                print(f"Remove column {colName} with {self.dfOrig[colName].nunique()} distinct values")
                self.dfOrig.drop(colName, axis=1, inplace=True)
                self.dfAnon.drop(colName, axis=1, inplace=True)
                del self.metadata['columns'][colName]

    def getQualityReports(self):
        self.qualityReport = sdmetrics.reports.single_table.QualityReport()
        self.qualityReport.generate(
            self.dfOrig, self.dfAnon, self.metadata)
        dfProperties = self.qualityReport.get_properties()
        fullReport = {'csvFile':self.resultsInfo['csvName'], 'synMethod':self.resultsInfo['dirName']}
        fullReport['overallScore'] = self.qualityReport.get_score()
        fullReport['properties'] = dfProperties.to_dict(orient='list')
        dfShapes = self.qualityReport.get_details(property_name='Column Shapes')
        fullReport['shapes'] = dfShapes.to_dict(orient='list')
        dfPairs = self.qualityReport.get_details(property_name='Column Pair Trends')
        fullReport['pairs'] = dfPairs.to_dict(orient='list')
        return fullReport

    def makeQualityVisuals(self, force):
        if self.qualityReport is None:
            return
        print("Make 1dim quality visual")
        name = self.resultsInfo['dirName'] + '.' + self.resultsInfo['csvName'] + '.1dim.png'
        figPath = os.path.join(self.qualDirPath, name)
        if force or not os.path.exists(figPath):
            fig = self.qualityReport.get_visualization(property_name='Column Shapes')
            try:
                fig.write_image(figPath)
            except:
                pass
        print("Make 2dim quality visual")
        name = self.resultsInfo['dirName'] + '.' + self.resultsInfo['csvName'] + '.2dim.png'
        figPath = os.path.join(self.qualDirPath, name)
        if force or not os.path.exists(figPath):
            fig = self.qualityReport.get_visualization(property_name='Column Pair Trends')
            try:
                fig.write_image(figPath)
            except:
                pass
        self.visual2dSubplots()

    def visual2dSubplots(self):
        baseColumns = list(self.metadata['columns'].keys())
        nDim = len(baseColumns)
        fields = self.metadata['columns']
        if nDim == 1 or nDim > 10:
            return
        specs = []
        figs = []
        titles = []
        for i in range(0, nDim - 1):
            columnSpecs = []
            for j in range(1, nDim):
                if i < j:
                    combs = [baseColumns[j], baseColumns[i]]
                    pairsScore = self._getPairsScore(baseColumns[j], baseColumns[i])
                    if pairsScore:
                        colLabel = f"{', '.join(combs)} score = {pairsScore:.3f}"
                    else:
                        colLabel = f"{', '.join(combs)}"
                    if fields[baseColumns[j]]['type'] == 'categorical' and fields[baseColumns[i]]['type'] == 'categorical':
                        # For two categorical cols, the fig from sdmetrics is a subplot itself.
                        # We need to decompose it into two separate traces and add separately as two
                        # cells in our final subplot
                        fig = sdmetrics.reports.utils.get_column_pair_plot(
                            real_data=self.dfOrig,
                            synthetic_data=self.dfAnon,
                            metadata=self.metadata,
                            column_names=combs,
                        )
                        fig.update_layout(showlegend=False)
                        origFig = fig
                        anonFig = copy.deepcopy(fig)
                        origFig['data'] = [origFig['data'][0]]
                        anonFig['data'] = [anonFig['data'][1]]
                        figs.append(origFig)
                        figs.append(anonFig)
                        titles.append('Original')
                        titles.append('Synthetic ' + colLabel)
                        columnSpecs.append({})
                        columnSpecs.append({})
                    else:
                        # In other cases there's just one trace, so we add it and span two columns
                        fig = sdmetrics.reports.utils.get_column_pair_plot(
                            real_data=self.dfOrig,
                            synthetic_data=self.dfAnon,
                            metadata=self.metadata,
                            column_names=combs,
                        )
                        fig.update_layout(showlegend=False)
                        figs.append(fig)
                        titles.append(colLabel)
                        columnSpecs.append({'colspan': 2})
                        columnSpecs.append(None)
                else:
                    # Skip duplicate (transposed pair) cells
                    columnSpecs.append(None)
                    columnSpecs.append(None)
            specs.append(columnSpecs)
            
        # nDim - 1 because we don't want to show the empty diagonal for (c0, c0) pairs
        # * 2 the subplot columns because the categorical column pairs are two cells each
        subplots = cf.subplots(figs, shape=(nDim - 1, (nDim - 1) * 2), specs=specs, subplot_titles=titles)
        subplotsFig = go.Figure(data=subplots['data'], layout=subplots['layout'])
        subplotsFig.update_layout(showlegend=False)

        print("Make 2dim scatter visual")
        name = self.resultsInfo['dirName'] + '.' + self.resultsInfo['csvName'] + '.2dimScatter.png'
        figPath = os.path.join(self.qualDirPath, name)

        # For some reason lots of dimensions cause the markers to disappear. This is a workaround
        scale = 1.0 if nDim <= 8 else (0.5 if nDim <= 16 else 0.25)

        try:
            subplotsFig.write_image(figPath, 
                                width=nDim*512, height=nDim*512,
                                scale=scale)
        except:
            pass

    def runSynMlJob(self, jobNum, force=False):
        mc = measuresConfig(self.tu)
        mlJobs = mc.getMlJobs()
        if jobNum >= len(mlJobs):
            print(f"runSynMlJob: my jobNum {jobNum} is too large")
            return
        myJob = mlJobs[jobNum]
        # Check if the job is already done
        measuresFile = myJob['csvFile'] + '.' + myJob['method'] + '.' + myJob['column'] + '.json'
        measuresDir = os.path.join(self.tu.synMeasures, myJob['synMethod'])
        os.makedirs(measuresDir, exist_ok=True)
        measuresPath = os.path.join(self.tu.synMeasures, myJob['synMethod'], measuresFile)
        if not force and os.path.exists(measuresPath):
            print(f"{measuresPath} exists, skipping")
            print("oneSynMLJob: SUCCESS (skipped)")
            return
        myJob['scoreOrig'] = myJob['score']
        myJob['elapsedOrig'] = myJob['elapsed']
        # Get the required datasets
        resultsPathNoFocus = os.path.join(self.tu.synResults, myJob['synMethod'], myJob['resultsFile'])
        resultsFileFocus = myJob['csvFile'] + '.' + myJob['column'] + '.json'
        resultsPathFocus = os.path.join(self.tu.synResults, resultsFileFocus)
        if os.path.exists(resultsPathNoFocus):
            print(f"Found no focus file {resultsPathNoFocus}")
            resultsPath = resultsPathNoFocus
            focusColumn = 'noFocusColumn'
        elif os.path.exists(resultsPathFocus):
            print(f"Found focus file {resultsPathFocus}")
            resultsPath = resultsPathFocus
            focusColumn = myJob['column']
        else:
            print(f"Neither {resultsPathNoFocus} nor {resultsPathFocus} found, can't continue")
            return
        with open(resultsPath, 'r') as f:
            results = json.load(f)
        if 'originalTable' not in results:
            print(f"Missing 'originalTable' in {resultsPath}")
            return
        dfOrig = pd.DataFrame(results['originalTable'], columns=results['colNames'])
        dfAnon = pd.DataFrame(results['anonTable'], columns=results['colNames'])
        startTime = time.time()
        print(f"runSynMlJob: Starting job {myJob} at time {startTime}")
        dfOrigTest, dfOrigTrain = self._getTestAndTrain(dfOrig)
        print(f"    dfOrigTest shape {dfOrigTest.shape}, dfOrigTrain shape {dfOrigTrain.shape}")
        dfAnonTest, dfAnonTrain = self._getTestAndTrain(dfAnon)
        print(f"    dfAnonTest shape {dfAnonTest.shape}, dfAnonTrain shape {dfAnonTrain.shape}")
        metadata = self._getMetadataFromCsvFile(myJob['csvFile'])
        print("Metadata:")
        pp.pprint(metadata)
        score = self._runOneMlMeasure(dfOrigTest, dfAnonTrain, metadata, myJob['column'], myJob['method'], myJob['csvFile'])
        if score is None:
            print("Scores is None, quitting")
            quit()
        endTime = time.time()
        print(f"Score = {score}")
        myJob['score'] = score
        myJob['elapsedSyn'] = endTime - startTime
        myJob['elapsed'] = results['elapsedTime']
        myJob['focusColumn'] = focusColumn
        print("Job Information")
        pp.pprint(myJob)
        with open(measuresPath, 'w') as f:
            json.dump(myJob, f, indent=4)
        print("oneSynMLJob: SUCCESS")

    def runOrigMlJob(self, jobNum):
        if jobNum >= len(self.origMlJobs):
            print(f"oneOrigMlJob: my jobNum {jobNum} is too large")
            return
        myJob = self.origMlJobs[jobNum]
        startTime = time.time()
        print(f"oneOrigMlJob: Starting job {myJob} at time {startTime}")
        df = self._readCsv(myJob['csvFile'])
        dfTest, dfTrain = self._getTestAndTrain(df)
        print(f"    dfTest shape {dfTest.shape}, dfTrain shape {dfTrain.shape}")
        metadata = self._getMetadataFromCsvFile(myJob['csvFile'])
        print("Metadata:")
        pp.pprint(metadata)
        score = self._runOneMlMeasure(dfTest, dfTrain, metadata, myJob['column'], myJob['method'], myJob['csvFile'])
        endTime = time.time()
        print(f"Score = {score}")
        myJob['score'] = score
        myJob['elapsed'] = endTime - startTime
        jsonStr = json.dumps(myJob)
        print(jsonStr)
        origMlJobPath = os.path.join(self.tu.synMeasures, 'OrigMlJobs')
        with open(origMlJobPath, 'a') as f:
            f.write(jsonStr+'\n')
        print("oneOrigMlJob: SUCCESS")

    def _runOneMlMeasure(self, dfTest, dfTrain, metadata, column, method, csvFile):
        exec = self.exec[method]
        kwargs = self.kwargs[method]
        score = None
        if kwargs:
            exec.MODEL_KWARGS = { 'max_iter': 500 }
        try:
            score = exec.compute(
                test_data=dfTest,
                train_data=dfTrain,
                target=column,
                metadata=metadata
        )
        except Exception as e:
            print(f"exception on {csvFile}, {column}, {method}")
            print(e)
            pp.pprint(metadata)
            a=1/0
            quit()
        return score

    def _getTestAndTrain(self, df):
        dfShuffled = df.sample(frac=1)
        trainSize = int(dfShuffled.shape[0]/2)
        if trainSize > self.maxTrainingSize:
            trainSize = self.maxTrainingSize
        testSize = dfShuffled.shape[0] - trainSize - 1
        dfTrain = dfShuffled.head(trainSize)
        dfTest = dfShuffled.head(-testSize)
        return dfTest, dfTrain

    def _readCsv(self, csvFile):
        csvPath = os.path.join(self.tu.csvLib, csvFile)
        return pd.read_csv(csvPath, low_memory=False, skipinitialspace=True)

    def enumerateSynMlJobs(self):
        self.synMlJobs = []
        mc = measuresConfig(self.tu)
        mc.initOrigMlJobs()

    def getMethodTypeFromMethod(self, method):
        for config in self.mlConfig:
            if method in config['methods']:
                return config['methodType']
        print(f"ERROR: getMethodTypeFromMethod: failed to classify method {method}")
        return None

    def enumerateOrigMlJobs(self):
        self.origMlJobs = []
        mc = measuresConfig(self.tu)
        for csvFile, mlClassInfo in mc.getCsvOrderInfo():
            limits = {
                'binary':0,
                'numeric':0,
                'category':0,
            }
            for colInfo in mlClassInfo['colInfo']:
                methodType, methods = self._getMethodsFromColInfo(colInfo)
                if methodType is None:
                    continue
                if limits[methodType] > self.maxEvalsPerType:
                    continue
                limits[methodType] += 1
                for method in methods:
                    self.origMlJobs.append({'csvFile':csvFile, 'column':colInfo['column'], 'method':method})

    def _getMethodsFromColInfo(self, colInfo):
        for mlInfo in self.mlConfig:
            if (colInfo['colType'] == mlInfo['type'] and
                colInfo['numDistinct'] >= mlInfo['low'] and
                colInfo['numDistinct'] <= mlInfo['high']):
                return mlInfo['methodType'], mlInfo['methods']
        return None, []

    def _getMetadataFromCsvFile(self, csvFile):
        # Find mlInfo
        mc = measuresConfig(self.tu)
        mlInfo = mc.getMlInfoFromCsvOrder(csvFile)
        metadata = {'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1', 'columns':{}}
        for colInfo in mlInfo['colInfo']:
            whatToDo = 'category'
            if colInfo['colType'] == 'float':
                whatToDo = 'numeric'
                subType = 'float'
            else:
                for mlInfo in self.mlConfig:
                    if (colInfo['colType'] == mlInfo['type'] and
                        colInfo['numDistinct'] >= mlInfo['low'] and
                        colInfo['numDistinct'] <= mlInfo['high']):
                        if mlInfo['methodType'] in ['binary', 'category']:
                            whatToDo = 'category'
                        else:
                            whatToDo = 'numeric'
                            subType = 'integer'
            if whatToDo == 'category':
                metadata['columns'][colInfo['column']] = {"type": "categorical",}
            else:
                metadata['columns'][colInfo['column']] = {"type": "numerical", "subtype": subType}
        return metadata

class measuresConfig:
    def __init__(self, tu):
        self.tu = tu
        # The ML score on the original data has to be above this threshold to generate
        # a corresponding measure on the synthetic data
        self.origMlScoreThreshold = 0.8
        self.goodMlJobs = None
        self.methods = None

    def getMlJobs(self):
        mlJobsOrderPath = os.path.join(self.tu.runsDir, 'mlJobs.json')
        with open(mlJobsOrderPath, 'r') as f:
            return json.load(f)

    def makeMlJobsBatchScript(self, csvLib, measuresDir, resultsDir, runsDir):
        batchScriptPath = os.path.join(self.tu.runsDir, "batchMl")
        batchScript = f'''#!/bin/sh
#SBATCH --time=7-0
#SBATCH --array=0-{len(self.mlJobsOrder)-1}
arrayNum="${{SLURM_ARRAY_TASK_ID}}"
python3 /INS/syndiffix/nobackup/internal-strategy/playground/adaptive-buckets/tests/oneSynMLJob.py \\
    --jobNum=$arrayNum \\
    --force=False \\
    --csvLib={csvLib} \\
    --resultsDir={resultsDir} \\
    --runsDir={runsDir} \\
    --measuresDir={measuresDir}
    '''
        with open(batchScriptPath, 'w') as f:
            f.write(batchScript)

    def _makeFocusJobsFile(self):
        mlJobs = self.getMlJobs()
        csvCol = {}
        for mlJob in mlJobs:
            if mlJob['csvFile'] not in csvCol:
                csvCol[mlJob['csvFile']] = {}
            csvCol[mlJob['csvFile']][mlJob['column']] = True
        self.focusJobs = []
        for csv, cols in csvCol.items():
            for col in cols.keys():
                self.focusJobs.append({'csvFile':csv, 'column':col})
        return self.focusJobs

    def getFocusJobs(self):
        focusJobsPath = os.path.join(self.tu.runsDir, 'focusBuildJobs.json')
        if os.path.exists(focusJobsPath):
            with open(focusJobsPath, 'r') as f:
                return json.load(f)
        print(f"ERROR: getFocusJobs: focusBuildJobs.json doesn't exist")
        return None

    def getFocusFromJobNumber(self, jobNum):
        focusJobs = self.getFocusJobs()
        if focusJobs is None:
            return None, None
        if jobNum >= len(focusJobs) - 1:
            print(f"ERROR: getFocusFromJobNumber: jobNum {jobNum} out of range")
            return None, None
        return focusJobs[jobNum]['csvFile'], focusJobs[jobNum]['column']

    def makeFocusRunsScripts(self, csvLib, measuresDir, resultsDir, runsDir):
        # Start by making a jobs file
        self._makeFocusJobsFile()
        focusJobsPath = os.path.join(self.tu.runsDir, 'focusBuildJobs.json')
        with open(focusJobsPath, 'w') as f:
            json.dump(self.focusJobs, f, indent=4)

        batchScriptPath = os.path.join(self.tu.runsDir, "batchFocus")
        batchScript = f'''#!/bin/sh
#SBATCH --time=7-0
#SBATCH --array=0-{len(self.focusJobs)-1}
arrayNum="${{SLURM_ARRAY_TASK_ID}}"
python3 /INS/syndiffix/nobackup/internal-strategy/playground/adaptive-buckets/tests/oneModel.py \\
    --dataSourceNum=$arrayNum \\
    --dataDir={csvLib} \\
    --synResults={resultsDir} \\
    --runsDir={runsDir} \\
    --synMeasures={measuresDir} \\
    --model=syndiffix_focus \\
    --withFocusColumn=True
    '''
        with open(batchScriptPath, 'w') as f:
            f.write(batchScript)

    def makeQualJobsBatchScript(self, measuresDir, resultsDir, numJobs, synMethod):
        batchScriptPath = os.path.join(self.tu.runsDir, "batchQual")
        if synMethod:
            batchScript = f'''#!/bin/sh
#SBATCH --time=7-0
#SBATCH --array=0-{numJobs}
arrayNum="${{SLURM_ARRAY_TASK_ID}}"
python3 /INS/syndiffix/nobackup/internal-strategy/playground/adaptive-buckets/tests/oneQualityMeasure.py \\
    --jobNum=$arrayNum \\
    --resultsDir={resultsDir} \\
    --synMethod={synMethod} \\
    --force=False \\
    --measuresDir={measuresDir}
            '''
        else:
            batchScript = f'''#!/bin/sh
#SBATCH --time=7-0
#SBATCH --array=0-{numJobs}
arrayNum="${{SLURM_ARRAY_TASK_ID}}"
python3 /INS/syndiffix/nobackup/internal-strategy/playground/adaptive-buckets/tests/oneQualityMeasure.py \\
    --jobNum=$arrayNum \\
    --resultsDir={resultsDir} \\
    --force=False \\
    --measuresDir={measuresDir}
            '''
        with open(batchScriptPath, 'w') as f:
            f.write(batchScript)

    def makePrivJobsBatchScript(self, runsDir, measuresDir, resultsDir, controlDir, numAttacks):
        batchScriptPath = os.path.join(self.tu.runsDir, "batchPriv")
        allResults = self.tu.getResultsPaths()
        privJobs = []
        jobNum = 0
        for result in allResults:
            print(result)
            columns = self.tu.getColumnsFromResult(result)
            privJob = copy.deepcopy(result)
            privJob['task'] = 'singlingOut'
            privJob['subtask'] = 'univariate'
            privJob['label'] = 'singUni'
            privJob['numAttacks'] = numAttacks
            privJob['jobNum'] = jobNum
            jobNum += 1
            privJobs.append(privJob)

            if len(columns) > 3:
                # Can only do multivariate attack with more than 3 columns
                privJob = copy.deepcopy(result)
                privJob['task'] = 'singlingOut'
                privJob['subtask'] = 'multivariate'
                privJob['label'] = 'singMulti'
                privJob['numAttacks'] = numAttacks
                privJob['jobNum'] = jobNum
                jobNum += 1
                privJobs.append(privJob)

            for column in columns:
                privJob = copy.deepcopy(result)
                privJob['task'] = 'inference'
                temp = column.split()
                privJob['label'] = f"inf.{''.join(temp)}"
                privJob['numAttacks'] = numAttacks
                privJob['auxCols'] = [col for col in columns if col != column]
                privJob['secret'] = column
                privJob['jobNum'] = jobNum
                jobNum += 1
                privJobs.append(privJob)
        privJobsPath = os.path.join(self.tu.runsDir, "privJobs.json")
        with open(privJobsPath, 'w') as f:
            json.dump(privJobs, f, indent=4)
        batchScript = f'''#!/bin/sh
#SBATCH --time=7-0
#SBATCH --array=0-{len(privJobs)-1}
arrayNum="${{SLURM_ARRAY_TASK_ID}}"
python3 /INS/syndiffix/nobackup/internal-strategy/playground/adaptive-buckets/tests/onePrivMeasure.py \\
    --jobNum=$arrayNum \\
    --runsDir={runsDir} \\
    --resultsDir={resultsDir} \\
    --measuresDir={measuresDir} \\
    --controlDir={controlDir} \\
    --force=False
    '''
        with open(batchScriptPath, 'w') as f:
            f.write(batchScript)

    def makeAndSaveMlJobsOrder(self, synMethod):
        self.initGoodMlJobs()
        self.findMlMethods()
        self.mlJobsOrder = []
        pp.pprint(f"Found synMethods {self.methods}")
        if synMethod:
            print(f"Limiting methods to {synMethod}")
        for method in self.methods:
            if synMethod is not None and method != synMethod:
                print(f"   Skip {method}")
                continue
            methodResultsDir = os.path.join(self.tu.synResults, method)
            resultsFileNames = [f for f in os.listdir(methodResultsDir) if os.path.isfile(os.path.join(methodResultsDir, f))]
            focusColumnsPath = os.path.join(self.tu.runsDir, 'focusColumns.json')
            if os.path.exists(focusColumnsPath):
                print(f"Read in {focusColumnsPath}")
                with open(focusColumnsPath, 'r') as f:
                    focusColumns = json.load(f)
            else:
                focusColumns = {}
            for fileName in resultsFileNames:
                if fileName[-5:] != '.json':
                    continue
                if fileName[-9:] == '.csv.json':
                    focusColumn = None
                else:
                    if fileName in focusColumns:
                        print(f"Get {fileName} from focusColumns")
                        focusColumn = focusColumns[fileName]
                    else:
                        filePath = os.path.join(methodResultsDir, fileName)
                        print(f"    Load {filePath}")
                        with open(filePath, 'r') as f:
                            results = json.load(f)
                        if 'focusColumn' not in results:
                            print(f"ERROR: Could not find focusColumn in {filePath}")
                            quit()
                        focusColumn = results['focusColumn']
                        print(f"    Add {focusColumn} to {fileName}")
                        focusColumns[fileName] = focusColumn
                        with open(focusColumnsPath, 'w') as f:
                            json.dump(focusColumns, f, indent=4)
                        json.dump
                csvPos = fileName.find('.csv')
                dataSourceName = fileName[:csvPos+4]
                if dataSourceName not in self.goodMlJobs:
                    # Can happen with for instance 2dim tables
                    continue
                for job in self.goodMlJobs[dataSourceName]:
                    if focusColumn is not None and job['column'] != focusColumn:
                        continue
                    self.mlJobsOrder.append({**job,
                                             **{'resultsFile':fileName,
                                                'synMethod': method,
                                                }})
        mlJobsOrderPath = os.path.join(self.tu.runsDir, 'mlJobs.json')
        with open(mlJobsOrderPath, 'w') as f:
            json.dump(self.mlJobsOrder, f, indent=4)

    def findMlMethods(self):
        ''' Here we assume that there is one directory per method in the results
        directory
        '''
        self.methods = [ f.name for f in os.scandir(self.tu.synResults) if f.is_dir() ]

    def initGoodMlJobs(self):
        ''' This computes the ML measure jobs that should be run on each datasource
        '''
        origMlJobsPath = os.path.join(self.tu.synMeasures, 'OrigMlJobs')
        self.goodMlJobs = {}
        with open(origMlJobsPath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                try:
                    job = json.loads(line)
                except:
                    # Not sure why this happens. Could be some race conditions in the file
                    # append operations, though I didn't think that was supposed to happen
                    # Anyway, it is relatively rare...  TODO: fix (low priority)
                    print(f"Bad line: {line}")
                    continue
                if job['score'] is None or job['score'] < self.origMlScoreThreshold:
                    continue
                if job['csvFile'] in self.goodMlJobs:
                    self.goodMlJobs[job['csvFile']].append(job)
                else:
                    self.goodMlJobs[job['csvFile']] = [job]

    def getCsvOrderInfo(self):
        csvOrderPath = os.path.join(self.tu.synMeasures, 'csvOrder.json')
        with open(csvOrderPath, 'r') as f:
            csvOrderInfo = json.load(f)
        for fileInfo in csvOrderInfo:
            csvFile = fileInfo[0]
            mlClassInfo = fileInfo[1]
            yield csvFile, mlClassInfo

    def getMlInfoFromCsvOrder(self, csvFile):
        csvOrderPath = os.path.join(self.tu.synMeasures, 'csvOrder.json')
        with open(csvOrderPath, 'r') as f:
            csvOrderInfo = json.load(f)
            for fileInfo in csvOrderInfo:
                if fileInfo[0] == csvFile:
                    return fileInfo[1]
        return None

    def makeCsvOrder(self):
        csvOrderPath = os.path.join(self.tu.synMeasures, 'csvOrder.json')
        if os.path.exists(csvOrderPath):
            print(f"Read in {csvOrderPath}")
            with open(csvOrderPath, 'r') as f:
                csvOrderInfo = json.load(f)
        else:
            print(f"Initialize csvOrderInfo")
            csvOrderInfo = []
        def filesOrder(): 
            return [x[0] for x in csvOrderInfo]
        def colsOrder(): 
            return [x[1] for x in csvOrderInfo]
        csvFiles = self.tu.getDataSources()
        mls = testUtils.mlSupport(self.tu)
        for csvFile in csvFiles:
            if csvFile in filesOrder():
                print(f"Datasource {csvFile} already recorded, skipping")
                continue
            # Append csv file information
            csvPath = os.path.join(self.tu.csvLib, csvFile)
            df = pd.read_csv(csvPath, low_memory=False, skipinitialspace=True)
            mlClassInfo = mls.makeMlClassInfo(df, csvFile)
            csvOrderInfo.append([csvFile,mlClassInfo])
            print(f"Adding datasource {csvFile}")
        with open(csvOrderPath, 'w') as f:
            json.dump(csvOrderInfo, f, indent=4)

if __name__ == '__main__':
    # Just for testing
    tu = testUtils.testUtilities()
    tu.registerCsvLib('csvAb')
    tu.registerSynMeasure('measuresAb')

    sdmt = sdmTools(tu)
    sdmt.enumerateOrigMlJobs()
    #pp.pprint(mlJobs)
    print(f"Total {len(sdmt.origMlJobs)} jobs")