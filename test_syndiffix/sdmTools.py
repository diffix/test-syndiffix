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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import testUtils
import pprint
from misc.csvUtils import readCsv
pp = pprint.PrettyPrinter(indent=4)

class sdmTools:
    def __init__(self, tu):
        self.maxEvalsPerType = 20
        self.maxTrainingSize = 20000
        self.mlMeasureRepeats = 20
        self.synMethods = [
            'tvae',
            'gaussianCopula',
            'ctGan',
            'copulaGan',
            'syndiffix',
            'syndiffix_ns',
            'syndiffix_focus',
            'syndiffix_multi',
            'syndiffix_multifocus',
        ]
        self.mlConfig = [
            {'type': 'text', 'low': 2, 'high': 2,
             'methods': ['BinaryAdaBoostClassifier',
                         'BinaryLogisticRegression',
                         'BinaryMLPClassifier',
                         ],
             'methodType': 'binary',
             },
            {'type': 'text', 'low': 3, 'high': 20,
             'methods': ['MulticlassDecisionTreeClassifier',
                         'MulticlassMLPClassifier',
                         ],
             'methodType': 'category',
             },
            {'type': 'integer', 'low': 2, 'high': 2,
             'methods': ['BinaryAdaBoostClassifier',
                         'BinaryLogisticRegression',
                         'BinaryMLPClassifier',
                         ],
             'methodType': 'binary',
             },
            {'type': 'integer', 'low': 3, 'high': 20,
             'methods': ['MulticlassDecisionTreeClassifier',
                         'MulticlassMLPClassifier',
                         ],
             'methodType': 'category',
             },
            {'type': 'integer', 'low': 50, 'high': 1000000000000,
             'methods': ['LinearRegression',
                         'MLPRegressor',
                         ],
             'methodType': 'numeric',
             },
            {'type': 'float', 'low': 50, 'high': 1000000000000,
             'methods': ['LinearRegression',
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
        self.dfControl = pd.DataFrame(results['testTable'], columns=results['colNames'])
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
            numAttacks = min(privJob['numAttacks'] + 1, self.dfOrig.shape[0],
                             self.dfAnon.shape[0], self.dfControl.shape[0]) - 1
            origFileName = privJob['csvName'].replace('half1.', '')
            mls = testUtils.mlSupport(self.tu)
            mlClassInfo = mls.makeMlClassInfo(self.dfControl, None)
            metadata = self._getMetadataFromMlInfo(mlClassInfo)
            if metadata['columns'][privJob['secret']]['type'] == 'numerical':
                regression = True
            else:
                regression = False
            if privJob['secret'] == 'native-country':
                print("Got native-country!")
                if 'pob' in privJob['auxCols']:
                    privJob['auxCols'].remove('pob')
            if privJob['secret'] == 'sf_flag':
                print("Got sf_flag!")
                print(privJob['auxCols'])
                if 'vendor_id' in privJob['auxCols']:
                    privJob['auxCols'].remove('vendor_id')
                print(privJob['auxCols'])
            if privJob['secret'] == 'vendor_id':
                print("Got vendor_id!")
                print(privJob['auxCols'])
                if 'sf_flag' in privJob['auxCols']:
                    privJob['auxCols'].remove('sf_flag')
                print(privJob['auxCols'])
            evaluator = InferenceEvaluator(ori=self.dfOrig,
                                           syn=self.dfAnon,
                                           control=self.dfControl,
                                           aux_cols=privJob['auxCols'],
                                           secret=privJob['secret'],
                                           regression=regression,
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
            'attack_rate': evalRes.attack_rate,
            'baseline_rate': evalRes.baseline_rate,
            'control_rate': evalRes.control_rate,
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
        mls = testUtils.mlSupport(self.tu)
        mlClassInfo = mls.makeMlClassInfo(self.dfOrig, None)
        self.metadata = self._getMetadataFromMlInfo(mlClassInfo)
        print("Metadata:")
        pp.pprint(self.metadata)
        # Quality of text columns with lots of unique values should always be poor, so we
        # remove those from our data
        self._removeIdentifyingColumns()
        self.qualReports = self.getQualityReports()
        self.qualReports['metadata'] = self.metadata
        self.qualReports['elapsedTime'] = results['elapsedTime']
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
            if colInfo['sdtype'] == 'categorical' and self.dfOrig[colName].nunique() > 250:
                print(f"Remove column {colName} with {self.dfOrig[colName].nunique()} distinct values")
                self.dfOrig.drop(colName, axis=1, inplace=True)
                self.dfAnon.drop(colName, axis=1, inplace=True)
                del self.metadata['columns'][colName]

    def getQualityReports(self):
        self.qualityReport = sdmetrics.reports.single_table.QualityReport()
        self.qualityReport.generate(
            self.dfOrig, self.dfAnon, self.metadata)
        dfProperties = self.qualityReport.get_properties()
        fullReport = {'csvFile': self.resultsInfo['csvName'], 'synMethod': self.resultsInfo['dirName']}
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
                    if fields[baseColumns[j]]['sdtype'] == 'categorical' and fields[baseColumns[i]]['sdtype'] == 'categorical':
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
                                    width=nDim * 512, height=nDim * 512,
                                    scale=scale)
        except:
            pass

    def runSynMlJob(self, jobNum, sampleNum, force=False):
        mc = measuresConfig(self.tu)
        mlJobs = mc.getMlJobs()
        if jobNum >= len(mlJobs):
            print(f"runSynMlJob: my jobNum {jobNum} is too large")
            return
        myJob = mlJobs[jobNum]
        # Check if the job is already done
        measuresFile = myJob['csvFile'] + '.' + myJob['method'] + '.' + myJob['column'].replace(' ','') + '.part_' + str(sampleNum) + '.ml.json'
        measuresDir = os.path.join(self.tu.tempSynMeasures, myJob['synMethod'])
        os.makedirs(measuresDir, exist_ok=True)
        measuresPath = os.path.join(measuresDir, measuresFile)
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
        dfAnon = pd.DataFrame(results['anonTable'], columns=results['colNames'])
        dfTest = pd.DataFrame(results['testTable'], columns=results['colNames'])
        print(f"    dfTest shape {dfTest.shape}, dfAnon (train) shape {dfAnon.shape}")
        mls = testUtils.mlSupport(self.tu)
        mlClassInfo = mls.makeMlClassInfo(dfTest, None)
        metadata = self._getMetadataFromMlInfo(mlClassInfo)
        startTime = time.time()
        print(f"runSynMlJob: Starting job {myJob} at time {startTime}")
        score = self._runOneMlMeasure(dfTest, dfAnon, metadata,
                                    myJob['column'], myJob['method'], myJob['csvFile'])
        if score is None:
            print("Scores is None, quitting")
            quit()
        endTime = time.time()
        print(f"Score = {score}")
        myJob['score'] = score
        myJob['elapsedSyn'] = endTime - startTime
        myJob['elapsed'] = results['elapsedTime']
        myJob['focusColumn'] = focusColumn
        myJob['sampleNum'] = sampleNum
        if 'features' in results:
            myJob['features'] = results['features']
        print("Job Information")
        pp.pprint(myJob)
        with open(measuresPath, 'w') as f:
            json.dump(myJob, f, indent=4)
        print("oneSynMLJob: SUCCESS")

    def runOrigMlJob(self, jobNum, sampleNum, force):
        if jobNum >= len(self.tempOrigMlJobs):
            print(f"oneOrigMlJob: my jobNum {jobNum} is too large")
            return
        myJob = self.tempOrigMlJobs[jobNum]
        tempOrigMlJobName = f"{myJob['csvFile']}.{myJob['column'].replace(' ','')}.{myJob['method']}.part_{sampleNum}.json"
        tempOrigMlJobPath = os.path.join(self.tu.tempOrigMlDir, tempOrigMlJobName)
        if not force and os.path.exists(tempOrigMlJobPath):
            print(f"{tempOrigMlJobPath} exists, skipping")
            print("oneSynMLJob: SUCCESS (skipped)")
            return
        csvPath = os.path.join(self.tu.csvLib, myJob['csvFile'])
        dfTrain = readCsv(csvPath)
        csvPath = os.path.join(self.tu.csvLibTest, myJob['csvFile'])
        dfTest = readCsv(csvPath)
        print(f"    dfTest shape {dfTest.shape}, dfTrain shape {dfTrain.shape}")
        mls = testUtils.mlSupport(self.tu)
        mlClassInfo = mls.makeMlClassInfo(dfTest, None)
        metadata = self._getMetadataFromMlInfo(mlClassInfo)
        print("Metadata:")
        pp.pprint(metadata)
        startTime = time.time()
        print(f"oneOrigMlJob: Starting job {myJob} at time {startTime}")
        score = self._runOneMlMeasure(dfTest, dfTrain, metadata, myJob['column'], myJob['method'], myJob['csvFile'])
        endTime = time.time()
        print(f"Score = {score}")
        myJob['score'] = score
        myJob['elapsed'] = endTime - startTime
        myJob['sampleNum'] = sampleNum
        with open(tempOrigMlJobPath, 'w') as f:
            json.dump(myJob, f, indent=4)
        print("oneOrigMlJob: SUCCESS")

    def _runOneMlMeasure(self, dfTest, dfTrain, metadata, column, method, csvFile):
        # TODO: change this limit depending on memory limitations of measuring machine
        if dfTrain.shape[0] > 100000:
            print(f"Reducing training size from {dfTrain.shape[0]} rows to 100k rows")
            dfTrain = dfTrain.sample(n=100000)
            print(f"Shape after reduction {dfTrain.shape}")
        exec = self.exec[method]
        kwargs = self.kwargs[method]
        if kwargs:
            exec.MODEL_KWARGS = {'max_iter': 500}
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
            a = 1 / 0
            quit()
        return score

    def _getTestAndTrain(self, df):
        dfShuffled = df.sample(frac=1)
        trainSize = int(dfShuffled.shape[0] / 2)
        if trainSize > self.maxTrainingSize:
            trainSize = self.maxTrainingSize
        dfTrain = dfShuffled[:trainSize]
        dfTest = dfShuffled[trainSize:]
        return dfTest, dfTrain

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
        self.tempOrigMlJobs = []
        mc = measuresConfig(self.tu)
        for csvFile, mlClassInfo in mc.getCsvOrderInfo():
            limits = {
                'binary': 0,
                'numeric': 0,
                'category': 0,
            }
            for colInfo in mlClassInfo['colInfo']:
                methodType, methods = self._getMethodsFromColInfo(colInfo)
                if methodType is None:
                    continue
                if limits[methodType] > self.maxEvalsPerType:
                    continue
                limits[methodType] += 1
                for method in methods:
                    self.tempOrigMlJobs.append({'csvFile': csvFile, 'column': colInfo['column'], 'method': method})

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
        return self._getMetadataFromMlInfo(mlInfo)
    
    def _getMetadataFromMlInfo(self, mlInfo):
        metadata = {'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1', 'columns': {}}
        for colInfo in mlInfo['colInfo']:
            whatToDo = 'category'
            if colInfo['colType'] == 'float':
                whatToDo = 'numeric'
                subType = 'float'
            if colInfo['colType'] == 'boolean':
                whatToDo = 'boolean'
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
                metadata['columns'][colInfo['column']] = {"sdtype": "categorical", }
            elif whatToDo == 'boolean':
                metadata['columns'][colInfo['column']] = {"sdtype": "boolean", }
            else:
                metadata['columns'][colInfo['column']] = {"sdtype": "numerical", "computer_representation": subType}
        return metadata

    def gatherFeatures(self):
        kfeatures = {}
        inFileNames = [f for f in os.listdir(self.tu.featuresTypeDir) if os.path.isfile(os.path.join(self.tu.featuresTypeDir, f))]
        for inFile in inFileNames:
            if inFile[-5:] != '.json':
                continue
            inPath = os.path.join(self.tu.featuresTypeDir, inFile)
            with open(inPath, 'r') as f:
                feat = json.load(f)
            if feat['csvFile'] not in kfeatures:
                kfeatures[feat['csvFile']] = {}
            if feat['targetColumn'] not in kfeatures[feat['csvFile']]:
                kfeatures[feat['csvFile']][feat['targetColumn']] = {}
            for thing in feat['origMlScores']:
                kfeatures[feat['csvFile']][feat['targetColumn']][thing['alg']] = feat['kFeatures']
        outPath = os.path.join(self.tu.runsDir, 'kfeatures.json')
        print(f"Writing kfeatures to {outPath}")
        with open(outPath, 'w') as f:
            json.dump(kfeatures, f, indent=4)

    def mergeMlMeasures(self, synMethod):
        if synMethod:
            inDir = os.path.join(self.tu.tempSynMeasures, synMethod)
            outDir = os.path.join(self.tu.synMeasures, synMethod)
        else:
            inDir = os.path.join(self.tu.tempOrigMlDir)
            outDir = os.path.join(self.tu.origMlDir)
        allMeasures = {}
        inFileNames = [f for f in os.listdir(inDir) if os.path.isfile(os.path.join(inDir, f))]
        for inFile in inFileNames:
            if 'part_' not in inFile:
                print(f"ERROR: unexpected file name {inFile}")
                quit()
            inPath = os.path.join(inDir, inFile)
            with open(inPath, 'r') as f:
                oneMl = json.load(f)
            key = oneMl['csvFile'] + '_' + oneMl['column'] + '_' + oneMl['method']
            if key in allMeasures:
                allMeasures[key]['allScores'].append(oneMl['score'])
            else:
                allMeasures[key] = {
                    'allScores':[oneMl['score']],
                    'info':oneMl,
                }
        for key, stuff in allMeasures.items():
            ''' synMeasure csvFile.method.column.ml.json
                origMeasure csvFile.column.method.json
            '''
            info = stuff['info']
            fileName = f"{info['csvFile']}.{info['column'].replace(' ','')}.{info['method']}.ml.json"
            outPath = os.path.join(outDir, fileName)
            info['score'] = max(stuff['allScores'])
            info['allScores'] = stuff['allScores']
            with open(outPath, 'w') as f:
                json.dump(info, f, indent=4)

class measuresConfig:
    def __init__(self, tu):
        self.tu = tu
        # The ML score on the original data has to be above this threshold to generate
        # a corresponding measure on the synthetic data
        self.origMlScoreThreshold = 0.7
        self.goodMlJobs = None
        self.methods = None

    def _makeLogsDir(self, name):
        dirName = os.path.join(self.tu.runsDir, name)
        os.makedirs(dirName, exist_ok=True)

    def getMlJobs(self):
        mlJobsOrderPath = os.path.join(self.tu.runsDir, 'mlJobs.json')
        with open(mlJobsOrderPath, 'r') as f:
            return json.load(f)

    def getFeaturesJobs(self):
        featuresJobsPath = os.path.join(self.tu.runsDir, 'featuresJobs.json')
        with open(featuresJobsPath, 'r') as f:
            return json.load(f)

    def makeFeaturesJobsBatchScript(self, csvLib, runsDir, featuresDir, featuresType, numJobs):
        batchFileName = f"batch_{featuresType}"
        batchScriptPath = os.path.join(self.tu.runsDir, batchFileName)
        testPath = os.path.join(self.tu.pythonDir, 'oneFeaturesJob.py')
        batchLogDir = f"logs_{featuresType}"
        self._makeLogsDir(f'logs_{featuresType}')
        batchScript = f'''#!/bin/sh
#SBATCH --time=7-0
#SBATCH --array=0-{numJobs-1}
#SBATCH --output={batchLogDir}/slurm-%A_%a.out
arrayNum="${{SLURM_ARRAY_TASK_ID}}"
python3 {testPath} \\
    --jobNum=$arrayNum \\
    --csvLib={csvLib} \\
    --runsDir={runsDir} \\
    --featuresType={featuresType} \\
    --force=False \\
    --featuresDir={featuresDir}
    '''
        with open(batchScriptPath, 'w') as f:
            f.write(batchScript)

    def makeOrigMlJobsBatchScript(self, csvLib, measuresDir, origMlDir, numJobs, numSamples):
        batchScriptPath = os.path.join(self.tu.runsDir, "batchOrigMl")
        testPath = os.path.join(self.tu.pythonDir, 'oneOrigMlJob.py')
        self._makeLogsDir('logs_origml')
        batchScript = f'''#!/bin/sh
#SBATCH --time=7-0
#SBATCH --array=0-{(numJobs*numSamples)-1}
#SBATCH --output=logs_origml/slurm-%A_%a.out
arrayNum="${{SLURM_ARRAY_TASK_ID}}"
python3 {testPath} \\
    --jobNum=$arrayNum \\
    --csvLib={csvLib} \\
    --origMlDir={origMlDir} \\
    --force=False \\
    --numJobs={numJobs} \\
    --measuresDir={measuresDir}
    '''
        with open(batchScriptPath, 'w') as f:
            print(f"Writing to {batchScriptPath}")
            f.write(batchScript)

    def makeMlJobsBatchScript(self, csvLib, tempMeasuresDir, resultsDir, runsDir, numSamples):
        batchScriptPath = os.path.join(self.tu.runsDir, "batchMl")
        testPath = os.path.join(self.tu.pythonDir, 'oneSynMLJob.py')
        self._makeLogsDir('logs_synml')
        batchScript = f'''#!/bin/sh
#SBATCH --time=7-0
#SBATCH --array=0-{(len(self.mlJobsOrder)*numSamples)-1}
#SBATCH --output=logs_synml/slurm-%A_%a.out
arrayNum="${{SLURM_ARRAY_TASK_ID}}"
python3 {testPath} \\
    --jobNum=$arrayNum \\
    --force=False \\
    --numJobs={len(self.mlJobsOrder)} \\
    --csvLib={csvLib} \\
    --resultsDir={resultsDir} \\
    --runsDir={runsDir} \\
    --tempMeasuresDir={tempMeasuresDir}
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
                self.focusJobs.append({'csvFile': csv, 'column': col})
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
        testPath = os.path.join(self.tu.pythonDir, 'oneModel.py')
        self._makeLogsDir('logs_focus')
        batchScript = f'''#!/bin/sh
#SBATCH --time=7-0
#SBATCH --array=0-{len(self.focusJobs)-1}
#SBATCH --output=logs_focus/slurm-%A_%a.out
arrayNum="${{SLURM_ARRAY_TASK_ID}}"
python3 {testPath} \\
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
        testPath = os.path.join(self.tu.pythonDir, 'oneQualityMeasure.py')
        batchScriptPath = os.path.join(self.tu.runsDir, "batchQual")
        self._makeLogsDir('logs_qual')
        if synMethod:
            batchScript = f'''#!/bin/sh
#SBATCH --time=7-0
#SBATCH --array=0-{numJobs}
#SBATCH --output=logs_qual/slurm-%A_%a.out
arrayNum="${{SLURM_ARRAY_TASK_ID}}"
python3 {testPath} \\
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
#SBATCH --output=logs_qual/slurm-%A_%a.out
arrayNum="${{SLURM_ARRAY_TASK_ID}}"
python3 {testPath} \\
    --jobNum=$arrayNum \\
    --resultsDir={resultsDir} \\
    --force=False \\
    --measuresDir={measuresDir}
            '''
        with open(batchScriptPath, 'w') as f:
            f.write(batchScript)

    def makePrivJobsBatchScript(self, runsDir, measuresDir, resultsDir, numAttacks, numAttacksInference):
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
                privJob['numAttacks'] = numAttacksInference
                privJob['auxCols'] = [col for col in columns if col != column]
                privJob['secret'] = column
                privJob['jobNum'] = jobNum
                jobNum += 1
                privJobs.append(privJob)
        privJobsPath = os.path.join(self.tu.runsDir, "privJobs.json")
        os.makedirs(self.tu.runsDir, exist_ok=True)
        with open(privJobsPath, 'w') as f:
            json.dump(privJobs, f, indent=4)
        testPath = os.path.join(self.tu.pythonDir, 'onePrivMeasure.py')
        self._makeLogsDir('logs_priv')
        batchScript = f'''#!/bin/sh
#SBATCH --time=7-0
#SBATCH --array=0-{len(privJobs)-1}
#SBATCH --output=logs_priv/slurm-%A_%a.out
arrayNum="${{SLURM_ARRAY_TASK_ID}}"
python3 {testPath} \\
    --jobNum=$arrayNum \\
    --runsDir={runsDir} \\
    --resultsDir={resultsDir} \\
    --measuresDir={measuresDir} \\
    --force=False
    '''
        with open(batchScriptPath, 'w') as f:
            f.write(batchScript)

    def makeAndSaveFeaturesJobOrder(self):
        self.initGoodMlJobs()
        goodTableTargetCombs = {}
        for csvFile in self.goodMlJobs.keys():
            for mlJob in self.goodMlJobs[csvFile]:
                if mlJob['csvFile'] not in goodTableTargetCombs:
                    goodTableTargetCombs[mlJob['csvFile']] = {mlJob['column']:[{'alg':mlJob['method'], 'score':mlJob['score']}]}
                elif mlJob['column'] not in goodTableTargetCombs[mlJob['csvFile']]:
                    goodTableTargetCombs[mlJob['csvFile']][mlJob['column']] = [{'alg':mlJob['method'], 'score':mlJob['score']}]
                elif mlJob['method'] not in goodTableTargetCombs[mlJob['csvFile']][mlJob['column']]:
                    goodTableTargetCombs[mlJob['csvFile']][mlJob['column']].append({'alg':mlJob['method'], 'score':mlJob['score']})
        self.featuresJobs = []
        pp.pprint(goodTableTargetCombs)
        jobNum = 0
        for csvFile in goodTableTargetCombs.keys():
            for targetColumn,algInfo in goodTableTargetCombs[csvFile].items():
                self.featuresJobs.append({'jobNum':jobNum,
                                          'csvFile':csvFile,
                                          'targetColumn':targetColumn,
                                          'algInfo':algInfo,
                                          })
                jobNum += 1
        featuresOrderPath = os.path.join(self.tu.runsDir, 'featuresJobs.json')
        print(f"Writing file {featuresOrderPath}")
        with open(featuresOrderPath, 'w') as f:
            json.dump(self.featuresJobs, f, indent=4)

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
            resultsFileNames = [f for f in os.listdir(
                methodResultsDir) if os.path.isfile(os.path.join(methodResultsDir, f))]
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
                dataSourceName = fileName[:csvPos + 4]
                if dataSourceName not in self.goodMlJobs:
                    # Can happen with for instance 2dim tables
                    continue
                for job in self.goodMlJobs[dataSourceName]:
                    if focusColumn is not None and job['column'] != focusColumn:
                        continue
                    measuresFile = job['csvFile'] + '.' + job['method'] + '.' + job['column'] + '.ml.json'
                    measuresPath = os.path.join(self.tu.synMeasures, method, measuresFile)
                    print(f"checking for {measuresPath}")
                    if os.path.exists(measuresPath):
                        print(f"    Skipping {measuresPath}")
                        continue
                    self.mlJobsOrder.append({**job,
                                             **{'resultsFile': fileName,
                                                'synMethod': method,
                                                }})
        mlJobsOrderPath = os.path.join(self.tu.runsDir, 'mlJobs.json')
        with open(mlJobsOrderPath, 'w') as f:
            json.dump(self.mlJobsOrder, f, indent=4)

    def findMlMethods(self):
        ''' Here we assume that there is one directory per method in the results
        directory
        '''
        self.methods = [f.name for f in os.scandir(self.tu.synResults) if f.is_dir()]

    def initGoodMlJobs(self):
        ''' This computes the ML measure jobs that should be run on each datasource
        '''
        # Get all of the orig ml measures json files
        mlFiles = self.tu.getOrigMlFiles()
        self.goodMlJobs = {}
        for mlFile in mlFiles:
            mlPath = os.path.join(self.tu.origMlDir, mlFile)
            with open(mlPath, 'r') as f:
                job = json.load(f)
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
            df = readCsv(csvPath)
            mlClassInfo = mls.makeMlClassInfo(df, csvFile)
            csvOrderInfo.append([csvFile, mlClassInfo])
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
    # pp.pprint(mlJobs)
    print(f"Total {len(sdmt.origMlJobs)} jobs")
