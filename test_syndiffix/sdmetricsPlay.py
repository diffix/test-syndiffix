import os
import pprint
import json
import itertools
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import testUtils
import sdmetrics
import sdmetrics.single_table
import sdmetrics.reports.single_table
import sdmetrics.reports
import pandas as pd
import cufflinks as cf
import plotly.graph_objects as go
import copy

pp = pprint.PrettyPrinter(indent=4)


class abSdmetrics:
    def __init__(self,
                 baseColumns,
                 orig,
                 anon,
                 buckets,
                 refiningNodes,
                 used2dimPairs,
                 play2dim=None,
                 lcfBuckets=None,
                 list1dimNodes=None,
                 fileName='default',
                 dir='synResults',
                 abInfo=None,
                 doPrint=False,
                 matchTolerances=[0.001, 0.01, 0.05, 0.1], maxDim=5,
                 distances=None,
                 force=False):
        '''
        '''
        self.p = doPrint
        self.force = force
        self.list1dimNodes = list1dimNodes
        self.play2dim = play2dim
        self.lcfBuckets = lcfBuckets
        self.fileName = fileName

        self.dir = dir
        os.makedirs(self.dir, exist_ok=True)

        name = self.fileName + '_quality.json'
        self.jsonPath = os.path.join(
            os.environ['AB_RESULTS_DIR'], self.dir, name)
        if os.path.exists(self.jsonPath):
            with open(self.jsonPath, 'r') as f:
                self.fullReport = json.load(f)
        else:
            self.fullReport = {'abInfo': abInfo}
        self.dataSource = None
        self.targetCol = None
        if abInfo and 'params' in abInfo and abInfo['params']:
            if 'dataSource' in abInfo['params']:
                self.dataSource = abInfo['params']['dataSource']
            if 'targetCol' in abInfo['params']:
                self.targetCol = abInfo['params']['targetCol']
        self.maxDim = maxDim
        self.matchTolerances = matchTolerances
        self.orig = orig
        self.anon = anon
        self.buckets = buckets
        self.used2dimPairs = used2dimPairs
        self.refiningNodes = refiningNodes
        self.baseColumns = baseColumns
        self.makeAidColumns()
        self.tu = testUtils.testUtilities()
        self.mls = testUtils.mlSupport(self.tu)
        self.metadata = self.mls.makeMetadata(self.dforigAid)
        self.qualityReport = None
        self.diagnosticReport = None
        self.distances = distances

    def runAll(self):
        self.doAllBucketsPlots()
        self.makeVisuals()
        if self.force is True or not self.qualityReportIsFinished():
            print('Make quality report')
            self.makeQualityReport()
        else:
            print('Skipping quality report')
        if self.force is True or not self.diagnosticReportIsFinished():
            print('Make diagnostic report')
            self.makeDiagnosticReport()
        else:
            print('Skipping diagnostic report')

    def runSynthesis(self):
        # The following commented out because it takes a very long time when there are
        # lots of columns
        if len(self.baseColumns) <= 4:
            if self.force is True or not self.newRowSynthesisIsFinished():
                self.makeNewRowSynthesis()

    def runMl(self):
        if self.force is True or not self.mlReportIsFinished():
            print('Make ML report')
            self.makeMlReport()
        else:
            print('Skipping ML report')

    def mlReportIsFinished(self):
        if 'ml' in self.fullReport:
            self.makeMlVisuals()
            return True
        return False

    def makeSampledTables(self):
        # Use the table with the uniform AID, since the classifiers use that column
        # and this seems to kill some classifiers...
        # (This may cause an issue when we deal with time-series data)
        self.dforigShuffled = self.dforigNoAid.sample(frac=1)
        self.dfanonShuffled = self.dfanonNoAid.sample(frac=1)
        halfOrig = int(self.dforigShuffled.shape[0] / 2)
        halfAnon = int(self.dfanonShuffled.shape[0] / 2)
        self.dforigTrain = self.dforigShuffled.head(halfOrig)
        self.dforigTest = self.dforigShuffled.head(-halfOrig)
        self.dfanonTrain = self.dfanonShuffled.head(halfAnon)
        self.dfanonTest = self.dfanonShuffled.head(-halfAnon)
        # There is a problem whereby if the training set and test set do not have the
        # same set of categories, sdmetrics throws an exception. To avoid this, we
        # modify values to ensure that this doesn't occur. Note that this only happens
        # to infrequent category values, so the added distortion is minimal
        numChangedRows = self.cleanUniques()
        # Check to make sure I really got rid of the unique...
        if numChangedRows and self.cleanUniques():
            print(f"Failed to properly clean {numChangedRows} uniques!")

    def cleanUniques(self):
        rowNum = 0
        for classType, columns in self.mlClassInfo.items():
            for column in columns:
                if not classType == 'categorical':
                    continue
                origTestUniques = {x: True for x in list(self.dforigTest[column].unique())}
                origTrainUniques = {x: True for x in list(self.dforigTrain[column].unique())}
                anonTrainUniques = {x: True for x in list(self.dfanonTrain[column].unique())}
                # We use dforigTest in all modeling, so add the missing value just once to
                # dforigTest
                for val in origTrainUniques.keys():
                    if val not in origTestUniques:
                        self.dforigTest.at[rowNum, column] = val
                        rowNum += 1
                        print(f"Adding {val} to dforigTest due to dforigTrain")
                for val in anonTrainUniques.keys():
                    if val not in origTestUniques:
                        self.dforigTest.at[rowNum, column] = val
                        rowNum += 1
                        print(f"Adding {val} to dforigTest due to dfanonTrain")
        return rowNum

    def makeMlVisuals(self):
        if len(self.fullReport['ml']) == 0:
            return
        df = pd.DataFrame(self.fullReport['ml'])
        g = sns.catplot(
            data=df, kind="bar",
            x="score", y="alg", hue="measure",
            palette="dark", alpha=.6, height=6
        )
        g.despine(left=True)
        g.set_axis_labels("", "Column: Classifier")
        g.legend.set_title("")
        g.set(xlim=(-1.1, 1.1))
        name = self.fileName + '_ml.png'
        pltPath = os.path.join(os.environ['AB_RESULTS_DIR'], self.dir, name)
        plt.savefig(pltPath)
        plt.close()

    def makeMlReport(self):
        self.fullReport['ml'] = []
        if len(self.baseColumns) <= 1:
            self.saveFullReport()
            return
        self.mlClassInfo = self.mls.makeMlClassInfo(self.dforigNoAid, self.dataSource, columns=self.baseColumns)
        self.makeSampledTables()
        for classType, columns in self.mlClassInfo.items():
            for column in columns:
                if self.targetCol and column != self.targetCol:
                    # If targetCol is defined, only measure that column
                    continue
                if classType == 'binary':
                    self.doBinaryClassifiers(column)
                elif classType == 'numeric':
                    self.doNumericClassifiers(column)
                elif classType == 'categorical':
                    self.doCategoricalClassifiers(column)
        self.saveFullReport()
        self.makeMlVisuals()

    def doBinaryClassifiers(self, column):
        numUnique = self.dforigAid[column].nunique()
        if numUnique != 2:
            print(f"Binary classification column {column} has {numUnique} distinct values!")
            return
        print("BinaryAdaBoostClassifier anon")
        anon = None
        orig = None
        try:
            anon = sdmetrics.single_table.BinaryAdaBoostClassifier.compute(
                test_data=self.dforigTest,
                train_data=self.dfanonTrain,
                target=column,
                metadata=self.metadata
            )
        except:
            print("model.compute() exception")
        print("BinaryAdaBoostClassifier orig")
        try:
            orig = sdmetrics.single_table.BinaryAdaBoostClassifier.compute(
                test_data=self.dforigTest,
                train_data=self.dforigTrain,
                target=column,
                metadata=self.metadata
            )
        except:
            print("model.compute() exception")
        if anon is not None and orig is not None:
            improvement = anon - orig
            self.fullReport['ml'].append({'alg': f'{column}: BinaryAdaBoost',
                                          'measure': 'anon',
                                          'score': anon})
            self.fullReport['ml'].append({'alg': f'{column}: BinaryAdaBoost',
                                          'measure': 'orig',
                                          'score': orig})
            self.fullReport['ml'].append({'alg': f'{column}: BinaryAdaBoost',
                                          'measure': 'improve',
                                          'score': improvement})
        print("BinaryLogisticRegression anon")
        anon = None
        orig = None
        try:
            anon = sdmetrics.single_table.BinaryLogisticRegression.compute(
                test_data=self.dforigTest,
                train_data=self.dfanonTrain,
                target=column,
                metadata=self.metadata
            )
        except:
            print("model.compute() exception")
        print("BinaryLogisticRegression orig")
        try:
            orig = sdmetrics.single_table.BinaryLogisticRegression.compute(
                test_data=self.dforigTest,
                train_data=self.dforigTrain,
                target=column,
                metadata=self.metadata
            )
        except:
            print("model.compute() exception")
        if anon is not None and orig is not None:
            improvement = anon - orig
            self.fullReport['ml'].append({'alg': f'{column}: BinaryRegression',
                                          'measure': 'anon',
                                          'score': anon})
            self.fullReport['ml'].append({'alg': f'{column}: BinaryRegression',
                                          'measure': 'orig',
                                          'score': orig})
            self.fullReport['ml'].append({'alg': f'{column}: BinaryRegression',
                                          'measure': 'improve',
                                          'score': improvement})
        print("BinaryMLPClassifier anon")
        sdmetrics.single_table.BinaryMLPClassifier.MODEL_KWARGS = {'max_iter': 500}
        anon = None
        orig = None
        try:
            anon = sdmetrics.single_table.BinaryMLPClassifier.compute(
                test_data=self.dforigTest,
                train_data=self.dfanonTrain,
                target=column,
                metadata=self.metadata
            )
        except:
            print("model.compute() exception")
        print("BinaryMLPClassifier orig")
        try:
            orig = sdmetrics.single_table.BinaryMLPClassifier.compute(
                test_data=self.dforigTest,
                train_data=self.dforigTrain,
                target=column,
                metadata=self.metadata
            )
        except:
            print("model.compute() exception")
        if anon is not None and orig is not None:
            improvement = anon - orig
            self.fullReport['ml'].append({'alg': f'{column}: BinaryMLP',
                                          'measure': 'anon',
                                          'score': anon})
            self.fullReport['ml'].append({'alg': f'{column}: BinaryMLP',
                                          'measure': 'orig',
                                          'score': orig})
            self.fullReport['ml'].append({'alg': f'{column}: BinaryMLP',
                                          'measure': 'improve',
                                          'score': improvement})

    def doCategoricalClassifiers(self, column):
        print("MulticlassDecisionTreeClassifier anon")
        anon = None
        orig = None
        try:
            anon = sdmetrics.single_table.MulticlassDecisionTreeClassifier.compute(
                test_data=self.dforigTest,
                train_data=self.dfanonTrain,
                target=column,
                metadata=self.metadata
            )
        except:
            print("model.compute() exception")
        print("MulticlassDecisionTreeClassifier orig")
        try:
            orig = sdmetrics.single_table.MulticlassDecisionTreeClassifier.compute(
                test_data=self.dforigTest,
                train_data=self.dforigTrain,
                target=column,
                metadata=self.metadata
            )
        except:
            print("model.compute() exception")
        if anon is not None and orig is not None:
            improvement = anon - orig
            self.fullReport['ml'].append({'alg': f'{column}: MulticlassTree',
                                          'measure': 'anon',
                                          'score': anon})
            self.fullReport['ml'].append({'alg': f'{column}: MulticlassTree',
                                          'measure': 'orig',
                                          'score': orig})
            self.fullReport['ml'].append({'alg': f'{column}: MulticlassTree',
                                          'measure': 'improve',
                                          'score': improvement})
        print("MulticlassMLPClassifier anon")
        sdmetrics.single_table.MulticlassMLPClassifier.MODEL_KWARGS = {'max_iter': 500}
        anon = None
        orig = None
        try:
            anon = sdmetrics.single_table.MulticlassMLPClassifier.compute(
                test_data=self.dforigTest,
                train_data=self.dfanonTrain,
                target=column,
                metadata=self.metadata,
                # max_iter=500,
            )
        except:
            print("model.compute() exception")
        print("MulticlassMLPClassifier orig")
        try:
            orig = sdmetrics.single_table.MulticlassMLPClassifier.compute(
                test_data=self.dforigTest,
                train_data=self.dforigTrain,
                target=column,
                metadata=self.metadata,
                # max_iter=500,
            )
        except:
            print("model.compute() exception")
        if anon is not None and orig is not None:
            improvement = anon - orig
            self.fullReport['ml'].append({'alg': f'{column}: MulticlassMLP',
                                          'measure': 'anon',
                                          'score': anon})
            self.fullReport['ml'].append({'alg': f'{column}: MulticlassMLP',
                                          'measure': 'orig',
                                          'score': orig})
            self.fullReport['ml'].append({'alg': f'{column}: MulticlassMLP',
                                          'measure': 'improve',
                                          'score': improvement})

    def doNumericClassifiers(self, column):
        print("LinearRegression anon")
        anon = None
        orig = None
        try:
            anon = sdmetrics.single_table.LinearRegression.compute(
                test_data=self.dforigTest,
                train_data=self.dfanonTrain,
                target=column,
                metadata=self.metadata
            )
        except:
            print("model.compute() exception")
        print("LinearRegression orig")
        try:
            orig = sdmetrics.single_table.LinearRegression.compute(
                test_data=self.dforigTest,
                train_data=self.dforigTrain,
                target=column,
                metadata=self.metadata
            )
        except:
            print("model.compute() exception")
        if anon is not None and orig is not None:
            improvement = anon - orig
            self.fullReport['ml'].append({'alg': f'{column}: LinearRegression',
                                          'measure': 'anon',
                                          'score': anon})
            self.fullReport['ml'].append({'alg': f'{column}: LinearRegression',
                                          'measure': 'orig',
                                          'score': orig})
            self.fullReport['ml'].append({'alg': f'{column}: LinearRegression',
                                          'measure': 'improve',
                                          'score': improvement})
        print("MLPRegressor anon")
        sdmetrics.single_table.MLPRegressor.MODEL_KWARGS = {'max_iter': 500}
        anon = None
        orig = None
        try:
            anon = sdmetrics.single_table.MLPRegressor.compute(
                test_data=self.dforigTest,
                train_data=self.dfanonTrain,
                target=column,
                metadata=self.metadata,
                # max_iter=500,
            )
        except:
            print("model.compute() exception")
        print("MLPRegressor orig")
        try:
            orig = sdmetrics.single_table.MLPRegressor.compute(
                test_data=self.dforigTest,
                train_data=self.dforigTrain,
                target=column,
                metadata=self.metadata,
                # max_iter=500,
            )
        except:
            print("model.compute() exception")
        if anon is not None and orig is not None:
            improvement = anon - orig
            self.fullReport['ml'].append({'alg': f'{column}: MLPRegressor',
                                          'measure': 'anon',
                                          'score': anon})
            self.fullReport['ml'].append({'alg': f'{column}: MLPRegressor',
                                          'measure': 'orig',
                                          'score': orig})
            self.fullReport['ml'].append({'alg': f'{column}: MLPRegressor',
                                          'measure': 'improve',
                                          'score': improvement})

    def visual2d(self, colNames=None):
        if colNames is None:
            colNames = [[self.baseColumns[0], self.baseColumns[1]]]
        if len(self.baseColumns) == 1:
            return
        for combs in colNames:
            if combs[0] not in self.dforigAid.columns or combs[1] not in self.dforigAid.columns:
                continue
            name = self.fileName + f"_2d_{combs[0]}_{combs[1]}.png"
            try:
                fig = sdmetrics.reports.utils.get_column_pair_plot(
                    real_data=self.dforigAid,
                    synthetic_data=self.dfanonAid,
                    metadata=self.metadata,
                    column_names=combs,
                )
            except:
                print("Fail sdmetrics.reports.utils.get_column_pair_plot()")
                continue
            figPath = os.path.join(os.environ['AB_RESULTS_DIR'], self.dir, name)
            try:
                fig.write_image(figPath)
            except:
                pass

    def visual1d(self, colNames=None):
        if colNames is None:
            colNames = [self.baseColumns[0]]
        for colName in colNames:
            name = self.fileName + f"_1d_{colName}.png"
            fig = sdmetrics.reports.utils.get_column_plot(
                real_data=self.dforigAid,
                synthetic_data=self.dfanonAid,
                column_name=colName,
                metadata=self.metadata,
            )
            figPath = os.path.join(os.environ['AB_RESULTS_DIR'], self.dir, name)
            try:
                fig.write_image(figPath)
            except:
                pass

    def visual2dSubplots(self):
        nDim = len(self.baseColumns)
        fields = self.metadata['fields']
        if nDim == 1:
            return
        specs = []
        figs = []
        titles = []
        for i in range(0, nDim - 1):
            columnSpecs = []
            for j in range(1, nDim):
                if i < j:
                    combs = [self.baseColumns[j], self.baseColumns[i]]
                    colDistance = self.distances[str((i, j))] if self.distances else ""
                    colLabel = f"{', '.join(combs)} distance = {colDistance}"
                    if fields[self.baseColumns[j]]['type'] == 'categorical' and fields[self.baseColumns[i]]['type'] == 'categorical':
                        # For two categorical cols, the fig from sdmetrics is a subplot itself.
                        # We need to decompose it into two separate traces and add separately as two
                        # cells in our final subplot
                        fig = sdmetrics.reports.utils.get_column_pair_plot(
                            real_data=self.dforigAid,
                            synthetic_data=self.dfanonAid,
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
                            real_data=self.dforigAid,
                            synthetic_data=self.dfanonAid,
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

        name = self.fileName + "_2dsubplots.png"
        figPath = os.path.join(os.environ['AB_RESULTS_DIR'], self.dir, name)

        # For some reason lots of dimensions cause the markers to disappear. This is a workaround
        scale = 1.0 if nDim <= 8 else (0.5 if nDim <= 16 else 0.25)

        try:
            subplotsFig.write_image(figPath,
                                    width=nDim * 512, height=nDim * 512,
                                    scale=scale)
        except:
            pass

    def visual1dimNodePlots(self):
        if not self.list1dimNodes:
            return
        nDim = len(self.baseColumns)
        fig, axes = plt.subplots(nDim, 1,
                                 constrained_layout=True,
                                 figsize=(10, nDim * 5),
                                 # subplot_kw=dict(box_aspect=1)
                                 )
        for colName, ax in zip(self.list1dimNodes.keys(), axes.flat):
            self.doList1dimNodesPlot(ax, self.list1dimNodes[colName], colName)

        name = self.fileName + f"_1dnodeplots.png"
        figPath = os.path.join(os.environ['AB_RESULTS_DIR'], self.dir, name)
        fig.suptitle(f"1dim node ranges by depth (black>100, red>50, blue>10, green<=10)")
        plt.savefig(figPath)
        plt.close()

    def visual1dSubplots(self):
        nDim = len(self.baseColumns)
        figs = []
        for colName in self.baseColumns:
            try:
                fig = sdmetrics.reports.utils.get_column_plot(
                    real_data=self.dforigAid,
                    synthetic_data=self.dfanonAid,
                    column_name=colName,
                    metadata=self.metadata,
                )
            except:
                print("Fail on sdmetrics.reports.utils.get_column_plot()")
                return
            fig.update_layout(showlegend=False)
            figs.append(fig)

        subplots = cf.subplots(figs, shape=(nDim, 1))
        subplotsFig = go.Figure(data=subplots['data'], layout=subplots['layout'])
        subplotsFig.update_layout(showlegend=False)

        name = self.fileName + f"_1dsubplots.png"
        figPath = os.path.join(os.environ['AB_RESULTS_DIR'], self.dir, name)

        try:
            subplotsFig.write_image(figPath, width=1024, height=nDim * 256)
        except:
            pass

    def newRowSynthesisIsFinished(self):
        if 'synthesis' in self.fullReport:
            return True
        return False

    def makeNewRowSynthesis(self):
        ''' Use sdmetrics to compute how well much of the anon data has a match in the
            original data. We do this for 2 dimensions up to 5 (maxDim) dimensions.
        '''
        self.fullReport['synthesis'] = []
        maxNumCols = max(self.maxDim, len(self.baseColumns))
        for dim in range(1, maxNumCols + 1):
            for matchTolerance in self.matchTolerances:
                for columnComb in itertools.combinations(self.baseColumns, dim):
                    newDfColumns = ['aid'] + list(columnComb)
                    dfOrig = self.dforigNoAid[newDfColumns]
                    dfAnon = self.dfanonNoAid[newDfColumns]
                    metadata = self.mls.makeMetadata(dfOrig)
                    fracNew, matched, new = self.newRowSynthesisQuery(
                        dfOrig, dfAnon, metadata, matchTolerance)
                    self.fullReport['synthesis'].append({'columns': str(columnComb),
                                                         'score': fracNew,
                                                         'Num Matched Rows': matched,
                                                         'Num New Rows': new,
                                                         'dim': dim,
                                                         'matchTolerance': matchTolerance,
                                                         })
        self.makeSummaryPlots()
        self.saveFullReport()

    def newRowSynthesisQuery(self, orig, anon, metadata, matchTolerance):
        # work from tables with uniform AID columns to bypass sdmetrics use of
        # AID column in computing NewRowSynthesis
        fracNew = sdmetrics.single_table.NewRowSynthesis.compute(
            real_data=orig,
            synthetic_data=anon,
            metadata=metadata,
            numerical_match_tolerance=matchTolerance,
            synthetic_sample_size=None
        )
        numRows = len(self.anon)
        matched = int((1 - fracNew) * numRows)
        new = int(fracNew * numRows)
        return 1 - fracNew, matched, new

    def makeAidColumns(self):
        if self.p:
            print(f"First row of orig without aid column: {self.orig[0]}")
        if self.p:
            print(f"First row of anon without aid column: {self.anon[0]}")
        self.origAid = [[i] + self.orig[i] for i in range(len(self.orig))]
        if self.p:
            print(f"First row of orig with aid column: {self.origAid[0]}")
        if self.p:
            print(f"Last row of orig with aid column: {self.origAid[-1]}")
        offset = len(self.origAid) + 1
        self.anonAid = [[offset + i] + self.anon[i] for i in range(len(self.anon))]
        if self.p:
            print(f"First row of anon with aid column: {self.anonAid[0]}")
        if self.p:
            print(f"Last row of anon with aid column: {self.anonAid[-1]}")
        self.columns = ['aid'] + self.baseColumns
        if self.p:
            print(self.columns)
        self.dforigAid = pd.DataFrame(self.origAid, columns=self.columns)
        if self.p:
            print(self.dforigAid.describe())
        self.dfanonAid = pd.DataFrame(self.anonAid, columns=self.columns)
        if self.p:
            print(self.dfanonAid.describe())
        # These are the tables with uniform identical AID columns
        self.origNoAid = [[0] + self.orig[i] for i in range(len(self.orig))]
        self.anonNoAid = [[0] + self.anon[i] for i in range(len(self.anon))]
        self.dforigNoAid = pd.DataFrame(self.origNoAid, columns=self.columns)
        self.dfanonNoAid = pd.DataFrame(self.anonNoAid, columns=self.columns)

    def makeSummaryPlots(self):
        # Synthesis
        if len(self.fullReport['synthesis']) == 0:
            return
        df = pd.DataFrame.from_dict(self.fullReport['synthesis'])
        plt.figure()
        ax = sns.boxplot(x='dim', y='score', data=df, hue='matchTolerance')
        plt.xlabel('Number of matched columns', fontsize=13)
        plt.ylabel('Fraction of matching rows', fontsize=12)
        plt.ylim(0, 1.1)
        ax.legend()
        name = self.fileName + '_synthesis.png'
        pltPath = os.path.join(os.environ['AB_RESULTS_DIR'], self.dir, name)
        plt.savefig(pltPath)
        plt.close()
        # Buckets. (Let's limit ourselves to buckets that are more than 1% of the total space)

    def doAllBucketsPlots(self):
        if self.buckets is None or self.refiningNodes is None:
            return
        # self.doPlay2dimPlotWork()    # uncomment to do this  TODO: remove at some point
        self.plotLcfBuckets()
        plotable2dimPairs = []
        for pair in self.used2dimPairs:
            if str(pair) in self.refiningNodes and str(pair) in self.buckets:
                plotable2dimPairs.append(pair)
        numPlots = len(plotable2dimPairs)
        if numPlots == 0:
            return
        self.doAllBucketsPlotsWork('from2dim', numPlots, plotable2dimPairs)
        self.doAllBucketsPlotsWork('from3dim', numPlots, plotable2dimPairs)

    def getRangeIndices(self, comb, bigComb):
        ri = []
        if comb[0] in bigComb:
            ri.append(bigComb.index(comb[0]))
        if comb[1] in bigComb:
            ri.append(bigComb.index(comb[1]))
        if len(ri) == 2:
            return ri
        return None

    def prepLcfBuckets(self):
        # I want to make LCF buckets for each column pair, but different colors
        # for the number of dimensions in the tree where the buckets were gathered
        preppedBuckets = {}
        for comb in self.used2dimPairs:
            preppedBuckets[comb] = []
            # find lcf buckets that contain the comb
            for thingy in self.lcfBuckets.values():
                if tuple(thingy['comb']) == comb:
                    for bkt in thingy['buckets']:
                        if len(bkt['ranges'][0]) == 2 and len(bkt['ranges'][1]) == 2:
                            preppedBuckets[comb].append({'color': 'blue', 'bucket': bkt['ranges']})
                elif len(thingy['comb']) > 2:
                    # These lcf buckets might contain the comb
                    ri = self.getRangeIndices(comb, thingy['comb'])
                    if ri is None:
                        continue
                    for bkt in thingy['buckets']:
                        ranges = [bkt['ranges'][ri[0]], bkt['ranges'][ri[1]]]
                        if len(ranges[0]) == 2 and len(ranges[1]) == 2:
                            preppedBuckets[comb].append({'color': 'green', 'bucket': ranges})
        return preppedBuckets

    def plotLcfBuckets(self):
        if len(self.used2dimPairs) == 0 or self.lcfBuckets is None:
            return
        preppedBuckets = self.prepLcfBuckets()
        allPairs = list(preppedBuckets.keys())
        numPlots = len(allPairs)
        plotHorz = math.ceil(math.sqrt(numPlots))
        plotHorz = max(plotHorz, 1)
        plotVert = math.ceil(numPlots / plotHorz)
        plotVert = max(plotVert, 1)
        if plotHorz > 1 or plotVert > 1:
            fig, axes = plt.subplots(plotVert, plotHorz,
                                     figsize=(plotHorz * 4, plotVert * 4),
                                     constrained_layout=True,
                                     subplot_kw=dict(box_aspect=1))
            for comb, ax in zip(allPairs, axes.flat):
                self.plotLcfBucketsPlot(ax, comb[0], comb[1], preppedBuckets[comb])
        else:
            fig, ax = plt.subplots()
            comb = allPairs[0]
            self.plotLcfBucketsPlot(ax, comb[0], comb[1], preppedBuckets[comb])
        name = self.fileName + f"_lcf.png"
        figPath = os.path.join(os.environ['AB_RESULTS_DIR'], self.dir, name)
        fig.suptitle(f"Buckets that failed LCF (blue at 2dim, green at > 2dim")

        plt.savefig(figPath)
        plt.close()

    def doAllBucketsPlotsWork(self, slice, numPlots, plotable2dimPairs):
        plotHorz = math.ceil(math.sqrt(numPlots))
        plotHorz = max(plotHorz, 1)
        plotVert = math.ceil(numPlots / plotHorz)
        plotVert = max(plotVert, 1)
        if plotHorz > 1 or plotVert > 1:
            fig, axes = plt.subplots(plotVert, plotHorz,
                                     figsize=(plotHorz * 4, plotVert * 4),
                                     constrained_layout=True,
                                     subplot_kw=dict(box_aspect=1))
            for comb, ax in zip(plotable2dimPairs, axes.flat):
                refiningNodes, buckets = self.getSlice(slice, comb)
                self.doBucketsPlot(ax, comb[0], comb[1], refiningNodes, buckets)
        else:
            fig, ax = plt.subplots()
            comb = plotable2dimPairs[0]
            refiningNodes, buckets = self.getSlice(slice, comb)
            self.doBucketsPlot(ax, comb[0], comb[1], refiningNodes, buckets)
        name = self.fileName + f"_buckets_{slice}.png"
        figPath = os.path.join(os.environ['AB_RESULTS_DIR'], self.dir, name)
        fig.suptitle(f"Refining Nodes and Buckets ({slice})")

    def doPlay2dimPlotWork(self):
        all2dimPairs = list(self.play2dim.keys())
        numPlots = len(all2dimPairs)
        plotHorz = math.ceil(math.sqrt(numPlots))
        plotHorz = max(plotHorz, 1)
        plotVert = math.ceil(numPlots / plotHorz)
        plotVert = max(plotVert, 1)
        if plotHorz > 1 or plotVert > 1:
            fig, axes = plt.subplots(plotVert, plotHorz,
                                     figsize=(plotHorz * 4, plotVert * 4),
                                     constrained_layout=True,
                                     subplot_kw=dict(box_aspect=1))
            for combStr, ax in zip(all2dimPairs, axes.flat):
                comb = [int(x) for x in combStr[1:-1].split(',')]
                self.doPlay2dimPlot(ax, comb[0], comb[1], self.play2dim[combStr]['buckets'])
        else:
            fig, ax = plt.subplots()
            combStr = all2dimPairs[0]
            comb = [int(x) for x in combStr[1:-1].split(',')]
            self.doPlay2dimPlot(ax, comb[0], comb[1], self.play2dim[combStr]['buckets'])
        name = self.fileName + f"_play2dim.png"
        figPath = os.path.join(os.environ['AB_RESULTS_DIR'], self.dir, name)
        fig.suptitle(f"Test flexible 2dim prebuild")

        plt.savefig(figPath)
        plt.close()

    def doAllBucketsPlotsWork(self, slice, numPlots, plotable2dimPairs):
        plotHorz = math.ceil(math.sqrt(numPlots))
        plotHorz = max(plotHorz, 1)
        plotVert = math.ceil(numPlots / plotHorz)
        plotVert = max(plotVert, 1)
        if plotHorz > 1 or plotVert > 1:
            fig, axes = plt.subplots(plotVert, plotHorz,
                                     figsize=(plotHorz * 4, plotVert * 4),
                                     constrained_layout=True,
                                     subplot_kw=dict(box_aspect=1))
            for comb, ax in zip(plotable2dimPairs, axes.flat):
                refiningNodes, buckets = self.getSlice(slice, comb)
                self.doBucketsPlot(ax, comb[0], comb[1], refiningNodes, buckets)
        else:
            fig, ax = plt.subplots()
            comb = plotable2dimPairs[0]
            refiningNodes, buckets = self.getSlice(slice, comb)
            self.doBucketsPlot(ax, comb[0], comb[1], refiningNodes, buckets)
        name = self.fileName + f"_buckets_{slice}.png"
        figPath = os.path.join(os.environ['AB_RESULTS_DIR'], self.dir, name)
        fig.suptitle(f"Refining Nodes and Buckets ({slice})")

        plt.savefig(figPath)
        plt.close()

    def getSlice(self, slice, comb2dim):
        ''' comb2dim is a tuple '''
        if slice == 'from2dim':
            return self.refiningNodes[str(comb2dim)], self.buckets[str(comb2dim)]
        # If we get here, then slice is from a 3dim tree
        # First, find the 3dim combination that contains the 2dim comb
        ci = None
        for combStr in self.buckets.keys():
            comb = [int(x) for x in combStr[1:-1].split(',')]
            if len(comb) <= 2:
                continue
            if comb2dim[0] in comb and comb2dim[1] in comb:
                ci = [comb.index(comb2dim[0]), comb.index(comb2dim[1])]
                break
        if ci is None:
            # There is no Ndim (N>2) for this 2dim
            return [], []
        # Now combStr is the key to both self.buckets and self.refiningNodes that
        # contains comb2dim. comb is the list version of combStr. ci are the
        # list indices into comb for the 2dim column indices.
        refiningNodes = []
        if combStr not in self.refiningNodes:
            return [], []
        for node in self.refiningNodes[combStr]:
            refiningNodes.append({'noisyCount': node['noisyCount'],
                                  'ranges': [node['ranges'][ci[0]], node['ranges'][ci[1]]]})
        buckets = []
        for bucket in self.buckets[combStr]:
            buckets.append({'noisyCount': bucket['noisyCount'],
                            'ranges': [bucket['ranges'][ci[0]], bucket['ranges'][ci[1]]]})
        return refiningNodes, buckets

    def doList1dimNodesPlot(self, ax, list1dimNodes, colName):
        minx = float('inf')
        maxx = float('-inf')
        miny = float('inf')
        maxy = float('-inf')
        perLevelCount = [0 for _ in range(100)]
        for node in list1dimNodes:
            left = node['ranges'][0]
            if len(node['ranges']) == 1:
                right = left + 0.000001
            else:
                right = node['ranges'][1]
            bottom = node['depth']
            top = node['depth'] + 1
            perLevelCount[node['depth']] += node['noisyCount']
            minx = min(minx, left)
            maxx = max(maxx, right)
            miny = min(miny, bottom)
            maxy = max(maxy, top)
            if node['noisyCount'] > 100:
                color = 'black'
            elif node['noisyCount'] > 50:
                color = 'red'
            elif node['noisyCount'] > 10:
                color = 'blue'
            else:
                color = 'green'
            ax.add_patch(patches.Rectangle((left, bottom), right - left, top - bottom,
                                           facecolor='none',
                                           edgecolor=color,
                                           ))
        all = max(perLevelCount)
        for i in range(len(perLevelCount)):
            if perLevelCount[i]:
                label = str(int((perLevelCount[i] / all) * 100)) + '%'
                ax.text(minx, i, label, horizontalalignment='left', verticalalignment='bottom')
        ax.set_xlabel(f"Range for column {colName}")
        ax.set_ylabel("Tree depth")
        if minx != float('inf') and maxx != float('-inf') and miny != float('inf') and maxy != float('-inf'):
            ax.set_ylim(miny, maxy)
            ax.set_xlim(minx, maxx)

    def plotLcfBucketsPlot(self, ax, yi, xi, buckets):
        ''' xi and yi are the column indices for the x and y axis respectively
        '''
        colTypes = list(self.dforigAid.dtypes)
        minx = float('inf')
        maxx = float('-inf')
        miny = float('inf')
        maxy = float('-inf')
        if ((pd.api.types.is_integer_dtype(colTypes[xi]) or
            pd.api.types.is_float_dtype(colTypes[xi]) or
            pd.api.types.is_numeric_dtype(colTypes[xi])) and
            (pd.api.types.is_integer_dtype(colTypes[yi]) or
            pd.api.types.is_float_dtype(colTypes[yi]) or
                pd.api.types.is_numeric_dtype(colTypes[yi]))):
            x = [x[xi] for x in self.orig]
            y = [x[yi] for x in self.orig]
            minx = min(x)
            miny = min(y)
            maxx = max(x)
            maxy = max(y)
            ax.scatter(x, y, s=1, alpha=0.8)
        if type(minx) == str or type(miny) == str or type(maxx) == str or type(maxy) == str:
            minx = float('inf')
            maxx = float('-inf')
            miny = float('inf')
            maxy = float('-inf')
        for bkt in buckets:
            bucket = bkt['bucket']
            left = bucket[1][0]
            right = bucket[1][1]
            bottom = bucket[0][0]
            top = bucket[0][1]
            minx = min(minx, left)
            maxx = max(maxx, right)
            miny = min(miny, bottom)
            maxy = max(maxy, top)
            ax.add_patch(patches.Rectangle((left, bottom), right - left, top - bottom,
                                           facecolor='none',
                                           edgecolor=bkt['color'],
                                           alpha=0.5,
                                           ))
        ax.set_xlabel(self.baseColumns[xi], fontsize=12)
        ax.set_ylabel(self.baseColumns[yi], fontsize=12)
        if minx != float('inf') and maxx != float('-inf') and miny != float('inf') and maxy != float('-inf'):
            ax.set_ylim(miny, maxy)
            ax.set_xlim(minx, maxx)

    def doPlay2dimPlot(self, ax, yi, xi, buckets):
        ''' xi and yi are the column indices for the x and y axis respectively
        '''
        colTypes = list(self.dforigAid.dtypes)
        minx = float('inf')
        maxx = float('-inf')
        miny = float('inf')
        maxy = float('-inf')
        if ((pd.api.types.is_integer_dtype(colTypes[xi]) or
            pd.api.types.is_float_dtype(colTypes[xi]) or
            pd.api.types.is_numeric_dtype(colTypes[xi])) and
            (pd.api.types.is_integer_dtype(colTypes[yi]) or
            pd.api.types.is_float_dtype(colTypes[yi]) or
                pd.api.types.is_numeric_dtype(colTypes[yi]))):
            x = [x[xi] for x in self.orig]
            y = [x[yi] for x in self.orig]
            # minx = min(x)
            # miny = min(y)
            # maxx = max(x)
            # maxy = max(y)
            ax.scatter(x, y, s=1, alpha=0.8)
        for bucket in buckets:
            r1 = 1
            if len(bucket[1][0]) == 1:
                r1 = 0
            t1 = 1
            if len(bucket[0][0]) == 1:
                t1 = 0
            left = bucket[1][0][0]
            right = bucket[1][0][r1]
            bottom = bucket[0][0][0]
            top = bucket[0][0][t1]
            minx = min(minx, left)
            maxx = max(maxx, right)
            miny = min(miny, bottom)
            maxy = max(maxy, top)
            ax.add_patch(patches.Rectangle((left, bottom), right - left, top - bottom,
                                           facecolor='none',
                                           edgecolor='green',
                                           alpha=0.5,
                                           ))
        ax.set_xlabel(self.baseColumns[xi], fontsize=13)
        ax.set_ylabel(self.baseColumns[yi], fontsize=12)
        if minx != float('inf') and maxx != float('-inf') and miny != float('inf') and maxy != float('-inf'):
            ax.set_ylim(miny, maxy)
            ax.set_xlim(minx, maxx)

    def doBucketsPlot(self, ax, yi, xi, refiningNodes, buckets):
        ''' xi and yi are the column indices for the x and y axis respectively
        '''
        colTypes = list(self.dforigAid.dtypes)
        if ((pd.api.types.is_integer_dtype(colTypes[xi]) or
            pd.api.types.is_float_dtype(colTypes[xi]) or
            pd.api.types.is_numeric_dtype(colTypes[xi])) and
            (pd.api.types.is_integer_dtype(colTypes[yi]) or
            pd.api.types.is_float_dtype(colTypes[yi]) or
                pd.api.types.is_numeric_dtype(colTypes[yi]))):
            x = [x[xi] for x in self.orig]
            y = [x[yi] for x in self.orig]
            ax.scatter(x, y, s=1, alpha=0.2)
        minx = float('inf')
        maxx = float('-inf')
        miny = float('inf')
        maxy = float('-inf')
        for bucket in buckets:
            r1 = 1
            if len(bucket['ranges'][1]) == 1:
                r1 = 0
            t1 = 1
            if len(bucket['ranges'][0]) == 1:
                t1 = 0
            left = bucket['ranges'][1][0]
            right = bucket['ranges'][1][r1]
            bottom = bucket['ranges'][0][0]
            top = bucket['ranges'][0][t1]
            minx = min(minx, left)
            maxx = max(maxx, right)
            miny = min(miny, bottom)
            maxy = max(maxy, top)
            ax.add_patch(patches.Rectangle((left, bottom), right - left, top - bottom,
                                           facecolor='red',
                                           alpha=0.7,
                                           ))
        for bucket in refiningNodes:
            r1 = 1
            if len(bucket['ranges'][1]) == 1:
                r1 = 0
            t1 = 1
            if len(bucket['ranges'][0]) == 1:
                t1 = 0
            left = bucket['ranges'][1][0]
            right = bucket['ranges'][1][r1]
            bottom = bucket['ranges'][0][0]
            top = bucket['ranges'][0][t1]
            minx = min(minx, left)
            maxx = max(maxx, right)
            miny = min(miny, bottom)
            maxy = max(maxy, top)
            ax.add_patch(patches.Rectangle((left, bottom), right - left, top - bottom,
                                           facecolor='none',
                                           # edgecolor='aquamarine',
                                           edgecolor='black',
                                           linewidth=0.3,
                                           ))
        ax.set_xlabel(self.baseColumns[xi], fontsize=13)
        ax.set_ylabel(self.baseColumns[yi], fontsize=12)
        if minx != float('inf') and maxx != float('-inf') and miny != float('inf') and maxy != float('-inf'):
            ax.set_ylim(miny, maxy)
            ax.set_xlim(minx, maxx)

    def saveFullReport(self):
        with open(self.jsonPath, 'w') as f:
            json.dump(self.fullReport, f, indent=4)

    def makeVisuals(self):
        self.visual1dimNodePlots()
        self.visual2dSubplots()
        self.visual1dSubplots()
        # self.visual1d()
        # self.visual2d()

    def diagnosticReportIsFinished(self):
        if ('results' in self.fullReport and
            'coverage' in self.fullReport and
                'boundaries' in self.fullReport):
            return True
        return False

    def makeDiagnosticReport(self):
        self.diagnosticReport = sdmetrics.reports.single_table.DiagnosticReport()
        self.diagnosticReport.generate(
            self.dforigAid, self.dfanonAid, self.metadata)
        self.fullReport['results'] = self.diagnosticReport.get_results()
        self.dfCoverage = self.diagnosticReport.get_details(
            property_name='Coverage')
        self.fullReport['coverage'] = self.dfCoverage.to_dict(orient='list')
        self.dfBoundaries = self.diagnosticReport.get_details(
            property_name='Boundaries')
        self.fullReport['boundaries'] = self.dfBoundaries.to_dict(
            orient='list')
        self.saveFullReport()

    def qualityReportIsFinished(self):
        if ('properties' in self.fullReport and
            'shapes' in self.fullReport and
                'pairs' in self.fullReport):
            return True
        return False

    def makeQualityReport(self):
        self.qualityReport = sdmetrics.reports.single_table.QualityReport()
        self.qualityReport.generate(
            self.dforigAid, self.dfanonAid, self.metadata)
        self.dfProperties = self.qualityReport.get_properties()
        self.fullReport['overallScore'] = self.qualityReport.get_score()
        self.fullReport['properties'] = self.dfProperties.to_dict(
            orient='list')
        self.dfShapes = self.qualityReport.get_details(
            property_name='Column Shapes')
        self.fullReport['shapes'] = self.dfShapes.to_dict(orient='list')
        self.dfPairs = self.qualityReport.get_details(
            property_name='Column Pair Trends')
        self.fullReport['pairs'] = self.dfPairs.to_dict(orient='list')
        # make the visuals
        name = self.fileName + '_quality.png'
        figPath = os.path.join(os.environ['AB_RESULTS_DIR'], self.dir, name)
        fig = self.qualityReport.get_visualization(
            property_name='Column Shapes')
        try:
            fig.write_image(figPath)
        except:
            pass
        name = self.fileName + '_pairs.png'
        figPath = os.path.join(os.environ['AB_RESULTS_DIR'], self.dir, name)
        fig = self.qualityReport.get_visualization(
            property_name='Column Pair Trends')
        try:
            fig.write_image(figPath)
        except:
            pass
        self.saveFullReport()


if __name__ == "__main__":
    pass
