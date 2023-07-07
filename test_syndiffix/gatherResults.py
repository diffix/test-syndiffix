import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import testUtils
import sdmTools
import pandas as pd
import json
import pprint
from misc.csvUtils import readCsv

pp = pprint.PrettyPrinter(indent=4)

class resultsGather():
    def __init__(self, measuresDir='measuresAb', csvDir='csvAb'):
        self.tu = testUtils.testUtilities()
        self.sdmt = sdmTools.sdmTools(self.tu)
        self.tu.registerSynMeasure(measuresDir)
        self.tu.registerCsvLib(csvDir)
        self.csvCounts = {}
        self.elapsedDone = {}

    def gather(self):
        inDirPaths, inDirNames = self.tu.getSynMeasuresDirs()
        self.tabData = []
        for self.inDirPath, self.synMethod in zip(inDirPaths, inDirNames):
            inFiles = [f for f in os.listdir(self.inDirPath) if os.path.isfile(os.path.join(self.inDirPath, f))]
            for self.inFile in inFiles:
                if self.inFile[-5:] != '.json':
                    continue
                inPath = os.path.join(self.inDirPath, self.inFile)
                with open(inPath, 'r') as f:
                    testRes = json.load(f)
                if 'overallScore' in testRes:
                    # This is a 1dim and 2dim quality score file
                    self.addQualityScore(testRes)
                elif 'method' in testRes:
                    # This is an ML score file
                    self.addMlScore(testRes)
                elif 'privRisk' in testRes:
                    self.addPrivScore(testRes)
                else:
                    print(f"gather: don't understand file {inPath}")
                    quit()
        self.dfTab = pd.DataFrame.from_records(self.tabData)

    def addQualityScore(self, tr):
        row = self.initTabRow(tr)
        row['rowType'] = 'overallScore'
        row['rowValue'] = tr['overallScore']
        self.tabData.append(row)

        row = self.initTabRow(tr)
        row['rowType'] = 'overallPairsScore'
        row['rowValue'] = tr['properties']['Score'][1]
        self.tabData.append(row)

        row = self.initTabRow(tr)
        row['rowType'] = 'overallColumnsScore'
        row['rowValue'] = tr['properties']['Score'][0]
        self.tabData.append(row)

        for i in range(len(tr['shapes']['Column'])):
            column = tr['shapes']['Column'][i]
            score = tr['shapes']['Quality Score'][i]
            row = self.initTabRow(tr)
            row['rowType'] = 'columnScore'
            row['rowValue'] = score
            row['targetColumn'] = column
            self.tabData.append(row)

        for i in range(len(tr['pairs']['Column 1'])):
            column1 = tr['pairs']['Column 1'][i]
            column2 = tr['pairs']['Column 2'][i]
            score = tr['pairs']['Quality Score'][i]
            row = self.initTabRow(tr)
            row['rowType'] = 'pairScore'
            row['rowValue'] = score
            row['targetColumn'] = column1
            row['targetColumn2'] = column2
            self.tabData.append(row)

    def addPrivScore(self, tr):
        row = self.initTabRow(tr)
        if tr['privRisk'][1][1] - tr['privRisk'][1][0] < 0.2:
            row['rowType'] = 'privRisk'
        else:
            row['rowType'] = 'privRiskHigh'
        row['rowValue'] = tr['privRisk'][0]
        row['privMethod'] = tr['privJob']['task']
        if row['privMethod'] == 'inference':
            row['targetColumn'] = tr['privJob']['secret']
        self.tabData.append(row)

    def setElapsedTime(self, tr, elapsedTime):
        if tr['synMethod'] not in self.elapsedDone:
            self.elapsedDone[tr['synMethod']] = {}
        if tr['csvFile'] not in self.elapsedDone[tr['synMethod']]:
            # only need to do this once
            self.elapsedDone[tr['synMethod']][tr['csvFile']] = True
            row = self.initTabRow(tr)
            row['rowType'] = 'elapsedTime'
            row['rowValue'] = elapsedTime
            # Now do total elapsed for when we have features computation
            if 'features' in tr and 'elapsedTime' in tr['features']:
                row['totalElapsedTime'] = elapsedTime + tr['features']['elapsedTime']
            else:
                row['totalElapsedTime'] = elapsedTime
            self.tabData.append(row)

    def computeMlPenality(self, synScore, origScore):
        origScore = max(origScore,0)
        synScore = max(synScore,0)
        return (origScore-synScore)/max(origScore,synScore)

    def computeFeaturesWithoutMax(self, featuresJob, featureThreshold):
        if 'kFeatures' in featuresJob:
            if featureThreshold:
                featuresColumns = self.getMlFeaturesByThreshold(featuresJob, featureThreshold)
            else:
                featuresColumns = featuresJob['kFeatures']
                print(f"There are {len(featuresJob['kFeatures'])} K features")
        else:
            if featureThreshold:
                featuresColumns = self.getUniFeaturesByThreshold(featuresJob, featureThreshold)
        return len(featuresColumns)

    def getMlFeaturesByThreshold(self, featuresJob, featureThreshold):
        k = featuresJob['k']
        # We always include the top feature
        features = [featuresJob['features'][0]]
        topScore = featuresJob['cumulativeScore'][k-1]
        for index in range(1,len(featuresJob['features'])):
            thisScore = featuresJob['cumulativeScore'][index]
            if abs(thisScore - topScore) > featureThreshold:
                features.append(featuresJob['features'][index])
            else:
                break
        return features

    def getUniFeaturesByThreshold(self, featuresJob, featureThreshold):
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

    def addMlScore(self, tr):
        self.setElapsedTime(tr, tr['elapsed'])
        row = self.initTabRow(tr)
        row['rowType'] = 'synMlScore'
        row['rowValue'] = max(tr['score'],0)
        row['targetColumn'] = tr['column']
        if tr['column'] in self.csvCounts[row['csvFile']]['nunique']:
            row['targetCardinality'] = self.csvCounts[row['csvFile']]['nunique'][tr['column']]
        row['mlMethod'] = tr['method']
        row['mlMethodType'] = self.sdmt.getMethodTypeFromMethod(tr['method'])
        row['origMlScore'] = max(tr['scoreOrig'],0)
        row['mlPenalty'] = self.computeMlPenality(tr['score'], tr['scoreOrig'])
        if 'features' in tr and 'params' in tr['features']:
            params = tr['features']['params']
            row['featureThreshold'] = params['featureThreshold']
            row['usedFeatures'] = len(params['usedFeatures'])
            if 'featuresWithoutMax' in params:
                row['featuresWithoutMax'] = params['featuresWithoutMax']
            else:
                row['featuresWithoutMax'] = self.computeFeaturesWithoutMax(tr['features'], params['featureThreshold'])
            if 'maxClusterSize' in params:
                row['maxClusterSize'] = params['maxClusterSize']
            row['numClusters'] = 1
            if 'numClusters' in params:
                row['numClusters'] = params['numClusters']
        self.tabData.append(row)

    def initTabRow(self, tr):
        if 'csvFile' in tr:
            csvName = tr['csvFile']
        elif 'privJob' in tr:
            csvName = tr['privJob']['csvName']
            csvName = csvName.replace('.half1', '')
        if csvName not in self.csvCounts:
            sourcePath = os.path.join(self.tu.csvLib, csvName)
            df = readCsv(sourcePath)
            self.csvCounts[csvName] = {'rows': df.shape[0], 'cols': df.shape[1]}
            colProfile = {}
            for colName in list(df.columns):
                colProfile[colName] = df[colName].nunique()
            self.csvCounts[csvName]['nunique'] = colProfile
        if 'synMethod' not in tr and 'privJob' not in tr:
            print(self.inFile)
            print(self.inDirPath)
            return {'rowType': None,
                    'rowValue': None,
                    'targetColumn': None,
                    'targetColumn2': None,
                    'targetCardinality': None,
                    'mlMethod': None,
                    'mlMethodType': None,
                    'origMlScore': None,
                    'featureThreshold': None,
                    'usedFeatures': None,
                    'totalElapsedTime': None,
                    'featuresWithoutMax': None,
                    'maxClusterSize': None,
                    'numClusters': None,
                    'mlPenalty': None,
                    'privMethod': None,
                    'csvFile': None,
                    'numRows': None,
                    'numColumns': None,
                    'synMethod': None,
                    }
        if 'privJob' in tr:
            return {'rowType': None,
                    'rowValue': None,
                    'targetColumn': None,
                    'targetColumn2': None,
                    'targetCardinality': None,
                    'mlMethod': None,
                    'mlMethodType': None,
                    'origMlScore': None,
                    'featureThreshold': None,
                    'usedFeatures': None,
                    'totalElapsedTime': None,
                    'featuresWithoutMax': None,
                    'maxClusterSize': None,
                    'numClusters': None,
                    'mlPenalty': None,
                    'privMethod': None,
                    'csvFile': csvName,
                    'numRows': self.csvCounts[csvName]['rows'],
                    'numColumns': self.csvCounts[csvName]['cols'],
                    'synMethod': tr['privJob']['dirName'],
                    }
        if self.csvCounts[csvName]['cols'] <= 10:
            numColumns = str(self.csvCounts[csvName]['cols'])
        else:
            numColumns = '>10'
        return {'rowType': None,
                'rowValue': None,
                'targetColumn': None,
                'targetColumn2': None,
                'targetCardinality': None,
                'mlMethod': None,
                'mlMethodType': None,
                'origMlScore': None,
                'featureThreshold': None,
                'usedFeatures': None,
                'totalElapsedTime': None,
                'featuresWithoutMax': None,
                'maxClusterSize': None,
                'numClusters': None,
                'mlPenalty': None,
                'privMethod': None,
                'csvFile': csvName,
                'numRows': self.csvCounts[csvName]['rows'],
                'numColumns': self.csvCounts[csvName]['cols'],
                'numColumnsCat': numColumns,
                'synMethod': tr['synMethod'],
                }


if __name__ == '__main__':
    rg = resultsGather()
    rg.gather()
    # print(rg.dfTab.head())
    print(rg.prettyExplain())
    # print(rg.dfTab.to_string())
    # pp.pprint(rg.tabData)
    dfComp = rg.compare('py.for_g0_v3.har_v5.cl_t01_v4.md_v2', 'py.for_g0_v3.har_v5.cl_t02_v4.md_v2')
    print(f"dfComp has {dfComp.shape[0]} entries")
    dfComp2 = rg.compare('py.for_g0_v3.har_v5.cl_t04_v4.md_v2', 'py.for_g0_v3.har_v5.cl_t03_v4.md_v2')
    print(f"dfComp2 has {dfComp2.shape[0]} entries")
    print(dfComp.head())
