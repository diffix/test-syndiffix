import os
import sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import testUtils
import sdmTools
import pandas as pd
import json
import pprint

pp = pprint.PrettyPrinter(indent=4)

class resultsGather():
    def __init__(self,measuresDir='measuresAb', csvDir='csvAb'):
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
                inPath = os.path.join(self.inDirPath,self.inFile)
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
        row['rowType'] = 'privRisk'
        row['rowValue'] = tr['privRisk'][0]
        row['privMethod'] = tr['privJob']['task']
        if row['privMethod'] == 'inference':
            row['targetColumn'] = tr['privJob']['secret']
        self.tabData.append(row)

    def addMlScore(self, tr):
        if tr['synMethod'] not in self.elapsedDone:
            self.elapsedDone[tr['synMethod']] = {}
        if tr['csvFile'] not in self.elapsedDone[tr['synMethod']]:
            # only need to do this once
            self.elapsedDone[tr['synMethod']][tr['csvFile']] = True
            row = self.initTabRow(tr)
            row['rowType'] = 'elapsedTime'
            row['rowValue'] = tr['elapsed']
            self.tabData.append(row)
        row = self.initTabRow(tr)
        row['rowType'] = 'synMlScore'
        row['rowValue'] = tr['score']
        row['targetColumn'] = tr['column']
        if tr['column'] in self.csvCounts[row['csvFile']]['nunique']:
            row['targetCardinality'] = self.csvCounts[row['csvFile']]['nunique'][tr['column']]
        row['mlMethod'] = tr['method']
        row['mlMethodType'] = self.sdmt.getMethodTypeFromMethod(tr['method'])
        self.tabData.append(row)

        row = self.initTabRow(tr)
        row['rowType'] = 'origMlScore'
        row['rowValue'] = tr['scoreOrig']
        row['targetColumn'] = tr['column']
        if tr['column'] in self.csvCounts[row['csvFile']]['nunique']:
            row['targetCardinality'] = self.csvCounts[row['csvFile']]['nunique'][tr['column']]
        row['mlMethod'] = tr['method']
        row['mlMethodType'] = self.sdmt.getMethodTypeFromMethod(tr['method'])
        self.tabData.append(row)

    def initTabRow(self, tr):
        if 'csvFile' in tr:
            csvName = tr['csvFile']
        elif 'privJob' in tr:
            csvName = tr['privJob']['csvName']
            csvName = csvName.replace('.half1','')
        if csvName not in self.csvCounts:
            sourcePath = os.path.join(self.tu.csvLib, csvName)
            df = pd.read_csv(sourcePath,index_col=False,low_memory=False)
            self.csvCounts[csvName] = {'rows':df.shape[0], 'cols':df.shape[1]}
            colProfile = {}
            for colName in list(df.columns):
                colProfile[colName] = df[colName].nunique()
            self.csvCounts[csvName]['nunique'] = colProfile
        if 'synMethod' not in tr and 'privJob' not in tr:
            print(self.inFile)
            print(self.inDirPath)
            return {'rowType':None,
                   'rowValue':None,
                   'targetColumn':None,
                   'targetColumn2':None,
                   'targetCardinality':None,
                   'mlMethod':None,
                   'mlMethodType':None,
                   'privMethod':None,
                   'csvFile':None,
                   'numRows':None,
                   'numColumns':None,
                   'synMethod':None,
            }
        if 'privJob' in tr:
            return {'rowType':None,
                   'rowValue':None,
                   'targetColumn':None,
                   'targetColumn2':None,
                   'targetCardinality':None,
                   'mlMethod':None,
                   'mlMethodType':None,
                   'privMethod':None,
                   'csvFile':csvName,
                   'numRows': self.csvCounts[csvName]['rows'],
                   'numColumns': self.csvCounts[csvName]['cols'],
                   'synMethod':tr['privJob']['dirName'],
            }
        if self.csvCounts[csvName]['cols'] <= 10:
            numColumns = str(self.csvCounts[csvName]['cols'])
        else:
            numColumns = '>10'
        return {'rowType':None,
               'rowValue':None,
               'targetColumn':None,
               'targetColumn2':None,
               'targetCardinality':None,
               'mlMethod':None,
               'mlMethodType':None,
               'privMethod':None,
               'csvFile':csvName,
               'numRows': self.csvCounts[csvName]['rows'],
               'numColumns': self.csvCounts[csvName]['cols'],
               'numColumnsCat': numColumns,
               'synMethod': tr['synMethod'],
        }


if __name__ == '__main__':
    rg = resultsGather()
    rg.gather()
    #print(rg.dfTab.head())
    print(rg.prettyExplain())
    #print(rg.dfTab.to_string())
    #pp.pprint(rg.tabData)
    dfComp = rg.compare('py.for_g0_v3.har_v5.cl_t01_v4.md_v2', 'py.for_g0_v3.har_v5.cl_t02_v4.md_v2')
    print(f"dfComp has {dfComp.shape[0]} entries")
    dfComp2 = rg.compare('py.for_g0_v3.har_v5.cl_t04_v4.md_v2', 'py.for_g0_v3.har_v5.cl_t03_v4.md_v2')
    print(f"dfComp2 has {dfComp2.shape[0]} entries")
    print(dfComp.head())