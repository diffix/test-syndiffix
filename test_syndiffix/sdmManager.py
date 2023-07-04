import fire
import sys
import os
import json
import statistics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sdmTools
import testUtils
import pprint
pp = pprint.PrettyPrinter(indent=4)


class SdmManager(object):
    def __init__(self):
        pass

    def updateCsvInfo(self, csvLib='csvAb', measuresDir='measuresAb'):
        ''' Run this utility whenever a new CSV data source is added to csvLib
        '''
        tu = testUtils.testUtilities()
        tu.registerCsvLib(csvLib)
        tu.registerSynMeasure(measuresDir)
        mc = sdmTools.measuresConfig(tu)
        mc.makeCsvOrder()

    def makeOrigMlRuns(self, csvLib='csvAb', measuresDir='measuresAb', runsDir='runAb', origMlDir='origMlAb'):
        tu = testUtils.testUtilities()
        tu.registerCsvLib(csvLib)
        tu.registerSynMeasure(measuresDir)
        tu.registerRunsDir(runsDir)
        tu.registerOrigMlDir(origMlDir)
        sdmt = sdmTools.sdmTools(tu)
        sdmt.enumerateOrigMlJobs()
        mc = sdmTools.measuresConfig(tu)
        mc.makeOrigMlJobsBatchScript(csvLib, measuresDir, origMlDir, len(sdmt.origMlJobs))

    def measureMlVariance(self, origMlDir='origMlAb'):
        tu = testUtils.testUtilities()
        tu.registerOrigMlDir(origMlDir)
        mlFiles = tu.getOrigMlFiles()
        results = {}
        for mlFile in mlFiles:
            mlPath = os.path.join(self.tu.origMlDir, mlFile)
            with open(mlPath, 'r') as f:
                job = json.load(f)
            #if 'allScores' not in job:
                #print(f"Missing allScores on {mlPath}")
                #quit()
            if job['method'] not in results:
                results[job['method']] = {
                    'max':[],
                    'avg':[],
                    'sd':[],
                    'max-min':[],
                    'posNeg':[],
                }
            results[job['method']]['max'].append(max(job['allScores']))
            results[job['method']]['avg'].append(statistics.mean(job['allScores']))
            if len(job['allScores']) > 1:
                results[job['method']]['sd'].append(statistics.stdev(job['allScores']))
            else:
                results[job['method']]['sd'].append(None)
            results[job['method']]['max-min'].append(max(job['allScores'])-min(job['allScores']))
            if max(job['allScores']) > 0 and min(job['allScores']) < 0:
                results[job['method']]['posNeg'].append(1)
            else:
                results[job['method']]['posNeg'].append(0)
        for method, res in results.items():
            self._printMlStats(method, res)

    def _printMlStats(self, method, res):
        print(f"{method}:")
        print(f"    Total samples: {len(res['max'])}")
        print(f"    Average max: {statistics.mean(res['max'])}")
        print(f"    Stddev max: {statistics.stdev(res['max'])}")
        print(f"    Average avg: {statistics.mean(res['avg'])}")
        print(f"    Stddev avg: {statistics.stdev(res['avg'])}")
        print(f"    Average stdev: {statistics.mean(res['sd'])}")
        print(f"    Stddev stdev: {statistics.stdev(res['sd'])}")
        print(f"    Average max-min gap: {statistics.mean(res['max-min'])}")
        print(f"    Stddev max-min gap: {statistics.stdev(res['max-min'])}")
        print(f"    {sum(res['posNeg'])} of {len(res['posNeg'])} have both positive and negative scores")

    def makeFeatures(self, csvLib='csvAb', featuresType='univariate', featuresDir='featuresAb', resultsDir='resultsAb', runsDir='runAb', origMlDir='origMlAb', synMethod=None):
        ''' This creates a set of jobs that can be run by oneSynMLJob.py, posts the jobs at
        runsDir/mlJobs.json, and puts the needed SLURM script in runsDir as runsDir/batchMl
        '''
        print(
            f"Running with csvLib={csvLib},featuresType={featuresType},featuresDir={featuresDir},resultsDir={resultsDir},runsDir={runsDir},synMethod={synMethod}")
        tu = testUtils.testUtilities()
        tu.registerCsvLib(csvLib)
        tu.registerFeaturesDir(featuresDir)
        tu.registerFeaturesType(featuresType)
        tu.registerRunsDir(runsDir)
        tu.registerSynResults(resultsDir)
        tu.registerOrigMlDir(origMlDir)
        mc = sdmTools.measuresConfig(tu)
        mc.makeAndSaveFeaturesJobOrder()
        mc.makeFeaturesJobsBatchScript(csvLib, runsDir, featuresDir, featuresType, len(mc.featuresJobs))

    def makeMlRuns(self, csvLib='csvAb', measuresDir='measuresAb', resultsDir='resultsAb', runsDir='runAb', origMlDir='origMlAb', synMethod=None):
        ''' This creates a set of jobs that can be run by oneSynMLJob.py, posts the jobs at
        runsDir/mlJobs.json, and puts the needed SLURM script in runsDir as runsDir/batchMl
        '''
        print(
            f"Running with csvLib={csvLib},measuresDir={measuresDir},resultsDir={resultsDir},runsDir={runsDir},synMethod={synMethod}")
        tu = testUtils.testUtilities()
        tu.registerCsvLib(csvLib)
        tu.registerSynMeasure(measuresDir)
        tu.registerRunsDir(runsDir)
        tu.registerSynResults(resultsDir)
        tu.registerOrigMlDir(origMlDir)
        mc = sdmTools.measuresConfig(tu)
        mc.makeAndSaveMlJobsOrder(synMethod)
        mc.makeMlJobsBatchScript(csvLib, measuresDir, resultsDir, runsDir)

    def makeQualRuns(self, measuresDir='measuresAb', resultsDir='resultsAb', runsDir='runAb', synMethod=None):
        ''' This creates a set of jobs that can be run by oneSynQualJob.py, and puts the needed
        SLURM script in runsDir as runsDir/batchQual
        '''
        print(f"Running with measuresDir={measuresDir},resultsDir={resultsDir},runsDir={runsDir},synMethod={synMethod}")
        tu = testUtils.testUtilities()
        tu.registerSynMeasure(measuresDir)
        tu.registerSynResults(resultsDir)
        tu.registerRunsDir(runsDir)
        allResults = tu.getResultsPaths(synMethod=synMethod)
        mc = sdmTools.measuresConfig(tu)
        mc.makeQualJobsBatchScript(measuresDir, resultsDir, len(allResults) - 1, synMethod)

    def makePrivRuns(self, measuresDir='measuresAb', resultsDir='resultsAb', runsDir='runAb', numAttacks=5000, numAttacksInference=500):
        ''' This creates a set of jobs that can be run by onePrivJob.py, and puts the needed
        SLURM script in runsDir as runsDir/batchPriv
        '''
        tu = testUtils.testUtilities()
        tu.registerSynMeasure(measuresDir)
        tu.registerSynResults(resultsDir)
        tu.registerRunsDir(runsDir)
        mc = sdmTools.measuresConfig(tu)
        mc.makePrivJobsBatchScript(runsDir, measuresDir, resultsDir, numAttacks, numAttacksInference)

    def makeFocusRuns(self, csvLib='csvAb', measuresDir='measuresAb', resultsDir='resultsAb', runsDir='runAb'):
        tu = testUtils.testUtilities()
        tu.registerSynMeasure(measuresDir)
        tu.registerSynResults(resultsDir)
        tu.registerRunsDir(runsDir)
        tu.registerCsvLib(csvLib)
        mc = sdmTools.measuresConfig(tu)
        mc.makeFocusRunsScripts(csvLib, measuresDir, resultsDir, runsDir)


def main():
    fire.Fire(SdmManager)


if __name__ == '__main__':
    main()
