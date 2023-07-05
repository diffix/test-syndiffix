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

    def _addJobToResults(self, method, job, results, mlFile):
        if method not in results:
            results[method] = {
                'max':[],
                'avg':[],
                'sd':[],
                'max-min':[],
                'max-first':[],
                'posNeg':[],
                '0.01ofMax':[],
                'mlFile':[],
            }
        maxScore = max(job['allScores'])
        results[method]['mlFile'].append(mlFile)
        results[method]['max'].append(maxScore)
        results[method]['avg'].append(statistics.mean(job['allScores']))
        if len(job['allScores']) > 1:
            results[method]['sd'].append(statistics.stdev(job['allScores']))
        results[method]['max-min'].append(maxScore-min(job['allScores']))
        results[method]['max-first'].append(maxScore-job['allScores'][0])
        if maxScore > 0 and min(job['allScores']) < 0:
            results[method]['posNeg'].append(1)
        else:
            results[method]['posNeg'].append(0)
        if maxScore > 0:
            for i in range(len(job['allScores'])):
                if maxScore - job['allScores'][i] < 0.01:
                    results[method]['0.01ofMax'].append(i)
                    break

    def measureMlVariance(self, origMlDir='origMlAb'):
        tu = testUtils.testUtilities()
        tu.registerOrigMlDir(origMlDir)
        mlFiles = tu.getOrigMlFiles()
        results = {}
        for mlFile in mlFiles:
            mlPath = os.path.join(tu.origMlDir, mlFile)
            with open(mlPath, 'r') as f:
                job = json.load(f)
            if 'allScores' not in job:
                print(f"Missing allScores on {mlPath}")
                quit()
            self._addJobToResults(job['method'], job, results, mlFile)
            if max(job['allScores']) > 0.5:
                self._addJobToResults(job['method']+'_good', job, results, mlFile)
        for method, res in results.items():
            self._printMlStats(method, res)

    def _printMlStats(self, method, res):
        print(f"{method}:")
        print(f"    Total samples: {len(res['max'])}")
        print(f"    Max max: {max(res['max'])}")
        print(f"    Min max: {min(res['max'])}")
        print(f"    Average max: {statistics.mean(res['max'])}")
        print(f"    Stddev max: {statistics.stdev(res['max'])}")
        print(f"    Average avg: {statistics.mean(res['avg'])}")
        print(f"    Stddev avg: {statistics.stdev(res['avg'])}")
        print(f"    Average stdev: {statistics.mean(res['sd'])}")
        print(f"    Stddev stdev: {statistics.stdev(res['sd'])}")
        print(f"    Max max-min gap: {max(res['max-min'])}")
        print(f"         {res['mlFile'][res['max-min'].index(max(res['max-min']))]}")
        print(f"    Average max-min gap: {statistics.mean(res['max-min'])}")
        print(f"    Stddev max-min gap: {statistics.stdev(res['max-min'])}")
        print(f"    Max max-first gap: {max(res['max-first'])}")
        print(f"         {res['mlFile'][res['max-first'].index(max(res['max-first']))]}")
        print(f"    Average max-first gap: {statistics.mean(res['max-first'])}")
        print(f"    Max 0.01ofMax: {max(res['0.01ofMax'])}")
        print(f"         {res['mlFile'][res['0.01ofMax'].index(max(res['0.01ofMax']))]}")
        print(f"    Average 0.01ofMax: {statistics.mean(res['0.01ofMax'])}")
        print(f"    Stddev 0.01ofMax: {statistics.stdev(res['0.01ofMax'])}")
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
