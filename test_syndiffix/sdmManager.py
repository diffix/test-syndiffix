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

    def updateCsvInfo(self, expDir='exp_base'):
        ''' Run this utility whenever a new CSV data source is added to csvLib
        '''
        tu = testUtils.testUtilities()
        tu.registerExpDir(expDir)
        mc = sdmTools.measuresConfig(tu)
        mc.makeCsvOrder()

    def gatherFeatures(self, expDir='exp_base', featuresType='ml'):
        print(
            f"Running gatherFeatures with featuresType={featuresType}")
        tu = testUtils.testUtilities()
        tu.registerExpDir(expDir)
        tu.registerFeaturesType(featuresType)
        sdmt = sdmTools.sdmTools(tu)
        sdmt.gatherFeatures()

    def mergeMlMeasures(self, expDir='exp_base', synMethod=None):
        ''' This can be used to either merge original ML measures or the measures for
            synthetic data. (The defaults showsn here are for the original data measures.)
            This overwrites the measures with whatever files are in the temp measures.
            By doing it this way, we allow new temp measures to be added to an existing
            set of temp measures, and then merged.
            If doing original ML measures, then leave synMethod as None.
            Else, synMethod must be assigned.
        '''
        tu = testUtils.testUtilities()
        tu.registerExpDir(expDir)
        sdmt = sdmTools.sdmTools(tu)
        sdmt.mergeMlMeasures(synMethod)

    def makeOrigMlRuns(self, expDir='exp_base', numSamples=20):
        tu = testUtils.testUtilities()
        tu.registerExpDir(expDir)
        sdmt = sdmTools.sdmTools(tu)
        sdmt.enumerateOrigMlJobs()
        mc = sdmTools.measuresConfig(tu)
        mc.makeOrigMlJobsBatchScript(len(sdmt.origMlJobs), numSamples)

    def _addJobToResults(self, method, job, results, mlFile):
        if method not in results:
            results[method] = {
                'max': [],
                'avg': [],
                'sd': [],
                'max-min': [],
                'max-first': [],
                'posNeg': [],
                '0.01ofMax': [],
                'mlFile': [],
            }
        maxScore = max(job['allScores'])
        results[method]['mlFile'].append(mlFile)
        results[method]['max'].append(maxScore)
        results[method]['avg'].append(statistics.mean(job['allScores']))
        if len(job['allScores']) > 1:
            results[method]['sd'].append(statistics.stdev(job['allScores']))
        results[method]['max-min'].append(maxScore - min(job['allScores']))
        results[method]['max-first'].append(maxScore - job['allScores'][0])
        if maxScore > 0 and min(job['allScores']) < 0:
            results[method]['posNeg'].append(1)
        else:
            results[method]['posNeg'].append(0)
        for i in range(len(job['allScores'])):
            if maxScore - job['allScores'][i] < 0.01:
                results[method]['0.01ofMax'].append(i)
                break

    def measureMlVariance(self, expDir='exp_base'):
        tu = testUtils.testUtilities()
        tu.registerExpDir(expDir)
        mlFiles = tu.getOrigMlFiles()
        results = {}
        for mlFile in mlFiles:
            mlPath = os.path.join(tu.origMlDir, mlFile)
            with open(mlPath, 'r') as f:
                job = json.load(f)
            if 'allScores' not in job:
                print(f"Missing allScores on {mlPath}")
                sys.exit()
            self._addJobToResults(job['method'], job, results, mlFile)
            if max(job['allScores']) > 0.5:
                self._addJobToResults(job['method'] + '_good', job, results, mlFile)
        pp.pprint(results)
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

    def makeFeatures(self, expDir='exp_base', featuresType='univariate', synMethod=None):
        ''' This creates a set of jobs that can be run by oneSynMLJob.py, posts the jobs at
        runsDir/mlJobs.json, and puts the needed SLURM script in runsDir as runsDir/batchMl
        '''
        print(
            f"Running with expDir={expDir},featuresType={featuresType},synMethod={synMethod}")
        tu = testUtils.testUtilities()
        tu.registerExpDir(expDir)
        tu.registerFeaturesType(featuresType)
        mc = sdmTools.measuresConfig(tu)
        mc.makeAndSaveFeaturesJobOrder()
        mc.makeFeaturesJobsBatchScript(featuresType, len(mc.featuresJobs))

    def makeMlRuns(self, expDir='exp_base', synMethod=None, numSamples=20, limitToFeatures=False):
        ''' This creates a set of jobs that can be run by oneSynMLJob.py, posts the jobs at
        runsDir/mlJobs.json, and puts the needed SLURM script in runsDir as runsDir/batchMl
        '''
        print(
            f"Running with expDir={expDir},synMethod={synMethod}")
        tu = testUtils.testUtilities()
        tu.registerExpDir(expDir)
        mc = sdmTools.measuresConfig(tu)
        mc.makeAndSaveMlJobsOrder(synMethod)
        mc.makeMlJobsBatchScript(numSamples, limitToFeatures)

    def makeQualRuns(self, expDir='exp_base', synMethod=None):
        ''' This creates a set of jobs that can be run by oneSynQualJob.py, and puts the needed
        SLURM script in runsDir as runsDir/batchQual
        '''
        print(f"Running with expDir={expDir},synMethod={synMethod}")
        tu = testUtils.testUtilities()
        tu.registerExpDir(expDir)
        allResults = tu.getResultsPaths(synMethod=synMethod)
        mc = sdmTools.measuresConfig(tu)
        mc.makeQualJobsBatchScript(len(allResults) - 1, synMethod)

    def makePrivRuns(self, expDir='exp_base', synMethod=None, numAttacks=5000, numAttacksInference=500):
        ''' This creates a set of jobs that can be run by onePrivJob.py, and puts the needed
        SLURM script in runsDir as runsDir/batchPriv
        '''
        tu = testUtils.testUtilities()
        tu.registerExpDir(expDir)
        mc = sdmTools.measuresConfig(tu)
        mc.makePrivJobsBatchScript(synMethod, numAttacks, numAttacksInference)

    def makeFocusRuns(self, expDir='exp_base'):
        tu = testUtils.testUtilities()
        tu.registerExpDir(expDir)
        mc = sdmTools.measuresConfig(tu)
        mc.makeFocusRunsScripts()

    def help(self):
        print('''
    def updateCsvInfo(self, expDir='exp_base'):
    def mergeMlMeasures(self, expDir='exp_base', synMethod=None):
    def makeOrigMlRuns(self, expDir='exp_base', numSamples=20):
    def makeFeatures(self, expDir='exp_base', featuresType='univariate', synMethod=None):
    def makeMlRuns(self, expDir='exp_base', synMethod=None, numSamples=20, limitToFeatures=False):
    def makeQualRuns(self, expDir='exp_base', synMethod=None):
    def makePrivRuns(self, expDir='exp_base', synMethod=None, numAttacks=5000, numAttacksInference=500):
    def makeFocusRuns(self, expDir='exp_base'):
        ''')


def main():
    fire.Fire(SdmManager)


if __name__ == '__main__':
    main()
