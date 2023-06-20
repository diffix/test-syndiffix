import fire
import sys
import os
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
