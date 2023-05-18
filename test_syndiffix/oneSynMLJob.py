import os
import sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import testUtils
import fire
import pprint
import sdmTools

pp = pprint.PrettyPrinter(indent=4)

''' This is used in SLURM to run one ML measure on some synthetic data.
'''

def oneSynMlJob(jobNum=0, csvLib='csvAb', measuresDir='measuresAb', resultsDir='resultsAb', runsDir='runAb', force=False):
    tu = testUtils.testUtilities()
    tu.registerCsvLib(csvLib)
    tu.registerSynMeasure(measuresDir)
    tu.registerSynResults(resultsDir)
    tu.registerRunsDir(runsDir)
    sdmt = sdmTools.sdmTools(tu)
    sdmt.runSynMlJob(jobNum, force=force)

def main():
    fire.Fire(oneSynMlJob)

if __name__ == '__main__':
    main()