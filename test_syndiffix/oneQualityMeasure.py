import os
import sys
import pandas as pd
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import testUtils
import fire
import pprint
import sdmTools

pp = pprint.PrettyPrinter(indent=4)

''' This is used in SLURM to run column quality measures on one test.
'''

def oneQualityMeasure(jobNum=0, measuresDir='measuresAb', resultsDir='resultsAb', showResultsListOnly=False, force=False, doVisuals=True, synMethod=None):
    tu = testUtils.testUtilities()
    tu.registerSynMeasure(measuresDir)
    tu.registerSynResults(resultsDir)
    allResults = tu.getResultsPaths(synMethod)
    if showResultsListOnly:
        print(f"results to run:")
        pp.pprint(allResults)
        print(f"There are {len(allResults)} jobs that need to be run")
        return
    if jobNum >= len(allResults):
        print("oneQualityMeasure: SUCCESS (jobNum too high)")
        return
    print("Using Job:")
    pp.pprint(allResults[jobNum])
    sdmt = sdmTools.sdmTools(tu)
    sdmt.runQualityMeasureJob(allResults[jobNum], force)
    if doVisuals:
        sdmt.makeQualityVisuals(force)
    print("oneQualityMeasure: SUCCESS")

def main():
    fire.Fire(oneQualityMeasure)

if __name__ == '__main__':
    main()