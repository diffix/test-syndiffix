import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import testUtils
import fire
import pprint
import sdmTools

pp = pprint.PrettyPrinter(indent=4)

''' This is used in SLURM to run column quality measures on one test.
'''


def onePrivMeasure(jobNum=0, runsDir='runAb', measuresDir='measuresAb', resultsDir='resultsAbHalf1', controlDir='csvAbHalf2', force=False):
    tu = testUtils.testUtilities()
    tu.registerSynMeasure(measuresDir)
    tu.registerSynResults(resultsDir)
    tu.registerControlDir(controlDir)
    tu.registerRunsDir(runsDir)
    privJobsPath = os.path.join(tu.runsDir, 'privJobs.json')
    with open(privJobsPath, 'r') as f:
        privJobs = json.load(f)
    if jobNum >= len(privJobs):
        print("onePrivMeasure: SUCCESS (jobNum too high)")
        return
    sdmt = sdmTools.sdmTools(tu)
    sdmt.runPrivMeasureJob(privJobs[jobNum], force)
    print("onePrivMeasure: SUCCESS")


def main():
    fire.Fire(onePrivMeasure)


if __name__ == '__main__':
    main()
