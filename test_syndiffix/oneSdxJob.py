import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import testUtils
import fire
import pprint
import sdmTools

pp = pprint.PrettyPrinter(indent=4)

''' This is used in SLURM to run one SynDiffix job, and produce a csv file
'''

def runSdx(tu, job):
    # Make the file name
    tableName = job['tableBase']
    if len(job['synColumns']) < 6:
        for colName in job['synColumns']:
            colName = colName.replace(' ','_')
            colName = colName.replace('-','_')
    resFileName = tableName + '.csv'
    resPath = os.path.join(tu.synResults, resFileName)
    print("Generating synthetic data at:")
    print(resPath)
    pass

def oneSdxJob(jobNum=0, expDir='exp_base', force=False):
    tu = testUtils.testUtilities()
    tu.registerExpDir(expDir)
    jobsPath = os.path.join(tu.runsDir, 'colCombJobs.json')
    print(f"jobsPath:{jobsPath}")
    with open(jobsPath, 'r') as f:
        jobs = json.load(f)
    if len(jobs) > jobNum+1:
        print(f"SUCCESS: ERROR: jobNum too high")
        quit()
    runSdx(tu, jobs[jobNum])


def main():
    fire.Fire(oneSdxJob)


if __name__ == '__main__':
    main()
