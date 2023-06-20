import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import testUtils
import fire
import pprint
import sdmTools

pp = pprint.PrettyPrinter(indent=4)

''' This is used in SLURM to run one features measure
'''


def oneFeaturesJob(jobNum=0, csvLib='csvAb', runsDir='runAb', featuresType='univariate', featuresDir='uniFeatAb', force=False):
    tu = testUtils.testUtilities()
    tu.registerCsvLib(csvLib)
    tu.registerFeaturesDir(featuresDir)
    tu.registerRunsDir(runsDir)
    mc = sdmTools.measuresConfig(tu)
    featuresJobs = mc.getFeaturesJobs()
    if jobNum >= len(featuresJobs):
        print(f"ERROR: jobNum {jobNum} too high")
        quit()
    job = featuresJobs[jobNum]
    featuresFileName = f"{featuresType}.{job['csvFile']}.{job['targetColumn']}.json"
    featuresPath = os.path.join(tu.featuresDir, featuresFileName)
    csvPath = os.path.join(tu.csvLib, job['csvFile'])
    jobInfo = {
        'featuresPath':featuresPath,
        'csvPath':csvPath,
        'targetColumn':job['targetColumn'],
    }
    print(jobInfo)
    


def main():
    fire.Fire(oneFeaturesJob)


if __name__ == '__main__':
    main()
