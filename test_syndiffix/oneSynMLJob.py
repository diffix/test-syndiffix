import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import testUtils
import fire
import pprint
import sdmTools

pp = pprint.PrettyPrinter(indent=4)

''' This is used in SLURM to run one ML measure on some synthetic data.
'''


def oneSynMlJob(jobNum=0, expDir='exp_base', numJobs=None, limitToFeatures=False, force=False):
    tu = testUtils.testUtilities()
    tu.registerExpDir(expDir)
    sdmt = sdmTools.sdmTools(tu)
    if numJobs:
        realJobNum = jobNum % numJobs
        sampleNum = int(jobNum / numJobs)
    else:
        realJobNum = jobNum
        sampleNum = 0
    sdmt.runSynMlJob(realJobNum, sampleNum, limitToFeatures, force=force)


def main():
    fire.Fire(oneSynMlJob)


if __name__ == '__main__':
    main()
