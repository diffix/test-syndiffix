import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import testUtils
import fire
import pprint
import sdmTools

pp = pprint.PrettyPrinter(indent=4)

''' This is used in SLURM to run one ML measure on the original data. The purpose is to
determine which ML measures run well on the original data, so that we know which measures
to run on the synthetic data later on.
'''


def oneOrigMlJob(jobNum=0, csvLib='csvAb', measuresDir='measuresAb', origMlDir='origMlAb', numJobs=None, force=False):
    tu = testUtils.testUtilities()
    tu.registerCsvLib(csvLib)
    tu.registerSynMeasure(measuresDir)
    tu.registerOrigMlDir(origMlDir)
    sdmt = sdmTools.sdmTools(tu)
    sdmt.enumerateOrigMlJobs()
    jobid = i%nj
    sample = int(i/nj)
    if numJobs:
        realJobNum = jobNum%numJobs
        sampleNum = int(jobNum/numJobs)
    else:
        realJobNum = jobNum
        sampleNum = 0
    sdmt.runOrigMlJob(realJobNum, sampleNum, force)


def main():
    fire.Fire(oneOrigMlJob)


if __name__ == '__main__':
    main()
