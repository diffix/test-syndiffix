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

def oneFeaturesJob(jobNum=0, expDir='exp_base', featuresType='univariate', force=False):
    tu = testUtils.testUtilities()
    tu.registerExpDir(expDir)
    tu.registerFeaturesType(featuresType)
    mc = sdmTools.measuresConfig(tu)
    featuresJobs = mc.getFeaturesJobs()
    if jobNum >= len(featuresJobs):
        print(f"ERROR: jobNum {jobNum} too high")
        sys.exit()
    job = featuresJobs[jobNum]
    featuresFileName = f"{featuresType}.{job['csvFile']}.{job['targetColumn']}.json"
    featuresPath = os.path.join(tu.featuresTypeDir, featuresFileName)
    csvPath = os.path.join(tu.csvLib, job['csvFile'])
    jobInfo = {
        'featuresPath': featuresPath,
        'csvPath': csvPath,
        'targetColumn': job['targetColumn'],
        'algInfo': job['algInfo']
    }
    # algInfo is a list of dicts with {'alg':mlAlg, 'score':score}
    # Here is an example `jobInfo` dict:
    '''
        {
        "jobNum": 125,
        "csvFile": "intrusion.csv",
        "targetColumn": "srv_serror_rate",
        "algInfo": [
            {
                "alg": "BinaryAdaBoostClassifier",
                "score": 0.99956057392866
            },
            {
                "alg": "BinaryAdaBoostClassifier",
                "score": 0.99956057392866
            },
            {
                "alg": "BinaryLogisticRegression",
                "score": 0.9986998087954111
            },
            {
                "alg": "BinaryMLPClassifier",
                "score": 0.9969720199310081
            }
        ]
    },
    '''
    # TODO: Edon your code gets called here with the info in `jobInfo`
    print(jobInfo)


def main():
    fire.Fire(oneFeaturesJob)


if __name__ == '__main__':
    main()
