import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from misc.csvUtils import readCsv
import numpy as np
import pprint
import fire

'''
This splits the given csv file into training and test parts, according to the ration `trainRatio`.
It assumes a directory structure with three directories:
    `csvAb/original`
    `csvAb/train`
    `csvAb/test`
`original` contains the original data as input, and `train` and `test` contain the train and test data as output.
'''

pp = pprint.PrettyPrinter(indent=4)
trainRatio = 0.7


def splitFiles():
    # dataSources = ['adult.data.csv', 'BankChurnersNoId.csv', 'census_small.csv']
    dataSources = []
    # Configure the following three directories
    inDir = os.path.join(os.environ['AB_RESULTS_DIR'], 'csvAttackRarePair', 'original')
    trainDir = os.path.join(os.environ['AB_RESULTS_DIR'], 'csvAttackRarePair', 'train')
    testDir = os.path.join(os.environ['AB_RESULTS_DIR'], 'csvAttackRarePair', 'test')
    os.makedirs(trainDir, exist_ok=True)
    os.makedirs(testDir, exist_ok=True)
    csvFiles = [f for f in os.listdir(inDir) if os.path.isfile(os.path.join(inDir, f))]
    for csvFile in csvFiles:
        if csvFile[-4:] != '.csv':
            continue
        if len(dataSources) > 0 and csvFile not in dataSources:
            continue
        print("-----------------------------------------")
        print(f"Datasource: {csvFile}")
        fullPath = os.path.join(inDir, csvFile)
        df = readCsv(fullPath)
        print(f"Number of rows: {df.shape[0]}")
        print("Before shuffle:")
        print(df.head())

        np.random.seed(0)
        dfShuffled = df.sample(frac=1)

        print("After shuffle:")
        print(dfShuffled.head())
        trainPart = int(dfShuffled.shape[0] * trainRatio)
        dfTrain = dfShuffled[:trainPart]
        dfTest = dfShuffled[trainPart:]
        print(f"Length of two splits: {dfTrain.shape[0]}, {dfTest.shape[0]}")
        pathTrain = os.path.join(trainDir, csvFile)
        dfTrain.to_csv(pathTrain, index=False, header=df.columns)
        pathTest = os.path.join(testDir, csvFile)
        dfTest.to_csv(pathTest, index=False, header=df.columns)


if __name__ == "__main__":
    fire.Fire(splitFiles)
