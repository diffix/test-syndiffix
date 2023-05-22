import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import pprint
import fire

'''
This splits the given csv file into two halves, randomly selected. For use in
measuring privacy.
'''

pp = pprint.PrettyPrinter(indent=4)


def splitFiles():
    # dataSources = ['adult.data.csv', 'BankChurnersNoId.csv', 'census_small.csv']
    dataSources = []
    # Configure the following three directories
    inDir = os.path.join(os.environ['AB_RESULTS_DIR'], 'csvAb')
    outDir1 = os.path.join(os.environ['AB_RESULTS_DIR'], 'csvAbHalf1')
    outDir2 = os.path.join(os.environ['AB_RESULTS_DIR'], 'csvAbHalf2')
    os.makedirs(outDir1, exist_ok=True)
    os.makedirs(outDir2, exist_ok=True)
    csvFiles = [f for f in os.listdir(inDir) if os.path.isfile(os.path.join(inDir, f))]
    for csvFile in csvFiles:
        if csvFile[-4:] != '.csv':
            continue
        if len(dataSources) > 0 and csvFile not in dataSources:
            continue
        print("-----------------------------------------")
        print(f"Datasource: {csvFile}")
        fullPath = os.path.join(inDir, csvFile)
        df = pd.read_csv(fullPath, index_col=False)
        print(f"Number of rows: {df.shape[0]}")
        print("Before shuffle:")
        print(df.head())
        dfShuffled = df.sample(frac=1)
        print("After shuffle:")
        print(dfShuffled.head())
        half = int(dfShuffled.shape[0] / 2)
        df1 = df.head(half)
        df2 = df.head(-half)
        print(f"Length of two splits: {df1.shape[0]}, {df2.shape[0]}")
        name1 = csvFile[:-4] + '.half1.csv'
        path1 = os.path.join(outDir1, name1)
        df1.to_csv(path1, index=False, header=df.columns)
        name2 = csvFile[:-4] + '.half2.csv'
        path2 = os.path.join(outDir2, name2)
        df2.to_csv(path2, index=False, header=df.columns)


if __name__ == "__main__":
    fire.Fire(splitFiles)
