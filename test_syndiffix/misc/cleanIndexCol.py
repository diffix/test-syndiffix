import os
import fire
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from misc.csvUtils import readCsv


def clean(csvDir):
    """
    Removes the index column from all CSV files which have one found in `csvDir`.
    """
    files = [f for f in os.listdir(csvDir) if os.path.isfile(os.path.join(csvDir, f))]

    for csvFile in files:
        if csvFile[-4:] != '.csv':
            print(f"{csvFile} is not a csv file")
            continue
        anonPath = os.path.join(csvDir, csvFile)
        dfAnon = readCsv(anonPath)
        if dfAnon.columns[0] == 'Unnamed: 0':
            dfAnon = pd.read_csv(anonPath, index_col=0)
            dfAnon.to_csv(anonPath, index=False, sep=',', header=True)


if __name__ == "__main__":
    fire.Fire(clean)
