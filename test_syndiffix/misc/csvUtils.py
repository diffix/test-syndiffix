import pandas as pd

from pandas._libs.parsers import STR_NA_VALUES as DEFAULT_NA_VALUES


def readCsv(csvPath, **pdKwargs):
    return pd.read_csv(csvPath,
                       low_memory=False,
                       skipinitialspace=True,
                       index_col=False,
                       **pdKwargs)
