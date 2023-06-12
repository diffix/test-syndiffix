import pandas as pd

from pandas._libs.parsers import STR_NA_VALUES as DEFAULT_NA_VALUES

# Removes ones which are expected to be found also in non-numeric columns.
NA_VALUES = list(set(DEFAULT_NA_VALUES) - set(['', '#N/A', '#N/A N/A',
                 '#NA', '<NA>', 'N/A', 'NA', 'NULL', 'None', 'n/a', 'null']))


def readCsv(csvPath):
    return pd.read_csv(csvPath,
                       low_memory=False,
                       skipinitialspace=True,
                       index_col=False,
                       keep_default_na=False,
                       na_values=NA_VALUES)
