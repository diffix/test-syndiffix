import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import json
import pprint
from misc.csvUtils import readCsv

pp = pprint.PrettyPrinter(indent=4)

expDir = 'exp_combs'
synMethod = 'sdx_release'

resultsDir = os.path.join(os.environ['AB_RESULTS_DIR'], expDir, 'results', synMethod)
files = [f for f in os.listdir(resultsDir) if os.path.isfile(os.path.join(resultsDir, f))]
pp.pprint(files)
quit()
