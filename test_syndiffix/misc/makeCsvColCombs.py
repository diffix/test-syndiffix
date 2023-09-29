import pandas
import os

def buildCombs(baseCsvName, pairs, aid, inDir, outDir):
    inPath = os.path.join(inDir, baseCsvName + '.csv')
    df = pandas.read_csv(inPath, low_memory=False)
    colNames = list(df.columns.values)
    print(colNames)
    quit()
    df['Address'].to_csv('boo')
    pass

print(os.environ['AB_RESULTS_DIR'])
inDir = os.path.join(os.environ['AB_RESULTS_DIR'], 'csvAb', 'original')
outDir = os.path.join(os.environ['AB_RESULTS_DIR'], 'csvCombs')

baseCsvName = 'taxi-one-day'
pairs = [['pickup_longitude', 'pickup_latitude'],
         ['dropoff_longitude', 'dropoff_latitude']]
aid = 'hack'
maxComb = 4

buildCombs(baseCsvName, pairs, aid, inDir, outDir)


