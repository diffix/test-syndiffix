import os
import pandas as pd
import csv

''' Change the order of columns in the CSV files for synthpop (small to high cardinality)'''

def doColumnSort(csvFile, inPath, outPath):
    csvInPath = os.path.join(inPath, csvFile)
    df = pd.read_csv(csvInPath, index_col=False, low_memory=False)
    numUniques = []
    for column in list(df.columns):
        numUniques.append([df[column].nunique(), column])
    sortedUniques = sorted(numUniques, key=lambda x: x[0])
    sortedColumns = [x[1] for x in sortedUniques]
    df = df[sortedColumns]
    csvOutPath = os.path.join(outPath, csvFile)
    df.to_csv(csvOutPath, index=False)

outDir = os.path.join(os.environ['AB_RESULTS_DIR'], 'csvSynthpop')
os.makedirs(outDir, exist_ok=True)
outOrig = os.path.join(outDir, 'original')
os.makedirs(outOrig, exist_ok=True)
outTest = os.path.join(outDir, 'test')
os.makedirs(outTest, exist_ok=True)
outTrain = os.path.join(outDir, 'train')
os.makedirs(outTrain, exist_ok=True)
inDir = os.path.join(os.environ['AB_RESULTS_DIR'], 'csvAb')
inOrig = os.path.join(inDir, 'original')
inTest = os.path.join(inDir, 'test')
inTrain = os.path.join(inDir, 'train')

csvFiles = [f for f in os.listdir(inOrig) if os.path.isfile(os.path.join(inOrig, f))]
for csvFile in csvFiles:
    for inPath, outPath in [(inOrig, outOrig), (inTest, outTest), (inTrain, outTrain)]:
        doColumnSort(csvFile, inPath, outPath)
