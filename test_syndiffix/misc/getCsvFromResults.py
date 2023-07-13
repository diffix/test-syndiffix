import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import pprint
import fire
import csv

'''
This goes through the results and writes the anonymous data as CSV
'''

pp = pprint.PrettyPrinter(indent=4)


def getCsv():
    outDir = os.path.join(os.environ['AB_RESULTS_DIR'], 'resultsCsvAb')
    os.makedirs(outDir, exist_ok=True)
    inDir = os.path.join(os.environ['AB_RESULTS_DIR'], 'resultsAb')
    dirs = [entry.name for entry in os.scandir(inDir) if entry.is_dir()]

    for synMethod in dirs:
        synMethodPath = os.path.join(inDir, synMethod)
        synMethodOutPath = os.path.join(outDir, synMethod)
        os.makedirs(synMethodOutPath, exist_ok=True)
        print(f"Doing {synMethodPath}")
        resultsFiles = [f for f in os.listdir(synMethodPath) if os.path.isfile(os.path.join(synMethodPath, f))]
        for resultsFile in resultsFiles:
            if resultsFile[-5:] != '.json':
                continue
            resultsPath = os.path.join(synMethodPath, resultsFile)
            print(f"   {resultsPath}")
            with open(resultsPath, 'r') as f:
                results = json.load(f)
            if 'anonTable' not in results:
                print("ERROR: no anonTable")
            csvName = resultsFile[:-5] + '.csv'
            csvPath = os.path.join(outDir, synMethod, csvName)
            with open(csvPath, 'w') as f:
                write = csv.writer(f)
                write.writerow(results['colNames'])
                write.writerows(results['anonTable'])
    quit()

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
    fire.Fire(getCsv)
