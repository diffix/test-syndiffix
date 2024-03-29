import os
import sys
import pandas as pd
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import testUtils
import gatherResults
import fire
import numpy as np

import json
import pprint

pp = pprint.PrettyPrinter(indent=4)

setLabelCountsGlobal = False
defaultHeight=3.5

def swrite(f, wstr):
    f.write(wstr)
    f.write('\n')


def summarize(expDir='exp_base',
              setLabelCounts=False,
              applesToApplesOnly=True,
              fontscale = 1.5,
              flush=False,       # re-gather
              force=False):      # overwrite existing plot
    global setLabelCountsGlobal
    global fontScale
    setLabelCountsGlobal = setLabelCounts
    fontScale = fontscale
    tu = testUtils.testUtilities()
    tu.registerExpDir(expDir)
    os.makedirs(tu.summariesDir, exist_ok=True)
    dfPath = os.path.join(tu.summariesDir, "summParquet")
    rg = gatherResults.resultsGather(expDir=tu.expDir)
    if flush is False and os.path.exists(dfPath):
        print(f"Reading dfAll from {dfPath}")
        dfAll = pd.read_parquet(dfPath)
    else:
        print(f"Gathering data...")
        rg.gather()
        dfAll = rg.dfTab
        dfAll.to_parquet(dfPath)

    print(list(dfAll.columns))
    jobs = None
    if os.path.exists('summarize.json'):
        with open('summarize.json', 'r') as f:
            jobs = json.load(f)
    print("Before ignore:")
    print(dfAll.columns)
    print("synMethods in dfAll:")
    synMethods = sorted(list(pd.unique(dfAll['synMethod'])))
    print(synMethods)
    makeCsvFiles(dfAll, tu)
    if jobs and 'colors' in jobs:
        boxColors = {}
        rgbColors = sns.color_palette('husl', len(jobs['colors']))
        for i in range(len(jobs['colors'])):
            boxColors[jobs['colors'][i]] = rgbColors[i]
    else:
        boxColors = None
    if jobs and 'ignore' in jobs:
        for synMethod in jobs['ignore']:
            print(f"Ignoring {synMethod}")
            query = f"synMethod != '{synMethod}'"
            dfAll = dfAll.query(query)
    if jobs and 'getBest' in jobs:
        for job in jobs['getBest']:
            print(f"Getting best of {job['from']}, renaming as {job['to']}")
            dfAll = getBest(dfAll, job['from'][0], job['from'][1], job['to'])
    if jobs and 'rename' in jobs:
        for job in jobs['rename']:
            print(f"Renaming {job['from']} to {job['to']}")
            # First remove the new name in case it is in dfAll
            query = f"synMethod != '{job['to']}'"
            dfAll = dfAll.query(query)
            # Then rename
            dfAll['synMethod'] = np.where((dfAll['synMethod'] == job['from']), job['to'], dfAll['synMethod'])
    # Make a column that tags large and small 2dim tables
    print(dfAll.columns)
    dfAll['2dimSizeTag'] = 'none'
    dfAll['2dimSizeTag'] = np.where(((dfAll['numColumns'] == 2) & (
        dfAll['numRows'] < 8000)), '7k rows', dfAll['2dimSizeTag'])
    dfAll['2dimSizeTag'] = np.where(((dfAll['numColumns'] == 2) & (
        dfAll['numRows'] > 27000)), '28k rows', dfAll['2dimSizeTag'])
    print("synMethods in dfAll:")
    synMethods = sorted(list(pd.unique(dfAll['synMethod'])))
    print(synMethods)
    print(f"Privacy plot")
    dfReal = dfAll.query(f"numColumns != 2")
    df2col = dfAll.query(f"numColumns == 2")
    doPrivPlot(tu, dfReal, force, boxColors)
    doPrivPlot(tu, dfReal, force, boxColors, privMethod='inference')
    doPrivPlot(tu, dfReal, force, boxColors, what='all')
    print("synMethods in dfReal:")
    synMethodsReal = sorted(list(pd.unique(dfReal['synMethod'])))
    print(synMethodsReal)
    print("synMethods in df2col:")
    synMethods2col = sorted(list(pd.unique(df2col['synMethod'])))
    print(synMethods2col)

    do2dimPlots(tu, df2col, synMethods, boxColors, force=force, doElapsed=True)
    doRealPlots(tu, dfReal, synMethods, boxColors, force=force, doElapsed=True)

    if jobs and 'combs' in jobs:
        for job in jobs['combs']:
            doRealPlots(tu, dfReal, job['columns'], boxColors, force=force, doElapsed=True,
                        scatterHues=job['scatterHues'], basicHues=job['basicHues'])
            if len(job['columns']) > 2:
                do2dimPlots(tu, df2col, job['columns'], boxColors, force=force, doElapsed=True,
                        scatterHues=job['scatterHues'], basicHues=job['basicHues'])
    dfBadPriv = dfAll.query("rowType == 'privRisk' and rowValue > 0.5")
    if dfBadPriv.shape[0] > 0:
        print("Bad privacy scores:")
        print(dfBadPriv[['rowValue', 'privMethod', 'targetColumn', 'csvFile', 'synMethod']].to_string())


def removeExtras(df):
    ''' This cleans out csv files that are not represented by all methods '''
    distinctCsv = list(pd.unique(df['csvFile']))
    distinctMethods = list(pd.unique(df['synMethod']))
    for csv, meth in itertools.product(distinctCsv, distinctMethods):
        df1 = df.query(f"synMethod == '{meth}' and csvFile == '{csv}'")
        if df1.shape[0] == 0:
            # combination doesn't exist, so get rid of csvFile
            df = df.query(f"csvFile != '{csv}'")
    return df


def makeCsvFiles(df, tu):
    for scoreType in ['columnScore', 'pairScore', 'synMlScore', 'elapsedTime', ]:
        dfTemp = df.query(f"rowType == '{scoreType}'")
        for column in dfTemp.columns:
            if dfTemp[column].isnull().all():
                dfTemp.drop(column, axis=1, inplace=True)
        csvPath = os.path.join(tu.summariesDir, f"{scoreType}.csv")
        print(f"Writing {csvPath}")
        dfTemp.to_csv(csvPath, index=False, header=dfTemp.columns)


def do2dimPlots(tu, dfIn, synMethods, boxColors, apples=True, force=False, scatterHues=[None], basicHues=[None], doElapsed=False):
    print(f"-------- do2dimPlots for synMethods '{synMethods}'")
    query = ''

    for synMethod in synMethods:
        query += f"synMethod == '{synMethod}' or "
    query = query[:-3]
    print(query)
    df = dfIn.query(query)

    title = f"Datasets with 2 columns"
    print(title)
    for hueCol in scatterHues:
        makeScatter(df, tu, synMethods, hueCol, 'equalAxis', f"2col", title, force)
    for hueCol in basicHues:
        makeAccuracyGraph(df, tu, hueCol, f"2col", title, force, boxColors, apples=apples)
    if doElapsed:
        for hueCol in basicHues:
            makeElapsedGraph(df, tu, hueCol, f"2col", title, force, boxColors, apples=apples)


def doRealPlots(tu, dfIn, synMethods, boxColors, apples=True, force=False, scatterHues=[None], basicHues=[None], doElapsed=False):
    print(f"-------- doRealPlots for synMethods '{synMethods}'")
    query = ''
    for synMethod in synMethods:
        query += f"synMethod == '{synMethod}' or "
    query = query[:-3]
    print(query)
    df = dfIn.query(query)

    # Now for only the real datasets
    title = "Real datasets only"
    print(title)
    for hueCol in scatterHues:
        makeScatter(df, tu, synMethods, hueCol, 'equalAxis', f"real", title, force)
    for hueCol in basicHues:
        makeMlGraph(df, tu, hueCol, 'real', title, force, boxColors, apples=apples)
    if doElapsed:
        for hueCol in basicHues:
            makeElapsedGraph(df, tu, hueCol, 'real', title, force, boxColors, apples=apples)


def makeScatter(df, tu, synMethods, hueCol, axisType, fileTag, title, force):
    if len(synMethods) != 2:
        return
    if hueCol:
        figPath = getFilePath(tu, synMethods, f"scatter..{hueCol}", f"{fileTag}.{axisType}")
    else:
        figPath = getFilePath(tu, synMethods, f"scatter.", f"{fileTag}.{axisType}")
    if not force and os.path.exists(figPath):
        print(f"Skipping scatter {figPath}")
        return
    print(f"    Scatter plots")
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))
    for ax0, ax1, rowType, axisLabel, rowVal, doLog, limit in zip([0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1],
                                                                  ['columnScore', 'pairScore', 'synMlScore',
                                                                   'synMlScore', 'elapsedTime', 'elapsedTime', ],
                                                                  ['Marginals Score', 'Pairs Score', 'ML Score',
                                                                      'ML Penality', 'Elapsed Time', 'Total Elapsed Time', ],
                                                                  ['rowValue', 'rowValue', 'rowValue', 'mlPenalty',
                                                                      'rowValue', 'totalElapsedTime', ],
                                                                  [False, False, False, False, True, True, ],
                                                                  [None, None, [0, 1], [-.25, 1], None, None, ]):
        dfTemp = df.query(f"rowType == '{rowType}'")
        if dfTemp.shape[0] > 0 and len(list(pd.unique(dfTemp['synMethod']))) == 2:
            dfBase = dfTemp.query(f"synMethod == '{synMethods[0]}'")
            dfOther = dfTemp.query(f"synMethod == '{synMethods[1]}'")
            print(f"Methods {synMethods}, score {rowType}:")
            makeScatterWork(dfBase, dfOther, synMethods, axs[ax0][ax1],
                            rowType, axisLabel, rowVal, hueCol, doLog, limit, axisType)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(figPath)
    plt.close()


def makeScatterWork(dfBase, dfOther, synMethods, ax, rowType, axisLabel, rowVal, hueCol, doLog, limit, axisType):
    legendDone = False
    dfMerged = pd.merge(dfBase, dfOther, how='left', on=['csvFile', 'targetColumn', 'targetColumn2', 'mlMethod'])
    print(f"dfMerged shape {dfMerged.shape}")
    # Let's count the number of times that X is greater than Y
    countX = len(dfMerged[dfMerged['rowValue_x'] > dfMerged['rowValue_y']])
    countY = len(dfMerged[dfMerged['rowValue_x'] < dfMerged['rowValue_y']])
    print(f"    All models with X>Y = {countX}, with Y>X = {countY}")
    # And for top-scoring models only:
    countX = len(dfMerged[(dfMerged['rowValue_x'] > dfMerged['rowValue_y']) &
                          ((dfMerged['rowValue_x'] >= 0.8) |
                           (dfMerged['rowValue_y'] >= 0.8))])
    countY = len(dfMerged[(dfMerged['rowValue_x'] < dfMerged['rowValue_y']) &
                          ((dfMerged['rowValue_x'] >= 0.8) |
                           (dfMerged['rowValue_y'] >= 0.8))])
    print(f"    >0.8 models with X>Y = {countX}, with Y>X = {countY}")
    # The columns get renamed after merging, so hueCol needs to be modified (to either
    # hueCol_x or hueCol_y). So long as the hueCol applies identically to the base and the other
    # data, it doesn't matter which.
    hueCol = hueCol + '_x' if hueCol else None
    hueDf,boxColorsLoc = getHueDfAndColors(dfMerged, hueCol, None, None)
    hue_order = sorted(list(pd.unique(dfMerged[hueCol]))) if hueCol else None
    rowValue_x = f"{rowVal}_x"
    rowValue_y = f"{rowVal}_y"
    g = sns.scatterplot(x=dfMerged[rowValue_x], y=dfMerged[rowValue_y], hue=hueDf, hue_order=hue_order, s=20, ax=ax)
    sns.set(font_scale=fontScale)
    if axisType == 'equalAxis':
        low = min(dfMerged[rowValue_x].min(), dfMerged[rowValue_y].min())
        high = max(dfMerged[rowValue_x].max(), dfMerged[rowValue_y].max())
        low = max(low, 0)
    else:
        low = max(dfMerged[rowValue_x].min(), dfMerged[rowValue_y].min())
        high = min(dfMerged[rowValue_x].max(), dfMerged[rowValue_y].max())
    ax.plot([low, high], [low, high], color='red')
    if ax.get_legend() is not None:
        if legendDone:
            ax.get_legend().remove()
        legendDone = True
    # ax.plot([dfMerged['rowValue_x'].min(), dfMerged['rowValue_x'].max()], [dfMerged['rowValue_y'].min(), dfMerged['rowValue_y'].max()], color='red')
    if limit is not None:
        ax.set_xlim(limit[0], limit[1])
        ax.set_ylim(limit[0], limit[1])
    if doLog:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(f"{synMethods[0]} {axisLabel} (log)")
        ax.set_ylabel(f"{synMethods[1]} {axisLabel} (log) ({dfMerged.shape[0]})")
    else:
        ax.set_xlabel(f"{synMethods[0]} {axisLabel}")
        ax.set_ylabel(f"{synMethods[1]} {axisLabel} ({dfMerged.shape[0]})")


def setLabelSampleCount(s, labels):
    newLabels = []
    sdropped = s.dropna()
    sCounts = sdropped.value_counts()
    for label in labels:
        if label not in sCounts:
            continue
        count = sCounts[label]
        if setLabelCountsGlobal:
            newLabels.append(f"{label}  ({count})")
        else:
            newLabels.append(f"{label}")
    return newLabels

def getBest(df, from1, from2, rename):
    df1 = df.query(f"synMethod == '{from1}'")
    df2 = df.query(f"synMethod == '{from2}'")
    if df1.shape[0] == 0 or df2.shape[0] == 0:
        return df
    dfMerged = pd.merge(df1, df2, how='inner', on=['csvFile', 'targetColumn', 'targetColumn2', 'mlMethod', 'numColumns', 'rowType', 'mlMethodType'])
    dfMerged['rowValue'] = np.where(dfMerged['rowValue_x'] > dfMerged['rowValue_y'],
                                    dfMerged['rowValue_x'], dfMerged['rowValue_y'])
    dfMerged['mlPenalty'] = np.where(dfMerged['rowValue_x'] > dfMerged['rowValue_y'],
                                    dfMerged['mlPenalty_x'], dfMerged['mlPenalty_y'])
    dfMerged['totalElapsedTime'] = np.where(dfMerged['rowValue_x'] > dfMerged['rowValue_y'],
                                    dfMerged['totalElapsedTime_x'], dfMerged['totalElapsedTime_y'])
    dfMerged['numRows'] = np.where(dfMerged['rowValue_x'] > dfMerged['rowValue_y'],
                                    dfMerged['numRows_x'], dfMerged['numRows_y'])
    dfMerged['synMethod'] = rename
    df1 = df[['synMethod', 'rowValue', 'csvFile', 'targetColumn', 'targetColumn2', 'mlMethod', 'mlPenalty', 'numColumns', 'numRows', 'rowType', 'totalElapsedTime', 'mlMethodType']]
    df2 = dfMerged[['synMethod', 'rowValue', 'csvFile', 'targetColumn', 'targetColumn2', 'mlMethod', 'mlPenalty', 'numColumns', 'numRows', 'rowType', 'totalElapsedTime', 'mlMethodType']]
    return pd.concat([df1, df2], axis=0)


def doPrivPlot(tu, df, force, boxColors, what='lowBounds', hueCol=None, privMethod='all'):
    if what == 'lowBounds' and privMethod == 'all':
        dfTemp = df.query("rowType == 'privRisk'")
        xaxis = 'Privacy Risk'
        printStats(dfTemp, hueCol, "priv high confidence, all methods")
        figPath = os.path.join(tu.summariesDir, 'priv.png')
    elif what == 'all' and privMethod == 'all':
        dfTemp = df.query("rowType == 'privRisk' or rowType == 'privRiskHigh'")
        xaxis = 'Privacy Risk (including low confidence scores)'
        printStats(dfTemp, hueCol, "priv high and low confidence, all methods")
        figPath = os.path.join(tu.summariesDir, 'privLowConf.png')
    elif what == 'lowBounds' and privMethod == 'inference':
        dfTemp = df.query("rowType == 'privRisk' and privMethod == 'inference'")
        xaxis = 'Privacy Risk'
        printStats(dfTemp, hueCol, "priv high confidence, inference only")
        figPath = os.path.join(tu.summariesDir, 'privInference.png')
    elif what == 'all' and privMethod == 'inference':
        dfTemp = df.query("rowType == 'privRisk' or rowType == 'privRiskHigh' and privMethod == 'inference'")
        xaxis = 'Privacy Risk (including low confidence scores)'
        printStats(dfTemp, hueCol, "priv high and low confidence, inference only")
        figPath = os.path.join(tu.summariesDir, 'privLowConfInference.png')
    if dfTemp.shape[0] == 0:
        return
    synMethods = sorted(list(pd.unique(dfTemp['synMethod'])))
    if 'noAnon' in synMethods:
        synMethods.remove('noAnon')
        synMethods.append('noAnon')
        print("New synMethods:")
        print(synMethods)
    hueDf,boxColorsLoc = getHueDfAndColors(dfTemp, hueCol, boxColors, 'synMethod')
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7, defaultHeight))
    sns.boxplot(x=dfTemp['rowValue'], y=dfTemp['synMethod'], palette=boxColorsLoc, hue=hueDf, order=synMethods, ax=axs)
    #sns.boxplot(x=dfTemp['rowValue'], y=dfTemp['synMethod'], palette=boxColorsLoc, hue=hueDf, order=synMethods)
    sns.set(font_scale=fontScale)
    plt.tight_layout()
    plt.xlim(0, 1)
    plt.xticks([0.01, 0.1, 0.2, 0.5, 1.0], ['0.01', '0.1', '0.2', '0.5', '1.0'])
    # plt.xscale('symlog')
    plt.xlabel(xaxis)
    plt.savefig(figPath)
    plt.close()


def makeMlGraph(df, tu, hueCol, fileTag, title, force, boxColors, apples=True):
    print("    ML plots")
    synMethods = sorted(list(pd.unique(df['synMethod'])))
    if 'noAnon' in synMethods:
        synMethods.remove('noAnon')
        synMethods.append('noAnon')
        print("New synMethods:")
        print(synMethods)
    if apples:
        figPath = getFilePath(tu, synMethods, 'ml', f"{fileTag}.{hueCol}")
    else:
        figPath = getFilePath(tu, synMethods, 'ml', f"{fileTag}.{hueCol}.noapples")
        title += " (not apples-to-apples)"
    if not force and os.path.exists(figPath):
        print(f"Skipping {figPath}")
        return
    height = max(defaultHeight, len(synMethods) * 0.8)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, height))

    dfTemp = df.query("rowType == 'synMlScore'")
    if dfTemp.shape[0] > 0:
        if apples:
            dfTemp = removeExtras(dfTemp)
        xaxis = 'ML Score'
        hueDf,boxColorsLoc = getHueDfAndColors(dfTemp, hueCol, boxColors, 'synMethod')
        print(figPath)
        print(title)
        print(xaxis)
        printStats(dfTemp, hueCol, "quality")
        sns.boxplot(x=dfTemp['rowValue'], y=dfTemp['synMethod'], palette=boxColorsLoc, hue=hueDf, order=synMethods, ax=axs[0])
        sns.set(font_scale=fontScale)
        sampleCounts = setLabelSampleCount(dfTemp['synMethod'], synMethods)
        if len(sampleCounts) == len(synMethods):
            axs[0].yaxis.set_ticklabels(setLabelSampleCount(dfTemp['synMethod'], synMethods))
        axs[0].set_xlabel(xaxis)
        low = dfTemp['rowValue'].min()
        if hueDf is not None:
            axs[0].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        axs[0].set_xlim(max(0, low), 1.0)

        xaxis = 'ML Penalty'
        hueDf,boxColorsLoc = getHueDfAndColors(dfTemp, hueCol, boxColors, 'synMethod')
        print(figPath)
        print(title)
        print(xaxis)
        printStats(dfTemp, hueCol, "quality", measureField='mlPenalty')
        sns.boxplot(x=dfTemp['mlPenalty'], y=dfTemp['synMethod'], palette=boxColorsLoc, hue=hueDf, order=synMethods, ax=axs[1])
        sns.set(font_scale=fontScale)
        sampleCounts = setLabelSampleCount(dfTemp['synMethod'], synMethods)
        if len(sampleCounts) == len(synMethods):
            axs[1].yaxis.set_ticklabels(setLabelSampleCount(dfTemp['synMethod'], synMethods))
        axs[1].set_xlabel(xaxis)
        low = dfTemp['rowValue'].min()
        if hueDf is not None:
            axs[1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        # axs[1].set_xlim(max(0, low), 1.0)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(figPath)
    plt.close()


def makeElapsedGraph(df, tu, hueCol, fileTag, title, force, boxColors, apples=True):
    print("    Elapsed plots")
    synMethods = sorted(list(pd.unique(df['synMethod'])))
    if apples:
        figPath = getFilePath(tu, synMethods, 'Elapsed', f"{fileTag}.{hueCol}")
    else:
        figPath = getFilePath(tu, synMethods, 'Elapsed', f"{fileTag}.{hueCol}.noapples")
        title += " (not apples-to-apples)"
    if not force and os.path.exists(figPath):
        print(f"Skipping {figPath}")
        return
    height = max(defaultHeight, len(synMethods) * 0.8)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, height))

    dfTemp = df.query("rowType == 'elapsedTime'")
    if dfTemp.shape[0] > 0:
        if apples:
            dfTemp = removeExtras(dfTemp)
        xaxis = 'Elapsed Time (seconds) (log)'
        hueDf,boxColorsLoc = getHueDfAndColors(dfTemp, hueCol, boxColors, 'synMethod')
        print(figPath)
        print(title)
        print(xaxis)
        printStats(dfTemp, hueCol, "time")

        sns.boxplot(x=dfTemp['rowValue'], y=dfTemp['synMethod'], palette=boxColorsLoc, hue=hueDf, order=synMethods, ax=axs[0])
        sns.set(font_scale=fontScale)
        sampleCounts = setLabelSampleCount(dfTemp['synMethod'], synMethods)
        if len(sampleCounts) == len(synMethods):
            axs[0].yaxis.set_ticklabels(setLabelSampleCount(dfTemp['synMethod'], synMethods))
        axs[0].set_xlim(left=0.1)
        axs[0].set_xscale('log')  # zzzz
        if hueDf is not None:
            axs[0].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        axs[0].set_xlabel(xaxis)

        xaxis = 'Elapsed Time with Feature Selection'
        sns.boxplot(x=dfTemp['totalElapsedTime'], y=dfTemp['synMethod'], palette=boxColorsLoc, hue=hueDf, order=synMethods, ax=axs[1])
        sns.set(font_scale=fontScale)
        sampleCounts = setLabelSampleCount(dfTemp['synMethod'], synMethods)
        if len(sampleCounts) == len(synMethods):
            axs[1].yaxis.set_ticklabels(setLabelSampleCount(dfTemp['synMethod'], synMethods))
        axs[1].set_xlim(left=0.1)
        axs[1].set_xscale('log')  # zzzz
        if hueDf is not None:
            axs[1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        axs[1].set_xlabel(xaxis)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(figPath)
    plt.close()


def makeBasicGraph(df, tu, hueCol, fileTag, title, force, boxColors, apples=True):
    print("    Basic plots")
    synMethods = sorted(list(pd.unique(df['synMethod'])))
    if apples:
        figPath = getFilePath(tu, synMethods, 'basicStats', f"{fileTag}.{hueCol}")
    else:
        figPath = getFilePath(tu, synMethods, 'basicStats', f"{fileTag}.{hueCol}.noapples")
        title += " (not apples-to-apples)"
    if not force and os.path.exists(figPath):
        print(f"Skipping {figPath}")
        return
    height = max(defaultHeight, len(synMethods) * 1.8)
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, height))

    dfTemp = df.query("rowType == 'columnScore'")
    if dfTemp.shape[0] > 0:
        if apples:
            dfTemp = removeExtras(dfTemp)
        # dfMerged = pd.merge(dfBase, dfOther, how='inner', on = ['csvFile','targetColumn','mlMethod'])
        xaxis = 'Marginal columns quality'
        hueDf,boxColorsLoc = getHueDfAndColors(dfTemp, hueCol, boxColors, 'synMethod')
        print(figPath)
        print(title)
        print(xaxis)
        printStats(dfTemp, hueCol, "quality")
        sns.boxplot(x=dfTemp['rowValue'], y=dfTemp['synMethod'], palette=boxColorsLoc, hue=hueDf, order=synMethods, ax=axs[0][0])
        sns.set(font_scale=fontScale)
        sampleCounts = setLabelSampleCount(dfTemp['synMethod'], synMethods)
        if len(sampleCounts) == len(synMethods):
            axs[0][0].yaxis.set_ticklabels(setLabelSampleCount(dfTemp['synMethod'], synMethods))
        axs[0][0].set_xlabel(xaxis)
        # axs[0][0].set_xscale('function', functions=(partial(np.power, 10.0), np.log10))
        low = dfTemp['rowValue'].min()
        if hueDf is not None:
            axs[0][0].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        axs[0][0].set_xlim(max(0.8, low), 1.0)

    dfTemp = df.query("rowType == 'pairScore'")
    if dfTemp.shape[0] > 0:
        if apples:
            dfTemp = removeExtras(dfTemp)
        xaxis = 'Column pairs quality'
        hueDf,boxColorsLoc = getHueDfAndColors(dfTemp, hueCol, boxColors, 'synMethod')
        print(figPath)
        print(title)
        print(xaxis)
        printStats(dfTemp, hueCol, "quality")
        sns.boxplot(x=dfTemp['rowValue'], y=dfTemp['synMethod'], palette=boxColorsLoc, hue=hueDf, order=synMethods, ax=axs[0][1])
        sns.set(font_scale=fontScale)
        sampleCounts = setLabelSampleCount(dfTemp['synMethod'], synMethods)
        if len(sampleCounts) == len(synMethods):
            axs[0][1].yaxis.set_ticklabels(setLabelSampleCount(dfTemp['synMethod'], synMethods))
        axs[0][1].set_xlabel(xaxis)
        # axs[0][1].set_xscale('function', functions=(partial(np.power, 10.0), np.log10))
        low = dfTemp['rowValue'].min()
        if hueDf is not None:
            axs[0][1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        axs[0][1].set_xlim(max(0.8, low), 1.0)
        # axs[0][1].set(yticklabels = [], ylabel = None)

    dfTemp = df.query("rowType == 'synMlScore'")
    if dfTemp.shape[0] > 0:
        if apples:
            dfTemp = removeExtras(dfTemp)
        xaxis = 'ML Score'
        hueDf,boxColorsLoc = getHueDfAndColors(dfTemp, hueCol, boxColors, 'synMethod')
        print(figPath)
        print(title)
        print(xaxis)
        printStats(dfTemp, hueCol, "quality")
        sns.boxplot(x=dfTemp['rowValue'], y=dfTemp['synMethod'], palette=boxColorsLoc, hue=hueDf, order=synMethods, ax=axs[1][0])
        sns.set(font_scale=fontScale)
        sampleCounts = setLabelSampleCount(dfTemp['synMethod'], synMethods)
        if len(sampleCounts) == len(synMethods):
            axs[1][0].yaxis.set_ticklabels(setLabelSampleCount(dfTemp['synMethod'], synMethods))
        axs[1][0].set_xlabel(xaxis)
        low = dfTemp['rowValue'].min()
        if hueDf is not None:
            axs[1][0].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        axs[1][0].set_xlim(max(0, low), 1.0)

        xaxis = 'ML Penalty'
        hueDf,boxColorsLoc = getHueDfAndColors(dfTemp, hueCol, boxColors, 'synMethod')
        print(figPath)
        print(title)
        print(xaxis)
        printStats(dfTemp, hueCol, "quality", measureField='mlPenalty')
        sns.boxplot(x=dfTemp['mlPenalty'], y=dfTemp['synMethod'], palette=boxColorsLoc, hue=hueDf, order=synMethods, ax=axs[1][1])
        sns.set(font_scale=fontScale)
        sampleCounts = setLabelSampleCount(dfTemp['synMethod'], synMethods)
        if len(sampleCounts) == len(synMethods):
            axs[1][1].yaxis.set_ticklabels(setLabelSampleCount(dfTemp['synMethod'], synMethods))
        axs[1][1].set_xlabel(xaxis)
        low = dfTemp['rowValue'].min()
        if hueDf is not None:
            axs[1][1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        # axs[1][1].set_xlim(max(0, low), 1.0)

    dfTemp = df.query("rowType == 'elapsedTime'")
    if dfTemp.shape[0] > 0:
        if apples:
            dfTemp = removeExtras(dfTemp)
        xaxis = 'Elapsed Time (seconds) (log)'
        hueDf,boxColorsLoc = getHueDfAndColors(dfTemp, hueCol, boxColors, 'synMethod')
        print(figPath)
        print(title)
        print(xaxis)
        printStats(dfTemp, hueCol, "time")
        sns.boxplot(x=dfTemp['rowValue'], y=dfTemp['synMethod'], palette=boxColorsLoc, hue=hueDf, order=synMethods, ax=axs[2][0])
        sns.set(font_scale=fontScale)
        sampleCounts = setLabelSampleCount(dfTemp['synMethod'], synMethods)
        if len(sampleCounts) == len(synMethods):
            axs[2][0].yaxis.set_ticklabels(setLabelSampleCount(dfTemp['synMethod'], synMethods))
        axs[2][0].set_xlim(left=0.1)
        axs[2][0].set_xscale('log')  # zzzz
        if hueDf is not None:
            axs[2][0].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        axs[2][0].set_xlabel(xaxis)
        # axs[2][0].set(yticklabels = [], ylabel = None)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(figPath)
    plt.close()


def makeAccuracyGraph(df, tu, hueCol, fileTag, title, force, boxColors, apples=True):
    print("    Accuracy plots")
    synMethods = sorted(list(pd.unique(df['synMethod'])))
    if apples:
        figPath = getFilePath(tu, synMethods, 'Accuracy', f"{fileTag}.{hueCol}")
    else:
        figPath = getFilePath(tu, synMethods, 'Accuracy', f"{fileTag}.{hueCol}.noapples")
        title += " (not apples-to-apples)"
    if not force and os.path.exists(figPath):
        print(f"Skipping {figPath}")
        return
    height = max(defaultHeight, len(synMethods) * 0.8)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, height))

    dfTemp = df.query("rowType == 'columnScore'")
    if dfTemp.shape[0] > 0:
        if apples:
            dfTemp = removeExtras(dfTemp)
        # dfMerged = pd.merge(dfBase, dfOther, how='inner', on = ['csvFile','targetColumn','mlMethod'])
        xaxis = 'Marginal columns quality'
        hueDf,boxColorsLoc = getHueDfAndColors(dfTemp, hueCol, boxColors, 'synMethod')
        synMethods = sorted(list(pd.unique(dfTemp['synMethod'])))
        print(figPath)
        print(title)
        print(xaxis)
        printStats(dfTemp, hueCol, "quality")
        print('just before:', list(pd.unique(dfTemp['synMethod'])))
        print('just before:', synMethods)
        sns.boxplot(x=dfTemp['rowValue'], y=dfTemp['synMethod'], palette=boxColorsLoc, hue=hueDf, order=synMethods, ax=axs[0])
        sns.set(font_scale=fontScale)
        sampleCounts = setLabelSampleCount(dfTemp['synMethod'], synMethods)
        if len(sampleCounts) == len(synMethods):
            axs[0].yaxis.set_ticklabels(setLabelSampleCount(dfTemp['synMethod'], synMethods))
        axs[0].set_xlabel(xaxis)
        # axs[0].set_xscale('function', functions=(partial(np.power, 10.0), np.log10))
        low = dfTemp['rowValue'].min()
        if hueDf is not None:
            axs[0].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        axs[0].set_xlim(max(0.8, low), 1.0)

    dfTemp = df.query("rowType == 'pairScore'")
    if dfTemp.shape[0] > 0:
        if apples:
            dfTemp = removeExtras(dfTemp)
        xaxis = 'Column pairs quality'
        hueDf,boxColorsLoc = getHueDfAndColors(dfTemp, hueCol, boxColors, 'synMethod')
        synMethods = sorted(list(pd.unique(dfTemp['synMethod'])))
        print(figPath)
        print(title)
        print(xaxis)
        printStats(dfTemp, hueCol, "quality")
        sns.boxplot(x=dfTemp['rowValue'], y=dfTemp['synMethod'], palette=boxColorsLoc, hue=hueDf, order=synMethods, ax=axs[1])
        sns.set(font_scale=fontScale)
        sampleCounts = setLabelSampleCount(dfTemp['synMethod'], synMethods)
        if len(sampleCounts) == len(synMethods):
            axs[1].yaxis.set_ticklabels(setLabelSampleCount(dfTemp['synMethod'], synMethods))
        axs[1].set_xlabel(xaxis)
        # axs[1].set_xscale('function', functions=(partial(np.power, 10.0), np.log10))
        low = dfTemp['rowValue'].min()
        if hueDf is not None:
            axs[1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        axs[1].set_xlim(max(0.8, low), 1.0)
        # axs[1].set(yticklabels = [], ylabel = None)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(figPath)
    plt.close()


def computeImprovements(dfTemp, measureType, measureField):
    targets = []
    methods = []
    for synMethod in list(pd.unique(dfTemp['synMethod'])):
        if 'syndiffix' in synMethod:
            targets.append(synMethod)
        else:
            methods.append(synMethod)
    for statType in ['median', 'average']:
        computeImprovementsWork(dfTemp, measureType, targets, methods, statType, measureField)


def computeImprovementsWork(dfTemp, measureType, targets, methods, statType, measureField):
    for target in targets:
        for method in methods:
            if measureType == 'quality':
                if statType == 'median':
                    targetErr = 1 - dfTemp[dfTemp['synMethod'] == target][measureField].median()
                    methodErr = 1 - dfTemp[dfTemp['synMethod'] == method][measureField].median()
                else:
                    targetErr = 1 - dfTemp[dfTemp['synMethod'] == target][measureField].mean()
                    methodErr = 1 - dfTemp[dfTemp['synMethod'] == method][measureField].mean()
                if targetErr > methodErr:
                    if methodErr == 0:
                        methodErr = 0.001
                    improvement = round(targetErr / methodErr, 2) * -1
                else:
                    if targetErr == 0:
                        targetErr = 0.001
                    improvement = round(methodErr / targetErr, 2)
            else:
                if statType == 'median':
                    targetTime = dfTemp[dfTemp['synMethod'] == target][measureField].median()
                    methodTime = dfTemp[dfTemp['synMethod'] == method][measureField].median()
                else:
                    targetTime = dfTemp[dfTemp['synMethod'] == target][measureField].mean()
                    methodTime = dfTemp[dfTemp['synMethod'] == method][measureField].mean()
                if targetTime > methodTime:
                    if methodTime == 0:
                        methodTime = 0.001
                    improvement = round(targetTime / methodTime, 2) * -1
                else:
                    if targetTime == 0:
                        targetTime = 0.001
                    improvement = round(methodTime / targetTime, 2)
            print(f"Improvement for {statType} of {target} over {method} = {improvement}")


def printStats(dfTemp, hueCol, measureType, measureField='rowValue'):
    if hueCol:
        dfGroupby = dfTemp.groupby(['synMethod', hueCol])[measureField].describe()
        print(f"groupby {hueCol}")
    else:
        dfGroupby = dfTemp.groupby(['synMethod'])[measureField].describe()
        computeImprovements(dfTemp, measureType, measureField)
    if dfGroupby.shape[0] == 0:
        return
    print(dfGroupby.to_string())
    pp.pprint(list(pd.unique(dfTemp['csvFile'])))
    print('Num csv files:', len(list(pd.unique(dfTemp['csvFile']))))


def getHueDfAndColors(dfTemp, hueCol, boxColors, catCol):
    if boxColors is not None:
        boxColorsLoc = boxColors.copy()
        cats = list(pd.unique(dfTemp[catCol]))
        print("cats:", cats)
        for cat in cats:
            if cat not in boxColors.keys():
                boxColorsLoc = None
                break
    else:
        boxColorsLoc = None
    if hueCol is None:
        print(f"getHueDfAndColors1: boxColors {boxColorsLoc}")
        return None, boxColorsLoc
    hues = list(pd.unique(dfTemp[hueCol]))
    if len(hues) <= 1:
        print(f"getHueDfAndColors2: boxColors {boxColorsLoc}")
        return None, boxColorsLoc
    print(f"getHueDfAndColors3: boxColors None")
    return dfTemp[hueCol], None


def getFilePath(tu, synMethods, part1, part2):
    localSorted = sorted(synMethods)
    if len(localSorted) > 6:
        init = "all_synMeasures"
    else:
        init = "_".join([str(x) for x in localSorted])
    if part2:
        return os.path.join(tu.summariesDir, f"{init}.{part1.replace(' ','_')}.by.{part2.replace(' ','_')}.png")
    else:
        return os.path.join(tu.summariesDir, f"{init}.{part1.replace(' ','_')}.png")


def main():
    fire.Fire(summarize)


if __name__ == '__main__':
    main()
