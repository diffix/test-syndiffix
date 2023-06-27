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

violinPlots = False
setLabelCountsGlobal = False


def swrite(f, wstr):
    f.write(wstr)
    f.write('\n')


def summarize(measuresDir='measuresAb',
              outDir='summAb',
              withViolinPlots=False,
              doSkipMethods=True,
              dumpDataOnly=False,
              setLabelCounts=False,
              applesToApplesOnly=True,
              whatToDo='general',    # 'synDiffix'
              flush=False,       # re-gather
              force=False):      # overwrite existing plot
    violinPlots = withViolinPlots
    global setLabelCountsGlobal
    setLabelCountsGlobal = setLabelCounts
    tu = testUtils.testUtilities()
    tu.registerSummariesDirCore(outDir)
    os.makedirs(tu.summariesDir, exist_ok=True)
    dfPath = os.path.join(tu.summariesDir, "summParquet")
    rg = gatherResults.resultsGather(measuresDir=measuresDir)
    if flush is False and os.path.exists(dfPath):
        print(f"Reading dfAll from {dfPath}")
        dfAll = pd.read_parquet(dfPath)
    else:
        print(f"Gathering data...")
        rg.gather()
        dfAll = rg.dfTab
        dfAll.to_parquet(dfPath)

    if dumpDataOnly:
        dumpMlData(dfAll)
        quit()
    print(list(dfAll.columns))
    # We are removing these two tables because they have a lot of onehotencoded columns
    # which 1) we don't deal well with, and 2) we don't need such encodings in the first place
    print("Remove one hot encoded data (covtype.csv and mnist12.csv) from gathered data")
    dfAll = dfAll.query("csvFile != 'covtype.csv' and csvFile != 'mnist12.csv'")
    jobs = None
    if os.path.exists('summarize.json'):
        with open('summarize.json', 'r') as f:
            jobs = json.load(f)
    if jobs and 'ignore' in jobs:
        for synMethod in jobs['ignore']:
            print(f"Ignoring {synMethod}")
            query = f"synMethod != '{synMethod}'"
            dfAll = dfAll.query(query)
    # Make a column that tags large and small 2dim tables
    print(dfAll.columns)
    dfAll['2dimSizeTag'] = 'none'
    dfAll['2dimSizeTag'] = np.where(((dfAll['numColumns'] == 2) & (
        dfAll['numRows'] < 8000)), '7k rows', dfAll['2dimSizeTag'])
    dfAll['2dimSizeTag'] = np.where(((dfAll['numColumns'] == 2) & (
        dfAll['numRows'] > 27000)), '28k rows', dfAll['2dimSizeTag'])
    synMethods = sorted(list(pd.unique(dfAll['synMethod'])))
    print(synMethods)
    if doSkipMethods:
        skipMethods = ['copulaGan', 'gaussianCopula', 'tvae', 'syndiffix_ns']
        for skipMethod in skipMethods:
            if skipMethod in synMethods:
                synMethods.remove(skipMethod)
    print(f"synMethods after skips: {synMethods}")
    print(f"Privacy plot")
    makeCsvFiles(dfAll, tu)
    doPrivPlot(tu, dfAll, force)
    doPrivPlot(tu, dfAll, force, what='all')
    doMlPlot(tu, dfAll, force)
    doPlots(tu, dfAll, synMethods, force=force)
    if whatToDo == 'general':
        if 'syndiffix_focus' in synMethods:
            doPlots(tu, dfAll, ['syndiffix_focus', 'ctGan', 'mostly'], force=force)
        if 'syndiffix' in synMethods:
            doPlots(tu, dfAll, ['syndiffix', 'ctGan', 'mostly'], force=force)
    if not applesToApplesOnly:
        doPlots(tu, dfAll, synMethods, apples=False, force=force)
    withoutMostly = synMethods.copy()
    withoutMostly.remove('mostly')
    # doPlots(tu, dfAll, withoutMostly, force=force)
    if jobs and 'combs' in jobs:
        for job in jobs['combs']:
            doPlots(tu, dfAll, job['columns'], force=force)
    if whatToDo == 'general' and 'syndiffix' in synMethods and 'syndiffix_focus' in synMethods:
        for compareMethod in ['syndiffix', 'syndiffix_focus']:
            for synMethod in synMethods:
                if synMethod == compareMethod:
                    continue
                doPlots(tu, dfAll, [compareMethod, synMethod], force=force)
    if whatToDo == 'synDiffix' and ('syndiffix' in synMethods or 'syndiffix_focus' in synMethods):
        if 'syndiffix_focus' in synMethods:
            compareMethod = 'syndiffix_focus'
        else:
            compareMethod = 'syndiffix'
        for synMethod in synMethods:
            if synMethod == compareMethod:
                continue
            if 'syndiffix' not in synMethod:
                continue
            doPlots(tu, dfAll, [compareMethod, synMethod], force=force)
    dfBadPriv = dfAll.query("rowType == 'privRisk' and rowValue > 0.5")
    if dfBadPriv.shape[0] > 0:
        print("Bad privacy scores:")
        print(dfBadPriv[['rowValue', 'privMethod', 'targetColumn', 'csvFile', 'synMethod']].to_string)


def dumpMlData(dfAll):
    print(list(dfAll.columns))
    print(list(pd.unique(dfAll['rowType'])))
    dfMl = dfAll.query("rowType == 'synMlScore'")
    dfMl = dfMl.rename(columns={'rowValue': 'synMlScore'})
    dfMl = dfMl[['synMethod', 'targetColumn', 'csvFile', 'synMlScore', 'mlMethod']]
    print(f"Shape of dfMl: {dfMl.shape}")
    dfDiffix = dfMl.query("synMethod == 'syndiffix'")
    dfDiffix = dfDiffix[['targetColumn', 'csvFile', 'synMlScore', 'mlMethod']]
    dfDiffix = dfDiffix.sort_values(by=['csvFile', 'targetColumn', 'mlMethod'])
    print(f"Shape of dfDiffix: {dfDiffix.shape}")
    print(dfDiffix.to_string())

    dfCtGan = dfMl.query("synMethod == 'ctGan'")
    dfCtGan = dfCtGan[['targetColumn', 'csvFile', 'synMlScore', 'mlMethod']]
    dfCtGan = dfCtGan.sort_values(by=['csvFile', 'targetColumn', 'mlMethod'])
    print(f"Shape of dfCtGan: {dfCtGan.shape}")
    print(dfCtGan.to_string())

    dfMostly = dfMl.query("synMethod == 'mostly'")
    dfMostly = dfMostly[['targetColumn', 'csvFile', 'synMlScore', 'mlMethod']]
    dfMostly = dfMostly.sort_values(by=['csvFile', 'targetColumn', 'mlMethod'])
    print(f"Shape of dfMostly: {dfMostly.shape}")
    print(dfMostly.to_string())

    dfRaw = dfAll.query("rowType == 'origMlScore'")
    dfRaw = dfRaw.rename(columns={'rowValue': 'origMlScore'})
    dfRaw = dfRaw[['targetColumn', 'csvFile', 'origMlScore', 'mlMethod']]
    dfRaw = dfRaw.drop_duplicates()
    print(f"Shape of dfRaw: {dfRaw.shape}")

    dfMerged = pd.merge(dfDiffix, dfCtGan, how='inner', on=['csvFile', 'targetColumn', 'mlMethod'])
    dfMerged = dfMerged.rename(columns={'synMlScore_x': 'diffix', 'synMlScore_y': 'ctGan'})
    print(f"Shape after first merge = {dfMerged.shape}")
    dfMerged = dfMerged.sort_values(by=['csvFile', 'targetColumn', 'mlMethod'])
    print(dfMerged.to_string())

    dfMerged = pd.merge(dfMerged, dfMostly, how='inner', on=['csvFile', 'targetColumn', 'mlMethod'])
    dfMerged = dfMerged.rename(columns={'synMlScore': 'mostly'})
    dfMerged = dfMerged[['csvFile', 'targetColumn', 'diffix', 'mostly', 'ctGan', 'mlMethod']]
    print(f"Shape after second merge = {dfMerged.shape}")
    dfMerged = dfMerged.sort_values(by=['csvFile', 'targetColumn', 'mlMethod'])
    print(dfMerged.to_string())

    dfMerged = pd.merge(dfMerged, dfRaw, how='inner', on=['csvFile', 'targetColumn', 'mlMethod'])
    dfMerged = dfMerged[['csvFile', 'targetColumn', 'origMlScore', 'diffix', 'mostly', 'ctGan', 'mlMethod']]
    print(f"Shape = {dfMerged.shape}")
    print(list(dfMerged.columns))
    print("After raw merge")
    print(dfMerged.to_string())

    dfMerged.to_csv('mlScores.csv', index=False)


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

def doPlots(tu, dfIn, synMethods, apples=True, force=False):
    print(f"-------- doPlots for synMethods '{synMethods}'")
    query = ''
    for synMethod in synMethods:
        query += f"synMethod == '{synMethod}' or "
    query = query[:-3]
    print(query)
    df = dfIn.query(query)

    title = "All datasets (real and parameterized)"
    print(title)
    hueColsScatter = [None, 'mlMethodType',]
    # hueColsScatter = [None, 'mlMethodType', 'targetCardinality']
    if len(synMethods) == 2:
        for hueCol in hueColsScatter:
            makeScatter(df, tu, synMethods, hueCol, 'equalAxis', 'all', title, force)
            # makeScatter(df, tu, synMethods, hueCol, 'compressedAxis', 'all', title, force)
    hueColsBasic = [None, 'mlMethodType',]
    for hueCol in hueColsBasic:
        makeBasicGraph(df, tu, hueCol, 'all', title, force, apples=apples)
        makeBasicViolin(df, tu, 'all', title)

    title = f"Datasets with 2 columns"
    print(title)
    dfTemp = df.query(f"numColumns == 2")
    hueColsScatter = [None]
    if len(synMethods) == 2:
        for hueCol in hueColsScatter:
            makeScatter(dfTemp, tu, synMethods, hueCol, 'equalAxis', f"2col", title, force)
    hueColsBasic = [None, '2dimSizeTag',]
    for hueCol in hueColsBasic:
        makeBasicGraph(dfTemp, tu, hueCol, f"2col", title, force, apples=apples)
        makeBasicViolin(df, tu, f"2col", title)

    # Now for only the real datasets
    title = "Real datasets only"
    print(title)
    dfTemp = df.query(f"numColumns != 2")
    hueColsScatter = [None, 'mlMethodType',]
    if len(synMethods) == 2:
        for hueCol in hueColsScatter:
            makeScatter(dfTemp, tu, synMethods, hueCol, 'equalAxis', 'real', title, force)
    hueColsBasic = [None, 'mlMethodType']
    for hueCol in hueColsBasic:
        makeBasicGraph(dfTemp, tu, hueCol, 'real', title, force, apples=apples)
        makeBasicViolin(dfTemp, tu, 'real', title)


def makeScatter(df, tu, synMethods, hueCol, axisType, fileTag, title, force):
    if len(synMethods) != 2:
        return
    if hueCol:
        figPath = getFilePath(tu, synMethods, f"scatter..{hueCol}", f"{fileTag}.{axisType}")
    else:
        figPath = getFilePath(tu, synMethods, f"scatter.", f"{fileTag}.{axisType}")
    if not force and os.path.exists(figPath):
        print(f"Skipping {figPath}")
        return
    print(f"    Scatter plots")
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    for ax0, ax1, score, doLog, limit in zip([0, 0, 1, 1], [0, 1, 0, 1], ['columnScore', 'pairScore', 'synMlScore', 'elapsedTime', ], [False, False, False, True, ], [None, None, [0, 1], None, ]):
        dfTemp = df.query(f"rowType == '{score}'")
        if dfTemp.shape[0] > 0 and len(list(pd.unique(dfTemp['synMethod']))) == 2:
            dfBase = dfTemp.query(f"synMethod == '{synMethods[0]}'")
            dfOther = dfTemp.query(f"synMethod == '{synMethods[1]}'")
            print(f"Methods {synMethods}, score {score}:")
            makeScatterWork(dfBase, dfOther, synMethods, axs[ax0][ax1], score, hueCol, doLog, limit, axisType)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(figPath)
    plt.close()


def makeScatterWork(dfBase, dfOther, synMethods, ax, score, hueCol, doLog, limit, axisType):
    legendDone = False
    dfMerged = pd.merge(dfBase, dfOther, how='inner', on=['csvFile', 'targetColumn', 'mlMethod'])
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
    hueDf = getHueDf(dfMerged, hueCol)
    hue_order = sorted(list(pd.unique(dfMerged[hueCol]))) if hueCol else None
    g = sns.scatterplot(x=dfMerged['rowValue_x'], y=dfMerged['rowValue_y'], hue=hueDf, hue_order=hue_order, s=15, ax=ax)
    if axisType == 'equalAxis':
        low = min(dfMerged['rowValue_x'].min(), dfMerged['rowValue_y'].min())
        high = max(dfMerged['rowValue_x'].max(), dfMerged['rowValue_y'].max())
        low = max(low, 0)
    else:
        low = max(dfMerged['rowValue_x'].min(), dfMerged['rowValue_y'].min())
        high = min(dfMerged['rowValue_x'].max(), dfMerged['rowValue_y'].max())
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
        ax.set_xlabel(f"{synMethods[0]} {score} (log)")
        ax.set_ylabel(f"{synMethods[1]} {score} (log) ({dfMerged.shape[0]})")
    else:
        ax.set_xlabel(f"{synMethods[0]} {score}")
        ax.set_ylabel(f"{synMethods[1]} {score} ({dfMerged.shape[0]})")


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


def getBestSyndiffix(df):
    dfNonFocus = df.query("synMethod == 'syndiffix'")
    dfFocus = df.query("synMethod == 'syndiffix_focus'")
    if dfNonFocus.shape[0] == 0 or dfFocus.shape[0] == 0:
        return df
    dfMerged = pd.merge(dfNonFocus, dfFocus, how='inner', on=['csvFile', 'targetColumn', 'mlMethod'])
    dfMerged['rowValue'] = np.where(dfMerged['rowValue_x'] > dfMerged['rowValue_y'],
                                    dfMerged['rowValue_x'], dfMerged['rowValue_y'])
    dfMerged['synMethod'] = 'syndiffix_best'
    df1 = df[['synMethod', 'rowValue']]
    df2 = dfMerged[['synMethod', 'rowValue']]
    return pd.concat([df1, df2], axis=0)


def doMlPlot(tu, df, force, hueCol=None):
    figPath = os.path.join(tu.summariesDir, 'ml.png')
    if not force and os.path.exists(figPath):
        print(f"Skipping {figPath}")
        return

    # dfTemp = df.query("rowType == 'synMlScore'")
    dfTemp = df.query("rowType == 'synMlScore' and numColumns > 2")

    # dfTemp = getBestSyndiffix(dfTemp)
    print("doMlPlot stats:")
    printStats(dfTemp, hueCol, "quality")
    xaxis = 'ML scores'
    hueDf = getHueDf(dfTemp, hueCol)
    sns.boxplot(x=dfTemp['rowValue'], y=dfTemp['synMethod'], hue=hueDf)
    plt.xlim(0, 1)
    plt.xlabel(xaxis)
    plt.savefig(figPath)
    plt.close()


def doPrivPlot(tu, df, force, what='lowBounds', hueCol=None):
    if what == 'lowBounds':
        dfTemp = df.query("rowType == 'privRisk'")
        xaxis = 'Privacy Risk'
        printStats(dfTemp, hueCol, "priv high confidence")
        figPath = os.path.join(tu.summariesDir, 'priv.png')
    else:
        dfTemp = df.query("rowType == 'privRisk' or rowType == 'privRiskHigh'")
        xaxis = 'Privacy Risk (including low confidence scores)'
        printStats(dfTemp, hueCol, "priv high and low confidence")
        figPath = os.path.join(tu.summariesDir, 'privLowConf.png')
    if dfTemp.shape[0] == 0:
        return
    synMethods = sorted(list(pd.unique(dfTemp['synMethod'])))
    hueDf = getHueDf(dfTemp, hueCol)
    sns.boxplot(x=dfTemp['rowValue'], y=dfTemp['synMethod'], hue=hueDf, order=synMethods)
    plt.tight_layout()
    plt.xlim(0,1)
    plt.xticks([0.01,0.1,0.2,0.5,1.0],['0.01','0.1','0.2','0.5','1.0'])
    #plt.xscale('symlog')
    plt.xlabel(xaxis)
    plt.savefig(figPath)
    plt.close()


def makeBasicGraph(df, tu, hueCol, fileTag, title, force, apples=True):
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
    height = max(5, len(synMethods) * 1.8)
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, height))

    dfTemp = df.query("rowType == 'columnScore'")
    if dfTemp.shape[0] > 0:
        if apples:
            dfTemp = removeExtras(dfTemp)
        # dfMerged = pd.merge(dfBase, dfOther, how='inner', on = ['csvFile','targetColumn','mlMethod'])
        xaxis = 'Marginal columns quality'
        hueDf = getHueDf(dfTemp, hueCol)
        print(figPath)
        print(title)
        print(xaxis)
        printStats(dfTemp, hueCol, "quality")
        sns.boxplot(x=dfTemp['rowValue'], y=dfTemp['synMethod'], hue=hueDf, order=synMethods, ax=axs[0][0])
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
        hueDf = getHueDf(dfTemp, hueCol)
        print(figPath)
        print(title)
        print(xaxis)
        printStats(dfTemp, hueCol, "quality")
        sns.boxplot(x=dfTemp['rowValue'], y=dfTemp['synMethod'], hue=hueDf, order=synMethods, ax=axs[0][1])
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
        hueDf = getHueDf(dfTemp, hueCol)
        print(figPath)
        print(title)
        print(xaxis)
        printStats(dfTemp, hueCol, "quality")
        sns.boxplot(x=dfTemp['rowValue'], y=dfTemp['synMethod'], hue=hueDf, order=synMethods, ax=axs[1][0])
        sampleCounts = setLabelSampleCount(dfTemp['synMethod'], synMethods)
        if len(sampleCounts) == len(synMethods):
            axs[1][0].yaxis.set_ticklabels(setLabelSampleCount(dfTemp['synMethod'], synMethods))
        axs[1][0].set_xlabel(xaxis)
        low = dfTemp['rowValue'].min()
        if hueDf is not None:
            axs[1][0].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        axs[1][0].set_xlim(max(0, low), 1.0)

        xaxis = 'ML Penalty'
        hueDf = getHueDf(dfTemp, hueCol)
        print(figPath)
        print(title)
        print(xaxis)
        printStats(dfTemp, hueCol, "quality")
        sns.boxplot(x=dfTemp['mlPenalty'], y=dfTemp['synMethod'], hue=hueDf, order=synMethods, ax=axs[1][1])
        sampleCounts = setLabelSampleCount(dfTemp['synMethod'], synMethods)
        if len(sampleCounts) == len(synMethods):
            axs[1][1].yaxis.set_ticklabels(setLabelSampleCount(dfTemp['synMethod'], synMethods))
        axs[1][1].set_xlabel(xaxis)
        low = dfTemp['rowValue'].min()
        if hueDf is not None:
            axs[1][1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        axs[1][1].set_xlim(max(0, low), 1.0)

    dfTemp = df.query("rowType == 'elapsedTime'")
    if dfTemp.shape[0] > 0:
        if apples:
            dfTemp = removeExtras(dfTemp)
        xaxis = 'Elapsed Time (seconds) (log)'
        hueDf = getHueDf(dfTemp, hueCol)
        print(figPath)
        print(title)
        print(xaxis)
        printStats(dfTemp, hueCol, "time")
        sns.boxplot(x=dfTemp['rowValue'], y=dfTemp['synMethod'], hue=hueDf, order=synMethods, ax=axs[2][0])
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


def computeImprovements(dfTemp, measureType):
    targets = []
    methods = []
    for synMethod in list(pd.unique(dfTemp['synMethod'])):
        if 'syndiffix' in synMethod:
            targets.append(synMethod)
        else:
            methods.append(synMethod)
    for statType in ['median', 'average']:
        computeImprovementsWork(dfTemp, measureType, targets, methods, statType)


def computeImprovementsWork(dfTemp, measureType, targets, methods, statType):
    for target in targets:
        for method in methods:
            if measureType == 'quality':
                if statType == 'median':
                    targetErr = 1 - dfTemp[dfTemp['synMethod'] == target]['rowValue'].median()
                    methodErr = 1 - dfTemp[dfTemp['synMethod'] == method]['rowValue'].median()
                else:
                    targetErr = 1 - dfTemp[dfTemp['synMethod'] == target]['rowValue'].mean()
                    methodErr = 1 - dfTemp[dfTemp['synMethod'] == method]['rowValue'].mean()
                if targetErr > methodErr:
                    if methodErr == 0: methodErr = 0.001
                    improvement = round(targetErr / methodErr, 2) * -1
                else:
                    if targetErr == 0: targetErr = 0.001
                    improvement = round(methodErr / targetErr, 2)
            else:
                if statType == 'median':
                    targetTime = dfTemp[dfTemp['synMethod'] == target]['rowValue'].median()
                    methodTime = dfTemp[dfTemp['synMethod'] == method]['rowValue'].median()
                else:
                    targetTime = dfTemp[dfTemp['synMethod'] == target]['rowValue'].mean()
                    methodTime = dfTemp[dfTemp['synMethod'] == method]['rowValue'].mean()
                if targetTime > methodTime:
                    if methodTime == 0: methodTime = 0.001
                    improvement = round(targetTime / methodTime, 2) * -1
                else:
                    if targetTime == 0: targetTime = 0.001
                    improvement = round(methodTime / targetTime, 2)
            print(f"Improvement for {statType} of {target} over {method} = {improvement}")


def printStats(dfTemp, hueCol, measureType):
    if hueCol:
        dfGroupby = dfTemp.groupby(['synMethod', hueCol])['rowValue'].describe()
        print(f"groupby {hueCol}")
    else:
        dfGroupby = dfTemp.groupby(['synMethod'])['rowValue'].describe()
        computeImprovements(dfTemp, measureType)
    if dfGroupby.shape[0] == 0:
        return
    print(dfGroupby.to_string())
    pp.pprint(list(pd.unique(dfTemp['csvFile'])))
    print('Num csv files:', len(list(pd.unique(dfTemp['csvFile']))))


def getHueDf(dfTemp, hueCol):
    if hueCol is None:
        return None
    hues = list(pd.unique(dfTemp[hueCol]))
    if len(hues) <= 1:
        return None
    return dfTemp[hueCol]


def getViolinDf(dfValues, dfLabels, order):
    dfConcat = pd.concat([dfValues, dfLabels], axis=1, ignore_index=True)
    dfConcat.columns = ['val', 'label']
    order.sort(reverse=True)
    allDf = [None for _ in range(len(order))]
    for index, col in zip(range(len(order)), order):
        allDf[index] = dfConcat.query(f"label == '{col}'")['val']
        allDf[index] = allDf[index].dropna()
    return allDf


def makeBasicViolin(df, tu, fileTag, title):
    if violinPlots is False:
        return
    print("    Violin plots")
    synMethods = sorted(list(pd.unique(df['synMethod'])))
    dfTemp = df.query("rowType == 'columnScore'")
    dfTemp = removeExtras(dfTemp)
    xaxis = 'Marginal columns quality'
    height = max(5, len(synMethods) * 1.5)
    fig, axs = plt.subplots(2, 2, figsize=(10, height))
    # At this point, dfTemp['rowValue'] is the data, and dfTemp['synMethod'] is the labels
    # I need to make a dataframe that has the labels as columns...
    dfViolin = getViolinDf(dfTemp['rowValue'], dfTemp['synMethod'], synMethods)
    pos = list(range(1, len(synMethods) + 1))
    quant = [[0.25, 0.5, 0.75] for _ in range(len(dfViolin))]
    axs[0][0].violinplot(dfViolin, pos, vert=False, quantiles=quant)
    axs[0][0].set_yticks(pos)
    axs[0][0].yaxis.set_ticklabels(setLabelSampleCount(dfTemp['synMethod'], synMethods))
    axs[0][0].set_xlabel(xaxis)

    dfTemp = df.query("rowType == 'pairScore'")
    dfTemp = removeExtras(dfTemp)
    xaxis = 'Column pairs quality'
    dfViolin = getViolinDf(dfTemp['rowValue'], dfTemp['synMethod'], synMethods)
    pos = list(range(1, len(synMethods) + 1))
    quant = [[0.25, 0.5, 0.75] for _ in range(len(dfViolin))]
    axs[0][1].set_yticks(pos)
    axs[0][1].yaxis.set_ticklabels(setLabelSampleCount(dfTemp['synMethod'], synMethods))
    axs[0][1].violinplot(dfViolin, pos, vert=False, quantiles=quant)

    dfTemp = df.query("rowType == 'synMlScore'")
    dfTemp = removeExtras(dfTemp)
    xaxis = 'ML Score'
    dfViolin = getViolinDf(dfTemp['rowValue'], dfTemp['synMethod'], synMethods)
    pos = list(range(1, len(synMethods) + 1))
    quant = [[0.25, 0.5, 0.75] for _ in range(len(dfViolin))]
    axs[1][0].violinplot(dfViolin, pos, vert=False, quantiles=quant)
    axs[1][0].set_yticks(pos)
    axs[1][0].yaxis.set_ticklabels(setLabelSampleCount(dfTemp['synMethod'], synMethods))
    axs[1][0].set_xlabel(xaxis)
    low = dfTemp['rowValue'].min()
    axs[1][0].set_xlim(max(0, low), 1.0)

    dfTemp = df.query("rowType == 'elapsedTime'")
    dfTemp = removeExtras(dfTemp)
    xaxis = 'Elapsed Time (seconds)'
    dfViolin = getViolinDf(dfTemp['rowValue'], dfTemp['synMethod'], synMethods)
    pos = list(range(1, len(synMethods) + 1))
    quant = [[0.25, 0.5, 0.75] for _ in range(len(dfViolin))]
    axs[1][1].set_yticks(pos)
    axs[1][1].violinplot(dfViolin, pos, vert=False, quantiles=quant)
    axs[1][1].yaxis.set_ticklabels(setLabelSampleCount(dfTemp['synMethod'], synMethods))
    axs[1][1].set_xscale('log')  # zzzz
    axs[1][1].set_xlabel(xaxis)

    figPath = getFilePath(tu, synMethods, 'basicViolin', fileTag)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(figPath)
    plt.close()


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
