import os

f = open('synAll.R', 'w')
baseDir = os.path.join(os.environ['AB_RESULTS_DIR'])
# Configure these two and the two at lines 11 and 12:
csvInPath = os.path.join(baseDir, 'csvAb')
toDirPath = os.path.join(baseDir, 'synthpop')
script = f'''
library("synthpop")
thisDir = getwd()
fromDirPath <- "C:/paul/abData/csvAb/"
toDirPath <- "C:/paul/abData/synthpop/"
'''
f.write(script)
files = [x for x in os.listdir(csvInPath) if os.path.isfile(os.path.join(csvInPath, x))]
for fileName in files:
    script = f'''
csvFile <- '{fileName}'
csvPath = paste(fromDirPath,csvFile,sep='')
elapsedFile <- paste(csvFile,'.json',sep='')
elapsedPath = paste(toDirPath,elapsedFile,sep='')
print(csvPath)
orig <- read.csv(csvPath)
startTime <- Sys.time()
anon <- syn(orig, smoothing='spline', cart.minbucket=10)
endTime <- Sys.time()
elapsed <- endTime - startTime
print(elapsed)
print(summary(anon))
setwd(toDirPath)
write.syn(anon,file = csvFile, filetype = "csv")
elapsedJson = paste('[', elapsed, ']', sep='')
cat(elapsedJson, file=elapsedPath)
setwd(thisDir)
'''
    f.write(script)

f.close()
