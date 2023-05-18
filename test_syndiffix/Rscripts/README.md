
# Scripts for running synthpop in R

Run `makeRscript.py` to build `synAll.R`.

Run `synAll.R` from Rstudio. This reads in the csv files from (default directory) `csvAb`, and puts the output in directory `synthpop`.

Run `extractSynthpop.py` to extract the needed data from `synthpop`, and place it in `.json` files that the measurement software can read.