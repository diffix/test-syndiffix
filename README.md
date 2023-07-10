## Environment variables

* `AB_RESULTS_DIR` is the path to the test configuration directory
* `AB_PYTHON_DIR` is the path to the directory containing the python test programs
* `AB_SHARP_DIR` is the path to the directory containing AB Sharp SynDiffix code (the `adaptive-buckets-sharp` directory)

## Tests directory structure

These tests all assume a root directory whose location is defined by the environment variable `AB_RESULTS_DIR`. The `AB_RESULTS_DIR` has a number of default subdirectories (which must be configured prior to running the tests):

There are eight different types of directories under the root directory:

* csv library: contains CSV datasets for testing. The datasets must be split into `train` and `test` files, each containing half of the data, randomly selected (see `misc/splitFile.py`). These two sets are in directories labeled `train` and `test`.
* synthetic data building results: contains the results of building synthetic data. Has one sub-directory per method (or different set of method parameters).
* measurement results: contains the results of measuring data quality from the synthetic data.
* measurement summaries: contains graphs that summarize the measurements
* run commands: contains the commands used to run the tests, including SLURM jobs
* original ML measures: the ML model scores against the original data
* temporary directory for the separate measurement samples for original ML measures
* temporary directory for the separate measurement samples for the ML measures of the synthetic data

The code is setup so that there is one group of these five directories per distinct set of CSV files. They can be named anything, but by convention are csvXXX, resultsXXX, measuresXXX, summXXX, runXXX, origMlXXX, measuresXXXTemp, and origMLXXXTemp.

### Synthetic data methods

To build synthetic data with SynDiffix, `github/diffix/syndiffix` needs to be installed.

We use the SDV library to build synthetic with learning-based methods (see `sdmManager.py`, and `sdmTools.py`).

## Creating CSV datasets

`quicktester.py` can build CSV datasets according to a variety of specifications.

Other than this, one can put any CSV dataset they wish.

## Building synthetic data

`quicktester.py` can also build synthetic data and run quality measures on it.

`oneModel.py` is used to build synthetic data on a SLURM cluster.

### Running syndiffix with focus columns

To build synthetic data with focus columns, it is necessary to first run `sdmManager.py makeFocusRuns`. This will build the files `focusBuildJobs.json` and `batchFocus` in the run directory. Then run `sbatch batchFocus`. This will place the resulting output in the directory `syndiffix_focus` in the results directory. (Note that prior to running `sdmManager.py makeFocusRuns`, you must have already run `sdmManager.py makeMlRuns`. This is because `makeFocusRuns` requires the file `mlJobs.json`.)

### Running syndiffix with features measures

This is currently just test software (prior to integrating features measure into syndiffix).

Run `sdmManager.py makeFeatures --featuresDir=featuresDir --featuresType=featuresType` where <featuresDir> is the directory holding the features json files, and <featuresType> is 'univariate' or 'ml' or whatever else we decide. This creates the featuresDir, a file called `featuresJobs.json` in the runs directory, a SLURM batchfile called `batch_<featuresType>` in the runs directory, and a directory `logs_<featuresType>` in the runs directory.

Run `sbatch batch_<featuresType>`. This creates the SLURM jobs with `oneFeaturesJob.py`

## Measuring quality

`quicktester.py` can measure quality.

To do measures on the SLURM cluster, we have the following workflow:

* Use `sdmManager.py updateCsvInfo` whenever new tables are added to a csv directory, or when code has changed and you want to rerun the original ML measures. This creates the file `csvOrder.json` in the measures directory. Note that if a table is removed, then `csvOrder.json` should be removed first. Also remove `focusColumns.json` in the runs directory before running this.
* Run `sdmManager.py makeOrigMlRuns` to create `batchOrigMl`. Select a temporary directory to hold the measure samples.
* Run `sbatch batchOrigMl` to run ML measures over the original (not synthesized) datasets. The purpose of this is to determine which ML measures (i.e. column and method) have the best quality. These make the measures to use for comparison with synthetic data. This creates the original ML measures directory and populates it with one json file per model. Note that `batchOrigMlRuns` creates multiple measures per table/column/method combination, each such measure in a separate file.
* Run `sdmManager.py mergeMlMeasures` to select the best score of multiple ml measures, create a `json` file containing that score, and place them in the original ML measures directory.
* Run `sdmManager.py makeMlRuns` to build the SLURM configuration information for doing ML measures (creates the files `mlJobs.json`, `batchMl`, and `batchMl.sh` in the run commands directory).
  * The cmd line parameter `--synMethod=method` can be used to limit the created jobs to those of the synMethod only. In any event, `makeMlRuns` will not schedule runs if measures already exist. You must manually remove existing measures if you want to rerun them.
  * The cmd line parameter `--limitToFeatures=True|False` determines whether the measure is made over the entire table, or only over the K features found with `makeFeatures`. If `--limitToFeatures=True`, then the file `gatherFeatures.sh` is additionally created in the runs directory.
  * NOTE: `makeMlRuns` needs to be run on a machine with more memory than the "submit" machines. Suggest pinky03 or brain03.
* Run `batchMl.sh` to do the ML measures. Note that this makes multiple measures per table/column/method combination.
* Run `sdmManager.py mergeMlMeasures` to select the best score of multiple ml measures, create a `json` file containing that score, and move them to the measures directory.
* In the run commands directory, do `sbatch batchMl` to do the ML measures
* Run `sdmManager.py makeQualRuns` to build the SLURM configuration information for doing quality measures (creates the file `batchQual` in the run commands directory)
* In the run commands directory, do `sbatch batchQual` to do the 1dim and 2dim quality measures.
* Run `summarize.py` to summarize the performance measures and place them in various plots.

    doPlots(tu, dfAll, ['syndiffix_multi', 'syndiffix', 'ctGan', 'mostly'], force=force)
## Measuring privacy

We use the python package `anonymeter` to measure privacy. This requires that we split all of our data files into `train` and `test` subsets. The basic idea is that `train` is synthesized, and then we use `test` as a control to test the effectiveness of attacks.

* Run `misc/splitFiles.py` to generate the csv split files. These are placed into directories `train` and `test` under `csvAb` respectively
* Generate synthesized data (e.g. using `oneModel.py`) from `train` and put them in some directory (`resultsAb` by default).
* (Run `sdmManager.py updateCsvInfo` if you haven't before.)
* Run `sdmManager.py makePrivRuns` to create a jobs file `privJobs.json` and a batch script `batchPriv`, both placed in the `runAb` directory. Do `sbatch batchPriv`. This places the privacy measures in the `measuresAb` directory.

## Summarizing quality and privacy

`summarize.py` reads in data from a measures directory and produces a set of summary graphs. Note that this was particularly designed to compare different PPoC parameter settings with each other.

-----------------------------------------------------------

# Documentation for early development of PPoC (retired)

### `csvLib`

The tests run off of prebuilt tables in `.csv` files. These files are by default in the directory `csvLib`. They can be in another directory as well.

### `synResults`

This contains the output of the adaptive buckets programs. This directory has its own structure defined below. The test routines `syntest.py` and `quicktester.py` (among possibly others) by default place their output here.

Note that `syntest.py` operates either with command line parameters or from a `json` config file. The file can be found in the `AB_RESULTS_DIR`, and by default is named `syntest.json`. An example of the file can be found in `sampleConfigs`.

### `synMeasures`

This contains the output of tests that measure the quality of the data in `synResults`.

Note that `synmeasure.py` operates either with command line parameters or from a `json` config file. The file can be found in the `AB_RESULTS_DIR`.  An example of the file can be found in `sampleConfigs/synmeasure.json`.

## Structure of data in `synResults`

Each test operates over a given data source, with given versions of the algorithms (forest, harvest, clustering), given parameters for each algorithm, and the implementation (python or F#).

`synResults` contains one directory for each combination of versions, parameters, and implementation. The directory name is formatted as:

`implementation.for_param_version.har_version.cl_param_version.md_version`

`implementation` is either 'py' or 'fs' (for python or F#). Later we will have 'pg' as well.

Each `param` is a tag defining the parameter set used for the respective algorithm (`for` for forest, `har` for harvest, `cl` for clustering, and `md` for microdata). (Note that at the moment, the default parameters have the tag `g0`.)

Each `version` is the version of the algorithm (`v1`, `v2`, ...).

An example is `py.for_g0_v3.har_v5.cl_g0_v1.md_v1`.

## Structure of results file names

The json files containing the results have the format:

`sourceDataFile.directoryName.json`

An example is `4dimAirportCluster.csv.py.for_g0_v3.har_v5.cl_g0_v1.md_v1.json`.

## Structure of the results file

The results file produced by `controller.py` has the following structure:

```
{
    "elapsedTime": 0.5984406471252441,
    "colNames": [
        "'c0'",
        "'c1'",
        "'c2'",
        "'c3'"
    ],
    "originalRows": [
        [
            "'a'",
            "'b'",
            0.9787379841057392,
            2.240893199201458
        ],
        [
            "'a'",
            "'b'",
            0.9500884175255894,
            -0.1513572082976979
        ],
...
    ],
    "synRows": [
        [
            "'a'",
            "'a'",
            -0.48094763821008923,
            1.3420975576293672
        ],
        [
            "'a'",
            "'a'",
            1.6458741734699058,
            -1.6682721485003236
        ],
...
    "params": {
        "forest": {
            "sing": 10,
            "range": 50,
            "lcf": 5,
            "dependence": 10,
            "threshSd": 1.0,
            "noiseSd": 1.0,
            "lcdBound": 2,
            "pName": "g0",
            "version": "v3"
        },
        "cluster": {
            "thresholds": [
                null,
                null,
                0.05,
                0.2,
                0.35
            ],
            "pName": "g0"
        },
        "baseConfig": "4dimDependentTextWithNoise.csv",
        "clustered": true,
        "fileName": "4dimDependentTextWithNoise.csv.py.for_g0_v3.har_v5.cl_g0_v1.md_v1",
        "harvest": {
            "version": "v5"
        },
        "microdata": {
            "version": "v1"
        }
    },
... other stuff
}
```

## Maintenance

### Formatting

Python code is formatted using:

```
autopep8 --in-place --recursive test_syndiffix/
```
