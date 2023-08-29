import os
import sys
import subprocess
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import testUtils
import fire
import pprint
import sdmTools

pp = pprint.PrettyPrinter(indent=4)

''' This is used in SLURM to run one synthpop job

    The R package "synthpop" must be installed first:
        If you don't have root privledges:
            Create directory `~/R/.lib/`
            Run R and do `install.packages("synthpop", lib="~/R/.lib/")`
            Do `export R_LIBS_USER="$HOME/R/.lib/:/usr/local/lib/R/site-library/"`
                (or whatever is appropriate for your setup)
                (can put this in ~\.bash_profile)
'''

def oneSynthPopJob(jobNum=0, expDir='exp_base', force=False):
    tu = testUtils.testUtilities()
    tu.registerExpDir(expDir)
    scripts = tu.getSynthpopScripts()
    if len(scripts) < jobNum:
        print(f"ERROR: jobNum {jobNum} bigger than number of scripts {len(scripts)}")
        sys.exit()
    scriptPath = os.path.join(tu.synthpopScriptsDir, scripts[jobNum])
    print(f"Running oneSynthPopJob with jobNum {jobNum}, script {scriptPath}")
    subprocess.run(['Rscript', '--verbose', f"{scriptPath}"])

def main():
    fire.Fire(oneSynthPopJob)


if __name__ == '__main__':
    main()
