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
'''

def oneSynthPopJob(jobNum=0, expDir='exp_base', force=False):
    tu = testUtils.testUtilities()
    tu.registerExpDir(expDir)
    scripts = tu.getSynthpopScripts()
    if len(scripts) < jobNum:
        print(f"ERROR: jobNum {jobNum} bigger than number of scripts {len(scripts)}")
        sys.exit()
    scriptPath = os.path.join(tu.synthpopScriptsDir, scripts[jobNum])
    print(f"Running oneSynthPopJob with jobNum {jobNum}, script {scripts[jobNum]}")
    subprocess.run(['R', 'CMD', 'BATCH', f"{scriptPath}"])

def main():
    fire.Fire(oneSynthPopJob)


if __name__ == '__main__':
    main()
