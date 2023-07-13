ns = 3
nj = 9

tot = ns * nj
check = []

for i in range(tot):
    jobid = i % nj
    sample = int(i / nj)
    checkVal = str(sample) + '.' + str(jobid)
    if checkVal in check:
        print(f"Failed on {i}, {sample}, {jobid}")
        quit()
    check.append(checkVal)
    print(f"{i}, job:{jobid}, sample:{sample}")
