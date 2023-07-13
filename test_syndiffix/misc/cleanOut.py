import os

# Get the list of all files in the directory
files = os.listdir()

deleteFiles = []
successes = {}
for filename in files:
    if filename[-3:] != 'out':
        continue
    # Open the file for reading
    with open(filename, 'r') as file:
        # Read all lines of the file
        lines = file.readlines()

    for line in lines:
        if 'SUCCESS' in line:
            if line in successes:
                successes[line] += 1
            else:
                successes[line] = 1
            deleteFiles.append(filename)
            break

for filename in deleteFiles:
    os.remove(filename)

print(f"Cleaned out {len(deleteFiles)} files")
for key, num in successes.items():
    print(f"{key}: {num}")
