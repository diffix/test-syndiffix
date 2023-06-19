import os

# Get the list of all files in the directory
files = os.listdir()

deleteFiles = []
for filename in files:
    if filename[-3:] != 'out':
        continue
    # Open the file for reading
    with open(filename, 'r') as file:
        # Read all lines of the file
        lines = file.readlines()

    for line in lines:
        if 'SUCCESS' in line:
            deleteFiles.append(filename)
            break

for filename in deleteFiles:
    os.remove(filename)
