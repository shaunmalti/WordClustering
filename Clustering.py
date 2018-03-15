import numpy as np
import os
import random


def ingest_data(root,f):
    file = root+f
    info = []
    with open(file) as fp:
        lines = fp.readlines()
        for line in lines:
            info.append(line.split(" "))
    info = fix_data(info)
    return info

def fix_data(info):
    fixed = dict()
    for i in range(0,len(info)):
        fixed[info[i][0]] = info[i][1:]
    return fixed

def input_data():
    data = []
    source = 'clustering-data/'
    for root, dirs, filenames in os.walk(source):
        for f in filenames:
            data.append(ingest_data(root,f))

    return data

def main():
    info = input_data()

# Step 1 - Pick K random points as cluster centers called centroids.
# Step 2 - Assign each xi to nearest cluster by calculating its distance to each centroid.
# Step 3 - Find new cluster center by taking the average of the assigned points.
# Step 4 - Repeat Step 2 and 3 until none of the cluster assignments change.

    # picking K random points as centroids - 1 from each file
    for i in range(0,4):
        randint = random.randint(0,59)


if __name__ == "__main__":
    main()