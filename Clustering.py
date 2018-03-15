import numpy as np
import os
import random
import math


# pull data from csv
def ingest_data(root,f):
    file = root+f
    info = []
    with open(file) as fp:
        lines = fp.readlines()
        for line in lines:
            info.append(line.split(" "))
    return info


# pull data for each file in clustering-data/ folder
def input_data():
    data = []
    source = 'clustering-data/'
    for root, dirs, filenames in os.walk(source):
        for f in filenames:
            data.append(ingest_data(root,f))
    return data


# define distance measure
def distance(x,y):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))


# Step 1 - Pick K random points as cluster centers called centroids.
# Step 2 - Assign each xi to nearest cluster by calculating its distance to each centroid.
# Step 3 - Find new cluster center by taking the average of the assigned points.
# Step 4 - Repeat Step 2 and 3 until none of the cluster assignments change.
def main():
    info = []
    info.append(input_data())

    # picking a random point as a centroid - 1 from each file
    randint = random.randint(0,59)
    centroid_1 = np.asarray(info[0][0][randint][1:]).astype(float)
    centroid_2 = np.asarray(info[0][0][randint+1][1:]).astype(float)

    # finding the euclidean distance between 2 points
    dist = distance((centroid_1[1:]),centroid_2[1:])
    print(dist)

if __name__ == "__main__":
    main()