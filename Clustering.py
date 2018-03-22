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
def find_distance(x,y):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))

def find_minmax(info):
    min = 0
    max = 0
    for i in range(0,len(info)):
        for j in range(0,len(info[i])):
            for ij in range(1,len(info[i][j])):
                if min > float(info[i][j][ij]):
                    min = float(info[i][j][ij])
                if max < float(info[i][j][ij]):
                    max = float(info[i][j][ij])
    return min,max


def fill_centroid(Min,Max):
    centroid = []
    for j in range(0, 300):
        centroid.append(random.uniform(Min, Max))
    return centroid


def grouped_distance(centroids,info):
    data = np.array(info)
    for i in range(0,len(data)):
        for j in range(0,len(data[i])):
            res_array = []
            # perform distance for each centroid
            for w in range(0,len(centroids)):
                res_array.append(find_distance(centroids[w],map(float,data[i][j][1:])))
            min_dist = res_array.index(min(res_array))
            # append index of cluster that is closest
            data[i][j].append(min_dist)
            # should change this to append the features to a new array, with index corresponding to
            # the centroid
    return data

def main():
    k = 4
    info = input_data()

    # picking a random point as a centroid - 1 from each file
    # randint = random.randint(0,59)
    # centroid_1 = np.asarray(info[0][randint][1:]).astype(float)
    # centroid_2 = np.asarray(info[0][randint+1][1:]).astype(float)

    # finding the euclidean distance between 2 points
    # dist = distance((centroid_1[1:]),centroid_2[1:])
    # print(dist)

    # defining the number of created centroids
    centroids = []
    Min,Max = find_minmax(info)
    for i in range(0,k):
        centroids.append(fill_centroid(Min,Max))

    labelled_info = grouped_distance(centroids,info)

    # next step is to get averaged value of cluster


# starting with 4 clusters -- 1 for each category
# process:
# randomly choose k points - create k points with 300 randomly generated features each
# calculate distance of points to all other points - placing closest into bins
# will have 4 bins at the end of an iteration
# take average of bins, that is the new centroid


if __name__ == "__main__":
    main()