import numpy as np
import os
import random
import operator as op
import functools
import itertools
import matplotlib.pyplot as plt
import scipy.spatial.distance as scp

# pull data from csv
def ingest_data(root,f,i):
    file = root+f
    info = []
    x = 0
    with open(file) as fp:
        lines = fp.readlines()
        for line in lines:
            info.append(line.split(" "))
            info[x][0] = i
            x += 1
    return info


# pull data for each file in clustering-data/ folder
def input_data(option):
    i = 0
    data = []
    source = 'clustering-data/'
    for root, dirs, filenames in os.walk(source):
        for f in filenames:
            i += 1
            # append the i as a label to the data, on a different dimension
            data.append(ingest_data(root,f,i))
    if option == "2":
        normalised = Normalise(data)
        return normalised
    return data


# perform normalisation of input data on a class by class basis
def Normalise(data):
    new_data = [[] for i in range(len(data))]
    output_array = [[] for i in range(len(data))]
    for i in range(0,len(data)):
        for j in range(0,len(data[i])):
            new_data[i].append(list(map(float,data[i][j][1:])))
    new_array = np.asarray(new_data)
    x = 1
    for i in range(0,len(new_data)):
        new_array[i] /= np.linalg.norm(new_array[i])
        for j in range(0,len(new_array[i])):
            output_array[i].append([x]+list(new_array[i][j]))
        x+=1
    return output_array


# define distance measure and carry out distance calculation
def find_distance(x,y,option):
    if option == "1":
        return np.linalg.norm(x-y)
    elif option == "2":
        return scp.cityblock(x,y)
    elif option == "3":
        return scp.cosine(x,y)


# return max of a nested list's column, to be used
# in generating centroids
def get_max_by_col(li, col):
    return max(li, key=lambda x: x[col])[col]

# return min of a nested list's column, to be used
# in generating centroids
def get_min_by_col(li, col):
    return min(li, key=lambda x: x[col])[col]

# place all data in one nested list and find min and max
# of each row for centroid generation
def find_minmax(info):
    total_array = []
    array = np.zeros(shape=(300,2))
    for i in range(0,len(info)):
        for j in range(0,len(info[i])):
            total_array.append(info[i][j][1:])
    for i in range (0,len(total_array[0])):
        array[i][1] = get_max_by_col(total_array,i)
        array[i][0] = get_min_by_col(total_array,i)
    return array


# fill centroids depending on minimum and maximum of each column
def fill_centroid(min_max_array):
    centroid = np.zeros(300)
    for i in range(0, len(min_max_array)):
        centroid[i] = random.uniform(min_max_array[i][0],min_max_array[i][1])
    return centroid


# find the distances from each centroid to each point and pick the smallest
def grouped_distance(centroids,info,k,option):
    data = np.array(info)
    list_new = [[] for i in range(k)]
    for i in range(0,len(data)):
        for j in range(0,len(data[i])):
            res_array = []
            # perform distance for each centroid
            for w in range(0,len(centroids)):
                res_array.append(find_distance(centroids[w],list(map(float,data[i][j][1:])),option))
            min_dist = res_array.index(min(res_array))
            list_new[min_dist].append(data[i][j])
    return list_new


# perform combination calculation for precision, recall and f-score
def ncr(n, r):
    r = min(r, n - r)
    numer = functools.reduce(op.mul, range(n, n - r, -1), 1)
    denom = functools.reduce(op.mul, range(1, r + 1), 1)
    return numer // denom


# get both true and false positives
def Positives(data_length):
    cent_lens = []
    totalpos = []
    for i in range(0, len(data_length)):
        cent_lens.append(len(data_length[i]))
        totalpos.append(ncr(cent_lens[i], 2))
    return sum(totalpos)

# get true positives
def TruePositives(data_length,k):
    vals = []
    for i in range(0,len(data_length)):
        vals.append([sum(1 for x in data_length[i] if x[0] == 1),
                    sum(1 for x in data_length[i] if x[0] == 2),
                    sum(1 for x in data_length[i] if x[0] == 3),
                    sum(1 for x in data_length[i] if x[0] == 4)])
    comb_vals = []
    for i in range(0,len(vals)):
        for j in range(0,len(vals[i])):
            if vals[i][j] >= 2:
                comb_vals.append(ncr(vals[i][j],2))
    return sum(comb_vals)


# perform accuracy calculations
def calculations(data_length,k):
    allPositives = Positives(data_length)
    tp = TruePositives(data_length,k)
    fp = allPositives - tp
    fn = FalseNegatives(data_length)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f_score = (2*precision*recall)/(precision+recall)
    print("Precision for k =",k,"is",precision)
    print("Recall for k =",k,"is",recall)
    print("F-Score for k =",k,"is",f_score)
    return  precision,recall,f_score


# get false negatives
def FalseNegatives(data_length):
    vals = []
    for i in range(0, len(data_length)):
        vals.append([sum(1 for x in data_length[i] if x[0] == 1),
                     sum(1 for x in data_length[i] if x[0] == 2),
                     sum(1 for x in data_length[i] if x[0] == 3),
                     sum(1 for x in data_length[i] if x[0] == 4)])
    combs = list(map(list,zip(*vals)))
    allcombs = []
    for i in range(0, len(combs)):
        allcombs.append(list(itertools.combinations(combs[i],2)))
    multiplicands = []
    for i in range(0, len(allcombs)):
        for j in range(0,len(allcombs[i])):
            if 0 not in allcombs[i][j]:
                multiplicands.append(np.prod(np.array(allcombs[i][j])))
    return sum(multiplicands)


# main method
def main():
    k = [1,2,3,4,5,6,7,8,9,10]
    option = input("Normalise Data? 1 = No, 2 = Yes ")
    info = input_data(option)

    min_max_list = find_minmax(info)
    prec_arr = []
    rec_arr = []
    f_array = []
    option_dist = input("Choose distance measure: 1 = Euclidean, 2 = Manhatten Distance, 3 = Cosine Similarity ")
    for wz in range(0,len(k)):
        centroids = []
        for i in range(0,k[wz]):
            centroids.append(fill_centroid(min_max_list))

        centroids_old_pos = centroids
        centroids_new_pos = np.zeros(shape=(4, 300))

        while np.array_equal(centroids_old_pos,centroids_new_pos) == False:
            centroids_old_pos = centroids
            data_length = grouped_distance(centroids,info,k[wz], option_dist)
            # aggregated = [np.zeros(301),np.zeros(301),np.zeros(301),np.zeros(301)]
            aggregated = [np.zeros(301) for i in range(k[wz])]
            for i in range(0,len(data_length)):

                if len(data_length[i]) < 1:
                    aggregated[i][1:] = np.asarray(centroids[i])
                else:
                    array = np.asarray(np.asfarray(data_length[i]))
                    # array = [item[1:] for item in array[i]]
                    aggregated[i] = np.mean(array,axis=0)
            centroids = [item[1:] for item in aggregated]
            centroids_new_pos = [item[1:] for item in aggregated]

        prec,rec,f = calculations(data_length, k[wz])
        print("Cluster with", k[wz], "centroids has reached minimum")
        prec_arr.append(prec)
        rec_arr.append(rec)
        f_array.append(f)

    plt.plot(k,prec_arr,label="Precision")
    plt.plot(k,rec_arr,label="Recall")
    plt.plot(k,f_array,label="F-Score")
    plt.xlabel("K")
    plt.ylabel("Metric Score")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()