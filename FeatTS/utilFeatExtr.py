import random
from datetime import datetime

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from pyclustering.cluster.kmedoids import kmedoids
import networkx as nx
from tsfresh import extract_features
from . import SLPA
import numpy as np
from itertools import combinations
import memory_profiler

matrixSym = []


def adaptTimeSeriesUCR(input_data):
    # Initialize empty lists to store the data
    id_list = []
    time_list = []
    value_list = []

    # Loop through the input data
    for i, sublist in enumerate(input_data):
        for j, value in enumerate(sublist):
            id_list.append(i)
            time_list.append(j + 1)  # Adding 1 to start time from 1
            value_list.append(value)

    # Create a DataFrame
    data = {"id": id_list, "time": time_list, "value": value_list}
    df = pd.DataFrame(data)
    return df


def choose_and_exclude_indices_by_percentage(classes, percentage):
    # Create a dictionary to store the chosen indices for each class
    class_indices = {}

    # Loop through the classes and assign indices
    for i, class_val in enumerate(classes):
        if class_val not in class_indices:
            class_indices[class_val] = []
        class_indices[class_val].append(i)

    # Determine the number of indices to choose for each class based on the percentage
    chosen_indices = []
    excluded_indices = []
    for class_val, indices in class_indices.items():
        num_indices_to_choose = int(len(indices) * percentage)
        random.shuffle(indices)
        chosen_indices.extend(indices[:num_indices_to_choose])
        excluded_indices.extend(indices[num_indices_to_choose:])

    # Sort the chosen and excluded indices
    chosen_indices.sort()
    excluded_indices.sort()

    return chosen_indices, excluded_indices


def getDataframeAcc(appSeries, perc):
    listClassExtr = list(appSeries.drop_duplicates())
    series = appSeries
    allAccInd = []
    allNotAccInd = []
    for x in listClassExtr:
        sommaClasse = sum(list(series.str.count(x)))
        accepted = int(sommaClasse * perc)
        listIndexAccepted = []
        listIndexNotAccepted = []
        for i in range(len(series)):
            if series[i] == x:
                if len(listIndexAccepted) <= accepted:
                    listIndexAccepted.append(i)
                    allAccInd.append(i)
                else:
                    listIndexNotAccepted.append(i)
                    allNotAccInd.append(i)
    return list(sorted(allAccInd)), list(sorted(allNotAccInd))


def extractFeature(listOut, external_feat=None, n_jobs=1):
    features_filtered_direct = extract_features(
        listOut, column_id="id", column_sort="time", n_jobs=n_jobs
    )
    if external_feat is not None:
        features_filtered_direct = features_filtered_direct.join(external_feat)
    features_filtered_direct = normalization_data(features_filtered_direct)

    return features_filtered_direct


def normalization_data(features_filtered_direct):
    features_filtered_direct = features_filtered_direct.dropna(axis="columns")
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()
    # Fit and transform the DataFrame using the scaler
    normalized_data = scaler.fit_transform(features_filtered_direct)
    # Create a new DataFrame with the normalized data
    features_filtered_direct = pd.DataFrame(
        normalized_data, columns=features_filtered_direct.columns
    )
    # Calculate variance for each column
    variance = features_filtered_direct.var()

    # Sort the Series based on values
    sorted_data = variance.sort_values()

    # Reshape the sorted Series into a 2D array
    X = sorted_data.values.reshape(-1, 1)

    # Perform K-Means clustering with k=2
    kmeans = KMeans(n_clusters=2, n_init="auto")
    kmeans.fit(X)
    variance_useless_column = variance[: (list(kmeans.labels_).count(0) - 1)].index
    features_filtered_direct.drop(columns=variance_useless_column, inplace=True)

    return features_filtered_direct


from concurrent.futures import ProcessPoolExecutor


def getTabNonSym(setCluster, listId, n_jobs=1):
    w = len(listId)
    matrixSym = np.zeros((w, w))

    def matrixCalcParal(result):
        for val in result:
            matrixSym[val["i"], val["j"]] = val["value"]

    totRig = len(listId) // n_jobs

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = []
        for i in range(n_jobs):
            start = i * totRig
            end = start + totRig if i < n_jobs - 1 else len(listId)
            futures.append(
                executor.submit(getValueMatrix, start, listId, end - start, setCluster)
            )

        for future in futures:
            matrixCalcParal(future.result())

    for i in range(w):
        maxVal = np.max(matrixSym[i])
        matrixSym[i] = np.abs(matrixSym[i] - maxVal)

    return matrixSym


def getValueMatrix(start, listId, totRig, listOfClust):
    try:
        dictOfValueIJ = []
        # Pre-calcola i risultati di numOfClusterPres per evitare chiamate ripetute
        cluster_pres_cache = {
            listId[i + start]: numOfClusterPres(listOfClust, listId[i + start])
            for i in range(totRig)
        }

        for i in range(totRig):
            id_i = listId[i + start]
            resultCluster = cluster_pres_cache[id_i]
            for j in range(len(listId)):
                id_j = listId[j]
                resultCouple = numOfRipetitionCouple(id_i, id_j, listOfClust)
                if resultCluster[1] == resultCouple[1]:
                    value = 1
                elif resultCouple[1] == 0:
                    value = 0
                else:
                    value = resultCouple[0] / resultCluster[0]

                dictSingle = {"value": value, "i": i + start, "j": j}
                dictOfValueIJ.append(dictSingle)
        return dictOfValueIJ
    except Exception as e:
        print("Exception in getValueMatrix:")
        print(e)


def getCluster(matrixsym, setCluster, numClust):
    dictTotal = {}
    for x in setCluster:
        listOfDist = []
        for y in setCluster:
            if x != y:
                dictSing = {"id": y, "distance": matrixsym[x][y]}
                listOfDist.append(dictSing)
        dictTotal[x] = listOfDist

    idChoose = getInitialIndex(dictTotal, numClust)
    D = pairwise_distances(matrixsym, metric="correlation")

    kmedoids_instance = kmedoids(D, idChoose, tolerance=0.000001)
    kmedoids_instance.process()
    Cl = kmedoids_instance.get_clusters()
    # show allocated clusters

    dictClu = {}
    for i in range(0, len(Cl)):
        dictApp = {i: Cl[i]}
        dictClu.update(dictApp)

    listOfCommFind = []

    for label in dictClu:
        for point_idx in dictClu[label]:
            dictSing = {"label": label, "cluster": list(setCluster)[point_idx]}
            listOfCommFind.append(dictSing)
    return listOfCommFind


def numOfClusterPres(setCluster, id):
    countId = 0
    countTimes = 0
    for i in range(0, len(setCluster)):
        if id in (setCluster[i]["list"]):
            countId += setCluster[i]["weight"]
            countTimes += 1
    return countId, countTimes


def numOfRipetitionCouple(id1, id2, setCluster):
    countId = 0
    countTimes = 0
    for i in range(0, len(setCluster)):
        if id1 in (setCluster[i]["list"]) and id2 in (setCluster[i]["list"]):
            countId += setCluster[i]["weight"]
            countTimes += 1
    return countId, countTimes


def listOfId(setCluster):
    listId = set()
    for value in setCluster:
        for id in value:
            listId.add(id)
    return list(listId)


def createSet(listOfCommFind, clusterK):
    listOfCluster = []
    for i in range(0, clusterK):
        dictSing = {"cluster": [], "label": i}
        listOfCluster.append(dictSing)
    for value in listOfCommFind:
        listApp = listOfCluster[value["label"]]["cluster"]
        listApp.append(value["cluster"])
        listOfCluster.remove(listOfCluster[value["label"]])
        dictSing = {"cluster": listApp, "label": value["label"]}
        listOfCluster.insert(value["label"], dictSing)

    return listOfCluster


def randomFeat(ris, numberFeatUse):
    ris = ris.dropna(subset=["p_value"])

    indexNames = ris[ris["relevant"] == True].index

    ris.drop(indexNames, inplace=True)
    randomFeat = ris.sample(n=numberFeatUse)

    return randomFeat


def getCommunityDetectionTrain(
    features,
    features_filtered_direct,
    listOfId,
    threshold,
    clusterK,
    chooseAlgorithm,
):
    dictOfInfo = {}

    for feature in features:
        G = nx.Graph()
        H = nx.path_graph(listOfId)
        G.add_nodes_from(H)

        # Convert the feature column to a numpy array for efficient operations
        values = features_filtered_direct[feature].to_numpy()
        # Calculate the pairwise absolute differences using broadcasting
        diff_matrix = np.abs(values[:, np.newaxis] - values)
        # Extract the upper triangle of the matrix, excluding the diagonal
        listOfDistance = diff_matrix[np.triu_indices(len(values), k=1)]
        # Find the threshold index
        threshold_index = int(len(listOfDistance) * threshold)
        # Use numpy's partition to find the threshold value without full sorting
        distanceMinAccept = np.partition(listOfDistance, threshold_index)[
            threshold_index
        ]

        for i, j in combinations(range(len(listOfId)), 2):
            if diff_matrix[i, j] < distanceMinAccept:
                G.add_edge(i, j)

        try:
            if list(chooseAlgorithm.keys())[0] == "SLPA":
                extrC = SLPA.find_communities(
                    G,
                    chooseAlgorithm["SLPA"]["iteration"],
                    chooseAlgorithm["SLPA"]["radious"],
                )
                coms = []
                for val in extrC:
                    coms.append(frozenset(extrC[val]))
            elif list(chooseAlgorithm.keys())[0] == "kClique":
                coms = list(
                    nx.algorithms.community.k_clique_communities(
                        G, chooseAlgorithm["SLPA"]["trainClique"]
                    )
                )
            elif list(chooseAlgorithm.keys())[0] == "Greedy":
                coms = list(nx.algorithms.community.greedy_modularity_communities(G))

            if len(coms) > clusterK:
                dictOfInfo[feature] = {
                    "distance": distanceMinAccept,
                    "cluster": coms,
                    "weightFeat": clusterK / len(coms),
                }
            else:
                dictOfInfo[feature] = {
                    "distance": distanceMinAccept,
                    "cluster": coms,
                    "weightFeat": len(coms) / clusterK,
                }

        except Exception as e:
            print(e)
            pass
    return dictOfInfo


def cleaning(df: pd.DataFrame) -> pd.DataFrame:
    # Drop columns which contain nan, +inf, and -inf values
    df = df.dropna(axis=1, how="any")
    cond = ((df == float("inf")) | (df == float("-inf"))).any(axis=0)
    df = df.drop(df.columns[cond], axis=1)
    # Apply a simple variance threshold
    selector = VarianceThreshold()
    selector.fit(df)
    # Get only no-constant features
    top_features = selector.get_feature_names_out()
    df = df[top_features]
    return df


def max_num_in_list(list):
    max = list[0]
    index = 0
    for i in range(1, len(list)):
        if list[i] > max:
            max = list[i]
            index = i
    return max, index


def min_num_in_list(list):
    min = list[0]
    index = 0
    for i in range(1, len(list)):
        if list[i] < min:
            min = list[i]
            index = i
    return min, index


def getInitialIndex(dictTotal, cluster):
    listOfMinDist = list()
    listOfMinId = list()
    dictSumDistance = dict()
    for idX in dictTotal.keys():
        somma = 0
        for singleDict in dictTotal[idX]:
            somma += singleDict["distance"]
        dictSumDistance[idX] = somma
        if len(listOfMinDist) < cluster:
            listOfMinDist.append(dictSumDistance[idX])
            listOfMinId.append(idX)
        else:
            if max_num_in_list(listOfMinDist)[0] > somma:
                listOfMinDist[max_num_in_list(listOfMinDist)[1]] = dictSumDistance[idX]
                listOfMinId[listOfMinDist.index(dictSumDistance[idX])] = idX

    return listOfMinId
