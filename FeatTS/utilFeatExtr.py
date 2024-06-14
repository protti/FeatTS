import random
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from pyclustering.cluster.kmedoids import kmedoids
import networkx as nx
import multiprocessing as mp
from tsfresh import extract_features
from FeatTS import SLPA

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
    data = {'id': id_list, 'time': time_list, 'value': value_list}
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


def getDataframeAcc(appSeries,perc):
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
    return list(sorted(allAccInd)),list(sorted(allNotAccInd))



def getSubSetFeatures(df,allAccInd,allNotAccInd,listOfClass):
    df = df.drop(allNotAccInd, axis=0)
    df = df.reset_index()
    seriesAcc =  pd.Series((listOfClass[i] for i in allAccInd))
    return df,seriesAcc


def extractFeature(listOut, listOfClass,trainFeatDataset,features_filtered_direct = None):

    if features_filtered_direct is None:
        features_filtered_direct = extract_features(listOut, column_id='id', column_sort='time')
        features_filtered_direct = normalization_data(features_filtered_direct)

    allAcc,allNotAcc = choose_and_exclude_indices_by_percentage(listOfClass, trainFeatDataset)
    # allAcc,allNotAcc = getDataframeAcc(series,trainFeatDataset)
    filtreFeat,seriesAcc = getSubSetFeatures(features_filtered_direct,allAcc,allNotAcc,listOfClass)
    if 'id' in filtreFeat.keys():
        filtreFeat = filtreFeat.drop('id',axis='columns')
    else:
        filtreFeat = filtreFeat.drop('index',axis='columns')
    return filtreFeat,seriesAcc,features_filtered_direct


def normalization_data(features_filtered_direct):
    features_filtered_direct = features_filtered_direct.dropna(axis='columns')
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()
    # Fit and transform the DataFrame using the scaler
    normalized_data = scaler.fit_transform(features_filtered_direct)
    # Create a new DataFrame with the normalized data
    features_filtered_direct = pd.DataFrame(normalized_data, columns=features_filtered_direct.columns)
    # Calculate variance for each column
    variance = features_filtered_direct.var()

    # Sort the Series based on values
    sorted_data = variance.sort_values()

    # Reshape the sorted Series into a 2D array
    X = sorted_data.values.reshape(-1, 1)

    # Perform K-Means clustering with k=2
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)
    variance_useless_column = variance[:(list(kmeans.labels_).count(0)-1)].index
    features_filtered_direct.drop(columns=variance_useless_column, inplace=True)


    return features_filtered_direct


def getMedianDistance(threshold,listOfValue):
    listOfDistance = []
    try:
        for i in range(0,len(listOfValue)):
            for j in range(i + 1, len(listOfValue)):
                listOfDistance.append(abs(listOfValue[i] - listOfValue[j]))
        listOfDistance.sort(reverse=False)

    except Exception as e:
        print(e)
    if listOfDistance[int(len(listOfDistance) * threshold)] == 0:
        print("All the values are equals")
    return listOfDistance[int(len(listOfDistance) * threshold)]





def getTabNonSym(setCluster,listId):
    w = len(listId)
    matrixSym = [[0 for x in range(w)] for y in range(w)]
    def matrixCalcParal(result):
        for val in result:
            matrixSym[val["i"]][val["j"]] = val["value"]

    pool = mp.Pool(mp.cpu_count())
    totRig = int(len(listId)/mp.cpu_count())

    for i in range(0,mp.cpu_count()):
        start = i * int((len(listId)/mp.cpu_count()))
        if i == mp.cpu_count() - 1:
            totRig += int(len(listId)%mp.cpu_count())
        pool.apply_async(getValueMatrix, args=(start,listId,totRig,setCluster),callback=matrixCalcParal)

    pool.close()
    pool.join()
    for i in range(len(matrixSym)):
        maxVal = max(matrixSym[i])
        for j in range(len(matrixSym)):
            matrixSym[i][j] = abs(matrixSym[i][j] - maxVal)
    return matrixSym



def getValueMatrix(start,listId,totRig,listOfClust):
    try:
        dictOfValueIJ = []
        for i in range(0,totRig):
            for j in range(0, len(listId)):
                resultCouple = numOfRipetitionCouple(listId[i+start], listId[j], listOfClust)
                resultCluster = numOfClusterPres(listOfClust, listId[i+start])
                if resultCluster[1] == resultCouple[1]:
                    value = 1
                elif resultCouple[1] == 0:
                    value = 0
                else:
                    value = resultCouple[0] / resultCluster[0]

                dictSingle = {"value":value,
                              "i":i+start,"j":j}

                dictOfValueIJ.append(dictSingle)
        return dictOfValueIJ
    except Exception as e:
        print("Exception in getValueMatrix:")


def getCluster(matrixsym,setCluster,numClust):

    dictTotal = {}
    for x in setCluster:
        listOfDist = []
        for y in setCluster:
            if x != y:
                dictSing = {"id":y,"distance":matrixsym[x][y]}
                listOfDist.append(dictSing)
        dictTotal[x] = listOfDist


    idChoose = getInitialIndex(dictTotal,numClust)
    D = pairwise_distances(matrixsym, metric='correlation')


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

def numOfClusterPres(setCluster,id):
    countId = 0
    countTimes = 0
    for i in range(0,len(setCluster)):
        if id in (setCluster[i]["list"]):
            countId += (setCluster[i]["weight"])
            countTimes += 1
    return countId, countTimes

def numOfRipetitionCouple(id1,id2,setCluster):
    countId = 0
    countTimes = 0
    for i in range(0,len(setCluster)):
        if id1 in (setCluster[i]["list"]) and id2 in (setCluster[i]["list"]):
            countId += setCluster[i]["weight"]
            countTimes += 1
    return countId,countTimes


def listOfId(setCluster):
    listId = set()
    for value in setCluster:
        for id in value:
            listId.add(id)
    return list(listId)

def createSet(listOfCommFind,clusterK):
    listOfCluster = []
    for i in range(0,clusterK):
        dictSing = {"cluster":[],"label":i}
        listOfCluster.append(dictSing)
    for value in listOfCommFind:
        listApp = listOfCluster[value["label"]]["cluster"]
        listApp.append(value["cluster"])
        listOfCluster.remove(listOfCluster[value["label"]])
        dictSing = {"cluster":listApp,"label":value["label"]}
        listOfCluster.insert(value["label"],dictSing)

    return listOfCluster


def randomFeat(ris,numberFeatUse):

    ris = ris.dropna(subset=['p_value'])

    indexNames = ris[ris['relevant'] == True].index

    ris.drop(indexNames, inplace=True)
    randomFeat = ris.sample(n=numberFeatUse)


    return randomFeat


def getCommunityDetectionTrain(feature, features_filtered_direct, listOfId, threshold, clusterK, chooseAlgorithm):

    dictOfInfo = {}
    G = nx.Graph()
    H = nx.path_graph(listOfId)
    G.add_nodes_from(H)
    distanceMinAccept = getMedianDistance(threshold, features_filtered_direct[feature])

    for i in range(0, len(listOfId)):
        for j in range(i + 1, len(listOfId)):
            if abs(features_filtered_direct[feature][i] - features_filtered_direct[feature][j]) < distanceMinAccept:
                G.add_edge(i, j)

    try:
        if list(chooseAlgorithm.keys())[0] == 'SLPA':
            extrC = SLPA.find_communities(G, chooseAlgorithm['SLPA']['iteration'], chooseAlgorithm['SLPA']['radious'])
            coms = []
            for val in extrC:
                coms.append(frozenset(extrC[val]))
        elif list(chooseAlgorithm.keys())[0] == 'kClique':
            coms = list(nx.algorithms.community.k_clique_communities(G, chooseAlgorithm['SLPA']['trainClique']))
        elif list(chooseAlgorithm.keys())[0] == 'Greedy':
            coms = list(nx.algorithms.community.greedy_modularity_communities(G))

        if len(coms) > clusterK:
            dictOfInfo[feature] = {"distance": distanceMinAccept, "cluster": coms, "weightFeat": clusterK / len(coms)}
        else:
            dictOfInfo[feature] = {"distance": distanceMinAccept, "cluster": coms, "weightFeat": len(coms) / clusterK}

    except Exception as e:
        print(e)
        pass

    return dictOfInfo


def cleaning(df: pd.DataFrame) -> pd.DataFrame:
    # Drop columns which contain nan, +inf, and -inf values
    df = df.dropna(axis=1, how='any')
    cond = ((df == float('inf')) | (df == float('-inf'))).any(axis=0)
    df = df.drop(df.columns[cond], axis=1)
    # Apply a simple variance threshold
    selector = VarianceThreshold()
    selector.fit(df)
    # Get only no-constant features
    top_features = selector.get_feature_names_out()
    df = df[top_features]
    return df

def max_num_in_list( list ):
    max = list[0]
    index = 0
    for i in range(1, len(list)):
        if list[i] > max:
            max = list[i]
            index = i
    return max, index

def min_num_in_list( list ):
    min = list[ 0 ]
    index = 0
    for i in range(1,len(list)):
        if list[i] < min:
            min = list[i]
            index = i
    return min,index


def getInitialIndex(dictTotal,cluster):
    listOfMinDist = list()
    listOfMinId = list()
    dictSumDistance = dict()
    for idX in dictTotal.keys():
        somma = 0
        for singleDict in dictTotal[idX]:
            somma += singleDict["distance"]
        dictSumDistance[idX]=somma
        if len(listOfMinDist) < cluster:
            listOfMinDist.append(dictSumDistance[idX])
            listOfMinId.append(idX)
        else:
            if max_num_in_list(listOfMinDist)[0] > somma:
                listOfMinDist[max_num_in_list(listOfMinDist)[1]] = dictSumDistance[idX]
                listOfMinId[listOfMinDist.index(dictSumDistance[idX])] = idX

    return listOfMinId