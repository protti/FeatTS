import csv
import os
import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import pairwise_distances
from pyclustering.cluster.kmedoids import kmedoids
from scipy.special import comb
import pickle
import networkx as nx
import multiprocessing as mp

from tsfresh import extract_features

import SLPA
import utilityUCR as util
matrixSym = []




def adaptTimeSeries(path):
    with open(path, 'r') as csvFile:
        reader = csv.reader(csvFile)
        id = 0
        listOfValue = []
        listOfId = []
        listOfTime = []
        listOfClass = []
        listGeneric = []
        listForDTW = []
        startPoint = 1
        splitClass = 0


        for row in reader:
            listValueApp = []
            # print(row)
            splitted = row[0].split('\t')

            if "AsphaltObstacles" in path or "AsphaltRegularity" in path:
                splitClass = len(splitted) - 1
                startPoint = 0

            listOfClass.append(splitted[splitClass])
            for i in range(startPoint,len(splitted)):
                if splitted[i] != "NaN":
                    listOfValue.append(float(splitted[i]))
                    listValueApp.append(float(splitted[i]))
                    listOfTime.append(i)
                    listOfId.append(id)
                    listGeneric.append((id,i,(float(splitted[i]))))
            listForDTW.append(listValueApp)
            id += 1

        df = pd.DataFrame(listGeneric, columns=['id', 'time','value'])
        series = pd.Series((i for i in listOfClass))


        return df,series,listOfClass,listForDTW

def getDataframeAcc(appSeries,perc):
    listClassExtr = list(appSeries.drop_duplicates())
    series = appSeries
    dictIndexAcc = {}
    dictIndexNotAcc = {}
    print(listClassExtr)
    # print(series)
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


def extractFeature(listOut, series,listOfClass,trainFeatDataset,features_filtered_direct = None):

    if features_filtered_direct is None:
        features_filtered_direct = extract_features(listOut, column_id='id', column_sort='time')
        features_filtered_direct = features_filtered_direct.dropna(axis='columns')


    allAcc,allNotAcc = getDataframeAcc(series,trainFeatDataset)
    filtreFeat,seriesAcc = getSubSetFeatures(features_filtered_direct,allAcc,allNotAcc,listOfClass)
    filtreFeat = filtreFeat.drop('id',axis='columns')
    return filtreFeat,seriesAcc,features_filtered_direct






def getMedianDistance(threshold,listOfValue):
    try:
        listOfDistance = []
        for i in range(0,len(listOfValue)):
            for j in range(i + 1, len(listOfValue)):
                listOfDistance.append(abs(listOfValue[i] - listOfValue[j]))
        listOfDistance.sort(reverse=False)

    except Exception as e:
        print(e)
    if listOfDistance[int(len(listOfDistance) * threshold)] == 0:
        print("Tutti i valori sono uguali")
    return listOfDistance[int(len(listOfDistance) * threshold)]



def getMedianDistanceDTW(threshold,listOfValue,matrixDistanceLoad = []):
    matrixDistance = [[-1 for x in range(len(listOfValue))] for y in range(len(listOfValue))]
    try:
        listOfDistance = []
        if matrixDistanceLoad == []:
            listOfDistance,matrixDistance = calculationDistDTW(listOfValue)
        else:
            listOfDistance,matrixDistance = calculationDistDTW(listOfValue,matrixDistanceLoad)
        listOfDistance.sort(reverse=False)
        print(listOfDistance)
    except Exception as e:
        print(e)
    if listOfDistance[int(len(listOfDistance) * threshold)] == 0:
        print("Tutti i valori sono uguali")
    if len(matrixDistanceLoad) != 0:
        return listOfDistance[int(len(listOfDistance) * threshold)], matrixDistanceLoad
    else:
        return listOfDistance[int(len(listOfDistance) * threshold)], matrixDistance



def calculationDistDTW(listOfValue,matrixDistanceLoaded = []):
    if matrixDistanceLoaded == []:
        w = len(listOfValue)
        matrixDistanceLoaded = [[0 for x in range(w)] for y in range(w)]

        def matrixCalcParal(result):
            # print(len(result))
            for val in result:
                matrixDistanceLoaded[val["i"]][val["j"]] = val["value"]

        listOfValueList = []
        for i in range(0, len(listOfValue)):
            listOfValueList.append(listOfValue[i:])


        pool = mp.Pool(mp.cpu_count())
        for ind in range(len(listOfValueList)):
            totRig = int(len(listOfValueList[ind]) / mp.cpu_count())
            for j in range(0, mp.cpu_count()):
                start = j * int((len(listOfValueList[ind]) / mp.cpu_count()))
                if j == mp.cpu_count() - 1:
                    totRig += int(len(listOfValueList[ind]) % mp.cpu_count())
                pool.apply_async(calcValueDTW, args=(ind,start, listOfValueList[ind], totRig), callback=matrixCalcParal)

        pool.close()
        pool.join()

    listDist = []
    for x in matrixDistanceLoaded:
        print(x)
    for i in range(0, len(listOfValue)):
        for j in range(i + 1, len(listOfValue)):
            listDist.append(matrixDistanceLoaded[i][j])
    print(listDist)
    return listDist,matrixDistanceLoaded


def calcValueDTW(indAna,start, listOfValue, totRig):
    dictOfValueIJ = []
    print(indAna,start,totRig)
    for val in range(start,start+totRig):
        value = fastdtw(listOfValue[0], listOfValue[val], dist=euclidean)[0]
        # print(value)
        dictSingle = {"value": value,
                      "i": indAna, "j": val+indAna}
        dictOfValueIJ.append(dictSingle)

    return dictOfValueIJ

def getTabNonSym(setCluster,listId):


    w = len(listId)
    matrixSym = [[0 for x in range(w)] for y in range(w)]


    def matrixCalcParal(result):
        # print(len(result))
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
    print("Lunghezza" + str(len(matrixSym)))
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


    idChoose = util.getInitialIndex(dictTotal,numClust)
    D = pairwise_distances(matrixsym, metric='correlation')


    kmedoids_instance = kmedoids(D, idChoose, tolerance=0.000001)
    kmedoids_instance.process()
    Cl = kmedoids_instance.get_clusters()
    # show allocated clusters

    dictClu = {}
    for i in range(0, len(Cl)):
        dictApp = {i: Cl[i]}
        dictClu.update(dictApp)
    print(dictClu)


    listOfCommFind = []

    for label in dictClu:
        for point_idx in dictClu[label]:
            # print('label {0}: {1}'.format(label, list(setCluster)[point_idx]))
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
    print(ris)
    randomFeat = ris.sample(n=numberFeatUse)
    print(randomFeat["p_value"])

    return randomFeat



def getCommunityDetectionTrain(feature,features_filtered_direct,listOfId,threshold,clusterK,chooseAlgorithm,trainKClique, nameDataset, algorithmFeat):

    listOfDictInfoFeat = {}
    if os.path.isdir("./" + nameDataset + "/" + algorithmFeat+"/KVal_"+str(clusterK) + "/CommunityDetection") == False:
        os.mkdir("./" + nameDataset + "/" + algorithmFeat+"/KVal_"+str(clusterK) + "/CommunityDetection")

    if os.path.isfile("./" + nameDataset + "/" + algorithmFeat+"/KVal_"+str(clusterK) + "/CommunityDetection/TrainListOfComm.pkl") == False:
        with open("./" + nameDataset + "/" + algorithmFeat+"/KVal_"+str(clusterK) + "/CommunityDetection/TrainListOfComm.pkl", 'wb') as f:
            pickle.dump(listOfDictInfoFeat, f)

    with open("./" + nameDataset + "/" + algorithmFeat+"/KVal_"+str(clusterK) + "/CommunityDetection/TrainListOfComm.pkl", 'rb') as f:
        listOfDictInfoFeat = pickle.load(f)

    if not feature in listOfDictInfoFeat.keys():
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

            if chooseAlgorithm == 0:
                coms = list(nx.algorithms.community.greedy_modularity_communities(G))
            elif chooseAlgorithm == 1:
                coms = list(nx.algorithms.community.k_clique_communities(G,trainKClique))
            else:
                extrC = SLPA.find_communities(G, 20, 0.01)
                coms = []
                for val in extrC:
                    coms.append(frozenset(extrC[val]))

            for value in coms:
                if len(coms) > clusterK:
                    dictOfInfo[feature] = {"distance": distanceMinAccept, "cluster": coms,
                                         "weightFeat": clusterK / len(coms)}
                else:
                    dictOfInfo[feature] = {"distance": distanceMinAccept, "cluster": coms,
                                         "weightFeat":  len(coms)/clusterK}


        except Exception as e:
            print(e)
            pass
        with open("./" + nameDataset + "/" + algorithmFeat+"/KVal_"+str(clusterK) + "/CommunityDetection/TrainListOfComm.pkl", 'wb') as f:
            listOfDictInfoFeat[feature] = dictOfInfo
            pickle.dump(listOfDictInfoFeat, f)
            f.close()
    else:
        dictOfInfo = listOfDictInfoFeat[feature]

    return dictOfInfo

def calcLocalGen(amiValues):
    past = -1
    for items in amiValues.iteritems():
        if past == -1:
            past = items[1]
        elif past <= items[1]:
            past = items[1]
        else:
            break

    return amiValues.max(),past

def getCommunityDetectionDTW(listForDTW,threshold,clusterK,chooseAlgorithm,trainKClique,listOfId,nameDataset,algorithmFeat):
    listOfDictInfoFeat = {}
    dictOfInfo = {}
    matrixDistance = []
    G = nx.Graph()
    H = nx.path_graph(listOfId)
    G.add_nodes_from(H)

    if os.path.isfile("./" +nameDataset +"/Train/distanceDTW") !=  True:
        distanceMinAccept,matrixDistance = getMedianDistanceDTW(threshold, listForDTW)
        pickle_out = open("./" +nameDataset +"/Train/distanceDTW", "wb")

        pickle.dump(matrixDistance, pickle_out)
        pickle_out.close()
    else:
        pickle_in = open("./" +nameDataset +"/Train/distanceDTW", "rb")
        matrixDistance = pickle.load(pickle_in)
        distanceMinAccept, matrixDistance = getMedianDistanceDTW(threshold, listForDTW,matrixDistance)

    print(distanceMinAccept)
    for i in range(0, len(listOfId)):
        for j in range(i + 1, len(listOfId)):
            if matrixDistance[i][j] < distanceMinAccept:
                G.add_edge(i, j)

    try:

        if chooseAlgorithm == 0:
            coms = list(nx.algorithms.community.greedy_modularity_communities(G))
        elif chooseAlgorithm == 1:
            coms = list(nx.algorithms.community.k_clique_communities(G, trainKClique))
        else:
            extrC = SLPA.find_communities(G, 20, 0.01)
            coms = []
            for val in extrC:
                coms.append(frozenset(extrC[val]))

        for value in coms:
            if len(coms) > clusterK:
                dictOfInfo["DTW"] = {"distance": distanceMinAccept, "cluster": coms,
                                       "weightFeat": clusterK / len(coms)}
            else:
                dictOfInfo["DTW"] = {"distance": distanceMinAccept, "cluster": coms,
                                       "weightFeat": len(coms) / clusterK}
    except Exception as e:
        print(e)
        pass
    return dictOfInfo

def rand_index_score(clusters, classes):
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)







