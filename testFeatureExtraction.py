import matplotlib.pyplot as plt
import pandas
from sklearn.metrics import adjusted_mutual_info_score
from tsfresh import feature_selection
import utilFeatExtr as util
import pandas as pd
import os
import multiprocessing as mp
import time
from PFA import PFA


if __name__ == '__main__':
    totalTime = time.time()
    # Choice of the number of features to use
    randomFeat = False
    calcDTW = False
    if calcDTW == True:
        numberFeatUse = 1
    else:
        numberFeatUse = 20

    featuNum = [x for x in range(1, numberFeatUse + 1)]
    # Choice of the number of clusters k
    clusterK = 2
    # Name of the dataset
    nameDataset = "ECG200"

    # Threshold of the distance
    threshold = 0.8

    trainFeatDataset = 0.2


    dictOfFeat = {}
    # Choice of the algorithm (Greedy Algorithm default)
    chooseAlgorithm = 0
    testKClique = 0
    trainKClique = 0
    if chooseAlgorithm == 0:
        algorithm = "Greedy Modularity"
    elif chooseAlgorithm == 1:
        testKClique = 10
        trainKClique = 5
        algorithm = "KClique(" + trainKClique + "," + testKClique + ")"
    else:
        algorithm = "SLPA.find_communities(G, 20, 0.01)"

    print("Choosed Algorithm: " + algorithm)

    if os.path.isfile(
            "./" + nameDataset + '/SFS/KVal_' + str(clusterK) + '/' + nameDataset + 'RankAlgorithm.csv'):
        os.remove("./" + nameDataset + '/SFS/KVal_' + str(clusterK) + '/' + nameDataset + 'RankAlgorithm.csv')


    # If necessary, adapt the time series for the software and I create the folder for the dataset
        # Take the path of the folders
    testPath = "./" + nameDataset + "/" + nameDataset + ".tsv"

    # Create the dataframe for the extraction of the features
    listOut, series,listOfClass, listForDTW = util.adaptTimeSeries(testPath)


    if os.path.isdir("./" + nameDataset + "/Train/") == False:
        os.mkdir("./" + nameDataset + "/Train/")

    # Extraction or loading of  features of the number of time series select for the train cross validation.

    if os.path.isfile(
            "./" + nameDataset + "/Train/featureALL" + nameDataset + ".pk1") == False:

        filtreFeat,seriesAcc,features_filtered_direct = util.extractFeature(listOut, series,listOfClass,trainFeatDataset)

        features_filtered_direct.to_pickle("./" + nameDataset + "/Train/featureALL" + nameDataset + ".pk1")
    else:
        features_filtered_direct = pd.read_pickle(
            "./" + nameDataset + "/Train/featureALL" + nameDataset + ".pk1")
        filtreFeat, seriesAcc,features_filtered_direct = util.extractFeature(listOut, series, listOfClass,trainFeatDataset,features_filtered_direct)

    # Extract the relevance for each features and it will be ordered by importance
    ris = feature_selection.relevance.calculate_relevance_table(filtreFeat, seriesAcc, ml_task="classification")
    print("Feature choosed: " + str(len(features_filtered_direct.keys())))
    ris = ris.sort_values(by='p_value')
    for x in ris["feature"]:
        print(x)
    if randomFeat == True:
        ris = util.randomFeat(ris, numberFeatUse)


    listOfFeatToUse = []
    for t in range(numberFeatUse):
        listOfFeatToUse.append(ris["feature"][t])



    dfFeatUs = pd.DataFrame()
    for x in range(len(listOfFeatToUse)):
        dfFeatUs[listOfFeatToUse[x]] = features_filtered_direct[listOfFeatToUse[x]]

    pfa = PFA()
    featPFA = pfa.fit(dfFeatUs)
    x = pfa.features_
    column_indices = pfa.indices_

    print("Threshold: " + str(threshold))
    timeTot = 0
    listDict = {}
    if os.path.isdir("./" + nameDataset + "/SFS/") == False:
        os.mkdir("./" + nameDataset + "/SFS")
        os.mkdir("./" + nameDataset + "/SFS/KVal_" + str(clusterK) + "/")
        os.mkdir("./" + nameDataset + "/SFS/KVal_" + str(clusterK) + "/SingleIterationInfo/")
    if os.path.isdir("./" + nameDataset + "/SFS/KVal_" + str(clusterK)) == False:
        os.mkdir("./" + nameDataset + "/SFS/KVal_" + str(clusterK) + "/")
        os.mkdir("./" + nameDataset + "/SFS/KVal_" + str(clusterK) + "/SingleIterationInfo/")

    f = open(
        "./" + nameDataset + "/Train/" + nameDataset + "__NF" + str(t) + "_th" + str(threshold) + "_algSFS.tsv",
        "w+")
    f.write("Threshold: \t" + str(threshold) + "\n")
    f.write("Algoritmo Usato:\t " + algorithm + " \n")
    listOfId = set(listOut["id"])
    originalClassNumb = len(set(list(series)))
    vettoreIndici = list(set(list(series)))
    print("Cluster Choosed:" + str(clusterK))

    dictOfT = {}

    # Create of dataframe where there are the values of the features take in consideration
    for value in set(list(series)):
        dictSing = {value: pd.Series([0], index=["count"])}
        dictOfT.update(dictSing)

    dictSing = {}
    df = pd.DataFrame(dictOfT)
    print(df)

    listOfNumCom = []
    dictOfInfoTrain = {}

    # Creation of the features that we want to use
    listOfFeat = featPFA

    if calcDTW != True:

        def collect_result_Train(result):
            dictOfInfoTrain.update(result)


        pool = mp.Pool(mp.cpu_count())

        listOfClustering = []

        start = time.time()
        # Creation of graph and extraction of community detection
        for feature in listOfFeat:
            pool.apply_async(util.getCommunityDetectionTrain, args=(feature, features_filtered_direct, listOfId,
                                                                    threshold, clusterK, chooseAlgorithm,
                                                                    trainKClique, nameDataset, "SFS"),
                             callback=collect_result_Train)

        pool.close()
        pool.join()
        end = time.time()
        print("Time Training: " + str(end - start))
        timeTot += end - start
        listDict.update({"TimeTrainCommDetect": end - start})
        f.write("Time Community Detection: \t" + str(end - start) + "\n")
    else:
        listOfFeat.append("DTW")
        dictOfInfoTrain = util.getCommunityDetectionDTW(listForDTW, threshold, clusterK, chooseAlgorithm,
                                                        trainKClique, listOfId, nameDataset, "SFS")

    print(str(len(dictOfInfoTrain.keys())))

    setCluster = list()
    # Creation of list with all the cluster and their weights, used for the creation of CoOccurrence Matrix
    for key in dictOfInfoTrain.keys():
        for clusterInside in dictOfInfoTrain[key]["cluster"]:
            dictSing = {'list': list(clusterInside), 'weight': dictOfInfoTrain[key]["weightFeat"]}
            print(dictSing)
            setCluster.append(dictSing)

    start = time.time()
    print("Computation Matrix....")
    # Creation of CoOccurrence Matrix
    matrixNsym = util.getTabNonSym(setCluster, list(listOfId))
    print(matrixNsym)
    end = time.time()
    timeTot += end - start
    print("Time Computation Matrix: " + str(end - start))
    listDict.update({"TimeTrainMatrix": end - start})

    f.write("Time Computation Matrix: \t" + str(end - start) + "\n")
    start = time.time()

    print("Compute Clustering....")
    # List of the cluster created in the training set. It will be used later for the intersaction
    # with the cluster extract from the testing.
    print("Train")
    listOfCommFindTest = util.getCluster(matrixNsym, listOfId, clusterK)
    end = time.time()

    timeTot += end - start
    print("Time Computation Cluster: " + str(end - start))
    listDict.update({"TimeTrainClus": end - start})
    print("Test")
    listOfCommFindTest = util.createSet(listOfCommFindTest, clusterK)
    listReal = [int(i) - int(min(list(series))) for i in list(series)]
    print(listReal)
    # Modify the index of the TimeSeries with their classes
    listOfProva = [0 for x in range(len(series))]
    for value in range(len(listOfCommFindTest)):
        listOfClass = []
        for ind in listOfCommFindTest[value]["cluster"]:
            listOfProva[ind] = value
    print(listReal)
    print(listOfProva)

    print("Real: " + str(adjusted_mutual_info_score([int(i) - 1 for i in list(series)], listOfProva)))
    resultCount = {}
    for value in set(list(series)):
        dictSingA = {value: pd.Series([0], index=["count"])}
        resultCount.update(dictSingA)
    dfA = pd.DataFrame(resultCount)

    # Creation of confusion matrix

    amiValue = adjusted_mutual_info_score([int(i) for i in list(series)], listOfProva)
    randIndex = util.rand_index_score([int(i) for i in list(series)], listOfProva)

    # Calculation of AMI and Rand Index
    print("AMI: " + str(amiValue))
    print("Rand Index: " + str(randIndex))

    countTP = 0
    for i in dfA.keys():
        countTP += dfA[i]["count"]
    listDict.update({"AMI": amiValue})
    listDict.update({"RandIndex" : randIndex})

    for i in range(0, len(listOfFeat)):
        f.write("Feature " + str(i) + ": \t" + str(listOfFeat[i]) + "\n")

    f.close()

    print()
    # Calculation and creation file for Ranking Algorithm
    listDict.update({"Threshold": threshold})
    listDict.update({"Cluster": clusterK})
    dictOfFeat.update({t: listDict})

    dbfile = open("./" + nameDataset + "/SFS/KVal_" + str(clusterK) + "/SingleIterationInfo/SummaryResults.csv",
                  'w+')

    for key in dictOfFeat.keys():
        dbfile.write("Feature")
        for keys in dictOfFeat[key]:
            dbfile.write("," + keys)
        dbfile.write("\n")
        break
    for key in dictOfFeat.keys():
        dbfile.write(str(key))
        for keys in dictOfFeat[key]:
            dbfile.write("," + str(dictOfFeat[key][keys]))
        dbfile.write("\n")

    dbfile.close()

    result = pandas.read_csv(
        "./" + nameDataset + "/SFS/KVal_" + str(clusterK) + "/SingleIterationInfo/SummaryResults.csv")
    print(type(result))
    ax = plt.gca()
    result.plot(kind='line', x='Feature', y='AMI', ax=ax)
    globalOpt, localOpt = util.calcLocalGen(result["AMI"])
    globalRI, localRI = util.calcLocalGen(result["RandIndex"])
    print("Local Optimum: " + str(localOpt))
    print("Global Optimum: " + str(globalOpt))
    print("Local RI: " + str(localRI))
    print("Global RI: " + str(globalRI))

    # plt.show()

    f = open("./" + nameDataset + "/SFS/KVal_" + str(clusterK) + "/SingleIterationInfo/" + nameDataset + "__NF" + str(
        t) + "_th" + str(threshold) + "_alg" + algorithm + "TIMETOTAL.tsv", "w+")
    f.write(str(time.time() - totalTime))



























