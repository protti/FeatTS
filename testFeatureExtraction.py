import matplotlib.pyplot as plt
import pandas
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from tsfresh import feature_selection
import utilFeatExtr as util
import pandas as pd
import os
import multiprocessing as mp
import time
from PFA import PFA
import csv


if __name__ == '__main__':
    listAlreadyDone = []
    if os.path.isfile("experiments.csv"):
        reader = csv.DictReader(open('experiments.csv'))
        listAlreadyDone = []
        for row in reader:
            listAlreadyDone.append(row["nameDataset"])
    listOfEntireDataset = list(set(os.listdir("./DatasetTS/")) - set(listAlreadyDone))
    # print(listOfEntireDataset)
    dictOfLength = {}
    for dataset in listOfEntireDataset:
        testPath = "./DatasetTS/" + dataset + "/" + dataset + ".tsv"
        with open(testPath, 'r+') as file:
            lines = file.readlines()
            dictOfLength[dataset] = len(lines)

    dictOfLength = dict(sorted(dictOfLength.items(), key=lambda item: item[1]))
    print("Datasets to Cluster:" + str(dictOfLength))

    with open('experiments.csv', 'a+', newline='') as csvfile:
        filesize = os.path.getsize("experiments.csv")
        fieldnames = ['nameDataset', 'AMI', 'ARI', 'RI']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if len(listOfEntireDataset) == len(os.listdir("./DatasetTS/")) and filesize == 0:
            writer.writeheader()
        csvfile.close()

    for dataset in dictOfLength.keys():

        totalTime = time.time()
        # Choice of the number of features to use
        randomFeat = False
        calcDTW = False

        if calcDTW:
            numberFeatUse = 1
        else:
            numberFeatUse = 20

        featuNum = [x for x in range(1, numberFeatUse + 1)]
        # Choice of the number of clusters k

        # Name of the dataset
        nameDataset = dataset
        print("Dataset: " + nameDataset)
        testPath = "./DatasetTS/" + nameDataset + "/" + nameDataset + ".tsv"

        # Create the dataframe for the extraction of the features
        listOut, series,listOfClass, listForDTW = util.adaptTimeSeries(testPath)
        clusterK = len(list(set(list(series))))

        # Threshold of the distance
        threshold = 0.8
        # Percentage of number of class to use
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



        if os.path.isfile(
                "./DatasetTS/" + nameDataset + '/SFS/KVal_' + str(clusterK) + '/' + nameDataset + 'RankAlgorithm.csv'):
            os.remove("./DatasetTS/" + nameDataset + '/SFS/KVal_' + str(clusterK) + '/' + nameDataset + 'RankAlgorithm.csv')

        if os.path.isdir("./DatasetTS/" + nameDataset + "/Train/") == False:
            os.mkdir("./DatasetTS/" + nameDataset + "/Train/")

        # Extraction or loading of  features of the number of time series select for the train cross validation.
        if os.path.isfile("./DatasetTS/" + nameDataset + "/Train/featureALL" + nameDataset + ".pk1") == False:
            filtreFeat,seriesAcc,features_filtered_direct = util.extractFeature(listOut, series,listOfClass,trainFeatDataset)

            features_filtered_direct.to_pickle("./DatasetTS/" + nameDataset + "/Train/featureALL" + nameDataset + ".pk1")
        else:
            features_filtered_direct = pd.read_pickle(
                "./DatasetTS/" + nameDataset + "/Train/featureALL" + nameDataset + ".pk1")
            filtreFeat, seriesAcc,features_filtered_direct = util.extractFeature(listOut, series, listOfClass,trainFeatDataset,features_filtered_direct)

        # Extract the relevance for each features and it will be ordered by importance
        ris = feature_selection.relevance.calculate_relevance_table(filtreFeat, seriesAcc, ml_task="classification")
        print("Number of Feature Extracted: " + str(len(features_filtered_direct.keys())))
        ris = ris.sort_values(by='p_value')
        if randomFeat:
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

        timeTot = 0
        if os.path.isdir("./DatasetTS/" + nameDataset + "/SFS/") == False:
            os.mkdir("./DatasetTS/" + nameDataset + "/SFS")
            os.mkdir("./DatasetTS/" + nameDataset + "/SFS/KVal_" + str(clusterK) + "/")
            os.mkdir("./DatasetTS/" + nameDataset + "/SFS/KVal_" + str(clusterK) + "/SingleIterationInfo/")
        if os.path.isdir("./DatasetTS/" + nameDataset + "/SFS/KVal_" + str(clusterK)) == False:
            os.mkdir("./DatasetTS/" + nameDataset + "/SFS/KVal_" + str(clusterK) + "/")
            os.mkdir("./DatasetTS/" + nameDataset + "/SFS/KVal_" + str(clusterK) + "/SingleIterationInfo/")


        listOfId = set(listOut["id"])
        originalClassNumb = len(set(list(series)))
        vettoreIndici = list(set(list(series)))

        dictOfT = {}

        # Create of dataframe where there are the values of the features take in consideration
        for value in set(list(series)):
            dictSing = {value: pd.Series([0], index=["count"])}
            dictOfT.update(dictSing)

        dictSing = {}
        df = pd.DataFrame(dictOfT)

        listOfNumCom = []
        dictOfInfoTrain = {}

        # Creation of the features that we want to use
        listOfFeat = featPFA

        print("Community Creation...")
        if not calcDTW:

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

        else:
            listOfFeat.append("DTW")
            dictOfInfoTrain = util.getCommunityDetectionDTW(listForDTW, threshold, clusterK, chooseAlgorithm,
                                                            trainKClique, listOfId, nameDataset, "SFS")


        setCluster = list()
        # Creation of list with all the cluster and their weights, used for the creation of CoOccurrence Matrix
        for key in dictOfInfoTrain.keys():
            for clusterInside in dictOfInfoTrain[key]["cluster"]:
                dictSing = {'list': list(clusterInside), 'weight': dictOfInfoTrain[key]["weightFeat"]}
                setCluster.append(dictSing)

        start = time.time()
        # Creation of CoOccurrence Matrix
        print("Matrix Creation...")
        matrixNsym = util.getTabNonSym(setCluster, list(listOfId))


        print("Clustering Creation....")
        # List of the cluster created in the training set. It will be used later for the intersaction
        # with the cluster extract from the testing.
        listOfCommFindTest = util.getCluster(matrixNsym, listOfId, clusterK)

        listOfCommFindTest = util.createSet(listOfCommFindTest, clusterK)
        listReal = [int(i) - int(min(list(series))) for i in list(series)]

        # Modify the index of the TimeSeries with their classes
        listOfProva = [0 for x in range(len(series))]
        for value in range(len(listOfCommFindTest)):
            listOfClass = []
            for ind in listOfCommFindTest[value]["cluster"]:
                listOfProva[ind] = value

        resultCount = {}
        for value in set(list(series)):
            dictSingA = {value: pd.Series([0], index=["count"])}
            resultCount.update(dictSingA)
        dfA = pd.DataFrame(resultCount)
        # Creation of confusion matrix

        amiValue = adjusted_mutual_info_score([int(i) for i in list(series)], listOfProva)
        randIndex = util.rand_index_score([int(i) for i in list(series)], listOfProva)
        adjRandInd = adjusted_rand_score([int(i) for i in list(series)], listOfProva)

        # Calculation of AMI and Rand Index
        print("Final Results:")
        print("AMI: " + str(amiValue))
        print("Rand Index: " + str(randIndex))
        print("Adjusted Rand Index: " + str(adjRandInd))

        with open('experiments.csv', 'a+', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(
                {'nameDataset': dataset, 'AMI': float(amiValue),
                 'ARI': float(adjRandInd),
                 'RI': float(randIndex)})
            csvfile.close()
        print("______________________________")

























