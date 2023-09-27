from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
from tsfresh import feature_selection
import src.utilFeatExtr as util
import pandas as pd
import os
import multiprocessing as mp
from src.PFA import PFA
from pyts import datasets

def preprocess_data(nameDataset, UCR=False):
    if UCR:
            datasetAll = list(datasets.fetch_ucr_dataset(nameDataset)['data_train']) + list(
                datasets.fetch_ucr_dataset(nameDataset)['data_test'])
            labelPred = list(datasets.fetch_ucr_dataset(nameDataset)['target_train']) + list(
                datasets.fetch_ucr_dataset(nameDataset)['target_test'])

            datasetAdapted = {"listOut":util.adaptTimeSeriesUCR(datasetAll),
                                           'series': pd.Series((str(i) for i in labelPred)), "listOfClass":list(str(i) for i in labelPred)}

    else:
            testPath = "./DatasetTS/" + nameDataset + "/" + nameDataset + ".tsv"
            if not os.path.isfile(testPath):
                util.mergeArffFiles(nameDataset)
            listOut, series, listOfClass = util.adaptTimeSeries(testPath)
            datasetAdapted = {"listOut": listOut, "series":series,"listOfClass":listOfClass}

    return datasetAdapted


def features_extraction_selection(nameDataset, datasetAdapted, trainFeatDataset=0.2, saveFeat=False, numberFeatUse=20, randomFeat=False):
        # Name of the dataset
        # print("Dataset: " + nameDataset)

        # Create the dataframe for the extraction of the features
        listOut = datasetAdapted["listOut"]
        series =datasetAdapted["series"]
        listOfClass = datasetAdapted["listOfClass"]

        if saveFeat:
            clusterK = len(list(set(list(series))))
            if os.path.isdir("./DatasetTS/" + nameDataset) == False:
                os.mkdir("./DatasetTS/" + nameDataset)

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
        else:
            filtreFeat, seriesAcc, features_filtered_direct = util.extractFeature(listOut, series, listOfClass,
                                                                                  trainFeatDataset)

        # Extract the relevance for each features and it will be ordered by importance
        ris = feature_selection.relevance.calculate_relevance_table(filtreFeat, seriesAcc, ml_task="classification")

        # print("Number of Feature Extracted: " + str(len(features_filtered_direct.keys())))
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
        return featPFA, features_filtered_direct

def community_and_matrix_creation(featPFA, datasetAdapted, features_filtered_direct, chooseAlgorithm={'Greedy':{}}, threshold = 0.8):

        clusterK = len(list(set(list(datasetAdapted["series"]))))
        listOfId = set(datasetAdapted["listOut"]["id"])
        dictOfT = {}
        # Create of dataframe where there are the values of the features take in consideration
        for value in set(list(datasetAdapted["series"])):
            dictSing = {value: pd.Series([0], index=["count"])}
            dictOfT.update(dictSing)

        dictOfInfoTrain = {}

        # Creation of the features that we want to use
        listOfFeat = featPFA

        # print("Community Creation...")

        def collect_result_Train(result):
            dictOfInfoTrain.update(result)


        pool = mp.Pool(mp.cpu_count())

        # Creation of graph and extraction of community detection
        for feature in listOfFeat:
            pool.apply_async(util.getCommunityDetectionTrain, args=(feature, features_filtered_direct, listOfId,
                                                                    threshold, clusterK, chooseAlgorithm),
                             callback=collect_result_Train)
        pool.close()
        pool.join()

        setCluster = list()
        # Creation of list with all the cluster and their weights, used for the creation of CoOccurrence Matrix
        for key in dictOfInfoTrain.keys():
            for clusterInside in dictOfInfoTrain[key]["cluster"]:
                dictSing = {'list': list(clusterInside), 'weight': dictOfInfoTrain[key]["weightFeat"]}
                setCluster.append(dictSing)


        # Creation of CoOccurrence Matrix
        # print("Matrix Creation...")
        matrixNsym = util.getTabNonSym(setCluster, list(listOfId))
        return matrixNsym

def cluster_evaluation(matrixNsym, datasetAdapted):
        # print("Clustering Creation....")
        # List of the cluster created in the training set. It will be used later for the intersaction
        # with the cluster extract from the testing.
        clusterK = len(list(set(list(datasetAdapted["series"]))))
        listOfId = set(datasetAdapted["listOut"]["id"])
        series = datasetAdapted["series"]

        listOfCommFindTest = util.getCluster(matrixNsym, listOfId, clusterK)

        listOfCommFindTest = util.createSet(listOfCommFindTest, clusterK)

        # Modify the index of the TimeSeries with their classes
        listOfProva = [0 for x in range(len(series))]
        for value in range(len(listOfCommFindTest)):
            for ind in listOfCommFindTest[value]["cluster"]:
                listOfProva[ind] = value

        resultCount = {}
        for value in set(list(series)):
            dictSingA = {value: pd.Series([0], index=["count"])}
            resultCount.update(dictSingA)

        # Creation of confusion matrix

        amiValue = adjusted_mutual_info_score([int(i) for i in list(series)], listOfProva)
        randIndex = util.rand_index_score([int(i) for i in list(series)], listOfProva)
        adjRandInd = adjusted_rand_score([int(i) for i in list(series)], listOfProva)
        normMutualInfo = normalized_mutual_info_score([int(i) for i in list(series)], listOfProva)
        return amiValue, normMutualInfo, randIndex, adjRandInd

def FeatTS(nameDataset, ucrDataset=False):
    print('FeatTS on going..')
    datasetAdapted = preprocess_data(nameDataset, ucrDataset)
    featPFA, features_filtered_direct = features_extraction_selection(nameDataset, datasetAdapted)
    matrixNsym = community_and_matrix_creation(featPFA, datasetAdapted, features_filtered_direct)
    amiValue, normMutualInfo, randIndex, adjRandInd = cluster_evaluation(matrixNsym, datasetAdapted)
    return amiValue, normMutualInfo, randIndex, adjRandInd





















