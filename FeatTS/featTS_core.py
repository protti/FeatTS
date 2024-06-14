import FeatTS.utilFeatExtr as util
from FeatTS.PFA import PFA
import pandas as pd
from tsfresh import feature_selection
import multiprocessing as mp


class FeatTS(object):
    """
    	FeatTS method for time series clustering.

    	Parameters
    	----------
    	n_clusters : int, optional
    		Number of clusters (default is 2).

    	n_jobs : int, optional
    		Number of jobs tun run in parallel the graph computation for each length
    		(default is 1).

    	value_PFA : float, optional
    		Value of explained variance
    		(default is 0.9).

    	max_numb_feat : int, optional
    		Number of features max to adopt for the graph
    		(default is 20).

    	random_feat : bool, optional
    		Pick features in a random way and not based on importance
    		(default is False).

    	threshold_community : float, optional
    		Threshold of closeness between two values
    		(default is 0.8).

    	algorithm_community : ['Greedy','kClique','SLPA'], optional
    		Type of community detecion
    		(default is Greedy).
    	"""

    def __init__(self, n_clusters, n_jobs=1, value_PFA=0.9, max_numb_feat=20,
                 random_feat=False, threshold_community=0.8, algorithm_community='Greedy') :
        """
        initialize kGraph method
        """
        self.n_clusters = n_clusters
        self.n_jobs = n_jobs
        self.value_PFA = value_PFA
        self.max_numb_feat = max_numb_feat
        self.random_feat = random_feat
        self.threshold_community = threshold_community
        self.algorithm_community = {algorithm_community:{}}

    def fit(self, X, y=[], train_semi_supervised=0):
        """
        compute kGraph on X

        Parameters
        ----------
        X : array of shape (n_samples, n_timestamps)
        	Training instances to cluster.

        y : array of labels (n_samples)

        train_perc : percentage of semi-supervision (float)

        Returns
        -------
        self : object
        	Fitted estimator.
        """
        if y != []:
            datasetAdapted = {"listOut": util.adaptTimeSeriesUCR(X),'series': pd.Series((str(i) for i in y)),
                              "listOfClass": list(str(i) for i in y)}
        else:
            datasetAdapted = {"listOut": util.adaptTimeSeriesUCR(X), 'series': pd.Series(list(str(-100) for i in range(X.shape[0]))),
                              "listOfClass": list(-100 for i in range(X.shape[0]))}

        featPFA, features_filtered_direct = self.__features_extraction_selection(datasetAdapted, train_semi_supervised)
        matrixNsym = self.__community_and_matrix_creation(featPFA, datasetAdapted, features_filtered_direct)
        self.labels_ = self.__cluster(matrixNsym, datasetAdapted)



    def __features_extraction_selection(self,datasetAdapted, train_semi_supervised):


        # Create the dataframe for the extraction of the features
        listOut = datasetAdapted["listOut"]
        listOfClass = datasetAdapted["listOfClass"]

        filtreFeat, seriesAcc, features_filtered_direct = util.extractFeature(listOut, listOfClass, train_semi_supervised)
        pfa = PFA()
        features_filtered_direct = util.cleaning(features_filtered_direct)
        if train_semi_supervised > 0:
            # Extract the relevance for each features and it will be ordered by importance
            ris = feature_selection.relevance.calculate_relevance_table(filtreFeat, seriesAcc, ml_task="classification")

            ris = ris.sort_values(by='p_value')

            if self.random_feat:
                ris = util.randomFeat(ris, self.max_numb_feat)

            listOfFeatToUse = []
            for t in range(self.max_numb_feat):
                listOfFeatToUse.append(ris["feature"][t])

            dfFeatUs = pd.DataFrame()
            for x in range(len(listOfFeatToUse)):
                dfFeatUs[listOfFeatToUse[x]] = features_filtered_direct[listOfFeatToUse[x]]
            featPFA = pfa.fit(dfFeatUs)
        else:
            featPFA = pfa.fit(features_filtered_direct)

        return featPFA, features_filtered_direct
    def __community_and_matrix_creation(self, featPFA, datasetAdapted, features_filtered_direct):

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


        pool = mp.Pool(self.n_jobs)

        # Creation of graph and extraction of community detection
        for feature in listOfFeat:
            pool.apply_async(util.getCommunityDetectionTrain, args=(feature, features_filtered_direct, listOfId,
                                                                    self.threshold_community, self.n_clusters, self.algorithm_community),
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

    def __cluster(self, matrixNsym, datasetAdapted):
        # print("Clustering Creation....")
        # List of the cluster created in the training set. It will be used later for the intersaction
        # with the cluster extract from the testing.
        listOfId = set(datasetAdapted["listOut"]["id"])
        series = datasetAdapted["series"]

        listOfCommFindTest = util.getCluster(matrixNsym, listOfId, self.n_clusters)

        listOfCommFindTest = util.createSet(listOfCommFindTest, self.n_clusters)

        # Modify the index of the TimeSeries with their classes
        y_pred = [0 for x in range(len(series))]
        for value in range(len(listOfCommFindTest)):
            for ind in listOfCommFindTest[value]["cluster"]:
                y_pred[ind] = value

        return y_pred