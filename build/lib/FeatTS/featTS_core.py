import pandas

import FeatTS.utilFeatExtr as util
from FeatTS.PFA import PFA
import pandas as pd
from tsfresh import feature_selection
import multiprocessing as mp
import numpy as np

class FeatTS(object):
    """
    	FeatTS method for time series clustering.

    	Parameters
    	----------
    	n_clusters : int, optional
    		Number of clusters (default is 2).

    	n_jobs : int, optional
    		Number of jobs tun run in parallel the graph computation for each length
    		(default is 4).

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
        initialize FeatTS method
        """
        self.n_clusters = n_clusters
        self.n_jobs = n_jobs
        self.value_PFA = value_PFA
        self.max_numb_feat = max_numb_feat
        self.random_feat = random_feat
        self.threshold_community = threshold_community
        self.algorithm_community = {algorithm_community:{}}
        self.feats_selected_ = []

    def fit(self, X, labels=None, external_feat: pd.DataFrame = None):
        """
        compute FeatTS on X

        Parameters
        ----------
        X : array of shape (n_samples, n_timestamps)
        	Training instances to cluster.

        y : dict of labels {idx:class}

        train_perc : percentage of semi-supervision (float)

        external_feat : features to use in combination with the features extracted (Dataframe)

        Returns
        -------
        self : object
        	Fitted estimator.
        	:param external_feat:
        	:param train_semi_supervised:
        """
        if external_feat is not None and X.shape[0] != external_feat.shape[0] :
            raise ValueError("The external features should have a feature value for each time series in input")

        if labels is not None:
            datasetAdapted = {"listOut": util.adaptTimeSeriesUCR(X),'labels': labels}

        else:
            datasetAdapted = {"listOut": util.adaptTimeSeriesUCR(X)}

        self.feats_selected_, features_filtered_direct = self.__features_extraction_selection(datasetAdapted, external_feat, self.value_PFA)


        matrixNsym = self.__community_and_matrix_creation(self.feats_selected_, datasetAdapted, features_filtered_direct)
        self.labels_ = self.__cluster(matrixNsym, datasetAdapted)

    def __features_extraction_selection(self,datasetAdapted, external_feat, value_PFA):

        # Create the dataframe for the extraction of the features
        listOut = datasetAdapted["listOut"]

        features_filtered_direct = util.extractFeature(listOut, external_feat=external_feat)

        if external_feat is not None:
            external_feat = features_filtered_direct[external_feat.columns.tolist()].copy()
            # features_filtered_direct.drop(columns=external_feat.columns.tolist(), inplace=True)

        pfa = PFA()
        features_filtered_direct = util.cleaning(features_filtered_direct)

        if 'labels' in list(datasetAdapted.keys()):
            allAcc = list(datasetAdapted["labels"].keys())
            seriesAcc = pd.Series((datasetAdapted["labels"][i] for i in allAcc))
            filtreFeat = features_filtered_direct.loc[allAcc].reset_index(drop=True)

            multiclass = False
            significant_class = 1
            if len(seriesAcc.unique()) > 2:
                multiclass = True
                significant_class = len(seriesAcc.unique())


            if 'id' in filtreFeat.keys():
                filtreFeat = filtreFeat.drop('id', axis='columns')
            elif 'index' in filtreFeat.keys():
                filtreFeat = filtreFeat.drop('index', axis='columns')


            # Extract the relevance for each features and it will be ordered by importance
            ris = feature_selection.relevance.calculate_relevance_table(filtreFeat, seriesAcc,
                                                                        ml_task="classification",
                                                                        n_jobs=self.n_jobs,
                                                                        multiclass=multiclass,
                                                                        n_significant=significant_class)
            if external_feat is not None:
                ris = ris[~ris['feature'].isin(external_feat.columns.tolist())]

            if multiclass:
                p_value_columns = [col for col in ris.columns if col.startswith('p_value')]
                # Replace NaN values with inf in the p_value columns
                ris[p_value_columns] = ris[p_value_columns].fillna(np.inf)
                # Sum the p_value columns
                ris['p_value'] = ris[p_value_columns].sum(axis=1)

            ris = ris.sort_values(by='p_value')

            if self.random_feat:
                ris = util.randomFeat(ris, self.max_numb_feat)

            listOfFeatToUse = []
            for t in range(self.max_numb_feat):
                listOfFeatToUse.append(ris["feature"][t])

            dfFeatUs = pd.DataFrame()
            for x in range(len(listOfFeatToUse)):
                dfFeatUs[listOfFeatToUse[x]] = features_filtered_direct[listOfFeatToUse[x]]
            featPFA = pfa.fit(dfFeatUs, value_PFA)
        else:
            featPFA = pfa.fit(features_filtered_direct, value_PFA)

        if external_feat is not None:
            featPFA.extend(external_feat.columns.tolist())
            # Identify columns in external_feat that are not in features_filtered_direct
            non_overlapping_columns = external_feat.columns.difference(features_filtered_direct.columns)
            # Select only the non-overlapping columns from external_feat
            external_feat_non_overlapping = external_feat[non_overlapping_columns]
            # Perform the join with the non-overlapping columns
            features_filtered_direct = features_filtered_direct.join(external_feat_non_overlapping)

        return featPFA, features_filtered_direct

    def __community_and_matrix_creation(self, featPFA, datasetAdapted, features_filtered_direct):
        listOfId = set(datasetAdapted["listOut"]["id"])
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

        listOfCommFindTest = util.getCluster(matrixNsym, listOfId, self.n_clusters)

        listOfCommFindTest = util.createSet(listOfCommFindTest, self.n_clusters)

        # Modify the index of the TimeSeries with their classes
        y_pred = [0 for x in range(len(listOfId))]
        for value in range(len(listOfCommFindTest)):
            for ind in listOfCommFindTest[value]["cluster"]:
                y_pred[ind] = value
        return y_pred