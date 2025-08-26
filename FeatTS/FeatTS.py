import os
import warnings
from typing import Optional

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import utilFeatExtr as util
from .PFA import PFA
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from tsfresh import feature_selection
import numpy as np
import time
from kneed import KneeLocator


class FeatTS(object):
    """
    featts method for time series clustering.

    Parameters
    ----------
    n_clusters: int, optional
            Number of clusters (default is 2).

    n_jobs: int, optional
            Number of jobs to run in parallel the graph computation for each length
            (default is 4).

    community_detection_jobs: int, optional
        Number of jobs to run in parallel the community detection algorithm.
        (default is 1).
        Be careful when increasing this number, as the community detection algorithm is memory-intensive.

    pfa_value: float, optional
            Value of explained variance
            (default is 0.9).

    max_num_feat: int, optional
            Number of features max to adopt for the graph
            (default is 20).

    random_feat: bool, optional
            Pick features in a random way and not based on importance
            (default is False).

    community_threshold: float, optional
            Threshold of closeness between two values
            (default is 0.8).

    community_algorithm: ['Greedy','kClique','SLPA'], optional
            Type of community detection algorithm to use
            (default is Greedy).
    """

    def __init__(
            self,
            n_clusters: Optional[int] = None,
            n_jobs: int = 4,
            community_detection_jobs: int = 4,
            pfa_value: float = 0.9,
            max_num_feat: int = 20,
            random_feat: bool = False,
            community_threshold: float = 0.8,
            community_algorithm: str = "Greedy",
            pfa_ext_feats: bool = False,
            verbose: bool = True,
    ):
        """
        initialize featts method
        """
        self.n_clusters = n_clusters
        self.n_jobs = n_jobs
        self.cm_jobs = community_detection_jobs
        self.pfa_value = pfa_value
        self.max_num_feat = max_num_feat
        self.random_feat = random_feat
        self.community_threshold = community_threshold
        self.community_algorithm = {community_algorithm: dict()}
        self.pfa_ext_feats = pfa_ext_feats
        self.feats_selected_ = list()
        self.ext_feats_selected_ = list()
        self.feats_variance_ = dict()
        self.ext_feats_variance_ = dict()
        self.sym_matrix_ = None
        self.labels_ = None
        self._adapted_dataset = None
        self.verbose = verbose

    def fit(
            self, x: np.ndarray, labels: dict = None, external_feat: pd.DataFrame = None
    ):
        """
        Apply featts on X.

        Parameters
        ----------
        x: array of shape (n_samples, n_timestamps)
            Training instances to cluster.

        labels: dict of labels {idx:class}

        external_feat: features to use in combination with the features extracted (Dataframe)

        Returns
        -------
        self : object
            Fit estimator.
        """
        if external_feat is not None and x.shape[0] != external_feat.shape[0]:
            raise ValueError(
                "The external features should have a feature value for each time series in input"
            )

        if labels is not None:
            adapted_dataset = {"listOut": util.adaptTimeSeriesUCR(x), "labels": labels}

        else:
            adapted_dataset = {"listOut": util.adaptTimeSeriesUCR(x)}

        start_feat_sel = time.time()

        self.feats_selected_, features_filtered_direct = (
            self.__features_extraction_selection(
                adapted_dataset, external_feat, self.pfa_value
            )
        )
        if self.verbose:
            print(f"Time to select features: {time.time() - start_feat_sel}")
        start_comm_detect = time.time()

        matrix_n_sym = self.__community_and_matrix_creation(
            self.feats_selected_, adapted_dataset, features_filtered_direct
        )

        self.sym_matrix_ = matrix_n_sym
        self._adapted_dataset = adapted_dataset

        if self.verbose:
            print(
                f"Time to Detect Communities and Create Matrix: {time.time() - start_comm_detect}"
            )

        # if the number of desired clusters was not given, use PCA on the selected features from PFA to determine it
        if self.n_clusters is None:
            scaler = StandardScaler()
            df_standardized = scaler.fit_transform(
                features_filtered_direct[self.feats_selected_]
            )

            # PCA on the selected features
            pca = PCA()
            pca.fit(df_standardized)

            # get the variance for the variables
            explained_variance = pca.explained_variance_ratio_

            self.n_clusters = KneeLocator(
                np.asarray([(range(len(explained_variance)))]).squeeze(),
                np.asarray(explained_variance).squeeze(),
                curve="convex",
                direction="decreasing",
            ).knee

    def predict(self) -> list[int]:
        """
        Predict the cluster for each instance in X (run the clustering algorithm).
        :return: list of cluster labels.
        """
        if self.verbose:
            print(f"Clustering with {self.n_clusters} clusters.")
        start_clustering = time.time()
        self.labels_ = self.__cluster()

        if self.verbose:
            print(f"Time to cluster: {time.time() - start_clustering}")

        return self.labels_

    def fit_predict(
            self, x: np.ndarray, labels: dict = None, external_feat: pd.DataFrame = None
    ) -> list[int]:
        """
        Fit the model and predict the cluster for each instance in X.
        :param x: array of shape (n_samples, n_timestamps)
            Training instances to cluster.
        :param labels: dict of labels {idx:class}
        :param external_feat: features to use in combination with the features extracted (Dataframe)
        :return: list of cluster labels.
        """
        self.fit(x, labels, external_feat)
        return self.predict()

    def __features_extraction_selection(
            self, adapted_dataset, external_feat, pfa_value
    ):

        # Create the dataframe for the extraction of the features
        output_list = adapted_dataset["listOut"]

        if self.pfa_ext_feats:
            features_filtered_direct = util.extractFeature(
                output_list, external_feat=None, n_jobs=self.n_jobs
            )
        else:
            features_filtered_direct = util.extractFeature(
                output_list, external_feat=external_feat, n_jobs=self.n_jobs
            )

        if external_feat is not None and self.pfa_ext_feats:
            external_feat = features_filtered_direct[
                external_feat.columns.tolist()
            ].copy()
            # features_filtered_direct.drop(columns=external_feat.columns.tolist(), inplace=True)

        os.makedirs("../data/unlabeled_data/EW_UD_wfeats", exist_ok=True)
        features_filtered_direct.to_csv(
            f"../data/unlabeled_data/EW_UD_wfeats/extracted_featts_sxf{self.pfa_ext_feats}_pfa{pfa_value}.csv"
        )

        pfa = PFA()
        features_filtered_direct = util.cleaning(features_filtered_direct)

        if "labels" in list(adapted_dataset.keys()):
            all_acc = list(adapted_dataset["labels"].keys())
            series_acc = pd.Series((adapted_dataset["labels"][i] for i in all_acc))
            filtre_feat = features_filtered_direct.loc[all_acc].reset_index(drop=True)

            multiclass = False
            significant_class = 1
            if len(series_acc.unique()) > 2:
                multiclass = True
                significant_class = len(series_acc.unique())

            if "id" in filtre_feat.keys():
                filtre_feat = filtre_feat.drop("id", axis="columns")
            elif "index" in filtre_feat.keys():
                filtre_feat = filtre_feat.drop("index", axis="columns")

            # Extract the relevance for each features and it will be ordered by importance
            ris = feature_selection.relevance.calculate_relevance_table(
                filtre_feat,
                series_acc,
                ml_task="classification",
                n_jobs=self.n_jobs,
                multiclass=multiclass,
                n_significant=significant_class,
            )
            if external_feat is not None:
                ris = ris[~ris["feature"].isin(external_feat.columns.tolist())]

            if multiclass:
                p_value_columns = [
                    col for col in ris.columns if col.startswith("p_value")
                ]
                # Replace NaN values with inf in the p_value columns
                ris[p_value_columns] = ris[p_value_columns].fillna(np.inf)
                # Sum the p_value columns
                ris["p_value"] = ris[p_value_columns].sum(axis=1)

            ris = ris.sort_values(by="p_value")

            if self.random_feat:
                ris = util.randomFeat(ris, self.max_num_feat)

            list_of_feats_to_use = []
            for t in range(self.max_num_feat):
                list_of_feats_to_use.append(ris["feature"][t])

            dfFeatUs = pd.DataFrame()
            for x in range(len(list_of_feats_to_use)):
                dfFeatUs[list_of_feats_to_use[x]] = features_filtered_direct[
                    list_of_feats_to_use[x]
                ]
            pfa_feats = pfa.fit(dfFeatUs, pfa_value)
        else:
            pfa_feats = pfa.fit(features_filtered_direct, pfa_value)

        pd.DataFrame(
            pfa.features_,
            index=range(features_filtered_direct.shape[0]),
            columns=pfa_feats,
        ).to_csv(
            f"../data/unlabeled_data/EW_UD_wfeats/selected_featts_sxf{self.pfa_ext_feats}_pfa{pfa_value}.csv"
        )

        print("Saved features to CSV.")

        selected_ext_feats = pd.DataFrame()

        if external_feat is not None:
            if self.pfa_ext_feats:
                ext_feats_pfa = pfa.fit(external_feat, pfa_value)
                selected_ext_feats = external_feat[ext_feats_pfa]
                if self.verbose:
                    print(
                        "Selected external feats: ",
                        ", ".join(selected_ext_feats.columns.tolist()),
                    )
            else:
                pfa_feats.extend(external_feat.columns.tolist())
                # Identify columns in external_feat that are not in features_filtered_direct
                non_overlapping_columns = external_feat.columns.difference(
                    features_filtered_direct.columns
                )
                # Select only the non-overlapping columns from external_feat
                selected_ext_feats = external_feat[non_overlapping_columns]

            # Perform the join with the non-overlapping columns or PFA-selected features
            features_filtered_direct = features_filtered_direct.join(selected_ext_feats)
            pfa_feats.extend(selected_ext_feats.columns.tolist())

        # Save selected features variance
        self.feats_variance_ = {
            feat: features_filtered_direct[feat].var()
            for feat in pfa_feats
            if feat in features_filtered_direct.columns
        }

        # Manage external features
        if external_feat is not None:
            if self.pfa_ext_feats:
                # If PFA was executed on the external features, save the selected features
                self.ext_feats_selected_ = selected_ext_feats.columns.tolist()
                # Save the variances of the selected external features
                self.ext_feats_variance_ = {
                    feat: selected_ext_feats[feat].var()
                    for feat in self.ext_feats_selected_
                }
            else:
                # If no PFA was executed on the external features, save all external features
                self.ext_feats_selected_ = external_feat.columns.tolist()
                # Save all variances of the external features
                self.ext_feats_variance_ = {
                    feat: external_feat[feat].var()
                    for feat in self.ext_feats_selected_
                }

        return pfa_feats, features_filtered_direct

    def __community_and_matrix_creation(
            self, pfa_feats, adapted_dataset, features_filtered_direct
    ):

        start_matrix_creation = time.time()
        list_of_ids = set(adapted_dataset["listOut"]["id"])
        dict_of_train_info = {}
        # Creation of the features that we want to use
        list_of_feats = pfa_feats

        def collect_train_result(result):
            dict_of_train_info.update(result)

        chunk_size = max(1, len(list_of_feats) // self.cm_jobs)

        feature_chunks = [
            list_of_feats[i: i + chunk_size]
            for i in range(0, len(list_of_feats), chunk_size)
        ]

        if self.verbose:
            print("Using {} workers for community detection".format(self.cm_jobs))

        with ProcessPoolExecutor(max_workers=self.cm_jobs) as executor:
            futures = [
                executor.submit(
                    util.getCommunityDetectionTrain,
                    feature_chunk,
                    features_filtered_direct,
                    list_of_ids,
                    self.community_threshold,
                    0 if self.n_clusters is None else self.n_clusters,
                    self.community_algorithm,
                )
                for feature_chunk in feature_chunks
            ]

            for future in as_completed(futures):
                collect_train_result(future.result())

        if self.verbose:
            print(f"Time to detect communities: {time.time() - start_matrix_creation}")

        cluster_list = list()
        # Creation of list with all the cluster and their weights, used for the creation of CoOccurrence Matrix
        for key in dict_of_train_info.keys():
            for clusterInside in dict_of_train_info[key]["cluster"]:
                dictSing = {
                    "list": list(clusterInside),
                    "weight": dict_of_train_info[key]["weightFeat"],
                }
                cluster_list.append(dictSing)

        # Creation of CoOccurrence Matrix
        # print("Matrix Creation...")

        matrixNsym = util.getTabNonSym(
            cluster_list, list(list_of_ids), n_jobs=self.n_jobs
        )
        if self.verbose:
            print(
                f"Time to Create Matrix General: {time.time() - start_matrix_creation}"
            )
        return matrixNsym

    def __cluster(self):
        # List of the cluster created in the training set. It will be used later for the intersaction
        # with the cluster extract from the testing.
        list_of_ids = set(self._adapted_dataset["listOut"]["id"])

        list_of_comm_find_test = util.getCluster(
            self.sym_matrix_, list_of_ids, self.n_clusters
        )

        list_of_comm_find_test = util.createSet(list_of_comm_find_test, self.n_clusters)

        # Modify the index of the TimeSeries with their classes
        y_pred = [0 for x in range(len(list_of_ids))]
        for value in range(len(list_of_comm_find_test)):
            for ind in list_of_comm_find_test[value]["cluster"]:
                y_pred[ind] = value
        return y_pred
