from collections import defaultdict

from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np


class PFA(object):
    def __init__(self, q=None):
        self.q = q

    def fit(self, X, expl_var=0.9):
        if not self.q:
            self.q = X.shape[1]

        sc = StandardScaler()
        X_trans = sc.fit_transform(X)
        # Choice of the Explained Variance
        pca = PCA(expl_var)
        pca.fit(X_trans)
        princComp = len(pca.explained_variance_ratio_)
        A_q = pca.components_.T

        kmeans = KMeans(n_clusters=princComp, n_init="auto").fit(A_q)
        clusters = kmeans.predict(A_q)
        cluster_centers = kmeans.cluster_centers_

        dists = defaultdict(list)
        for i, c in enumerate(clusters):
            dist = euclidean_distances([A_q[i, :]], [cluster_centers[c, :]])[0][0]
            dists[c].append((i, dist))

        self.indices_ = [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]
        self.features_ = X_trans[:, self.indices_]
        listFeat = []
        for x in self.indices_:
            listFeat.append(X.columns[x])

        return listFeat
