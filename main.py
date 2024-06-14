from aeon.datasets import load_classification
from sklearn.metrics import adjusted_mutual_info_score
import numpy as np
from FeatTS import FeatTS

if __name__ == '__main__':

    dataCof = load_classification("Coffee")
    X = np.squeeze(dataCof[0], axis=1)
    y = dataCof[1].astype(int)

    featTS = FeatTS(n_clusters=2)
    featTS.fit(X,y,train_semi_supervised=0.2)
    print(adjusted_mutual_info_score(featTS.labels_,y))