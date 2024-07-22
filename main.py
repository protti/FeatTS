import pandas as pd
from aeon.datasets import load_classification
from sklearn.metrics import adjusted_mutual_info_score
import numpy as np
from FeatTS import FeatTS
import time
import random
from collections import defaultdict

def select_random_percent(labels, perc):
    # Group indices by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    # Select 20% of indices randomly for each class
    selected_indices = {}
    for label, indices in class_indices.items():
        num_to_select = max(1, int(len(indices) * perc))  # At least one item should be selected
        selected_indices_for_class = random.sample(indices, num_to_select)
        for idx in selected_indices_for_class:
            selected_indices[idx] = label

    return selected_indices

if __name__ == '__main__':

    dataCof = load_classification("Coffee")
    X = np.squeeze(dataCof[0], axis=1)
    y = dataCof[1].astype(int)
    print(X.shape)
    # external_feat = pd.DataFrame({'LEN':y})
    labels = select_random_percent(y,0.2)
    scores = []
    for i in range(5):
        start = time.time()
        featTS = FeatTS(n_clusters=2, n_jobs=4)
        featTS.fit(X)
        scores.append(adjusted_mutual_info_score(featTS.labels_,y))
        end = time.time()
        print(end-start)
        print(adjusted_mutual_info_score(featTS.labels_,y))
    print(np.mean(scores))
