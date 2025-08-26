import pandas as pd
from aeon.datasets import load_classification
from sklearn.metrics import adjusted_mutual_info_score
import numpy as np
from featts.FeatTS import FeatTS
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


def main():
    coffee_dataset = load_classification("Coffee")
    X = np.squeeze(coffee_dataset[0], axis=1)
    y = coffee_dataset[1].astype(int)
    print("Input data shape:", X.shape)
    scores = []

    for i in range(5):
        print("Run", i)
        start = time.time()
        model = FeatTS(n_clusters=2, n_jobs=4)
        out_labels = model.fit_predict(X)
        scores.append(adjusted_mutual_info_score(out_labels, y))
        end = time.time()
        print(f"Clustering computed in {end - start:.2f} seconds.")
        print("AMI:", adjusted_mutual_info_score(out_labels, y))
    print("Average score:", np.mean(scores))


if __name__ == '__main__':
    main()
