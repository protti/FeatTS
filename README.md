# FeatTS

## Paper

At this link you can find the paper related at this code: http://openproceedings.org/2021/conf/edbt/p270.pdf

## Running 

The package could be installed with the following command:

```python
pip install FeatTS
```

## Usage

In order to play with FeatTS, please check the [UCR Archive](https://www.timeseriesclassification.com/). We depict below a code snippet demonstrating how to use FeatTS.

```python
from aeon.datasets import load_classification
from sklearn.metrics import adjusted_mutual_info_score
import numpy as np
from featts import FeatTS

if __name__ == '__main__':
    dataCof = load_classification("Coffee")
    X = np.squeeze(dataCof[0], axis=1)
    y = dataCof[1].astype(int)
    labels = {0: y[0], 1: y[1], 5: y[5], 6: y[0]}  # semi-supervised mode
    n_clusters = 2  # Number of clusters

    featTS = FeatTS(n_clusters=2)
    featTS.fit(X, labels=labels)
    print(adjusted_mutual_info_score(featTS.labels_, y))
```

It is also possible to add some external features to the computation. These features will help the choice of the 
best features:

```python
from aeon.datasets import load_classification
from sklearn.metrics import adjusted_mutual_info_score
import numpy as np
import pandas as pd
from featts import FeatTS

if __name__ == '__main__':
    dataCof = load_classification("Coffee")
    X = np.squeeze(dataCof[0], axis=1)
    y = dataCof[1].astype(int)
    labels = {0: y[0], 1: y[1], 5: y[5], 6: y[0]}  # semi-supervised mode
    external_feat = pd.DataFrame({'LEN': y})

    featTS = FeatTS(n_clusters=2)
    featTS.fit(X, labels=labels, external_feat=external_feat)
    print(adjusted_mutual_info_score(featTS.labels_, y))
    print(featTS.feats_selected_)
```
