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
from FeatTS import FeatTS

if __name__ == '__main__':

    dataCof = load_classification("Coffee")
    X = np.squeeze(dataCof[0], axis=1)
    y = dataCof[1].astype(int)

    featTS = FeatTS(n_clusters=2)
    featTS.fit(X,y,train_semi_supervised=0.2)
    print(adjusted_mutual_info_score(featTS.labels_,y))
```