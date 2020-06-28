# cEnsemble

## Running 

In the `testFeatureExtraction.py` we can found the main file where we can set the parameter for launch the code. 

They could be found from the **23** to **32** rows.


```python
    # Choice of the number of clusters k
    clusterK = 2
    
    # Name of the dataset
    nameDataset = "ECG200"

    # Threshold of the distance
    threshold = 0.8
    
    # Percentage of number of class to use
    trainFeatDataset = 0.2
```

## Configuration File

For test some other dataset it's very important to create a *.tsv* file where the first column will be the class of the time series
and then all the points of the latter:
<table>
  <tr>
    <th>Classe</th>
    <th>1</th>
    <th>2</th>
    <th>3</th>
    <th>4</th>
    <th>5</th>
    <th>...</th>
    
  </tr>
  <tr>
    <td>0</td>
    <td>2.5</td>
    <td>2.8</td>
    <td>2.2</td>
    <td>2.1</td>
    <td>3.8</td>
    <td>...</td>
  </tr>
  
  <tr>
    <td>1</td>
    <td>10.5</td>
    <td>12.1</td>
    <td>11.2</td>
    <td>10.3</td>
    <td>14.8</td>
    <td>...</td>
  </tr> 
  
  <tr>
    <td>0</td>
    <td>1.5</td>
    <td>1.9</td>
    <td>2.2</td>
    <td>2.9</td>
    <td>3.3</td>
    <td>...</td>
  </tr> 
  <tr>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
  </tr> 
</table>

The `.tsv` should have the same name of the folder where it is contained, so if for example the name is `dataset.tsv` it should be in
the folder named `dataset`. And for test the code just put `dataset` in line **27** in this way:
```python
# Name of the dataset
nameDataset = ["dataset"]
```
