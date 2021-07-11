# FeatTS

## Running 

python_version > '3.7' 

In the `testFeatureExtraction.py` we can found the main file where we can set the parameter for launch the code. 
The dataset used for the test should be inside the _DatasetTS_ folder. Inside this folder, you have to create
a folder with the same name of the dataset.

At the end of the computation, a file called **experiments.tsv** will contain all the results obained on the datasets.


## Configuration File

For test some other dataset it's very important to create a *.tsv* file where the first column will be the class of the time series
and then all the points of the latter:
<table>
  <tr>
    <th>Class</th>
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
