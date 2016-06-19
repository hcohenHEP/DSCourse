# Faulty Version Detector

We present here an approach for the detection of faulty production versions using various time-series analysis and activity features. Using a three months data and a trained machine learning algorithm we were able to predict whether a deployed production version is considered faulty. This study was preformed on data provided by Microsoft compony regarding their product, Application Insights.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisities

The tutorial is written in IPython notebook. The code itself depends on scipy common packages such as numpy, pandas, sklearn etc..

The data files should be located as follows (or chagned via data_path variable in general module): 
* Initial three months data should be put in local folder named **data**
* New data for classification should be put in local folder named **new_data**

### Running
Running is through the IPython notebooks but the ruuning is also possible via the code itself.

Mining

```
import mining
import general
general.data_path="data/"
df = mining.preprocess_data()
```

DTW
```
import dtw
df = dtw.DTW_AJAXV()

```

Clustering/Labelling

```
import labeling
labeling.cluster_versions()

```

Training BDT

```
import mva
mva.train()
```

Predicting 

```
import mva
mva.predict(df)
```

## Authors

* **Hadar Cohen**
* **Yury Lechinsky**
