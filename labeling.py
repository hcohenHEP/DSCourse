"""
======================================================
Labelling
======================================================
This model is responsible for labelling the mined data for different classes.
It uses clustering in order to determine the label of each production version.

The result will be saved to new data-set: DataLabeld.csv
"""

import os
import sys
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
import gc

def cluster_versions():
    """Return the data-set (previously mined and saved at DataMined.csv) as a labelled
       data-set. Clustering the data in the parameter-space of RatioFS and Views.
       For now combining "Excellent" and "Good" labels to just one label.

       Present the final clustering result plot and saves the labelled data to
       DataLabeled.csv
    """

    df = pd.read_csv('DataMined.csv' ,encoding='latin_1', index_col=0, parse_dates=False)
    df["Class"] = np.ones(len(df))
    data = df.ix[df.RatioFS<0.05][['RatioFS','Views']]
    data = np.array(data)
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)
    X = data
    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)
    # create clustering estimators
    birch = cluster.Birch(n_clusters=3)
    birch.fit(X)
    if hasattr(birch, 'labels_'):
        y_pred = birch.labels_.astype(np.int)
    else:
        y_pred = birch.predict(X)

    # plot
    plt.figure(3)
    plt.suptitle("Versions Clustering", fontsize=15)
    plt.scatter(data[:, 0], data[:, 1], color=colors[y_pred].tolist(), s=10)
    plt.xlabel("Fails/Success")
    plt.ylabel("Views")
    plt.show()

    #Combine the 'Excellent' & 'Good' for now, future version should include
    #  these properties as well
    y_pred[y_pred<2]*=0
    y_pred[y_pred>1]=1
    df.ix[df.RatioFS<0.05,"Class"]=y_pred
    df.to_csv("DataLabeled.csv")
