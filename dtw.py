
"""
======================================================
Dynamic Time Warping
======================================================
Dynamic time warping (DTW) is an algorithm for measuring
similarity between two temporal sequences which may vary in time or speed.

Here it's beeing used for measuring how the ajax fail/Succes
rates are behaving against the views request the same version got

"""

import os
import sys
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from general import *
from math import *
import ajax
import gc
import concurrent.futures

def DTWDistance(s1, s2):
    """Returns the DTW between two time series.
    :param s1: Series 1
    :param s2: Series 2
    """
    DTW={}

    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return sqrt(DTW[len(s1)-1, len(s2)-1])

def get_ajax_series(prod,start_time,end_time):
    """Return the time series of the ajax behaviour for a given production version
        at a given time period

    :param prod: the production version
    :param state: True/False for Fail/Success states accordingly
    :param start_time: The start date for the time series
    :param end_time: The end date for the time series
    """
    df = load_df("ajax_events",start_time,end_time)
    df["IsFail"] = df["Result"]=="Fail"

    #Sucess
    df_kpi = df.query("IsFail==False" +  " & Prod=='" + prod + "'")
    dfi_kpi = df_kpi.resample("1h").count()
    vals_s = np.array(dfi_kpi.IsFail,dtype='float64')
    #Fails
    df_kpi = df.query("IsFail==True"  +  " & Prod=='" + prod + "'")
    dfi_kpi = df_kpi.resample("1h").count()
    vals_f = np.array(dfi_kpi.IsFail,dtype='float64')
    #No need for these anymore - clear the memory
    del df,df_kpi,dfi_kpi
    gc.collect()

    return vals_s/np.linalg.norm(vals_s),vals_f/np.linalg.norm(vals_f)

def get_view_series(prod,start_time,end_time):
    """Return the time series of the views behaviour for a given production version
        at a given time period

    :param prod: the production version
    :param start_time: The start date for the time series
    :param end_time: The end date for the time series
    """
    df = load_df("views",start_time,end_time)
    df_kpi = df.query("Prod=='" + prod + "'")
    dfi_kpi = df_kpi.resample("1h").count()
    vals = np.array(dfi_kpi.Prod,dtype='float64')
    #No need for these anymore - clear the memory
    del df
    gc.collect()

    return vals/np.linalg.norm(vals)

def DTW_AJAXV(df=None, new_data=False):
    """Return the df data  with the additional features of DTW

    :param df: data frame with initial mining features
    :param new_data: boolean that states if we're calculating for the first
                     time or ongoing version classifer.
                     Affects the saving of the data to the model files
    """
    if new_data == False:
        df = pd.read_csv('DataMined.csv',encoding='latin_1',
                         index_col=0, parse_dates=False, low_memory=False)

    #Create new column in the data for the specific DTW
    df.loc[:,"DTW_FV"] = np.zeros(len(df)) -9
    df.loc[:,"DTW_SV"] = np.zeros(len(df)) -9
    df.loc[:,"DTW_FS"] = np.zeros(len(df)) -9
    for prod in df.index:
        print("DTW Calculation for " + str(prod))
        start_time = np.array(df.ix[df.index==prod,'Start_Time'])[0]
        end_time = np.array(df.ix[df.index==prod,'End_Time'])[0]

        #Using concurrent allows us to control pandas & python bad memory managment
        aj_s = aj_f = vi =  None
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            aj_s,aj_f = executor.submit(get_ajax_series, prod,start_time,end_time).result()
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            vi = executor.submit(get_view_series, prod,start_time,end_time).result()

        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            df.loc[prod,"DTW_FV"] = executor.submit(DTWDistance,aj_f,vi).result()
            df.loc[prod,"DTW_SV"] = executor.submit(DTWDistance,aj_s,vi).result()
            df.loc[prod,"DTW_FS"] = executor.submit(DTWDistance,aj_s,aj_f).result()

        del aj_s,aj_f,vi
        gc.collect()

    df = df.replace([np.inf, -np.inf], 1.5)
    if new_data == False:
        df.to_csv("DataMined.csv")

    return df
