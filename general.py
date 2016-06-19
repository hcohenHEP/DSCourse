"""
======================================================
General
======================================================
General methods that are shared with the different modules in the software.
Contains different functions to loading the data and some other ploting
"""

import os
import sys
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import gc

#The path for the loading functions to use
data_path = 'data/'

def get_color_index(index):
    """Returns a unique color by index. Used when plotting diffrent series on
        a single plot

    :param index: running index from 0-12
    """
    if index == 0:
        return "blue"
    elif index == 1:
        return "red"
    elif index == 2:
        return "green"
    elif index == 3:
        return "fuchsia"
    elif index == 4:
        return "black"
    elif index == 5:
        return "yellow"
    elif index == 6:
        return "lime"
    elif index == 7:
        return "darkblue"
    elif index == 8:
        return "brown"
    elif index == 9:
        return "cyan"
    elif index == 10:
        return "indigo"
    elif index == 11:
        return "grey"
    elif index == 12:
        return "lightpink"

    return "blue"

def get_file_names(file_name,start_time='0',end_time='0'):
    """Returns a unique color by index. Used when plotting diffrent series on
        a single plot

    :param index: running index from 0-12
    """
    full_files = []
    ongoing = False
    for root,sub_dirs,tmp_files in os.walk(data_path):
        for r in sorted(sub_dirs):
            for r_sub,dirs,files in os.walk(data_path + r):
                for fn in files:
                    if fn == file_name + ".csv":
                        if r_sub.split('/')[-1].startswith("day_" + start_time) == True:
                            ongoing = True
                        if ongoing == True and r_sub.endswith(end_time) == False:
                            full_files.append(r_sub + "/" + fn)
                        if r_sub.endswith(end_time) == True:
                            full_files.append(r_sub + "/" + fn)
                            return full_files

    return full_files

def load_df(file_name,start_time='0',end_time='0'):
    """Returns pandas data frame for the requested data type

    :param file_name: data type to load (views/ajax_events)
    :param start_time: starting date to load from in format of YYYY-MM-DD
    :param end_time: the ending date of loading in format of YYYY-MM-DD
    """
    full_files = get_file_names(file_name,start_time,end_time)
    df = pd.DataFrame()
    for day in full_files:
        #print("Loading " + day)
        df = df.append(pd.read_csv(day ,encoding='latin_1', index_col=0, parse_dates=True,low_memory=False))

    return df



def plot_fails(df, interval,color,q_str,max_var,prev_ax=None):

    df_kpi = df
    if q_str != "":
        df_kpi = df.query(q_str)
    dfi_kpi = df_kpi.resample(interval).count()

    #Check if we even got something
    if dfi_kpi.shape[0] == 0:
        return prev_ax,-1

    ax = dfi_kpi[max_var].plot(grid=True, linewidth=2, figsize=(20, 8), fontsize=15,ax=prev_ax,color=color)
    return ax,np.max(dfi_kpi[max_var])
