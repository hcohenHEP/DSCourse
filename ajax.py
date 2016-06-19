"""
======================================================
Ajax
======================================================
This Ajax module is responsible for proper production
version plots.

Some functions are obselte and used for initial data studies.
"""

import os
import sys
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import general

def prep_data(df):
    """Returns the df with additional columns
        'IsFail' - boolean if the Result state of the request has faild
        'AjaxDuration' - splitting of the duration to Fast,Normal & Slow

    :param df: the data frame for ajax_events
    """
    df["IsFail"] = df["Result"]=="Fail"
    #df = df[["Prod","IsFail","StatusText","duration"]]
    labels=['Fast', 'Normal', 'Slow']
    rec_labels, bins= pd.qcut(df.duration, 3, retbins=True, labels=labels)
    df.loc[:,"AjaxDuration"] = np.array(rec_labels)
    for label in labels:
        df.loc[:,label] = df["AjaxDuration"] == label
    return df

def plot_ajax_dates(df, interval,name,versions,mode="Fail",start_time="2015-10-01",end_time="2016-10-31"):
    index = 0
    max_fails = 0
    ax = None
    leg_labels=[]
    for version in versions:
        if version != "nan":
            #print("Version" + version)
            color = get_color_index(index)
            if mode == "Fail":
                q_str = "IsFail==True & Prod=='" + version + "'"
            else:
                q_str = "IsFail==False & Prod=='" + version + "'"
            ax,tmp_max = plot_fails(df=df,interval=interval,color=color,
                                    prev_ax=ax,q_str=q_str,
                                    max_var="IsFail",start_time=start_time,end_time=end_time)
            if tmp_max != -1:
                leg_labels.append(str(version)[:11])
                index = index + 1
                if tmp_max > max_fails:
                    max_fails = tmp_max



    plt.legend(leg_labels,loc='best')
    plt.ylim([0,max_fails])
    plt.suptitle("Prod Ver " + mode + " For: " + name, fontsize=30)
    plt.ylabel(mode + " Count")
    plt.savefig("output/AjaxFull" + mode + "_" + name + "_" + interval + '.png')

def plot_ajax_version_errors_stacked(df,interval,version,start_time="2015-10-01",end_time="2016-10-31"):
    status_texts = df.drop_duplicates("StatusText")
    index = 0
    max_fails = 0
    ax = None
    leg_labels=[]
    hists = []
    for status in status_texts["StatusText"]:
        if str(status) != "nan":
            #print("Version" + version)
            color = get_color_index(index)
            q_str = "IsFail==True & Prod=='" + version + "' & StatusText=='" + str(status) + "'"
            df_kpi = df.query(q_str)
            df_kpi = df_kpi.ix[start_time:end_time].resample(interval, how=["count"])
            if df_kpi.shape[0] != 0:
                hists.append(np.array(df_kpi["StatusText"]))
                leg_labels.append(str(status))
                index = index + 1



    #check if we've got any data at all
    if len(leg_labels)!=0:
        plt.hist(hists,24,stacked=True,normed=False)
        plt.legend(leg_labels,loc='best')
        plt.ylim([0,max_fails])
        plt.suptitle(version[:11] + " Fails " + " For: " + start_time + ":" + end_time, fontsize=30)
        plt.ylabel("Counts")
        plt.savefig("output/AjaxFailStacked_" + version[:11]  + "_" + interval + '.png')

def plot_ajax_version_errors(df,interval,version,save=False):
    plt.figure()
    status_texts = df.drop_duplicates("StatusText")
    index = 0
    max_fails = 0
    ax = None
    leg_labels=[]
    for status in status_texts["StatusText"]:
        if str(status) != "nan":
            #print("Version" + version)
            color = general.get_color_index(index)
            q_str = "IsFail==True & Prod=='" + version + "' & StatusText=='" + str(status) + "'"
            ax,tmp_max = general.plot_fails(df=df,interval=interval,color=color,
                                    prev_ax=ax,q_str=q_str,
                                    max_var="IsFail")
            if tmp_max != -1:
                leg_labels.append(str(status))
                index = index + 1
                if tmp_max > max_fails:
                    max_fails = tmp_max


    #check if we've got any data at all
    if len(leg_labels)!=0:
        plt.legend(leg_labels,loc='best')
        plt.ylim([0,max_fails])
        plt.suptitle(version[:11] + " Fails ", fontsize=30)
        plt.ylabel("Counts")
        if save == True:
            plt.savefig("output/AjaxFail_" + version[:11]  + "_" + start_time[8:] + "-" + end_time[8:] + '.png')


def plot_ajax_day(df, interval,name,mode="Fail"):
    ver_df = df.drop_duplicates("Prod")
    index = 0
    max_fails = 0
    ax = None
    leg_labels=[]
    for version in ver_df["Prod"]:
        if str(version) != "nan":
            #print("Version" + version)
            color = get_color_index(index)
            if mode == "Fail":
                q_str = "IsFail==True & Prod=='" + str(version) + "'"
            else:
                q_str = "IsFail==False & Prod=='" + str(version) + "'"
            ax,tmp_max = plot_fails(df=df,interval=interval,color=color,
                                    prev_ax=ax,q_str=q_str,
                                    max_var="IsFail")
            if tmp_max != -1:
                leg_labels.append(str(version)[:11])
                index = index + 1
                if tmp_max > max_fails:
                    max_fails = tmp_max



    plt.legend(leg_labels,loc='best')
    plt.ylim([0,max_fails])
    plt.suptitle("Prod Ver " + mode + " For: " + name, fontsize=30)
    plt.ylabel(mode + " Count")
    plt.savefig("output/Ajax" + mode + "_" + name + "_" + interval + '.png')


def plot_ajax_slow_duration(df, interval,label):
    ver_df = df.drop_duplicates("Prod")
    index = 0
    max_fails = 0
    ax = None
    leg_labels=[]
    for version in ver_df["Prod"]:
        if str(version) != "nan":
            #print("host" + host)
            color = general.get_color_index(index)
            ax,tmp_max = general.plot_fails(df=df,interval=interval,color=color,
                                    q_str=label + "==True & Prod=='" + str(version) + "'",
                                    max_var="duration",prev_ax=ax)
            if tmp_max != -1:
                leg_labels.append(str(host))
                index = index + 1
                if tmp_max > max_fails:
                    max_fails = tmp_max



    plt.legend(leg_labels,loc='best')
    plt.ylim([0,max_fails])
    plt.suptitle("Slow " + version[:11], fontsize=30)
    plt.ylabel("Count")
    plt.savefig("output/AjaxSlow_" + version[:11]  + "_" + start_time[8:] + "-" + end_time[8:] + '.png')
