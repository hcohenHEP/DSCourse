#!/usr/bin/python

import os
import sys
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import general


def plot_views_day(df, interval,name):
    ver_df = df.drop_duplicates("Prod")
    index = 0
    max_fails = 0
    ax = None
    leg_labels=[]
    plt.figure()
    for version in ver_df["Prod"]:
        if str(version) != "nan":
            #print("Version" + version)
            color = get_color_index(index)
            q_str = "Prod=='" + str(version) + "'"
            ax,tmp_max = plot_fails(df=df,interval=interval,color=color,
                                    prev_ax=ax,q_str=q_str,
                                    max_var="Prod")
            if tmp_max != -1:
                leg_labels.append(str(version)[:11])
                index = index + 1
                if tmp_max > max_fails:
                    max_fails = tmp_max



    plt.legend(leg_labels,loc='best')
    plt.ylim([0,max_fails])
    plt.suptitle("Views Ver For: " + name, fontsize=30)
    plt.ylabel("Count")
    plt.savefig("output/Views" + "_" + name + "_" + interval + '.png')


def plot_views_version(df, interval,version,save=False):
    plt.figure()
    color = general.get_color_index(1)
    q_str = "Prod=='" + version + "'"
    ax,max_views = general.plot_fails(df=df,interval=interval,color=color,
                            prev_ax=None,q_str=q_str,
                            max_var="Prod")


    plt.legend([version],loc='best')
    plt.ylim([0,max_views ])
    plt.suptitle("Views For: " + version, fontsize=30)
    plt.ylabel("Count")
    if save == True:
        plt.savefig("Views" + "_" + version + "_" + interval + '.png')
