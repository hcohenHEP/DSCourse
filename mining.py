import general
import os
import sys
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import gc
import copy
import shutil

data = np.array([],dtype=[('Prod', object),('Success', 'int32'), ('Fail', 'int32'),
                          ('Views', 'int32'),('FailAvgDuration','float32'),
                          ('SuccessAvgDuration','float32'),('TotalAvgDuration','float32'),
                          ('Start_Time',object),('End_Time',object),
                          ('AvgSessionDuration','float32'),('Sessions','int32')])
features = list(data.dtype.names)[1:]

ver_counts = {}
ongoing = {}
last_day = ''

def add_new_record(prod,aj_suc, aj_fail,views,f_avgD,s_avgD,t_avgD,start_time,sid_avg,sid_len):
    global data
    data = np.append(data,np.array([(prod,aj_suc,aj_fail,views,f_avgD,s_avgD,
                                     t_avgD,start_time,'',sid_avg,sid_len)],dtype=data.dtype))

def reboot_ongoing():
    global ongoing
    for key in ongoing.keys():
        ongoing[key]=False

def day_mining_views(df):
    df = df[["Prod"]]
    df = df.fillna(-2)

    versions = df[["Prod"]].drop_duplicates("Prod")
    versions = versions[versions.Prod!=-2]

    for version in versions.Prod:
        df_ver = df.query("Prod=='" + version + "'")
        views = int(df_ver.Prod.count())

        row = data[data["Prod"]==version]
        if row.shape[0] != 0:
            row["Views"] += views
            data[data["Prod"]==version] = row

def day_mining_ajax(df,day):
    """Mining the specifc day's data of ajax events.
       Calculating the following features:
            - Start_Time - find the first day of the version appearance
    :param prod: the production version
    :param start_time: The start date for the time series
    :param end_time: The end date for the time series
    """
    #Prep the data and divide it to fails & Success
    df["IsFail"] = df["Result"]=="Fail"
    df = df[["Prod","IsFail","StatusText","duration","Sid"]]
    df = df.fillna(-2)

    versions = df[["Prod"]].drop_duplicates("Prod")
    versions = versions[versions.Prod!=-2]

    global ongoing,last_day,data
    reboot_ongoing()
    s_date = day.split('/')[-2].split(' - ')[0][4:]
    e_date = day.split('/')[-2].split(' - ')[1]
    last_day = e_date

    for version in versions.Prod:
        ongoing[version] = True
        df_ver = df.query("Prod=='" + version + "'")

        df_tmp = df_ver.query("IsFail==True")
        ajax_fail = int(df_tmp.Prod.count())
        f_avgD_n = len(df_tmp.duration)
        f_avgD =0
        if f_avgD_n > 0:
            f_avgD = np.array(df_tmp.duration).mean()

        df_tmp = df_ver.query("IsFail==False")
        ajax_success = int(df_tmp.Prod.count())
        s_avgD_n = len(df_tmp.duration)
        s_avgD = 0
        if s_avgD > 0:
            s_avgD = np.array(df_tmp.duration).mean()

        t_avgD = 0
        if s_avgD_n + f_avgD_n > 0:
            t_avgD = (f_avgD*f_avgD_n + s_avgD*s_avgD_n)/(s_avgD_n+f_avgD_n)


        #SessionID average time in version
        sid_a = df_ver.drop_duplicates("Sid",keep="first")
        sid_b = df_ver.drop_duplicates("Sid",keep="last")
        sid_len = len(sid_a)
        sid_avg = 0
        for sid in sid_a.Sid:
            sid_time = abs( (np.datetime64(sid_b[sid_b.Sid==sid].index) -
                      np.datetime64(sid_a[sid_a.Sid==sid].index)).item().total_seconds() )
            sid_avg+=sid_time
        sid_avg/=sid_len



        row = data[data["Prod"]==version]
        if row.shape[0] == 0:
            add_new_record(version,ajax_success,ajax_fail,0,f_avgD,s_avgD,t_avgD,s_date,sid_avg,sid_len)
            ver_counts[version] = {"Fail":f_avgD_n,"Success":s_avgD_n,"Total":s_avgD_n+f_avgD_n,"SidAvg":sid_len}
        else:
            row["Fail"] += ajax_fail
            row["Success"] += ajax_success
            row["FailAvgDuration"] = (row["FailAvgDuration"]*ver_counts[version]["Fail"] + f_avgD*f_avgD_n)/(f_avgD_n+ver_counts[version]["Fail"])
            row["SuccessAvgDuration"] = (row["SuccessAvgDuration"]*ver_counts[version]["Success"] + f_avgD*f_avgD_n)/(f_avgD_n+ver_counts[version]["Success"])
            row["TotalAvgDuration"] = (row["TotalAvgDuration"]*ver_counts[version]["Total"] + f_avgD*f_avgD_n)/(f_avgD_n+ver_counts[version]["Total"])
            row["AvgSessionDuration"] = (row["AvgSessionDuration"]*ver_counts[version]["SidAvg"] + sid_avg*sid_len)/(sid_len+ver_counts[version]["SidAvg"])
            row["Sessions"]+= sid_len
            data[data["Prod"]==version] = row

            ver_counts[version]["Fail"]+=f_avgD_n
            ver_counts[version]["Success"]+=s_avgD_n
            ver_counts[version]["Total"]+=f_avgD_n+s_avgD_n
            ver_counts[version]["SidAvg"]+=sid_len

    #Check the onoging for end date
    tmp_onoging = copy.copy(ongoing)
    for k in tmp_onoging.keys():
        if ongoing[k] == False:
            row = data[data["Prod"]==k]
            row["End_Time"]=e_date
            data[data["Prod"]==k] = row
            del ongoing[k]

def preprocess_data(save_csv=True):
    global data,ongoing,last_day
    #Read the data files
    full_files_aj = []
    full_files_vi = []
    for root,sub_dirs,tmp_files in os.walk(general.data_path):
        for r in sorted(sub_dirs):
            for r_sub,dirs,files in os.walk(general.data_path + r):
                for fn in files:
                    if fn == "ajax_events.csv":
                        full_files_aj.append(r_sub + "/" + fn)
                    elif fn == "views.csv":
                        full_files_vi.append(r_sub + "/" + fn)

    for day in full_files_aj:
        print("Ajax Mining for : " + day)
        df = pd.read_csv(day ,encoding='latin_1', index_col=0, parse_dates=True,
                         low_memory=False)
        day_mining_ajax(df,day)
        #No need for df anymore - clear the memory
        del df
        gc.collect()

    for day in full_files_vi:
        print("View Mining for : " + day)
        df = pd.read_csv(day ,encoding='latin_1', index_col=0, parse_dates=True,
                         low_memory=False)
        day_mining_views(df)
        #No need for df anymore - clear the memory
        del df
        gc.collect()

    #since the data is limited, we need to recognize the last day, simple patch
    for key in ongoing.keys():
        row = data[data["Prod"]==key]
        row["End_Time"]=last_day
        data[data["Prod"]==key] = row

    #Finished mining the data, preprocess it now
    index = data["Prod"]
    values = data[features]
    df = pd.DataFrame(values,index)
    df = df.fillna(0)
    #Calculate the different ratios for initial labeling
    df["RatioFS"] = df["Fail"]/(0.001+df["Success"])
    df["RatioFV"] = df["Fail"]/(0.001+df["Views"])

    df = df.sort_values(by='RatioFS',ascending=False)
    if save_csv == True:
        df.to_csv("DataMined.csv")

    return df

def continues_versions_merging(df):
    df_prev = pd.read_csv('DataLabeled.csv' ,encoding='latin_1', index_col=0,
                          parse_dates=False, low_memory=False)

    for prod in df.index:
        if df_prev.ix[df_prev.index==prod,"Fail"].shape[0] != 0:
            fails_prev = int(df_prev.ix[df_prev.index==prod,"Fail"])
            success_prev = int(df_prev.ix[df_prev.index==prod,"Success"])
            views_prev = int(df_prev.ix[df_prev.index==prod,"Views"])
            start_time = df_prev.ix[df_prev.index==prod,"Start_Time"].tolist()[0]
            fail_avg_prev = float(df_prev.ix[df_prev.index==prod,"FailAvgDuration"])
            success_avg_prev = float(df_prev.ix[df_prev.index==prod,"SuccessAvgDuration"])
            total_avg_prev = float(df_prev.ix[df_prev.index==prod,"TotalAvgDuration"])

            #Now remove this from the previous data
            df_prev=df_prev[df_prev.index!=prod]

            #Combine the data
            df.ix[df.index==prod,"FailAvgDuration"] = (df.ix[df.index==prod,"FailAvgDuration"]*df.ix[df.index==prod,"Fail"]+
                                                       fails_prev*fail_avg_prev)/(fails_prev+df.ix[df.index==prod,"Fail"] )
            df.ix[df.index==prod,"SuccessAvgDuration"] = (df.ix[df.index==prod,"SuccessAvgDuration"]*df.ix[df.index==prod,"Success"]+
                                                       success_prev*success_avg_prev)/(success_prev+df.ix[df.index==prod,"Success"] )
            df.ix[df.index==prod,"TotalAvgDuration"] = (df.ix[df.index==prod,"TotalAvgDuration"]*(df.ix[df.index==prod,"Fail"]+df.ix[df.index==prod,"Success"])+
                                                       (fails_prev+success_prev)*total_avg_prev)/(fails_prev+success_prev+df.ix[df.index==prod,"Fail"]
                                                                                                  +df.ix[df.index==prod,"Success"])

            df.ix[df.index==prod,"Fail"] = df.ix[df.index==prod,"Fail"] + fails_prev
            df.ix[df.index==prod,"Success"] = df.ix[df.index==prod,"Success"] + success_prev
            df.ix[df.index==prod,"Views"] = df.ix[df.index==prod,"Views"] + views_prev
            df.ix[df.index==prod,"Start_Time"] = start_time



    #Recalculate the ratios
    df["RatioFS"] = df["Fail"]/(0.001+df["Success"])
    df["RatioFV"] = df["Fail"]/(0.001+df["Views"])

    df_prev.to_csv("DataLabeled.csv")
    return df

def merge_new_data_directories():
    general.data_path = "data/"
    for root,sub_dirs,tmp_files in os.walk("new_data/"):
        for r in sorted(sub_dirs):
            for r_sub,dirs,files in os.walk("new_data/" + r):
                shutil.move(r_sub,"data/")

def add_new_labled_data(df):
    with open('DataLabeled.csv', 'a') as f:
        df.to_csv(f, header=False)
        f.close()
