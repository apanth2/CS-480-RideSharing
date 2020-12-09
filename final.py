#!/usr/bin/env python
# coding: utf-8

# # Taxi Ride Sharing 
# 
# The following code runs ride sharing algorithm for any given day (provided the path to the data folder is mentioned properly).The distances between the locations are precomputed in a different program.
# Following are the steps to run the program :
# 
# 1. Download all the data files from the location <a href="https://drive.google.com/drive/folders/1ah7pfjnnJkYiwHUqGLgznPU4u0u5dFme" > Data Files </a>.
# 2. The folder contains two main subfolders : Data folder and Distance Folder. Data ranges from Jan 2015 - Dec 2015.
# 3. Start the graphhopper server.
# 4. Input the path to the folders as mentioned by the program.
# 5. The code runs for two configuration : a.one day data b.For specified minutes.Select required option(note :This code can run for any year as specified by the user.But since the data downloaded in the 1st step corresponds to only 2015 Data input 2015 as year.
#
# 

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import requests
from h3 import h3
import json
from urllib.request import URLError, Request, urlopen
from itertools import combinations
from itertools import permutations
from dateutil import parser
from datetime import datetime, timedelta
import math
import networkx as nx
from tqdm import tqdm


# In[ ]:





# In[13]:





# In[6]:


# Check if graphhopper works
request_str = 'http://localhost:8989/route?point=' + str(40.760166) + '%2C' + str(-73.964760) + '&point=' + str(40.768780) + '%2C' + str(-73.867058) + '&vehicle=car'
request = Request(request_str)
res=requests.get(request_str)
print("Distance = {}".format(json.loads(res.text)['paths'][0]['distance']))
print("Time = {}".format(json.loads(res.text)['paths'][0]['time']))


# In[7]:


class Node:
    def __init__(self,idx,data):
        self.id = idx
        self.pickup_location = (data.pickup_latitude,data.pickup_longitude,data.pickup_h3)
        self.dropoff_location = (data.dropoff_latitude,data.dropoff_longitude,data.dropoff_h3)
        self.pickup_time = data.pickup_time
        self.dropoff_time = data.dropoff_time
        self.distance = data.trip_distance
        self.duration = data.duration
        self.delay = data.delay
        self.passenger_count = data.passenger_count


# In[8]:


def getDistanceAndDuration(node_a, node_b, trip_type):
    try:
        if trip_type==2: 
            e, f, g, h = node_a.pickup_location[0], node_a.pickup_location[1], node_b.pickup_location[0],node_b.pickup_location[1]
        else:
            e, f, g, h = node_a.dropoff_location[0], node_a.dropoff_location[1], node_b.dropoff_location[0],node_b.dropoff_location[1]
        request_str = 'http://localhost:8989/route?point=' + str(e) + '%2C' + str(f) + '&point=' + str(
            g) + '%2C' + str(h) + '&vehicle=car'
        request = Request(request_str)
        res = requests.get(request_str)
        if 'paths' in json.loads(res.text):
            distance = json.loads(res.text)['paths'][0]['distance']
            time = json.loads(res.text)['paths'][0]['time']
            minute, msec = divmod(time, 60000)
            return distance / 1609.344 , minute + (msec / 100000)
        else:
            return float('inf'),float('inf')
    except:
        return float('inf'),float('inf')
    


# In[9]:


def GetAllPairs(node_a, node_b, tripType):
    if tripType == 1:
        #if no distance call graphhopper
        if (node_a.dropoff_location[2],node_b.dropoff_location[2]) not in dfDistance.index:
            A_B_Distance,A_B_Duration = getDistanceAndDuration(node_a, node_b, tripType)
        else:
            A_B_Distance = dfDistance.loc[(node_a.dropoff_location[2], node_b.dropoff_location[2])]['distance']
            A_B_Duration = dfDistance.loc[(node_a.dropoff_location[2], node_b.dropoff_location[2])]['duration']
        
        start_A_dist = node_a.distance
        a_b_dist   = A_B_Distance
        start_a_dur  = node_a.duration
        a_b_dur    = A_B_Duration
        
        if (node_b.dropoff_location[2],node_a.dropoff_location[2]) not in dfDistance.index:
            B_A_Distance,B_A_duration = getDistanceAndDuration(node_a, node_b, tripType)
        else:
            B_A_Distance = dfDistance.loc[(node_b.dropoff_location[2], node_a.dropoff_location[2])]['distance']
            B_A_duration = dfDistance.loc[(node_b.dropoff_location[2], node_a.dropoff_location[2])]['duration']
            
        StartToB_dist = node_b.distance
        b_a_dist = B_A_Distance
        StartTo_B_dur = node_b.duration
        b_a_dur = B_A_duration
        
        path_1_TotalDistance,path_1_TotalDuration = start_A_dist + a_b_dist,start_a_dur + a_b_dur
        Path_1_A_duration,Path_1_B_duration = start_a_dur,path_1_TotalDuration
        
        Path_2_Total_distance,Path_2_TotalDuration = StartToB_dist+b_a_dist,StartTo_B_dur+b_a_dur
        Path_2_A_duration,Path_2_B_duration         = Path_2_TotalDuration ,StartTo_B_dur
               
    else:
        if (node_a.pickup_location[2],node_b.pickup_location[2]) not in dfDistance.index:
            A_B_Distance,A_B_Duration = getDistanceAndDuration(node_a, node_b, tripType)
        else:
            A_B_Distance = dfDistance.loc[(node_a.pickup_location[2], node_b.pickup_location[2])]['distance']
            A_B_Duration = dfDistance.loc[(node_a.pickup_location[2], node_b.pickup_location[2])]['duration']
        
        a_b_dist   = A_B_Distance
        b_end_dist = node_b.distance
        a_b_dur    = A_B_Duration
        b_end_dur  = node_b.duration
        
        if (node_b.pickup_location[2],node_a.pickup_location[2]) not in dfDistance.index:
            B_A_Distance,B_A_duration = getDistanceAndDuration(node_b, node_a, tripType)
        else:
            B_A_Distance = dfDistance.loc[(node_b.pickup_location[2], node_a.pickup_location[2])]['distance']
            B_A_duration = dfDistance.loc[(node_b.pickup_location[2], node_a.pickup_location[2])]['duration']
        
        b_a_dist   = B_A_Distance
        a_end_dist = node_a.distance
        b_a_dur    = B_A_duration
        a_end_dur  = node_a.duration
        
        path_1_TotalDistance,path_1_TotalDuration = a_b_dist + b_end_dist,a_b_dur + b_end_dur
        Path_1_A_duration,Path_1_B_duration = path_1_TotalDuration,b_end_dur
        
        Path_2_Total_distance,Path_2_TotalDuration, = b_a_dist+a_end_dist,b_a_dur+a_end_dur
        Path_2_A_duration,Path_2_B_duration         = a_end_dur,Path_2_TotalDuration
        
    return ((path_1_TotalDistance,path_1_TotalDuration,Path_1_A_duration,Path_1_B_duration),( Path_2_Total_distance,Path_2_TotalDuration,Path_2_A_duration,Path_2_B_duration))
    


# In[10]:


def CalculateEdgeWeight(node_a, node_b, trip_type):
    path1,path2 = GetAllPairs(node_a, node_b, trip_type)
    minimum_distance = float('inf')
    for path in (path1,path2):
        distanceContraint = (path[0] <= node_a.distance + node_b.distance)
        delayConstraint = (path[2] <= node_a.duration + node_a.delay) & (path[3] <= node_b.duration + node_b.delay)
        #add social constraint too...
        
        
        if distanceContraint and delayConstraint and path[0]< minimum_distance:
            minimum_distance = path[0]
    distance_saved = node_a.distance + node_b.distance - minimum_distance
    return distance_saved


# In[11]:


def getRSG(G, trip_type):
    for node_a,node_b in list(combinations(G,2)):
        if (node_a.passenger_count+node_b.passenger_count)<=4:
            distance_saved = CalculateEdgeWeight(node_a, node_b, trip_type)
            if distance_saved!= float('-inf') :
                G.add_edge(node_a,node_b, weight=distance_saved)
    return G


# # Average distance saved per pool as a % of total distance of individual rides

# In[12]:


def CalculateAverageDistanceSaved(merged_trips, Final_Graph):
    with_sharing , without_sharing = [],[]
    for i in range(len(merged_trips)):
        all_nodes =  set()
        total_dis_before_merging = 0
        total_dis_after_merging = 0
        for each_node in Final_Graph[i].nodes:
            total_dis_before_merging += each_node.distance
            all_nodes.add(each_node)
        #remove merged nodes from orginal rga graph
        for u,v in merged_trips[i].edges:
            all_nodes.remove(u)
            all_nodes.remove(v)
            total_dis_after_merging += Final_Graph[i].get_edge_data(u,v)['weight']
        #add unmerged solo trips also
        for solo in all_nodes:
            total_dis_after_merging += solo.distance
        with_sharing.append(total_dis_after_merging)
        without_sharing.append(total_dis_before_merging)
    return(sum([(1-x/y) for x, y in zip(with_sharing, without_sharing)])/len(without_sharing) * 100)   


# # Average number of trips saved per pool as a % of number of individual trips

# In[13]:


def AverageTripSaved(merged_trips, Final_Graph):
    savedRides = []
    for idx in range(len(merged_trips)):
        numIndv_trips = len(Final_Graph[idx].nodes)
        numPooled_trips = len(merged_trips[idx].edges)
        savedRides.append(numPooled_trips/numIndv_trips * 100)
    return(sum(savedRides)/len(savedRides))


# In[14]:


def MainAlgoritm(trip_type, excecution_time):
    start_execution_time = time.time()
    Final_Graph = []
    t = 0
    for _,trips in df.groupby(['pool_window']):
        nodes = []
        trips = trips.reset_index()
        for idx, row in trips.iterrows():
            nodes.append(Node(idx,trips.iloc[idx]))
        G = nx.Graph()
        G.add_nodes_from(nodes)
        Final_Graph.append(G)

    #Start of the code
    mergedTrips = []
    cn=0
    for individual_graph in tqdm(Final_Graph,total=len(Final_Graph)):
        if int(excecution_time) > 0 and ((time.time() - start_execution_time)/60 >= float(excecution_time)):
            break
        s = time.time()
        rideSharingGraph = getRSG(individual_graph, trip_type)
        #maximum weighted algorithm
        maximumWeightedGraph = nx.max_weight_matching(rideSharingGraph, maxcardinality=True)
        g_match = nx.Graph()
        for u,v in maximumWeightedGraph:
            g_match.add_edge(u,v)

        mergedTrips.append(g_match)
        t += time.time()-s
    if int(excecution_time)>0:
        print("Number of pools processed in {} min :{}".format(excecution_time,len(mergedTrips)))
    else:
        print("Number of pools processed :{}".format(len(mergedTrips)))
    print("Average computation time is {} sec".format(t/len(mergedTrips)))
    average_distance_saved = CalculateAverageDistanceSaved(mergedTrips, Final_Graph)
    average_trip_saved = AverageTripSaved(mergedTrips, Final_Graph)
    print("Average distance saved for poolwindow {} is :{}".format(pool_time_window,average_distance_saved))
    print("Average trip saved for poolwindow {} is :{}".format(pool_time_window,average_trip_saved)) 
    return pool_time_window,average_distance_saved,average_trip_saved,t/len(mergedTrips)


# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
def plotBar(df_results, label):
    a4_dims = (12, 9)
    sns.set(rc={'figure.figsize':(13,9)})
    ax = sns.barplot(x="Type", y="saved", hue="pool_window", data=df_results,palette="Blues_d")
    ax.set_xlabel('')
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., 1*height,
                    '{}%'.format(round(height,2)),
                    ha='center', va='bottom')
    for ticks in ax.xaxis.get_major_ticks():
        ticks.label.set_fontsize(18)
    ax.set_ylabel('Average {} saved'.format(label),fontsize = 18)
    ax.set_title('Average {} for the year 2015'.format(label),fontsize = 18)
    plt.show()
def plotLine(df_time):
    a4_dims = (12, 9)
    sns.set(rc={'figure.figsize':(13,9)})
    ax = sns.lineplot(x="pool_window", y="time", hue="Type", data=df_time,palette="Blues_d")
    ax.set_xlabel('')
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., 1*height,
                    '{}%'.format(round(height,2)),
                    ha='center', va='bottom')
    for ticks in ax.xaxis.get_major_ticks():
        ticks.label.set_fontsize(18)
    ax.set_xlabel('pool windows in minute',fontsize = 18)    
    ax.set_ylabel('Time in seconds',fontsize = 18)
    ax.set_title('Average computation per pool',fontsize = 18)
    plt.show()
    


# # Run This

# In[16]:


import time
def ceil_dt(dt, delta):
        return datetime.min + math.ceil((dt - datetime.min) / delta) * delta
excecution_time = 0
DataFolder = input("Enter the absolute path to Data Folder:")
distance_folder = input("Enter the absolute path to distance Folder:")
# mon = dict({'Jan':'1','Feb':'2','Mar':'3','Apr':'4','May':'5','June':'6','July':'7','Aug':'8','Sep':'9',\
#             'Oct':'10','Nov':'11','Dec':'12'})
year = input("Enter year:")
Month = input ("Enter month:")
Month_Name = input ("Enter month name:")
day = input("Enter start day:")
dayend = input("Enter end day:")
running_type='1'

# running_type = input("What input data you want to run:\n 1.Run one day's data\n 2.Run the algorithm for given minutes:")
if running_type == '2':
    excecution_time = input("Enter excecution time in minutes:")
    strt_time = input("Enter start time in military format (00:00:00):")
month = Month

#reading distance and data file
file_name = str(DataFolder) + '/Taxi_Data/green_tripdata_' + str(year) + '-' + str(Month) + '.csv'
distance_file_name = str(distance_folder)+'/Distance/'+year+'-'+month+'.csv'
df = pd.read_csv(file_name)
dfDistance = pd.read_csv(distance_file_name)
dfDistance.drop_duplicates(subset=['pickup_h3', 'dropoff_h3'], keep=False, inplace=True)
dfDistance.set_index(['pickup_h3', 'dropoff_h3'], inplace= True)
dfDistance = dfDistance.sort_index()
#######################################################################################################

columns = ['tpep_pickup_datetime', 'tpep_dropoff_datetime','passenger_count',       'trip_distance', 'pickup_longitude','pickup_latitude','dropoff_longitude', 'dropoff_latitude']
df = df[columns]
df.rename(columns={'tpep_pickup_datetime':'pickup_time',
       'tpep_dropoff_datetime':'dropoff_time'},inplace=True)
drop_index=df[(df.pickup_latitude==0)|(df.pickup_longitude==0)|(df.trip_distance==0)].index
df.drop(drop_index,inplace=True)
df['pickup_time'] = pd.to_datetime(df['pickup_time'])
df['dropoff_time'] = pd.to_datetime(df['dropoff_time'])
df['pickup_h3'] = df.apply(lambda x: h3.geo_to_h3(x['pickup_latitude'], x['pickup_longitude'], 15), axis=1)
df['dropoff_h3'] = df.apply(lambda x: h3.geo_to_h3(x['dropoff_latitude'], x['dropoff_longitude'], 15), axis=1)

if running_type =='1':
    start_date=str(year)+'-'+str(month)+'-'+str(day)+' 00:00:00'
    end_date=str(year)+'-'+str(month)+'-'+str(dayend)+' 23:59:59'
    df=df[(df['pickup_time'] >= start_date) & (df['dropoff_time'] <= end_date)]
else:
    start_date=str(year)+'-'+str(month)+'-'+str(day)+' '+str(strt_time)
    df=df[(df['pickup_time'] >= start_date)]

# query =  (f"SELECT * FROM stocks_daily_prices "
#              f"WHERE stock_ticker = '{ticker}' and "
#              f" Date BETWEEN '{start}' AND '{end}';")
    
df.reset_index(drop=True,inplace=True)
df['duration'] = (df['pickup_time']-df['dropoff_time']).dt.seconds
df['delay'] = df['duration'].apply(lambda x: x*0.20)
df_results_distance_from = pd.DataFrame(columns =  ['pool_window','saved','Type'])
df_results_trip_from= pd.DataFrame(columns =  ['pool_window','saved','Type'])
df_time_from = pd.DataFrame(columns = ['pool_window','time','Type'])
for i,pool_time_window in enumerate(list([2,5,7])):
    start_time = time.time()
    df['pool_window'] = df['pickup_time'].apply(lambda x: ceil_dt(x.to_pydatetime(), timedelta(minutes=pool_time_window)))
    print("\nStarting main algorithm...")
    pool_time_window,average_distance_saved,average_trip_saved,pool_com_time = main_algoritm(1,excecution_time)
    df_results_distance_from.loc[i] = [pool_time_window,average_distance_saved,Month_Name]
    df_results_trip_from.loc[i] = [ pool_time_window,average_trip_saved,Month_Name]
    df_time_from.loc[i] =  [pool_time_window,pool_com_time,Month_Name]
    total_time = (time.time()-start_time)/60.0
    print("algorithm time taken for {} pool window is :{} minutes\n ".format(pool_time_window,total_time))


# # Graphs

# In[20]:


plotBar(df_results_distance_from.append(df_results_distance_from), "distance")
plotBar(df_results_trip_from.append(df_results_trip_from), "trip")
# plotLine(df_time_to.append(df_time_from))


# In[ ]:








from flask import Flask, jsonify, request #import objects from the Flask model
import configparser
import os
from sqlalchemy import create_engine
import mysql.connector




cwd = os.getcwd()

config = configparser.ConfigParser()
config.read(cwd + '/practicum_my.cnf')

db_connection_str = """mysql+pymysql://{user}:{password}@{host}/{database}""".format(host=config['client']['host'],
                                                                                         database=config['client'][
                                                                                             'database'],
                                                                                         user=config['client']['user'],
                                                                                         password=config['client'][
                                                                                             'password']
                                                                                         )

    # Create engine
db_engine = create_engine(db_connection_str)





query = (f"SELECT * FROM yellow_taxi_2015 "
             f"WHERE tpep_pickup_datetime  BETWEEN '2015-02-01 12:00:00' AND '2015-02-01 12:05:00';") 
print(query)
df_json = pd.read_sql(str(query) , con=db_engine)
df_json.shape


# In[ ]:




