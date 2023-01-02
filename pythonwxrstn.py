from requests import get
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#lat/long of location to use
melat = 52 # latitude
melong = 00 # longitude

# url
url = "https://apex.oracle.com/pls/apex/raspberrypi/weatherstation/getallstations"

# urlwxr
urlwxr = "https://apex.oracle.com/pls/apex/raspberrypi/weatherstation/getlatestmeasurements/"

# read in json data
data = get(url).json()['items']

data = pd.DataFrame(data=data)

# rename colnames
data.rename(columns={list(data)[0]:'Id',list(data)[1]:'Name',list(data)[2]:'Lat',list(data)[3]:'Long'},inplace=True)

# remove where Long not numeric
data = data[~pd.isnull(data['Long'])]

#data['Lat'] = data['Lat'].astype(float)
#data['Long'] = data['Long'].astype(float)

def haversine(lon1, lat1, lon2, lat2): # use numpy (np) to use array values from df
    # convert degrees to radians
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    distance = 2 * np.arcsin(np.sqrt(a)) * 6371 #6371 is the radius of the Earth
    return distance

# create new column with data from latme,longme
data ['dis'] = haversine(melat,melong,data['Lat'],data['Long'])


# to get weather data we need to append the station id to the urlwxr
# set a condition e.g. only locations within 50km
datatouse = data

datatouse = datatouse[datatouse['dis']<=50]

# get a unique list of weather stations
lst = pd.Series(datatouse['Id']).drop_duplicates().to_list()

# initialise data3
data3 = {'weather_stn_id':[0],'ambient_temp':[0],'ground_temp':[0],'air_quality':[0],'air_pressure':[0],'humidity':[0],'wind_speed':[0],'wind_gust_speed':[0]}
data3 = pd.DataFrame(data=data3)

# for each item in lst, get the weather station data and make a new table

for i in lst:
    urlwxr1 = urlwxr+str(i)
    
    # read in json data
    data1 = get(urlwxr1).json()['items']

    data1 = pd.DataFrame(data=data1)
    
    # check if not empty
    try:
        if not data1['weather_stn_id'].empty:
            t = data1.columns
            
            data2 = data1[['weather_stn_id','ambient_temp','ground_temp','air_quality','air_pressure','humidity','wind_speed','wind_gust_speed']]
            
            data3 = pd.concat([data3,data2],axis=0) #axis = 1 for column join
    except:
        pass
    
# join lat long onto data3
data3 = pd.merge(data3,datatouse, how="left",left_on='weather_stn_id',right_on='Id')

# drop na
data3 = data3.dropna()

# add text filter to distinguish between stations
data3['stn_name_index'] = data3.index
data3['stn_name_index'] = data3['stn_name_index'].astype(str)
data3['stn_name'] = data3['stn_name_index'] +"stn"

cols = ['ambient_temp','air_pressure','humidity']

for coler in cols:
    fig, ax = plt.subplots()
    
    ax.bar(data3['stn_name'],data3[coler], 0.5,color='#CD2456')
    
    for x,y in zip(data3['stn_name'],data3[coler]):

        label = "{}".format(y) 
    
    
        plt.annotate(label, # this is the text
                     (x,y), # these are the coordinates to position the label
                     textcoords="offset points", # how to position the text
                     xytext=(0,10), # distance from text to points (x,y)
                     ha='center',
                     rotation='horizontal', fontsize=8) # horizontal alignment can be left, right or center
        
    # Customise some display properties
    ax.set_ylabel(coler,size=8)
    ax.set_xlabel('Station',size=8)
    #ax.set_title('Title',size=9)
    ax.set_xticks(data3['stn_name'])    # This ensures we have one tick per year, otherwise we get fewer
    ax.set_xticklabels(data3['stn_name'], rotation='horizontal',size=8)
    ax.tick_params(axis='y',labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    #set size of graph
    cmsize=1/2.54
    fig.set_size_inches(30*cmsize, 15*cmsize)
    
    #save plot
    #plt.savefig('C:\\Users\\kelvi\\Desktop\\figpath.png', dpi=400, bbox_inches='tight')



    


