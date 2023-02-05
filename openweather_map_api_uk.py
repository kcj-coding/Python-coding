import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import sklearn as sk
import xlsxwriter as xlw
import os
import html
import re

#################### configure lats/lons of interest, and enter api key ###########################
api_key = "" # api key for openweathermap.org

date_to_use = "04/02/2023" # today's date or date of data collection to use in graph title

lat1 = 49 # positive number means North, negative number means South
lat2 = 70 # positive number means North, negative number means South

lon1 = -16 # positive number means East, negative number means West
lon2 = 5 # positive number means East, negative number means West

# the lat2 - lat1 and lon2 - lon1 range should be the same
if (lat2-lat1) == (lon2-lon1):
    print("Lat2-Lat1 and Lon2-Lon1 match... continuing")
else:
    raise ValueError("Lat2-Lat1 and Lon2-Lon1 do not match. Please make them equal")

increment = 0.5 # to increase lat/lon co-ordinates by this amount
################################################################################

# create a dataframe of the latitude points of interest to query for weather stations
lats = pd.DataFrame(data={'lat':np.arange(lat1,lat2,increment)}) # increment in 0.5 steps

# create a numpy array of the longitude points of interest
lon_range=[]
rngr = int((lon2-lon1)/increment)

for i in range(0,rngr,1):
    lon_rng = np.arange((lon1+i),(lon2+i),increment) # increment in 0.5 steps
    lon_range.append(lon_rng)

# configuring df
d1 = pd.DataFrame(data={'data':[0]})
d2 = pd.DataFrame(data={'data':[0]})

# for each longitude array, join the latitude entries
for each in lon_range:
    lon_df = pd.DataFrame(data={'lon':each})
    d1 = pd.concat([lats,lon_df],axis=1)
    d2 = pd.concat([d2,d1])

# remove things we do not need
d2 = d2[~pd.isnull(d2['lat'])]
d2 = d2.drop(columns=['data'])
d2 = d2.reset_index()

################################################################################

# configuring df
dff = pd.DataFrame(data={'name':['xyz']})

# make an api call for each lat/long combination as created earlier and store results in a dataframe
for ii in range(0,len(d2),1):
        
        city = "https://api.openweathermap.org/data/2.5/weather?lat="+str(d2['lat'][ii])+"&lon="+str(d2['lon'][ii])+"&appid="+api_key+"&units=imperial"
        city = pd.read_json(city, lines=True)
        co_ords = pd.DataFrame([city.coord[0]])
        name = city.name
        wind = pd.DataFrame([city.wind[0]])
        df = pd.concat([co_ords,name,wind],axis=1)
        dff = pd.concat([dff,df])
        
dff = dff[dff['name']!='xyz']
dff = dff.reset_index()
################################################################################

# graph definitions
def base_map_plot(sdata,bar,title):
    # 1. Draw the map background
    fig = plt.figure(figsize=(10, 7))
    m = Basemap(projection='lcc', resolution='h', 
                lat_0=54, lon_0=-1, # set the lat_0 and lon_0 as required for your plot
                width=0.1E7, height=0.1E7)
    #m.shadedrelief()
    m.drawmapboundary(fill_color='#99ffff') # or hex colour
    m.fillcontinents(color='#FFFFFF', lake_color='#99ffff')
    m.drawcoastlines(color='#FFFFFF')
    m.drawcountries(color='#FFFFFF')
    #m.drawstates(color='gray')
    m.drawparallels(np.arange(-90, 100, 1))#, linewidth=0.5, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0 )
    m.drawmeridians(np.arange(-100,100,1))#, linewidth=0.5, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0 )
    # 2. scatter data
    # and size reflecting area
    m.scatter(lon, lat, latlon=True,
              c=area, s=sdata,
              cmap="Accent", zorder=2)#, alpha=0.5)

    # 3. create colorbar and legend
    m.drawparallels(range(-90, 100, 1), linewidth=0.5, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0 )
    m.drawmeridians(range(-100,100,1), linewidth=0.5, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0 )
    if bar==True:
        plt.colorbar(label="Wind speed/mph")
    plt.title(title)
################################################################################

# graph the data
lon = dff['lon'].values
lat = dff['lat'].values
area = dff['speed'].values
wind = dff['speed'].values

base_map_plot(sdata=wind*10,bar=True,
              title="Wind speed UK "+date_to_use)

#save plot
plt.savefig('C:\\Users\\kelvi\\Desktop\\wxr_map11_uk.png', dpi=400, bbox_inches='tight')
