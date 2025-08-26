
# to do
# add in pdf to histogram
# add in correlation and only graph compared columns if > some value
# save view of all histograms? in main folder


#import win32com.client # to interact with windows applications
import pandas as pd
#import scipy.stats as stats
#from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import LogisticRegression
#from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import re
import os
import glob
#import shutil
#import time
from mpl_toolkits.basemap import Basemap

# file folder location 
folder = r"C:\\Folder"

# if output folder exists delete everything inside of the folder
#if os.path.exists(folder):
#    files = glob.glob(f"{folder}/*")
#    for f in files:
#        shutil.rmtree(f) # removes files and folders

# check output folder exists and if not create it
if not os.path.exists(folder):
    os.makedirs(folder, exist_ok=True)
    

    
    
def number_format(num):
    
    if num > 0:
        digits = int(math.log10(num))+1
    elif num == 0:
        digits = 1
    else:
        digits = int(math.log10(-num))+2 # +1 if you don't count the '-' 
    #print(digits)
    if digits > 3 or digits < 0:
        return '{:.2e}'.format(num)
    else:
        return round(num,2)
    
#tst = number_format(50000)
#tst1 = number_format(0.000456578)
#tst2 = number_format(54323456)
#tst3 = number_format(45.23456)

def line_graph(x,y,xx,yy,title,file_loc):
    fig, ax = plt.subplots()
    
    # Plot each bar plot. Note: manually calculating the 'dodges' of the bars
    #ax = plt.axes()
    
    #ax.plot(np.arange(0,len(df.data[df.type=='a'])), df.c_number[df.type=='a'], label='Product a', color='#CD2456')
    #ax.plot(np.arange(0,len(df.data[df.type=='b'])), df.c_number[df.type=='b'], label='Product b', color='#14022E')
    
    #ax.scatter(x,y,alpha=0.5, c="black")
    #ax.plot(x,y)

    for i in pd.Series(dfs['type']).drop_duplicates().tolist():
            try:
               ax.scatter(x[dfs['type']==i], y[dfs['type']==i], label="Product "+i, color='#'+str(hex(random.randint(0,16777215)))[2:])
            except:
                   pass
    
    # Customise some display properties for line graph
    ax.set_ylabel(yy,size=8)
    #ax.set_ylim(0,200) #(0,max(df.Sales)+10)
    #ax.set_yticks(ax.get_yticks().astype(np.int64))
    #ax.set_yticklabels(ax.get_yticks().astype(np.int64),rotation='horizontal',size=8)
    #ax.set_xlim(0,8)
    ax.set_xlabel(xx,size=8)
    ax.set_title(title,size=9)
    #ax.set_xticks(np.arange(0,len(df.data[df.type=='a'])))    # This ensures we have one tick per year, otherwise we get fewer
    #ax.set_xticklabels(np.arange(0,len(df.data[df.type=='a'])), rotation='horizontal',size=8)
    #ax.fill(df.Doy[df.Product=='A'],df.Sales[df.Product=='A']-5,df.Sales[df.Product=='A']+5,color='k', alpha=.15)
    #ax.set_xticks([])
    #ax.set_yticks([])
    ax.legend(loc='upper left',fontsize=8)
    ax.legend(*[*zip(*{l:h for h,l in zip(*ax.get_legend_handles_labels())}.items())][::-1])
    #ax.set_axisbelow(True) # to put gridlines at back
    #ax.grid(linestyle='--',color='#CECECE')
    ax.tick_params(axis='y',labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    #set size of graph
    cmsize=1/2.54
    fig.set_size_inches(30*cmsize, 15*cmsize)
    
    #save plot
    plt.savefig(file_loc, dpi=400, bbox_inches='tight')
    
    # Ask Matplotlib to show the plot
    #plt.show()
    plt.close()
    
def dual_line_graph(x,y,xx,yy,y1,yy1,title,file_loc):
    fig, ax = plt.subplots()
    
    # Plot each bar plot. Note: manually calculating the 'dodges' of the bars
    #ax = plt.axes()
    
    #ax.plot(np.arange(0,len(df.data[df.type=='a'])), df.c_number[df.type=='a'], label='Product a', color='#CD2456')
    #ax.plot(np.arange(0,len(df.data[df.type=='b'])), df.c_number[df.type=='b'], label='Product b', color='#14022E')
    
    #ax.scatter(x,y,alpha=0.5, c="black")
    #ax.plot(x,y)

    ax.plot(x,y,'b-')
    ax1 = ax.twinx()
    ax1.plot(x,y1,'r-')
    
    # Customise some display properties for line graph
    ax.set_ylabel(yy,size=8)
    ax1.set_ylabel(yy1,size=8)
    #ax.set_ylim(0,200) #(0,max(df.Sales)+10)
    #ax.set_yticks(ax.get_yticks().astype(np.int64))
    #ax.set_yticklabels(ax.get_yticks().astype(np.int64),rotation='horizontal',size=8)
    #ax.set_xlim(0,8)
    ax.set_xlabel(xx,size=8)
    ax.set_title(title,size=9)
    #ax.set_xticks(np.arange(0,len(df.data[df.type=='a'])))    # This ensures we have one tick per year, otherwise we get fewer
    #ax.set_xticklabels(np.arange(0,len(df.data[df.type=='a'])), rotation='horizontal',size=8)
    #ax.fill(df.Doy[df.Product=='A'],df.Sales[df.Product=='A']-5,df.Sales[df.Product=='A']+5,color='k', alpha=.15)
    #ax.set_xticks([])
    #ax.set_yticks([])
    ax.legend(loc='upper left',fontsize=8)
    ax.legend(*[*zip(*{l:h for h,l in zip(*ax.get_legend_handles_labels())}.items())][::-1])
    #ax.set_axisbelow(True) # to put gridlines at back
    #ax.grid(linestyle='--',color='#CECECE')
    ax.tick_params(axis='y',labelsize=8)
    ax.spines[['top']].set_visible(False)
    ax1.spines[['top']].set_visible(False)

    
    #set size of graph
    cmsize=1/2.54
    fig.set_size_inches(30*cmsize, 15*cmsize)
    
    #save plot
    plt.savefig(file_loc, dpi=400, bbox_inches='tight')
    
    # Ask Matplotlib to show the plot
    #plt.show()
    plt.close()
    
# graph definitions
def base_map_plot(sdata,lat,lon,area,label,bar,ldg_name,title,name):
    # 1. Draw the map background
    fig = plt.figure(figsize=(10, 7))
    m = Basemap(projection='lcc', resolution='h', 
                lat_0=54, lon_0=-1,
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
              cmap="jet", zorder=2)#, alpha=0.5)
    
    for x,y,z,a in zip(lon, lat, label,np.arange(0,len(lon))):

        labels = label[int(a)]#"{}".format(label)
    
        plt.annotate(labels, # this is the text
                     m(x,y),
                     #(x,y), # these are the coordinates to position the label
                     xycoords='data',
                     textcoords="offset points", # how to position the text
                     xytext=(0,10), # distance from text to points (x,y)
                     ha='center',
                     rotation='horizontal', fontsize=8) # horizontal alignment can be left, right or center
    #m.pcolormesh(xi, yi, zi,latlon=True,
    #          cmap="jet")#, zorder=2)#, alpha=0.5)

    # 3. create colorbar and legend
    m.drawparallels(range(-90, 100, 1), linewidth=0.5, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0 )
    m.drawmeridians(range(-100,100,1), linewidth=0.5, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0 )
    if bar==True:
        plt.colorbar(label=f"{ldg_name}")
    plt.title(title)
    
    #set size of graph
    cmsize=1/2.54
    fig.set_size_inches(30*cmsize, 15*cmsize)
    
    #save plot
    plt.savefig(folder+f'/{name}.png', dpi=400, bbox_inches='tight')
    
    # Ask Matplotlib to show the plot
    #plt.show()
    #plt.close()
    
def hist_graph(df,xx,bins,savename):
    mean = df[xx].mean()
    std = df[xx].std()
    #iqr = np.subtract(*np.percentile(df_num[i], [75, 25]))
    iqr = number_format(df[xx].quantile(0.75) - df[xx].quantile(0.25))
    
    # what percentage of values are captured within the range 
    pct_val = round((len(df[(df[xx] < mean+(2*std)) & (df[xx] > mean-(2*std))])/len(df[xx]))*100,3)
    
    graph_title = str('Title'+" for column "+str(xx)+" Mean:"+str(number_format(mean))+"; percentage of values captured within range:"+str(number_format(pct_val))+"%"+"\nn="+str(len(df[xx]))+"; sd="+str(number_format(std))+"; IQR="+str(iqr))
    
    #### histogram
    fig, ax = plt.subplots()
    
    #ax.boxplot(x=df_num[i], labels=labels, showmeans = False)
    #df_num[[i]].hist(ax=ax)
    plt.hist(df[[xx]], bins=bins, edgecolor="white")
    plt.axvline(mean,linestyle="solid",c="red")
    plt.axvline(mean+(2*std),linestyle="dashed",c="red")
    plt.axvline(mean-(2*std),linestyle="dashed",c="red")
    ax.grid(linestyle='',color='#CECECE')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(str(xx),size=10)
    ax.set_ylabel("Frequency", size=10)
    ax.set_title(graph_title, size=12)
    
    # annotate graph
    #for x in zip()
    
    #set size of graph
    cmsize=1/2.54
    fig.set_size_inches(30*cmsize, 15*cmsize)
    
    # save
    plt.savefig(f'{folder}/{str(savename)}'+'_hist.png', dpi=400, bbox_inches='tight')
    plt.close()


# df = pd.DataFrame({"val":[1,2,3], "val_str":["1","2","3"]})

#df = pd.DataFrame({"val":[random.randint(0,1000) for p in range(0,900,1)], "val_str":np.repeat(["1","2","3"],300)})

# turn str into factors
#df_f = pd.factorize(df['val_str'])[0]

folder_file = "C:\\Users\\kelvi\\Desktop\\x-plane data\\"
filer = "Data_a330.txt"
file = folder_file+filer
with open(file, "r") as f:
    file1 = f.read()
    
df = pd.read_csv(file,sep="|")

for column in df.columns:
    # if column is not type int or float then turn into str
    #if df[column].dtype not in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:# or not in ["bifu"]:
    if df[column].dtype.kind not in "bifu":
        print(column)
        print(df[column].dtype)
        df[column+"_str"] = df[column]#.copy
        df[column+"_str"] = df[column+"_str"].astype(str)
        df[column+"_str"] = pd.factorize(df[column+"_str"])[0] # take the numeric values
        
# write as csv with converted columns
df.to_csv(f'{folder}/file.csv',index=False)


df.columns = df.columns.str.replace("[\s+]","", regex=True) # remove multiple white spaces
column_names = df.columns

column_name_select = ["X_real._time", "X_totl._time", "X_Vind._kias", "Vtrue._ktas", "X_alt.ftmsl", "pitch.__deg", "hding.__mag", "X__lat.__deg", "X__lon.__deg",
                 "X_fuel.___lb", "total.___lb", "fuel1.tankC", "p-alt.ftMSL"]

# remove X and replace . with ,
column_names_select = [str.replace(x,"X","") for x in column_name_select]
column_names_select = [str.replace(x,".",",") for x in column_names_select]

#df.columns = df.columns.str.replace("[^A-Za-z0-9]","", regex=True) # remove special characters from column names, make text and numbers


# select only the columns of interest
df = df[[c for c in df.columns if c in column_names_select]]

df_old = df

# order by time
df = df.sort_values(by=["_totl,_time"])
 

# select only the numeric columns
numerics= ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_num = df.select_dtypes(include=numerics)

# make new columns as variations of selected column
dfs = pd.DataFrame()
for column in df.columns:

    name2 = column+"_nxt"
    name3 = column+"_csum"
    name4 = column+"_chg"
    name5 = column+"_cchg"
    
    dfi = pd.DataFrame({column:df[column],name2:df[column].shift(-1), name3:df[column].cumsum()})
    dfi[name4] = dfi[name2] - dfi[column]
    dfi[name5] = dfi[name4].cumsum()
    dfs = pd.concat([dfs,dfi], axis=1)

dfs_alc = dfs.columns
# get altitude change in 1 sec, 1 min
dfs['altitude_chg1sec'] = dfs['p-alt,ftMSL_chg']/dfs['_totl,_time_chg']
dfs['altitude_chg1min'] = dfs['altitude_chg1sec']*60

# get distance covered in 1 sec, 1 min, 1 hr
dfs['dist_chg1sec'] = dfs['Vtrue,_ktas']/3600
dfs['dist_chg1min'] = dfs['dist_chg1sec'] * 60
dfs['dist_chg1hr'] = dfs['dist_chg1min'] * 60

# if fuel nxt != fuel (current) then mark as changed
dfs['ff_change'] = dfs.apply(lambda x: "changed" if x['_fuel,___lb_nxt'] != x['_fuel,___lb'] else "no change", axis=1)

# make a table of the changed times and left_join to data
dfs_ff_chg = dfs[dfs['ff_change'] == "changed"]
dfs_ff_chg['ff_time_chg'] = dfs_ff_chg['_real,_time'].shift(1)
dfs_ff_chg['ff_time_chg'] = dfs_ff_chg['ff_time_chg'] - dfs_ff_chg['_real,_time']

dfs_ff_chg = dfs_ff_chg[['_real,_time','ff_time_chg']]

dfs = pd.merge(dfs, dfs_ff_chg, how="left", left_on="_real,_time", right_on="_real,_time")

# dfs <- left_join(dfs, dfs_ff_chg, by = join_by("X_real._time"))

# divide fuel used by time, multiply by 3600 to get hourly fuel burn
dfs['ff_hr'] = (abs(dfs['_fuel,___lb_nxt'] - dfs['_fuel,___lb'])/dfs['ff_time_chg']) * 60 * 60 

# test
chk = lambda x: "yes" if x <= (dfs['p-alt,ftMSL'].values[0])+50 else "no"

# define flight phases based on altitude changes
def flight_condition(altitude,speed,altitude_chg, speed_cat, x):
  # if altitude within 50 of starting altitude and speed < speed_cat - departure
  if altitude <= ((dfs['p-alt,ftMSL'].values[0])+50) and speed <= speed_cat:
    #val <- "on ground"
    val = "on_ground"

  
  # if altitude_chg (nxt) > altitude + x - climbing
  elif altitude_chg > x:
    #val <- "climbing"
    val = "climbing"
  
  
  # if altitude_chg (nxt) >= altitude - x or altitude_chg <= altitude + x - level
  elif(((altitude_chg >= 0-x) and (altitude_chg <= x))):
    #val <- "level"
    val = "level"
  
  
  
  # if altitude_chg (nxt) < altitude - x - descending
  elif(altitude_chg < 0-x):
    #val <- "descending"
    val = "descending"
  
  
  # if altitude within 50 of ending altitude and speed < speed_car - arrival
  elif(altitude <= (altitude[len(altitude)]+50) and speed <= speed_cat):
    #val <- "on ground"
    val = "on_ground"
  
  
  else:
    #val <- "unknown"
    val = "unknown"
  
  return(val)

dfs['type'] = dfs.apply(lambda x: flight_condition(x['p-alt,ftMSL'],x['Vtrue,_ktas'],x['altitude_chg1min'],100,50),axis=1)

# graph the route, stages of flight / altitude, speed etc.

line_graph(dfs['__lon,__deg'], dfs['__lat,__deg'],"Longitude", "Latitude", "Graph of lat by lon", f"{folder}/lat_lon.png")

line_graph(dfs['_totl,_time'], dfs['p-alt,ftMSL'],"Time", "Altitude", "Graph of altitude by time", f"{folder}/alt_time.png")

hist_graph(dfs,'_Vind,_kias',None,"xyz")

hist_graph(dfs,'hding,__mag',36,"xyz1") # 10 degree intervals

# dual line graph of speed and altitude
dual_line_graph(dfs['_totl,_time'], dfs['p-alt,ftMSL'], "Time", "Altitude", dfs['_Vind,_kias'], "Speed", "Speed and Altitude", f"{folder}/dual_line.png")

# dual line graph of pitch and altitude
dual_line_graph(dfs['_totl,_time'], dfs['p-alt,ftMSL'], "Time", "Altitude", dfs['pitch,__deg'], "Pitch", "Pitch and Altitude", f"{folder}/dual_line1.png")

# dual line graph of fuel and altitude
dual_line_graph(dfs['_totl,_time'], dfs['p-alt,ftMSL'], "Time", "Altitude", dfs['_fuel,___lb'], "Fuel", "Fuel and Altitude", f"{folder}/dual_line2.png")

# dual line graph of fuel and speed
dual_line_graph(dfs['_totl,_time'], dfs['_Vind,_kias'], "Time", "Speed", dfs['_fuel,___lb'], "Fuel", "Fuel and Speed", f"{folder}/dual_line3.png")


# plot on basemap
base_map_plot(sdata=1,lat=dfs['__lat,__deg'],lon=dfs['__lon,__deg'],area=dfs['p-alt,ftMSL'],label="",ldg_name="Altitude",bar=True,
              title="Title",name="map")

###############################################################################