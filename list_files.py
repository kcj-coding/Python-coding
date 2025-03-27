import pandas as pd
import re
import os
import datetime

folder = r"C:\Users\kelvi\Desktop"

# get sub folder
folders = [x[0] for x in os.walk(folder)]

files = pd.DataFrame()
# get files in folders
for folder_name in folders:
    file = os.listdir(folder_name)
    files_x = pd.DataFrame({'folder':folder_name, 'files':file})
    files = pd.concat((files,files_x))
    
files = files.reset_index()
files = files.drop(columns={'index'})

files['file_link'] = files['folder']+"/"+files['files']

# get times of modification and creation, and size of the files
mod_times = []
mod_yrs = []
create_times = []
sizes = []
exts = []

for filer in files['file_link']:
    #folder1 = files[['folder']][files['files']==filer].reset_index()
    #folder1 = folder1[['folder']]
    #folder1=folder1['folder'][0]
    mod_time = os.path.getmtime(filer)#rf"{folder1}\{filer}")
    mod_time = datetime.datetime.fromtimestamp(mod_time)
    mod_times.append(mod_time)
    
    mod_yr = mod_time.year
    mod_yrs.append(mod_yr)
    
    create_time = os.path.getctime(filer)#rf"{folder1}\{filer}")
    create_time = datetime.datetime.fromtimestamp(create_time)
    create_times.append(create_time)
    
    size = os.stat(filer).st_size#rf"{folder1}\{filer}").st_size
    sizes.append(size) # size in Bytes
    
    ext = re.sub(".*(?<=\.)","",filer)
    exts.append(ext)

        
# add to df
df = pd.DataFrame({'folder':files['folder'], 'file':files['files'], 'created':create_times, 'modified':mod_times, 'size':sizes, 'file_link':files['file_link'], 'ext':exts, 'yr':mod_yrs})

# output csv
df.to_csv(folder+"\\"+"files_and_folders.csv", index=False)

# group by
df_yr = df.groupby(['yr'], as_index=False).count()
df_yr = df.groupby(['yr']).count().reset_index()
df_yr = df_yr[df_yr.columns[0:2]]

# filetypes
df_f = df.groupby(['ext'], as_index=False).count()
df_f = df.groupby(['ext']).count().reset_index()
df_f = df_f[df_f.columns[0:2]]
df_f = df_f.sort_values(df_f.columns[1], ascending=False)