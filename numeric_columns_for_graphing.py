#import win32com.client # to interact with windows applications
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import re
import os
import glob
import time

################# configure variables #########################################

# file folder location 
folder = r"C:\\Folder"

# remove any columns with score less than specified
corr_threshold = 0.001 # this is for column against column graphing

# only one of graph_by_ should be True
graph_by_column_name = True
graph_by_column_number = False


# check output folder exists and if not create it
if not os.path.exists(folder):
    os.makedirs(folder, exist_ok=True)
    
################# define functions ###########################################
    
def number_format(num):
    if math.isnan(num):
        digits = 0
    elif num > 0:
        digits = int(math.log10(num))+1
    elif num == 0:
        digits = 1
    else:
        digits = int(math.log10(-num))+2 # +1 if you don't count the '-' 

    if digits > 3 or digits < 0:
        return '{:.2e}'.format(num)
    elif digits == 0:
        return 0
    else:
        return round(num,2)
   
# qq plot from https://stackoverflow.com/questions/13865596/quantile-quantile-plot-using-scipy
def QQ_plot(data, save_loc):

    # Sort as increasing
    y = np.sort(data)
    
    # Compute sample mean and std
    mean, std = np.mean(y), np.std(y)
    
    # Compute set of Normal quantiles
    ppf = stats.norm(loc=mean, scale=std).ppf # Inverse CDF
    N = len(y)
    x = [ppf( i/(N+2) ) for i in range(1,N+1)]

    # Make the QQ scatter plot
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    
    # Plot diagonal line
    dmin, dmax = np.min([x,y]), np.max([x,y])
    diag = np.linspace(dmin, dmax, 1000)
    ax.plot(diag, diag, color='red', linestyle='--')
    plt.gca().set_aspect('equal')
    
    # Add labels
    ax.set_xlabel('Normal quantiles')
    ax.set_ylabel('Sample quantiles')
    
    #set size of graph
    cmsize=1/2.54
    fig.set_size_inches(30*cmsize, 15*cmsize)
    
    #save plot
    plt.savefig(save_loc, dpi=400, bbox_inches='tight')
    
def line_graph(x,y,xx,yy,title,file_loc):
    fig, ax = plt.subplots()
    
    # Plot each bar plot. Note: manually calculating the 'dodges' of the bars
    #ax = plt.axes()
    
    #ax.plot(np.arange(0,len(df.data[df.type=='a'])), df.c_number[df.type=='a'], label='Product a', color='#CD2456')
    #ax.plot(np.arange(0,len(df.data[df.type=='b'])), df.c_number[df.type=='b'], label='Product b', color='#14022E')
    
    ax.scatter(x,y,alpha=0.5, c="black")
    ax.plot(X, y_pred_lr, c="blue")
    ax.plot(X, y_pred_ply, c="red")
    
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
    #ax.legend(loc='upper left',fontsize=8)
    #ax.legend(*[*zip(*{l:h for h,l in zip(*ax.get_legend_handles_labels())}.items())][::-1])
    #ax.set_axisbelow(True) # to put gridlines at back
    #ax.grid(linestyle='--',color='#CECECE')
    ax.tick_params(axis='y',labelsize=8)
    ax.spines[['right','top']].set_visible(False)
    #ax.spines['right'].set_visible(False)
    
    #set size of graph
    cmsize=1/2.54
    fig.set_size_inches(30*cmsize, 15*cmsize)
    
    #save plot
    plt.savefig(file_loc, dpi=400, bbox_inches='tight')
    
    # Ask Matplotlib to show the plot
    #plt.show()
    #plt.close()

##################### modify data #############################################

# check
if (graph_by_column_name == True and graph_by_column_number==True):
    raise ValueError("Cannot both graph_by_ be true")
    
start_time = time.time()

df = pd.DataFrame({"data":[x for x in range(1,11,1)], "typer":["a","b"]*5})

#df = pd.DataFrame({"val":[random.randint(0,1000) for p in range(0,900,1)], "val_str":np.repeat(["1","2","3"],300)})

#df.columns = df.columns.str.replace("[^A-Za-z0-9]","", regex=True) # remove special characters from column names

df_old = df

for column in df.columns:
    # if column is not type int or float then turn into str
    #if df[column].dtype not in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:# or not in ["bifu"]:
    if df[column].dtype.kind not in "bifu":

        df[column+"_str"] = df[column]#.copy
        df[column+"_str"] = df[column+"_str"].astype(str)
        df[column+"_str"] = pd.factorize(df[column+"_str"])[0]
        
# write as csv with converted columns
df.to_csv(f'{folder}/file.csv',index=False)

# select only the numeric columns
numerics= ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_num = df.select_dtypes(include=numerics)

############## select columns for comparison by correlation scores ############

# get correlation scores
df_num_corr = df_num.corr()

# remove any columns which have a correlation greater than = -0.1 and less than = 0.1
df_num_corr_x = df_num_corr[(df_num_corr >= corr_threshold) | (df_num_corr <= -corr_threshold)]
df_num_corr_x_r = df_num_corr_x[[df_num_corr_x.columns[0]]]
df_num_corr_x_r = df_num_corr_x.dropna()

# since index and matrix, all rows have all possible values hence take first column and remove any na
# then take the column names from the index
# make the existing df only with these column names - for the column versus column part

corr_columns = df_num_corr_x_r.index.to_list()
df_corr_columns = df_num[[c for c in df_num.columns if c in corr_columns]]

# or make long df that can be queried
df_cor = df_num_corr_x.stack().rename_axis(('from','to')).reset_index(name='correlation')
# df_cor = df_cor[(df_cor['from'] != df_cor['to']) & (~df_cor['correllation'].isnull())] # filter away not same name and nan values

# save correlation scores
df_cor.to_csv(f'{folder}/corr.csv',index=False)

###############################################################################

# plot histogram of each column against other columns
fig, ax = plt.subplots()
df_num.hist()
#set size of graph
cmsize=1/2.54
fig.set_size_inches(30*cmsize, 15*cmsize)

# save
plt.savefig(f'{folder}/'+'_hist.png', dpi=400, bbox_inches='tight')

df_num.plot.hist()

################ update by column name ########################################
if graph_by_column_name == True:
    # graph each column against every other column
    for i in df_num.columns:
        
        # plot histogram and boxplot of this data
    
        if not os.path.exists(f'{folder}/{str(i)}'):
            os.makedirs(f'{folder}/{str(i)}', exist_ok=True)
    
        mean = df_num[i].mean()
        std = df_num[i].std()
        
        # what percentage of values are captured within the range 
        pct_val = round((len(df_num[(df_num[i] < mean+(2*std)) & (df_num[i] > mean-(2*std))])/len(df_num[i]))*100,3)
        
        #### histogram
        fig, ax = plt.subplots()
        
        #ax.boxplot(x=df_num[i], labels=labels, showmeans = False)
        df_num[[i]].hist(ax=ax)
        plt.axvline(mean,linestyle="solid",c="red")
        plt.axvline(mean+(2*std),linestyle="dashed",c="red")
        plt.axvline(mean-(2*std),linestyle="dashed",c="red")
        ax.grid(linestyle='',color='#CECECE')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(str(i),size=10)
        ax.set_ylabel("Frequency", size=10)
        ax.set_title('Title'+" for column "+str(i)+" Mean:"+str(round(mean,3))+"\n percentage of values captured within range:"+str(round(pct_val,3))+"%",size=12)
        
        # annotate graph
        #for x in zip()
        
        #set size of graph
        cmsize=1/2.54
        fig.set_size_inches(30*cmsize, 15*cmsize)
        
        # save
        plt.savefig(f'{folder}/{str(i)}/{str(i)}'+'_hist.png', dpi=400, bbox_inches='tight')
        
        ### histogram with pdf
        
        param = stats.norm.fit(df_num[i].dropna())
        x_pdf = np.linspace(*df_num[i].agg([min, max]), 100) # x-values
    
        
        
        fig, ax = plt.subplots()
        
        #ax.boxplot(x=df_num[i], labels=labels, showmeans = False)
        df_num[[i]].hist(ax=ax, density=True)
        plt.plot(x_pdf, stats.norm.pdf(x_pdf, *param), color = 'black')
        plt.axvline(mean,linestyle="solid",c="red")
        plt.axvline(mean+(2*std),linestyle="dashed",c="red")
        plt.axvline(mean-(2*std),linestyle="dashed",c="red")
        ax.grid(linestyle='',color='#CECECE')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(str(i),size=10)
        ax.set_ylabel("Frequency", size=10)
        ax.set_title('Title'+" for column "+str(i)+" Mean:"+str(round(mean,3))+"\n percentage of values captured within range:"+str(round(pct_val,3))+"%",size=12)
        
        # annotate graph
        #for x in zip()
        
        #set size of graph
        cmsize=1/2.54
        fig.set_size_inches(30*cmsize, 15*cmsize)
        
        # save
        plt.savefig(f'{folder}/{str(i)}/{str(i)}'+'_hist_pdf.png', dpi=400, bbox_inches='tight')
        
        
        
        #### boxplot
        fig, ax = plt.subplots()
        
        #ax.boxplot(x=df_num[i], labels=labels, showmeans = False)
        df_num[[i]].boxplot(ax=ax)
        ax.grid(linestyle='',color='#CECECE')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(str(i),size=10)
        ax.set_ylabel("Number", size=10)
        ax.set_title('Title'+" for column "+str(i)+" Mean:"+str(round(mean,3)),size=12)
        
        # annotate graph
        #for x in zip()
        
        #set size of graph
        cmsize=1/2.54
        fig.set_size_inches(30*cmsize, 15*cmsize)
        
        # save
        plt.savefig(f'{folder}/{str(i)}/{str(i)}'+'_boxplot.png', dpi=400, bbox_inches='tight')
        
        # qq plot
        QQ_plot(df_num[i],f'{folder}/{str(i)}/{str(i)}'+'_qqplot.png')
        
        for j in df_num.columns:
            #tst = df_cor['correlation'][(df_cor['from']==i) & (df_cor['to']==j)].values[0]
            try:
                if i != j and abs(df_cor['correlation'][(df_cor['from']==i) & (df_cor['to']==j)].values[0]) >= corr_threshold:
                    
                    #if not os.path.exists(f'{folder}/{str(i)}/{str(j)}'):
                    #    os.makedirs(f'{folder}/{str(i)}/{str(j)}', exist_ok=True)
                    
                    if not os.path.exists(f'{folder}/{str(i)}/{str("Columns")}'):
                         os.makedirs(f'{folder}/{str(i)}/{str("Columns")}', exist_ok=True)
                         
                    # make x and y array of this data
                    x_arr = df_num[i].to_numpy().reshape(-1,1)
                    y_arr = df_num[j].to_numpy().reshape(-1,1)
                    
                    shape = x_arr.shape
                    X = np.sort(x_arr.flatten())
                    
                    # linear regression
                    
                    # fit a linear regression model
                    model_lr = LinearRegression().fit(x_arr,y_arr)
                    r_sq_lr = model_lr.score(x_arr, y_arr)  # model acccuracy, e.g. is x a good predictor of y for this model with its parameters
                    
                    
                    lr_int = number_format(model_lr.intercept_[0])
                    lr_coef = number_format(model_lr.coef_[0][0])
                    
                    # predict y-values
                    y_pred_lr = model_lr.predict(X.reshape((shape)))
                    
    
                    # polynominal regression
                    
                    # set degree as number of polynomial regression function
                    degree_num = 3
                    
                    # transform the x data
                    transformer = PolynomialFeatures(degree=degree_num, include_bias=False)
                    
                    transformer.fit(x_arr)
                    
                    x_ = transformer.transform(x_arr)
                    
                    # or
                    
                    x_ = PolynomialFeatures(degree=degree_num, include_bias=False).fit_transform(x_arr)
                    
                    # fit a linear regression model
                    model_ply = LinearRegression().fit(x_, y_arr)
                    r_sq_ply = model_ply.score(x_, y_arr) # is x a good predictor of y for this model with its parameters
                    
                    
                    ply_int = number_format(model_ply.intercept_[0])
                    ply_coef1 = number_format(model_ply.coef_[0][0]) # 1st parameter
                    ply_coef2 = number_format(model_ply.coef_[0][1])# 2nd parameter
                    ply_coef3 = number_format(model_ply.coef_[0][2]) # 3rd parameter
                    
                    # predict y-values
                    y_pred_ply = model_ply.predict(PolynomialFeatures(degree=degree_num, include_bias=False).fit_transform(X.reshape((shape))))
                
                    # graph the data
                    #line_graph(df_num[i],df_num[j],xx=i,yy=j,title="Graph of "+i+" against "+j+" || LR: y="+str(lr_coef)+"x+"+str(lr_int)+" || PLY: y="+str(ply_coef1)+"x1+"+str(ply_coef2)+"x2+"+str(ply_coef3)+"x3+"+str(ply_int)+"\n lr_rsq: "+str(round(r_sq_lr,3))+"; ply_rsq: "+str(round(r_sq_ply,3))+" || target corr: "+str(corr_threshold)+"; actual corr: "+str(round(abs(df_cor['correlation'][(df_cor['from']==i) & (df_cor['to']==j)].values[0]),3)), file_loc=f'{folder}/{str(i)}/{str("Columns")}/{str(i)}_{str(j)}'+'_.png')
                    line_graph(df_num[i],df_num[j],xx=i,yy=j,title="Graph of "+i+" against "+j+" || LR: y="+str(lr_coef)+"x+"+str(lr_int)+" || PLY: y="+str(ply_coef1)+"x+"+str(ply_coef2)+"x\u00b2+"+str(ply_coef3)+"x\u00b3+"+str(ply_int)+"\n lr_rsq: "+str(round(r_sq_lr,3))+"; ply_rsq: "+str(round(r_sq_ply,3))+" || target corr: "+str(corr_threshold)+"; actual corr: "+str(round(abs(df_cor['correlation'][(df_cor['from']==i) & (df_cor['to']==j)].values[0]),3)), file_loc=f'{folder}/{str(i)}/{str("Columns")}/{str(i)}_{str(j)}'+'_.png')
                    
                    # boxplot of both columns
                    ttt = stats.ttest_ind(df_num[i],df_num[j]) # t-test
                    my_dict = {str(i): df_num[i], str(j): df_num[j]}
                    
                    fig, ax = plt.subplots()
                    ax.boxplot(my_dict.values(), labels=my_dict.keys())
                    ax.grid(linestyle='',color='#CECECE')
                    ax.spines[['right','top']].set_visible(False)
                    ax.set_xlabel(str(i),size=10)
                    ax.set_ylabel("Number", size=10)
                    ax.set_title('Title'+" for column "+str(i)+" Mean:"+str(round(mean,3))+
                                 "\nt-stat: "+str(ttt[0])+"; p-value: "+str(ttt[1]),size=12)
                    
                    # annotate graph
                    #for x in zip()
                    
                    #set size of graph
                    cmsize=1/2.54
                    fig.set_size_inches(30*cmsize, 15*cmsize)
                    
                    # save
                    plt.savefig(f'{folder}/{str(i)}/{str("Columns")}/{str(i)}'+'_boxplot.png', dpi=400, bbox_inches='tight')
                    
                    # k-means
            except:
                pass

################ code below achieves same outcome using column number ###########
################ update by column number ########################################            

if graph_by_column_number == True:

    df_num = df.select_dtypes(include=numerics)
    for i in range(0,len(df_num.columns),1):
        
        # name as string
        #df_num.columns[i]
        # values as series
        #df_num[df_num.columns[i]]
        # values as dataframe
        #df_num[df_num.columns[[i]]]
        
        # plot histogram and boxplot of this data
        
    
        if not os.path.exists(f'{folder}/{str(df_num.columns[i])}'):
            os.makedirs(f'{folder}/{str(df_num.columns[i])}', exist_ok=True)
    
        mean = df_num[df_num.columns[i]].mean()
        std = df_num[df_num.columns[i]].std()
        
        # what percentage of values are captured within the range
        pct_val = round((len(df_num[(df_num[df_num.columns[i]] < mean+(2*std)) & (df_num[df_num.columns[i]] > mean-(2*std))])/len(df_num[df_num.columns[i]]))*100,3)
        
        #### histogram
        fig, ax = plt.subplots()
        
        #ax.boxplot(x=df_num[i], labels=labels, showmeans = False)
        df_num[df_num.columns[i]].hist(ax=ax)
        plt.axvline(mean,linestyle="solid",c="red")
        plt.axvline(mean+(2*std),linestyle="dashed",c="red")
        plt.axvline(mean-(2*std),linestyle="dashed",c="red")
        ax.grid(linestyle='',color='#CECECE')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(str(i),size=10)
        ax.set_ylabel("Frequency", size=10)
        ax.set_title('Title'+" for column "+str(df_num.columns[i])+" Mean:"+str(round(mean,3))+"\n percentage of values captured within range:"+str(round(pct_val,3))+"%",size=12)
        
        
        #set size of graph
        cmsize=1/2.54
        fig.set_size_inches(30*cmsize, 15*cmsize)
        
        # save
        plt.savefig(f'{folder}/{str(df_num.columns[i])}/{str(df_num.columns[i])}'+'_hist.png', dpi=400, bbox_inches='tight')
        
        ### histogram with pdf
        
        param = stats.norm.fit(df_num[df_num.columns[i]].dropna())
        x_pdf = np.linspace(*df_num[df_num.columns[i]].agg([min, max]), 100) # x-values
    
        
        
        fig, ax = plt.subplots()
        
        #ax.boxplot(x=df_num[i], labels=labels, showmeans = False)
        df_num[df_num.columns[i]].hist(ax=ax, density=True)
        plt.plot(x_pdf, stats.norm.pdf(x_pdf, *param), color = 'black')
        plt.axvline(mean,linestyle="solid",c="red")
        plt.axvline(mean+(2*std),linestyle="dashed",c="red")
        plt.axvline(mean-(2*std),linestyle="dashed",c="red")
        ax.grid(linestyle='',color='#CECECE')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(str(i),size=10)
        ax.set_ylabel("Frequency", size=10)
        ax.set_title('Title'+" for column "+str(df_num.columns[i])+" Mean:"+str(round(mean,3))+"\n percentage of values captured within range:"+str(round(pct_val,3))+"%",size=12)
        
        # annotate graph
        #for x in zip()
        
        #set size of graph
        cmsize=1/2.54
        fig.set_size_inches(30*cmsize, 15*cmsize)
        
        # save
        plt.savefig(f'{folder}/{str(df_num.columns[i])}/{str(df_num.columns[i])}'+'_hist_pdf.png', dpi=400, bbox_inches='tight')
        
    
        #### boxplot
        fig, ax = plt.subplots()
        
        #ax.boxplot(x=df_num[i], labels=labels, showmeans = False)
        df_num[df_num.columns[[i]]].boxplot(ax=ax)
        ax.grid(linestyle='',color='#CECECE')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(str(i),size=10)
        ax.set_ylabel("Number", size=10)
        ax.set_title('Title'+" for column "+str(df_num.columns[i])+" Mean:"+str(round(mean,3)),size=12)
        
        # annotate graph
        #for x in zip()
        
        #set size of graph
        cmsize=1/2.54
        fig.set_size_inches(30*cmsize, 15*cmsize)
        
        # save
        plt.savefig(f'{folder}/{str(df_num.columns[i])}/{str(df_num.columns[i])}'+'_boxplot.png', dpi=400, bbox_inches='tight')# plot histogram and boxplot of this data
        
        # qq plot
        QQ_plot(df_num[df_num.columns[[i]]],f'{folder}/{str(df_num.columns[i])}/{str(df_num.columns[i])}'+'_qqplot.png')
        
        for j in range(0,len(df_num.columns),1):
            try:
                if i != j and abs(df_cor['correlation'][(df_cor['from']==df_num.columns[i]) & (df_cor['to']==df_num.columns[j])].values[0]) >= corr_threshold:
                    
                    #if not os.path.exists(f'{folder}/{str(df_num.columns[i])}/{str(df_num.columns[j])}'):
                    #    os.makedirs(f'{folder}/{str(df_num.columns[i])}/{str(df_num.columns[j])}', exist_ok=True)
                        
                    if not os.path.exists(f'{folder}/{str(df_num.columns[i])}/{str("Columns")}'):
                        os.makedirs(f'{folder}/{str(df_num.columns[i])}/{str("Columns")}', exist_ok=True)
                    
                    # make x and y array of this data
                    x_arr = df_num[df_num.columns[i]].to_numpy().reshape(-1,1)
                    y_arr = df_num[df_num.columns[j]].to_numpy().reshape(-1,1)
                    
                    shape = x_arr.shape
                    X = np.sort(x_arr.flatten())
                    
                    # linear regression
                    
                    # fit a linear regression model
                    model_lr = LinearRegression().fit(x_arr,y_arr)
                    r_sq_lr = model_lr.score(x_arr, y_arr)  # model acccuracy, e.g. is x a good predictor of y for this model with its parameters
                    
                    # get model parameters
                    lr_int = number_format(model_lr.intercept_[0])
                    lr_coef = number_format(model_lr.coef_[0][0])
                    
                    # predict y-values
                    y_pred_lr = model_lr.predict(X.reshape((shape)))
                    
    
                    # polynominal regression
                    
                    # set degree as number of polynomial regression function
                    degree_num = 3
                    
                    # transform the x data
                    transformer = PolynomialFeatures(degree=degree_num, include_bias=False)
                    
                    transformer.fit(x_arr)
                    
                    x_ = transformer.transform(x_arr)
                    
                    # or
                    
                    x_ = PolynomialFeatures(degree=degree_num, include_bias=False).fit_transform(x_arr)
                    
                    # fit a linear regression model
                    model_ply = LinearRegression().fit(x_, y_arr)
                    r_sq_ply = model_ply.score(x_, y_arr) # is x a good predictor of y for this model with its parameters
                    
                    # get model parameters
                    ply_int = number_format(model_ply.intercept_[0])
                    ply_coef1 = number_format(model_ply.coef_[0][0]) # 1st parameter
                    ply_coef2 = number_format(model_ply.coef_[0][1])# 2nd parameter
                    ply_coef3 = number_format(model_ply.coef_[0][2]) # 3rd parameter
                    
                    # predict y-values
                    y_pred_ply = model_ply.predict(PolynomialFeatures(degree=degree_num, include_bias=False).fit_transform(X.reshape((shape))))
                    
                    # graph the data
                    #line_graph(df_num[df_num.columns[i]],df_num[df_num.columns[j]],xx=df_num.columns[i],yy=df_num.columns[j],title="Graph of "+df_num.columns[i]+" against "+df_num.columns[j]+" || LR: y="+str(lr_coef)+"x+"+str(lr_int)+" || PLY: y="+str(ply_coef1)+"x1+"+str(ply_coef2)+"x2+"+str(ply_coef3)+"x3+"+str(ply_int)+"\n lr_rsq: "+str(round(r_sq_lr,3))+"; ply_rsq: "+str(round(r_sq_ply,3))+" || target corr: "+str(corr_threshold)+"; actual corr: "+str(round(abs(df_cor['correlation'][(df_cor['from']==df_num.columns[i]) & (df_cor['to']==df_num.columns[j])].values[0]),3)), file_loc=f'{folder}/{str(df_num.columns[i])}/{str("Columns")}/{str(df_num.columns[i])}_{str(df_num.columns[j])}'+'_.png')
                    line_graph(df_num[df_num.columns[i]],df_num[df_num.columns[j]],xx=df_num.columns[i],yy=df_num.columns[j],title="Graph of "+df_num.columns[i]+" against "+df_num.columns[j]+" || LR: y="+str(lr_coef)+"x+"+str(lr_int)+" || PLY: y="+str(ply_coef1)+"x+"+str(ply_coef2)+"x\u00b2+"+str(ply_coef3)+"x\u00b3+"+str(ply_int)+"\n lr_rsq: "+str(round(r_sq_lr,3))+"; ply_rsq: "+str(round(r_sq_ply,3))+" || target corr: "+str(corr_threshold)+"; actual corr: "+str(round(abs(df_cor['correlation'][(df_cor['from']==df_num.columns[i]) & (df_cor['to']==df_num.columns[j])].values[0]),3)), file_loc=f'{folder}/{str(df_num.columns[i])}/{str("Columns")}/{str(df_num.columns[i])}_{str(df_num.columns[j])}'+'_.png')
                    
                    # boxplot of both columns
                    ttt = stats.ttest_ind(df_num[df_num.columns[i]],df_num[df_num.columns[j]]) # t-test
                    my_dict = {str(df_num.columns[i]): df_num[df_num.columns[i]], str(df_num.columns[j]): df_num[df_num.columns[j]]}
                    
                    fig, ax = plt.subplots()
                    ax.boxplot(my_dict.values(), labels=my_dict.keys())
                    ax.grid(linestyle='',color='#CECECE')
                    ax.spines[['right','top']].set_visible(False)
                    ax.set_xlabel(str(i),size=10)
                    ax.set_ylabel("Number", size=10)
                    ax.set_title('Title'+" for column "+str(i)+" Mean:"+str(round(mean,3))+
                                 "\nt-stat: "+str(ttt[0])+"; p-value: "+str(ttt[1]),size=12)
                    
                    # annotate graph
                    #for x in zip()
                    
                    #set size of graph
                    cmsize=1/2.54
                    fig.set_size_inches(30*cmsize, 15*cmsize)
                    
                    # save
                    plt.savefig(f'{folder}/{str(df_num.columns[i])}/{str("Columns")}/{str(df_num.columns[i])}'+'_boxplot.png', dpi=400, bbox_inches='tight')
                    
                    # k-means
            except:
                pass
            
print("runtime of processing: " +str(round(time.time()-start_time,3))+" seconds")

# remove locally saved plots
# get sub folder
files_list = []
folders = [x[0] for x in os.walk(folder)]
for folder_ext in folders:
    files_list1 = [x for x in glob.glob(rf"{folder_ext}/*.png")]
    files_list.append(files_list1)
    
for file in files_list:
    for filer in file:
        os.remove(filer)