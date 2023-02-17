import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
import xlsxwriter as xlw
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
#from sklearn.metrics import mean_squared_error

################################# setup #######################################
# setup
file_name = 'C:\\Users\\kelvi\\Desktop\\egkk_passengers1.xlsx' #train/test on existing data
#file_name_test = 'C:\\Users\\kelvi\\Desktop\\dada.xlsx' # future predictions
output_folder = 'C:\\Users\\kelvi\\Desktop\\'
excel_file_name = 'egkk_forecast_python.xlsx'

################################# filters #######################################
#filter configurations
val_num = 10 # train/test split on number x e.g 10 rows into data
future_time = 5 # days,years,months etc

################################# load file #######################################
#open file
df = pd.read_excel(file_name)#, skiprows=4,sheet_name='test')

#rename col of interest to 'test'
df.rename(columns={'Passengers':'test'},inplace=True)

#get arbitiary time-series
df['year'] = df.index+1

def timeseriesmkr(value):
    if value['year'] < 10:
        return "0"+value['year'].astype(str)
    else:
        return value['year'].astype(str)
    
   
df['year'] = df.apply(timeseriesmkr,axis=1)
df['year'] = '20'+df['year'].astype(str) + '-01-01'
df['year'] = pd.to_datetime(df['year'])


#export to excel - if want just 1 file write the df/images from here
#with pd.ExcelWriter(output_folder+excel_file_name, engine = "xlsxwriter") as writer:


#define test and train data note that as time dependent test data should be in the future
val_num=str(val_num)
train = df[df['year'] <= '20'+val_num+'-01-01']
test = df[df['year'] > '20'+val_num+'-01-01']

################ model configuration #####################
y = train['test']
y1 = df['test']

ARMAmodel = SARIMAX(y, order = (1, 0, 1))
ARIMAmodel = ARIMA(y, order = (1, 1, 1))
SARIMAXmodel = SARIMAX(y, order = (5, 4, 2), seasonal_order=(2,2,2,12))

ARMAmodeltest = SARIMAX(y1, order = (1, 0, 1))
ARIMAmodeltest = ARIMA(y1, order = (1, 1, 1))
SARIMAXmodeltest = SARIMAX(y1, order = (5, 4, 2), seasonal_order=(2,2,2,12))
  
########################################################################################
def lwrtest(value):
    if value < 0:
        return 0
    else:
        return value

def lower_test():
    test1['lower test'] = test1['lower test'].apply(lwrtest)#,axis=1)
    test1['Predictions'] = test1['Predictions'].apply(lwrtest)#,axis=1)
    test1['upper test'] = test1['upper test'].apply(lwrtest)#,axis=1)
    
def lower_test_predict():
    df1['lower test'] = df1['lower test'].apply(lwrtest)#,axis=1)
    df1['Predictions'] = df1['Predictions'].apply(lwrtest)#,axis=1)
    df1['upper test'] = df1['upper test'].apply(lwrtest)#,axis=1)
    
############################ graph definitions #########################################
def testdatalinegraph(title,savename):
    # Initialise a figure. subplots() with no args gives one plot.
    fig, ax = plt.subplots()

    # Plot each bar plot. Note: manually calculating the 'dodges' of the bars
    #ax = plt.axes()

    ax.plot(range(0,len(test1.index),1), test1['test'], label='Real', color='#CD2456')
    ax.plot(range(0,len(test1.index),1), test1['Predictions'], label='Predicted', color='#14022E')
    ax.fill_between(range(0,len(test1.index),1),(test1['Predictions']-(test1['Predictions']-test1['lower test'])),(test1['Predictions']+(test1['upper test']-test1['Predictions'])),alpha=0.2)

    # Customise some display properties for line graph
    ax.set_ylabel('',size=8)
    #ax.set_ylim(0,200)
    #ax.set_xlim(0,8)
    ax.set_xlabel('',size=8)
    ax.set_title(title,size=9)
    ax.set_xticks(range(0,len(test1.index),1))    # This ensures we have one tick per year, otherwise we get fewer
    ax.set_xticklabels(range(0,len(test1.index),1), rotation='horizontal',size=8)
    #ax.fill(df.Doy[df.Product=='A'],df.Sales[df.Product=='A']-5,df.Sales[df.Product=='A']+5,color='k', alpha=.15)
    #ax.set_xticks([])
    #ax.set_yticks([])
    ax.legend(loc='upper left',fontsize=8)
    ax.set_axisbelow(True)
    ax.grid(linestyle='--',color='#CECECE')
    ax.tick_params(axis='y',labelsize=8)

    #set size of graph
    cmsize=1/2.54
    fig.set_size_inches(30*cmsize, 15*cmsize)

    #save plot
    plt.savefig(savename, dpi=400, bbox_inches='tight')
    
def futurelinegraph(title,savename):
    # Initialise a figure. subplots() with no args gives one plot.
    fig, ax = plt.subplots()

    # Plot each bar plot. Note: manually calculating the 'dodges' of the bars
    #ax = plt.axes()

    ax.plot(range(0,len(df1.index),1), df1['test'], label='Real', color='#CD2456')
    ax.plot(range(0,len(df1.index),1), df1['Predictions'], label='Predicted', color='#14022E')
    ax.fill_between(range(0,len(df1.index),1),(df1['Predictions']-(df1['Predictions']-df1['lower test'])),(df1['Predictions']+(df1['upper test']-df1['Predictions'])),alpha=0.2)

    # Customise some display properties for line graph
    ax.set_ylabel('',size=8)
    #ax.set_ylim(0,200)
    #ax.set_xlim(0,8)
    ax.set_xlabel('',size=8)
    ax.set_title(title,size=9)
    ax.set_xticks(range(0,len(df1.index),1))    # This ensures we have one tick per year, otherwise we get fewer
    ax.set_xticklabels(range(0,len(df1.index),1), rotation='horizontal',size=8)
    #ax.fill(df.Doy[df.Product=='A'],df.Sales[df.Product=='A']-5,df.Sales[df.Product=='A']+5,color='k', alpha=.15)
    #ax.set_xticks([])
    #ax.set_yticks([])
    ax.legend(loc='upper left',fontsize=8)
    ax.set_axisbelow(True)
    ax.grid(linestyle='--',color='#CECECE')
    ax.tick_params(axis='y',labelsize=8)

    #set size of graph
    cmsize=1/2.54
    fig.set_size_inches(30*cmsize, 15*cmsize)

    #save plot
    plt.savefig(savename, dpi=400, bbox_inches='tight')
    
############################### excel output definitions ############################

def exceltestout(name,imagename):
    with pd.ExcelWriter(output_folder+name+excel_file_name, engine = "xlsxwriter") as writer:
      test1.to_excel(writer, sheet_name='Sheet1',index=False) #index=false drops index rows of df
      worksheet = writer.sheets['Sheet1']
      worksheet.insert_image('K2',imagename)
      
def excelpredictout(filename,imagename):
    with pd.ExcelWriter(output_folder+filename, engine = "xlsxwriter") as writer:
      df1.to_excel(writer, sheet_name='Sheet1',index=False) #index=false drops index rows of df
      worksheet = writer.sheets['Sheet1']
      worksheet.insert_image('I2',imagename)
    
#######################################################################

############### arma ####################################
##########################################################

ARMAmodel = ARMAmodel.fit()

y_pred = ARMAmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = ARMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df['rmse'] =  np.sqrt(mean_squared_error(test["test"], y_pred_df["Predictions"]))
y_pred_df['accuracy'] = (y_pred_df["Predictions"]/test["test"])
y_pred_df['mae']=mean_absolute_error(test["test"].values, y_pred_df["Predictions"])
y_pred_df.index = test.index

#append to test data
test1 = pd.merge(test,y_pred_df,how="left",on=test.index)
    
lower_test()

################### line graph ###############################################
testdatalinegraph(title="ARMA on test data",savename=output_folder+'arma_test.png')

#export to excel
exceltestout(name="ARMA test",imagename=output_folder+'arma_test.png')


y_pred_out = y_pred_df["Predictions"]

# evaluate performance
arma_rmse = np.sqrt(mean_squared_error(test["test"].values, y_pred_df["Predictions"]))
print("RMSE: ",arma_rmse)

############### arima ####################################
##########################################################

ARIMAmodel = ARIMAmodel.fit()

y_pred = ARIMAmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = ARIMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df['rmse'] =  np.sqrt(mean_squared_error(test["test"], y_pred_df["Predictions"]))
y_pred_df['accuracy'] = (y_pred_df["Predictions"]/test["test"])
y_pred_df['mae']=mean_absolute_error(test["test"].values, y_pred_df["Predictions"])
y_pred_df.index = test.index
y_pred_out = y_pred_df["Predictions"] 

#append to test data
test1 = pd.merge(test,y_pred_df,how="left",on=test.index)

lower_test()

################### line graph ###############################################
testdatalinegraph(title="ARIMA on test data",savename=output_folder+'arima_test.png')

#export to excel
exceltestout(name="ARIMA test ",imagename=output_folder+'arima_test.png')

import numpy as np
from sklearn.metrics import mean_squared_error

arma_rmse = np.sqrt(mean_squared_error(test["test"].values, y_pred_df["Predictions"]))
print("RMSE: ",arma_rmse)

############### sarima ####################################
##########################################################

SARIMAXmodel = SARIMAXmodel.fit()

y_pred = SARIMAXmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = SARIMAXmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df['rmse'] =  np.sqrt(mean_squared_error(test["test"], y_pred_df["Predictions"]))
y_pred_df['accuracy'] = (y_pred_df["Predictions"]/test["test"])
y_pred_df['mae']=mean_absolute_error(test["test"].values, y_pred_df["Predictions"])
y_pred_df.index = test.index
y_pred_out = y_pred_df["Predictions"] 

#append to test data
test1 = pd.merge(test,y_pred_df,how="left",on=test.index)

lower_test()

################### line graph ###############################################
testdatalinegraph(title="SARIMA on test data",savename=output_folder+'sarima_test.png')

#export to excel
exceltestout(name="SARIMA test ",imagename=output_folder+'sarima_test.png')


#######################################################################################
################################# future predictions ##################################
df['Predictions']=np.nan
############### arma ####################################
##########################################################
ARMAmodeltest = ARMAmodeltest.fit()

y_pred = ARMAmodeltest.get_forecast((len(df.index)+future_time)-len(df.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = ARMAmodeltest.predict(start = y_pred_df.index[0], end = y_pred_df.index[0]+future_time)
#y_pred_df.index = (df.index+future_time)
y_pred_out = y_pred_df["Predictions"]

#join onto original data
df1=pd.concat([df,y_pred_df])
lower_test_predict()

#graph this
################### line graph ###############################################
futurelinegraph(title="ARMA Predictions",savename=output_folder+'arma_predictions.png')

#excel output
excelpredictout(filename=" ARMA predictions.xlsx",imagename=output_folder+'arma_predictions.png')

############### arima ####################################
##########################################################
ARIMAmodeltest = ARIMAmodeltest.fit()

y_pred = ARIMAmodeltest.get_forecast((len(df.index)+future_time)-len(df.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = ARIMAmodeltest.predict(start = y_pred_df.index[0], end = y_pred_df.index[0]+future_time)
#y_pred_df.index = (df.index+future_time)
y_pred_out = y_pred_df["Predictions"] 

#join onto original data
df1=pd.concat([df,y_pred_df])
lower_test_predict()

#graph this
################### line graph ###############################################
futurelinegraph(title="ARIMA Predictions",savename=output_folder+'arima_predictions.png')

#excel output
excelpredictout(filename=" ARIMA predictions.xlsx",imagename=output_folder+'arima_predictions.png')

############### sarima ####################################
##########################################################
SARIMAXmodeltest = SARIMAXmodeltest.fit()

y_pred = SARIMAXmodeltest.get_forecast((len(df.index)+future_time)-len(df.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = SARIMAXmodeltest.predict(start = y_pred_df.index[0], end = y_pred_df.index[0]+future_time)
#y_pred_df.index = (df.index+future_time)
y_pred_out = y_pred_df["Predictions"] 

#join onto original data
df1=pd.concat([df,y_pred_df])
lower_test_predict()

#graph this
################### line graph ###############################################
futurelinegraph(title="SARIMA Predictions",savename=output_folder+'sarima_predictions.png')

#excel output
excelpredictout(filename=" SARIMA predictions.xlsx",imagename=output_folder+'sarima_predictions.png')

# remove images
os.remove(output_folder+'arma_test.png')
os.remove(output_folder+'arima_test.png')
os.remove(output_folder+'sarima_test.png')
os.remove(output_folder+'arma_predictions.png')
os.remove(output_folder+'arima_predictions.png')
os.remove(output_folder+'sarima_predictions.png')