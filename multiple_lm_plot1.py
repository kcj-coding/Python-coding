# monte carlo estimate
import random
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


#import numpy as np
#import pandas as pd
#from scipy import stats
#import textwrap
#import statsmodels.formula.api as smf
#import statsmodels.api as sm
#from matplotlib import pyplot as plt

#from sklearn.linear_model import LinearRegression
#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.metrics import r2_score




#df <- data.frame(xx=rnorm(200), yy=rnorm(200), zz=rep(c("a","b", "c", "d"),times=50))
#df <- data.frame(xx=seq(1,200,1), yy=rep(c(1,2,3,2),each=50), zz=rep(c("a","b", "c", "d"),times=50))

df = pd.DataFrame({'xx':np.random.normal(0,1,200), 'yy':np.random.normal(0,1,200), 'zz':["a","b","c","d"]*50})
#df = pd.DataFrame({'xx':range(0,200,1), 'yy':[1,2,3,4]*50, 'zz':["a","b","c","d"]*50})

x_name = "xx"
group_var = "zz"

# graph by group

# for every column in df

for column in df.columns:
    # select the columns of interest
    df_tst = df[[x_name, column, group_var]]
    
    if (x_name is not column) & (df[column].dtype.kind in 'biufc'): # column type is numeric
    
        # get boxplot data
        
        labels = df_tst[group_var].unique()

        collection=[]
        for label in labels:
            test = df_tst[df_tst[group_var]==label]
            collection.append(df_tst[column][df_tst[group_var]==label].to_list())
    
        # boxplot
        fig, ax = plt.subplots()
        
        #ax.boxplot(x=df_num[i], labels=labels, showmeans = False)
        ax.boxplot(collection, tick_labels=labels)
        ax.grid(linestyle='',color='#CECECE')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(str(group_var),size=10)
        ax.set_ylabel("Number", size=10)
        ax.set_title('Title',size=12)
        
        # annotate graph
        #for x in zip()
        
        #set size of graph
        cmsize=1/2.54
        fig.set_size_inches(30*cmsize, 15*cmsize)
        
        plt.show()
        
        ########## plot scatter by group_var
        
        # get equations by type
        
        
        unq_cat = len(labels) # get count of how many unique group types
        fig, ax = plt.subplots(unq_cat)
        
        # for each group, make a separate graph
        for axi in range(0,unq_cat,1):
            ax[axi].scatter(df_tst[x_name][df_tst[group_var]==labels[axi]], df_tst[column][df_tst[group_var]==labels[axi]])
            
            xxx = df_tst[x_name][df_tst[group_var]==labels[axi]]
            yyy = df_tst[column][df_tst[group_var]==labels[axi]]
            
            # make smooth linear and polynominal estimates
            
            # lr
            poly_lr = np.polyfit(xxx, yyy, deg=1, rcond=None, full=False, w=None, cov=False)
            p_lr = np.poly1d(poly_lr)
            p_lr = r2_score(yyy, p_lr(xxx))
            xs = np.linspace(min(xxx), max(xxx), 100)
            ys = poly_lr[0] * xs + poly_lr[1]
            ax[axi].plot(xs, ys, color='blue')#, label=f'$({poly_lr[0]:.2f})x + ({poly_lr[1]:.2f}); r^2={p_lr}$')
            
            
            # poly
            poly_ply = np.polyfit(xxx, yyy, deg=3, rcond=None, full=False, w=None, cov=False)
            p_ply= np.poly1d(poly_ply)
            p_ply = r2_score(yyy, p_ply(xxx))
            xs_ply = np.linspace(min(xxx), max(xxx), 100)
            ys_ply = poly_ply[0] * xs ** 3 + poly_ply[1] * xs ** 2 + poly_ply[2] * xs + poly_ply[3]
            ax[axi].plot(xs_ply, ys_ply, color='red')#, label=f'$({poly_ply[0]:.2f})x^3+({poly_ply[1]:.2f})x^2+({poly_ply[2]:.2f})x + ({poly_ply[3]:.2f}); r^2={p_ply}$')
            
            # annotate equations
            #ax[axi].annotate(min(xxx),min(yyy),)
            
            ax[axi].annotate(f'y=$({poly_lr[0]:.2f})x + ({poly_lr[1]:.2f})$; $r^2={p_lr:.2f}$',
                        xy=(np.mean(xxx),max(yyy)),
                        xytext=(15, 0),  # 4 points vertical offset.
                        textcoords='offset points',
                        ha='center', va='bottom', color="blue")
            
            ax[axi].annotate(f'y=$({poly_ply[0]:.2f})x^3+({poly_ply[1]:.2f})x^2+({poly_ply[2]:.2f})x + ({poly_ply[3]:.2f})$; $r^2={p_ply:.2f}$',
                        xy=(np.mean(xxx),min(yyy)),
                        xytext=(0, 0),  # 4 points vertical offset.
                        textcoords='offset points',
                        ha='center', va='bottom', color="red")
            
        
            #ax.boxplot(x=df_num[i], labels=labels, showmeans = False)
            #ax.boxplot(collection, tick_labels=labels)
            ax[axi].grid(linestyle='',color='#CECECE')
            ax[axi].spines['top'].set_visible(False)
            ax[axi].spines['right'].set_visible(False)
            #ax[axi].set_xlabel(str(group_var),size=10)
            #ax[axi].set_ylabel("Number", size=10)
            #ax[axi].set_title('Title',size=12)
            ax[axi].set_title(f"\n{str(labels[axi])}",size=10, loc="right")
            
            # only show xlabs for bottom graph
            if axi != unq_cat-1:
                ax[axi].get_xaxis().set_visible(False)
                ax[axi].set_xlabel=""
                ax[axi].xlabs=""
        
        # annotate graph
        #for x in zip()
        ax[0].set_title("Title",size=10)
        
        #set size of graph
        cmsize=1/2.54
        fig.set_size_inches(30*cmsize, 15*cmsize)
        
        plt.show()