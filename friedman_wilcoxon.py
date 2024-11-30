# guidance found here: https://stats.stackexchange.com/questions/467467/how-to-do-friedman-test-and-post-hoc-test-on-python

from scipy.stats import friedmanchisquare, wilcoxon
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

# create df
nums = np.random.randint(0,100,(270))
labels = ["a","b","c"]*90
xyz = [1,2,3]*90

df = pd.DataFrame({'labels':labels, 'nums':nums, 'xyz':xyz})

# boxplot graph of data
labels = df['labels'].unique()

collection=[]
for label in labels:
    test = df[df['labels']==label]
    collection.append(df['nums'][df['labels']==label].to_list())

# make list
#for i in range(0,len(labels)):
#    labels[i] = collection[i]

def boxplot_graph(data, labels):
    fig, ax = plt.subplots()
    
    ax.boxplot(x=data, labels=labels, showmeans = False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Type',size=10)
    ax.set_ylabel("Number", size=10)
    ax.set_title('Title',size=12)
    
    # annotate graph
    #for x in zip()
    
    #set size of graph
    cmsize=1/2.54
    fig.set_size_inches(30*cmsize, 15*cmsize)
    
    return ax.boxplot(x=data, labels=labels, showmeans = False)


    # annotate graph
    #for x in zip()
    
    #set size of graph
    cmsize=1/2.54
    fig.set_size_inches(30*cmsize, 15*cmsize)

bp = boxplot_graph(collection, labels=labels)

# convert labels to factor
df['labels_fct'] = pd.factorize(df['labels'])[0]

# drop categorical column
df = df.drop('labels', axis="columns")

# run the friedman test through the columns
f_test = friedmanchisquare(df['labels_fct'],df['nums'],df['xyz'])
f_res = pd.DataFrame({'test':'Friedman','statistic':f_test[0],'pvalue':f_test[0]},index=[0])

# run pairwise wilcoxon
wilc_test = [wilcoxon(df[i],df[j]) for i,j in itertools.combinations(df.columns,2)]    
w_res = pd.DataFrame(wilc_test)
w_res['test'] = ["wilcoxon " + i+" vs "+j for i,j in itertools.combinations(df.columns,2)]

# concat the results
res = pd.concat([f_res,w_res])