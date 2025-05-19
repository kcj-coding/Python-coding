# monte carlo estimate
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

start_time = time.time()
########## configuration ######################################################

val1 = "heads"
prob1 = 0.9

val2 = "tails"
prob2 = 0.1

n_runs = 5
n_events = 250

# monte carlo says that as number of events increases the probability tends to the expected probability

# check values equal 1

if prob1 + prob2 != 1:
    raise ValueError("Probability values do not equal 1 - please adjust")
    
######## define functions #####################################################

def grapher(lister,prob,xxx,xxx1,word,n):
    plt.plot(lister)
    plt.axhline(y=prob, color='r', linestyle='dashed')
    [plt.axvline(_x, color='#111212',linewidth=1,alpha=0.1) for _x in xxx]
    [plt.axvline(_x, color='#42f5f2',linewidth=1,alpha=0.1) for _x in xxx1]
    plt.gca().spines[['right','top']].set_visible(False)
    plt.xlabel("Iterations")
    plt.ylabel("Probability")
    plt.title(f"Total Probability of {word} outcome from coin flip; n={n}\n{val1}={len(xxx1)}; {val2}={len(xxx)}")
    plt.show()
    
def coin_flip():
    val = random.choices([0,1],weights=(prob1,prob2))[0]#random.randint(0,10)/10 # shows last entry random.randint(0,10)/10
    if val > 0.5:
        typer = val2
    else:
        typer = val1
        
    return val, typer

dff = pd.DataFrame()

def monte_carlo_coin_xtra(runs, n, dff):
    for i in range(runs):
        results = 0
        for ii in range(n):
            flip_result, typer = coin_flip()
            results = results + flip_result
            
            prob_value = results/(ii+1)
            
            df1 = pd.DataFrame({'run':i,'event':ii,'result':[prob_value], 'type':typer})
            dff = pd.concat([dff,df1])
            
    dff = dff.reset_index()        
    #heads = [i for i, x in enumerate(list2) if x == "heads"] 
    #tails = [i for i, x in enumerate(list2) if x == "tails"]
    

    #fig,axs = plt.subplots(2)
    for runr in dff['run']:
        #plt.plot(dff['event'][dff['run']==runr], dff['result'][dff['run']==runr], c=np.random.rand(3,))
        
        x_valer = dff['event'][dff['run']==runr]
        y_valer = dff['result'][dff['run']==runr]
        y_valer1 = [1-x for x in y_valer]
        y_valer2 = [x for x in y_valer]
        
        for ix in range(2):
            if ix == 0:
               plt.figure(1)
               plt.plot(x_valer, y_valer1, c=np.random.rand(3,)) 
               plt.axhline(y=prob1, color='r', linestyle='dashed')
               plt.gca().spines[['right','top']].set_visible(False)
               plt.xlabel("Iterations")
               plt.ylabel("Probability")
               plt.title(f"Total Probability of {val1} outcome from coin flip; n={n}, runs={runs}", size=10)
            else:
                plt.figure(2)
                plt.plot(x_valer, y_valer2, c=np.random.rand(3,))
                plt.axhline(y=prob2, color='r', linestyle='--')
                plt.gca().spines[['right','top']].set_visible(False)
                plt.xlabel("Iterations")
                plt.ylabel("Probability")
                plt.title(f"Total Probability of {val2} outcome from coin flip; n={n}, runs={runs}", size=10)
        
    plt.show()
    

    # plot last event run
    if i == runs-1:
        list1 = dff['result'][dff['run']==i]
        heads = dff['event'][(dff['run']==i) & (dff['type']==val1)]
        tails = dff['event'][(dff['run']==i) & (dff['type']==val2)]
        
        #print("length of heads is:",len(heads))
        #print("length of tails is:",len(tails))
        
        heads_chkr = [1-x for x in list1]
        tails_chkr = [x for x in list1]
        
        grapher(heads_chkr,prob1,tails,heads,val1,n)
        grapher(tails_chkr,prob2,tails,heads,val2,n)
        
    return sum(dff['result']/(runs*n))

def monte_carlo_coin(n):
    results = 0
    for i in range(n):
        flip_result, typer = coin_flip()
        results = results + flip_result
        
        prob_value = results/(i+1)
        
        list1.append(prob_value)
        list2.append(typer)
        
    heads = [i for i, x in enumerate(list2) if x == val1] 
    tails = [i for i, x in enumerate(list2) if x == val2]
    
    heads_chkr = [1-x for x in list1]
    tails_chkr = [x for x in list1]
    
    grapher(heads_chkr,prob1,tails,heads,val1,n)
    grapher(tails_chkr,prob2,tails,heads,val2,n)
        
    return results/n

################## monte carlo coin flip probability multiple runs ##########################

tst = monte_carlo_coin_xtra(n_runs,n_events,dff)

################## monte carlo coin flip probability ##########################


list1 = []
list2 = []

tst1 = monte_carlo_coin(n_events)

heads_chkr = [list1[i] for i, x in enumerate(list2) if x == "heads"]
coin_abs = [1 if x == "heads" else 0 for x in list2]
coin_abs_clr = ["black" if x == "heads" else "green" for x in list2]
tails_chkr = [list1[i] for i, x in enumerate(list2) if x == "tails"]

###############################################################################

print("runtime:",round(time.time()-start_time,3),"seconds")