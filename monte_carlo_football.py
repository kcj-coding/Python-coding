# monte carlo estimate
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


teams = ["A","B","C","D","E"]
team_probs = [0.1,0.4,0.9,0.7,0.3]

outcomes = ["W","D","L"]
outcome_prob = [0.4,0.2,0.4]
points = [3,1,0]

play_loc = ["H","A"]
play_prob = [0.6,0.4] # more chance of winning or drawing if playing at home

win_chg = [0.1] # winning increases probability of winning again
loss_chg = [-0.1] # losing increases probability of losing again

# teams play all teams twice except themselves

n_runs = 50

goals = [0,1,2,3,4,5,6]
goals_prob = [0.5,0.4,0.3,0.2,0.1,0.03,0.01] # unlikely to score high number of goals

start_time = time.time()

# create matches
matches = pd.DataFrame()
for h in range(len(teams)):
    for a in range(len(teams)):
        if teams[h] != teams[a]:
            match = pd.DataFrame({'home':[teams[h]], 'away':[teams[a]]})
            matches = pd.concat([matches,match])

matches = matches.reset_index()
matches = matches.drop(columns=["index"])

# reorder so not all of home against different away
new_order = random.sample(range(0,len(matches)),len(matches))

matches_new = pd.DataFrame()
for x in new_order:
    xx = matches[matches.index.isin([x])]
    matches_new = pd.concat([matches_new,xx])


res = pd.DataFrame()

for run in range(n_runs):
    team_probs1 = team_probs.copy() # copy so does not update existing values
    team_probs1 = team_probs[:] # copy so does not update existing values
    for i in range(len(matches_new)):
    #for h in range(len(teams)):
    #    for a in range(len(teams)):
    #        if teams[h] != teams[a]:
                #print(teams[h]," v ", teams[a])
                #print("Home: ",teams[h], "Away: ",teams[a])
                
                # get result probability - create custom probabilities from defined starting points
                
        h_team = matches_new['home'][i]
        a_team = matches_new['away'][i]
        
        # find number in teams
        h_loc = [ii for ii, x in enumerate(teams) if x == h_team][0]
        a_loc = [ii for ii, x in enumerate(teams) if x == a_team][0]
        
        h_team_prob = team_probs1[h_loc] * play_prob[0]
        #outcome_prob_h = [x*h_team_prob for x in outcome_prob]
        a_team_prob = team_probs1[a_loc] * play_prob[1]
        
        outcome_prob_h = [h_team_prob,max(h_team_prob*0.2, a_team_prob*0.2),a_team_prob]
        
        result = random.choices(outcomes,weights=outcome_prob_h)[0]
        
        if result == "W":
            a_score = random.choices(goals[0:5],weights=goals_prob[0:5])[0]#random.randint(0,5)
            h_score = min(6,a_score+random.choices(goals[1:6],weights=goals_prob[1:6])[0]) # ensures score is higher than away
            # team_w prob increases
            # team_a prob decreases
            team_probs1[h_loc] = min(0.95,(1+win_chg[0]) * team_probs1[h])
            team_probs1[a_loc] = max(0.01,(1+loss_chg[0]) * team_probs1[a])
        elif result == "L":
            a_score = random.choices(goals[1:6],weights=goals_prob[1:6])[0]#random.randint(0,5)
            h_score = max(0,a_score-random.choices(goals[1:6],weights=goals_prob[1:6])[0]) # ensures score is lower than away
            # team_w prob decreases
            # team_a prob increases
            team_probs1[h_loc] = max(0.01,(1+loss_chg[0]) * team_probs1[h])
            team_probs1[a_loc] = min(0.95,(1+win_chg[0]) * team_probs1[a])
        else:
            a_score = random.choices(goals[0:4],weights=goals_prob[0:4])[0]#random.randint(0,5)
            h_score = a_score
        
        exp_result = ["W" if h_team_prob > a_team_prob else "L"][0]
        
        h_points = [3 if result=="W" else 0 if result=="L" else 1][0]
        a_points = [0 if result=="W" else 3 if result=="L" else 1][0]
        
        #print(result)
               
        # outcome probability
        dff = pd.DataFrame({'run':[run],'home':[teams[h_loc]],'away':[teams[a_loc]],'probs':[outcome_prob_h], 'result':[result], 'score':[str(h_score)+":"+str(a_score)],'goals_h':[h_score],'goals_a':[a_score],'exp_result':[exp_result],
                            'h_prob':[team_probs1[h_loc]], 'a_prob':[team_probs1[a_loc]], 'h_points':[h_points], 'a_points':[a_points]})
        res = pd.concat([res,dff])
        
# at the end, want to track team positions by run
df_teams = pd.DataFrame()

for team in teams:
    dfi = res[((res['home']==team) | (res['away']==team))]
    
    dfi_h = res[res['home']==team]
    dfi_a = res[res['away']==team]
    
    # want to know run, team, and total number of h_points
    dfi_hi = dfi_h.groupby(['run','home'])['h_points'].sum().reset_index()
    dfi_hi = dfi_hi.rename(columns={'run':'run', 'home':'team', 'h_points':'points'})
    
    # want to know run, team and total number of a_points
    dfi_ai = dfi_a.groupby(['run','away'])['a_points'].sum().reset_index()
    dfi_ai = dfi_ai.rename(columns={'run':'run', 'away':'team', 'a_points':'points'})
    
    # add points together and add to df run, team, total points
    df_xx = pd.concat([dfi_hi, dfi_ai])
    df_xx = df_xx.groupby(['run','team'])['points'].sum().reset_index()
    
    # get goals for, goals against and goal difference
    df_goals_for_h = dfi_h.groupby(['run'])['goals_h'].sum().reset_index()
    df_goals_for_h = df_goals_for_h.rename(columns={'run':'run','goals_h':'goals_for'})
    df_goals_for_a = dfi_a.groupby(['run'])['goals_a'].sum().reset_index()
    df_goals_for_a = df_goals_for_a.rename(columns={'run':'run','goals_a':'goals_for'})
    df_goals_for = pd.concat([df_goals_for_h, df_goals_for_a])
    df_goals_for = df_goals_for.groupby(['run'])['goals_for'].sum().reset_index()
    
    # goals against
    df_goals_agn_h = dfi_h.groupby(['run'])['goals_a'].sum().reset_index()
    df_goals_agn_h = df_goals_agn_h.rename(columns={'run':'run','goals_a':'goals_agn'})
    df_goals_agn_a = dfi_a.groupby(['run'])['goals_h'].sum().reset_index()
    df_goals_agn_a = df_goals_agn_a.rename(columns={'run':'run','goals_h':'goals_agn'})
    df_goals_agn = pd.concat([df_goals_agn_h, df_goals_agn_a])
    df_goals_agn = df_goals_agn.groupby(['run'])['goals_agn'].sum().reset_index()
    
    # combine
    df_xx = pd.merge(df_xx, df_goals_for, how="left", left_on="run", right_on="run")
    df_xx = pd.merge(df_xx, df_goals_agn, how="left", left_on="run", right_on="run")
    df_xx['goal_diff'] = df_xx['goals_for'] - df_xx['goals_agn']
    
    df_teams = pd.concat([df_teams,df_xx])

    
# then at end rank all teams in each run based on total points
df_teams = df_teams.sort_values(by=['run', 'points', 'goal_diff'], ascending=[True,False, False])

# get finishing position - include GD for determnination
#df_teams['rank'] = df_teams[["points", "goal_diff"]].apply(tuple,axis=1)\
#             .rank(method='dense',ascending=False).astype(int)
#df_teams['rank'] = df_teams.groupby('run')['points'].rank(ascending=False, method='dense').astype(int)
df_teams['rank'] = 1
df_teams['rank'] = df_teams.groupby(['run'])['rank'].cumsum()

# finally, for each team want to know the proportion percentage of time finishing at each position
# get the final positions for each team and calculate number of times (out of total) at each position

df_teams_f = df_teams.groupby(['team','rank'])['points'].count().reset_index()

# want to make df that has all possible positions, and join for each team
# fill in any missing or NA values with 0

dff = pd.DataFrame({'rank':range(1,len(teams)+1,1)})

df_teams_f = pd.merge(dff,df_teams_f, how="left", left_on="rank", right_on="rank")

df_teams_f['pct'] = df_teams_f['points']/df_teams_f.groupby(['team'])['points'].transform('sum')

# pivot longer for rank
dff_lng = df_teams_f.pivot(index="team", columns="rank", values="pct")
dff_lng = dff_lng.fillna(0)
dff_lng['team'] = dff_lng.index#teams

# drop team for heatmap, as team labels
dff_lng_mp = dff_lng.drop(columns=["team"]) # have just the pct by rank for the heatmap, team is index

# make heatmap
pct_np = dff_lng_mp.to_numpy()

#set size of graph
cmsize=1/2.54

fig, ax = plt.subplots(figsize=(30*cmsize, 15*cmsize))
im = ax.imshow(pct_np, vmin=0, vmax=max(df_teams_f['pct']), cmap="Reds")

# Show all ticks and label them with the respective list entries
ax.set_xticks(range(len(teams)), labels=range(1,len(teams)+1),
              rotation=45, ha="right", rotation_mode="anchor")
ax.set_yticks(range(len(teams)), labels=teams)#range(0,len(teams)))

# Loop over data dimensions and create text annotations.
for i in range(len(teams)):
    for j in range(len(teams)):
        text = ax.text(j, i, pct_np[i, j],
                       ha="center", va="center", color="w")
        
ax.spines[['top','right']].set_visible(False)
ax.set_xlabel("Rank",size=10)
ax.set_ylabel("Team", size=10)
ax.set_title('Title',size=12)

#ax.set_title("Title")
fig.tight_layout()

plt.show()


end_time = time.time()
print("runtime:",end_time-start_time,"seconds")