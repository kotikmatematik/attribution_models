#!/usr/bin/env python
# coding: utf-8

# In[25]:


#common
import itertools
import math
import pandas as pd
import random
from collections import defaultdict
import numpy as np


# ## Collecting and preprocessing data

# In[3]:


# simulated data
channels = ['cpc_google', 'cpc_yandex', 'cpa', 'criteo', 'email', 'social']
chain_lst = []
for i in range(20000):
    chain_len_lst = list(range(1,23))
    length = random.choices(chain_len_lst, [30,15,12,8,7,5,5,3,2,1,1,1,1,1,1,1,1,1,1,1,1,1], k=1)[0]
    chain = random.choices(channels, [12,8,30,3,24,23], k=random.randint(1,len(channels)))
    conversion = random.choices([0,1], [85, 15], k=1)[0]
    chain_lst.append(['->'.join(chain), conversion])
    
chain_df = pd.DataFrame(chain_lst, columns=('chain', 'conversion'))
chain_df['sorted_uniq_chain'] = chain_df['chain'].apply(lambda x:'->'.join(sorted(set(x.split('->')))))
chain_df.head(2)


# ### For Shapley vector

# In[8]:


# get dummies to build shapley model
shapley_chain_df  = pd.concat([chain_df.groupby('sorted_uniq_chain').agg({'conversion': 'sum'})                    .rename({'conversion':'conversions'}, axis=1),
          chain_df.groupby('sorted_uniq_chain').agg({'conversion': 'count'})\
                    .rename({'conversion':'visits'}, axis=1)], 
          axis=1).reset_index()

for channel in channels:
    shapley_chain_df[channel] = shapley_chain_df.sorted_uniq_chain.apply(lambda x: 1 if channel in x.split('->') else 0)

shapley_chain_df['conversion_rate'] = shapley_chain_df['conversions']/shapley_chain_df['visits']

shapley_chain_df.head(2)


# ### For Markov chains

# In[10]:


markov_data_df = chain_df[['chain', 'conversion']]
markov_data_df['user_id'] = list(markov_data_df.index+1)
markov_data_df['result'] = np.where(markov_data_df['conversion']==1, 'purchase', 'unsuccessful')
markov_data_df['total_chain'] = markov_data_df.apply(lambda x:['start']+x.chain.split('->')+[x.result], axis=1)

markov_data_lst = []
for i, ch in zip(markov_data_df['user_id'].values.tolist(), markov_data_df['total_chain'].values.tolist()):
    for j in ch:
        markov_data_lst.append([i, j])
        
markov_data_df = pd.DataFrame(markov_data_lst, columns=['user_id', 'step'])


# In[11]:


markov_data_df.head(2)


# ## Calculating values

# ### Shapley

# In[16]:


# function returns all the possible subsets of a set of channels.
# input: set of channels

def subsets(s):
    if len(s)==1:
        return s
    else:
        sub_channels=[]
        for i in range(1,len(s)+1):
            sub_channels.extend(map(list,itertools.combinations(s, i)))
    return map("->".join,map(sorted,sub_channels))

# function computes the value of each coalition.
# coalition: coalition of channels
# coal_value_dict: A dictionnary containing the number of conversions
# that each subset of channels has yielded.

def v_function(coalition, coal_value_dict):
    
    subsets_of_coal = list(subsets(coalition.split("->")))
    worth_of_coal = 0
    for sub in subsets_of_coal:
        if sub in coal_value_dict:
            worth_of_coal += coal_value_dict[sub]
    return worth_of_coal

# function calculates values for each channel
# inputs: df with two columns: unique sorted chains and orders,
# chain_col - column with chains, objective - column with orders or profit,
# channels - list of existed channels

def calculate_shapley_values(df, chain_col, objective, channels):

    # First, let's convert the dataframe "subsets_conversions" into a dictionnary
    coal_value_dict = df.set_index(chain_col).to_dict()[objective]

    #For each possible combination of channels coalition, we compute the total number of conversions yielded by every subset of coalition. 
    # Example : if coalition = {c1,c2}, then v(coalition) = C({c1}) + C({c2}) + C({c1,c2})
    v_values = {}
    for coalition in subsets(channels):
        v_values[coalition] = v_function(coalition,coal_value_dict)
    
    n=len(channels)
    shapley_values = defaultdict(int)

    for channel in channels:
        for coalition in v_values.keys():
            if channel not in coalition.split("->"):
                cardinal_coalition=len(coalition.split("->"))
                coalition_with_channel = coalition.split("->")
                coalition_with_channel.append(channel)            
                coalition_with_channel="->".join(sorted(coalition_with_channel))
                shapley_values[channel] += (v_values[coalition_with_channel]-v_values[coalition])*(math.factorial(cardinal_coalition)                                                            *math.factorial(n-cardinal_coalition-1)/math.factorial(n))
        # Add the term corresponding to the empty set
        shapley_values[channel]+= v_values[channel]/n 
        
    shapley_values_df = pd.DataFrame([[[k][0][0],[k][0][1]]  for k in shapley_values.items()],
                                     columns=('channel', 'values'))
    shapley_values_df['share'] = shapley_values_df['values']/shapley_values_df['values'].sum()
    return shapley_values_df


# In[17]:


# objective - orders
shap_alt_df_order = calculate_shapley_values(shapley_chain_df[['sorted_uniq_chain', 'conversions']],
                                       'sorted_uniq_chain', 'conversions', channels)
shap_alt_df_order


# ### Markov chains

# In[19]:


def get_data(data, touchpoints, start, conversion, nonconversion, user_ids):

    # Sort data and reindex
    data = data.reset_index()

    # Define conversion
    data['conversions'] = 0 
    data.loc[data[touchpoints]==conversion, 'conversions'] = 1

    # Count conversions
    data['conversion_count'] = data.groupby('conversions').cumcount()+1
    data.loc[data['conversions']!=True, 'conversion_count'] = np.nan
    data['conversion_count'] = data['conversion_count'].fillna(method='bfill')
    data['conversion_count'] = data['conversion_count'].fillna(data['conversion_count'].max()+1)

    # Split into conversion journeys
    data['journey_id'] = list(zip(data[user_ids], data['conversion_count']))

    # Initialize dict for temporary transition matrices and removal effects
    return data

def attribute(data, touchpoints, start, conversion, nonconversion, user_ids):
    journeys = data.copy()
    journeys['next_'+touchpoints] = journeys[touchpoints].shift(-1)
    journeys = journeys[~journeys[touchpoints].isin([nonconversion,conversion])].dropna()
    
    # Get transition probabilities
    states = journeys.pivot_table(index=[touchpoints],
                                                    values='journey_id',
                                                    aggfunc=len)
    transitions = journeys.pivot_table(index=[touchpoints, 'next_'+touchpoints],
                                                         values='journey_id',
                                                         aggfunc=len)
    transitions = transitions.reset_index()
    transitions = transitions.join(states, on=touchpoints, rsuffix='_total')
    transitions['probability'] = transitions['journey_id']/transitions['journey_id'+'_total']
    transitions = transitions.sort_values('probability')

    # Get transition matrix
    trans_matrix = transitions.pivot_table(index=touchpoints, 
                                                             columns='next_'+touchpoints, 
                                                             values='probability',
                                                             aggfunc=np.mean,
                                                             fill_value=0)
    # Add missing columns
    for index, row in trans_matrix.iterrows():
        if index not in trans_matrix.columns:
            trans_matrix[index] = 0

    # Add missing rows
    for col in trans_matrix.columns:
        if col not in trans_matrix.index.values:
            new_row = pd.Series()
            new_row.name = col
            trans_matrix = trans_matrix.append(new_row)


    # Fill in NAs with zero probabilities
    trans_matrix = trans_matrix.fillna(0)

    # Reorder columns to solve as linear equations
    trans_matrix = trans_matrix[trans_matrix.index.values]

    # Make sure probabilities sum to 1 (required for next step)
    for index, row in trans_matrix[trans_matrix.sum(axis=1)<1].iterrows():
        trans_matrix.loc[index, index] = 1
    # Set constant term to zero (on RHS)
    RHS = np.zeros(trans_matrix.shape[0])  

    # Set conversion probability at conversion to 1
    RHS[trans_matrix.index.get_loc(conversion)] = 1

    # Make equations' RHS equal the long-run transition probability of that variable to the conversion then subtract from both sides
    for index, row in trans_matrix.iterrows():
        if (index != conversion) & (index != nonconversion):
            trans_matrix.loc[index, index] -= 1
    # Solve system of equations
    x = np.linalg.solve(trans_matrix, RHS)
    return (trans_matrix, x, RHS)

def attribute_removal(remove, x, trans_matrix, RHS, nonconversion, start):
    temp_trans_matrix = {}
    temp_x = {}
    # Copy transition probability table if it exists or create it if it doesn't 
    temp_trans_matrix[remove] = trans_matrix.copy()
    # Set removed touchpoint probabilities to zero except for unsuccessful
    temp_trans_matrix[remove].loc[remove] = 0
    temp_trans_matrix[remove].loc[remove, nonconversion] = 1
        
    # Make equations' RHS for the removed touchpoint equal the long-run transition probability of that variable to the conversion then subtract from both sides
    temp_trans_matrix[remove].loc[remove, remove] -= 1
    
    # Solve system of equations
    temp_x[remove] = np.linalg.solve(temp_trans_matrix[remove], RHS)
    
    # Get conversion probability at start
    conv_prob = x[trans_matrix.index.get_loc(start)]
    conv_prob_remove = temp_x[remove][temp_trans_matrix[remove].index.get_loc(start)]
    removal_rate = 1 - conv_prob_remove/conv_prob
    return removal_rate  

def calculate_markov_chain(data, touchpoints, start, conversion, nonconversion, user_ids, channels):
    data_new = get_data(data, touchpoints, start, conversion, nonconversion, user_ids)
    trans_matrix, x, RHS = attribute(data_new, touchpoints, start, conversion, nonconversion, user_ids)
    removal_lst = []
    for channel in channels:
        r_rate = attribute_removal(channel, x, trans_matrix,RHS, nonconversion, start)
        removal_lst.append([channel,r_rate])
    
    markov_df = pd.DataFrame(removal_lst, columns=('channel', 'removal_rate'))
    if markov_df.removal_rate.min()<0:
        markov_df['share'] = (markov_df['removal_rate']+1)/(markov_df['removal_rate']+1).sum()
    else:
        markov_df['share'] = markov_df['removal_rate']/markov_df['removal_rate'].sum()
    
    return markov_df


# In[20]:


calculate_markov_chain(data=markov_data_df,
                      touchpoints='step',
                      start='start',
                      conversion='purchase',
                      nonconversion='unsuccessful',
                      user_ids='user_id', channels=channels)


# In[23]:


markov_chain_order = calculate_markov_chain(data=markov_data_df,
                      touchpoints='step',
                      start='start',
                      conversion='purchase',
                      nonconversion='unsuccessful',
                      user_ids='user_id', channels=channels)


# In[24]:


total_conversions = chain_df.conversion.sum()
markov_chain_order['conversions'] = markov_chain_order['share']*total_conversions
markov_chain_order


# ### Last click

# In[17]:


chain_df['last_channel'] = chain_df['chain'].apply(lambda x:x.split('->')[-1])
last_click_df = chain_df.groupby('last_channel').conversion.sum().reset_index().rename({'conversion':'last_click',
                                                                                       'last_channel':'channel'}, axis=1)


# ## Save to csv

# In[18]:


all_models = pd.concat([last_click_df.set_index('channel'), 
           shap_alt_df_order[['channel', 'values']].rename({'values':'shapley_value'}, axis=1).set_index('channel'),
          markov_chain_order[['channel', 'conversions']].rename({'conversions':'markov_chain'}, axis=1).set_index('channel')], axis=1)
all_models['shapley_diff_abs'] = all_models.shapley_value-all_models.last_click
all_models['markov_diff_abs'] = all_models.markov_chain-all_models.last_click
all_models['shapley_diff_share'] = all_models.shapley_value/all_models.last_click-1
all_models['markov_diff_share'] = all_models.markov_chain/all_models.last_click-1


# In[19]:


all_models.to_csv('attribution_models_1.csv', decimal=',', sep=';')

