'''
This file includes a set of functions that step by step attempt to identify and simulate optimal voting strategies using quadratic voting.

The function "~~~" runs the full simulation.

Here's how it was done in general:

1. We need to compare strategies, which is an n-element unique array.  To start, we create a function that for a given max element value, m,
returns all the unique strategies of length n using that number.  m is based on the max number of votes using a given vote credit limit.
m is the square root of the nearest perfect square lower than the vote credit limit.
'''

import pandas as pd
import numpy as np
from numpy import linalg as LA
from itertools import combinations_with_replacement, groupby
from math import floor
import re
import sys

# ----------------------------------------------------------------------------------------------------------------------------------------
# I. CREATE STRATEGY METRICS FOR A GIVEN N AND C
# ----------------------------------------------------------------------------------------------------------------------------------------

# 1. Create metrics for a given N (n) and floor of square root of C (m).  FIRST FILTER = remove all null votes.
def strategies_for_n_given_m(m, N):
    '''
    Creates a Table for voting metrics to identify best strategies.
    Removes all Null Voting strategies.
    
    Parameters:
        m = Top limit - max number of votes
        n = Number of Issues
    '''
        # Create an iterable for all values from 0 to m:
    numbers = list(range(m + 1))
    
    # Generate all unique combinations of length N from the numbers 0 to m
    combs = list(combinations_with_replacement(numbers, N))
    
    # Remove equivalent strategies (e.g. [2,1] == [1,2]).  !-set() looks at both numbers:
    combs = list(set(c for c in combs))
    
    # Sort each list in descending order - descending because the convention is to list them high to low
    combs = sorted(combs, key=lambda x: x[::-1], reverse = True)
    
    # Create return object - a list of dicts for easy df creation
    output = []
    
    # Get the vote credit total needed for each strategy
    for i in combs:
        # FIRST FILTER: Remove all strategies where the elements are equal (equilvalent to Null Vote)
        if i == i[0:1] * N:
            #print(f'Voting strategy {i} == Null Vote.  Removed.')
            continue
            
        # If not a null strategy, build an entry for output
        else:    
            i = sorted(i, reverse = True)
            
            # Create an object to hold the strategy and its total votes and vote_credits needed
            output_dict = {}
            output_dict['strategy'] = i
            output_dict['vote_sum'] = sum(i)
            output_dict['vote_credits'] = sum([p**2 for p in i])
            
            output.append(output_dict) 
    
    return output


# 2. Build the strategy set by creating m from vote credits and filtering out all stategies that do not work with the given vote credit amount.
def build_strategy_set(N, vote_credits):
    '''
    Use vote credits to filter out stragies from a complete N,m strategy set.
        
    Parameters:
        int N = Number of Issues
        int vote_credits = C = Maximum amount of vote credits to get N-length strategies for
        
    Returns:
        obj s_set = Dataframe with all possible strategies along with their vote totals and credit requirements up to C. 
    '''
    # First identify the value of m, which is the highest possible value for an individual Voting Item's total.
    m = floor(vote_credits**.5)
   
    # Now create and filter out all strategies with C greater than given vote_credits
    s_set = pd.DataFrame(strategies_for_n_given_m(m,N))
    s_set = s_set[s_set['vote_credits'] <= vote_credits]
    s_set = s_set.reset_index(drop = True)
    
    return s_set


# ----------------------------------------------------------------------------------------------------------------------------------------
# II. REDUCE STRAETGY SETS (GET RID OF SIMILAR, LOWER MAGNITUDE STRATEGIES)
# ----------------------------------------------------------------------------------------------------------------------------------------

# 1. "Arithmatic" reduction
def arithmatic_reduction(strategy_list):
    '''
    Reduce strategy set by going through strategies and seeing if each can be reduced, and if so appending
    a return list with that rather than the original strategy.
    
    Parameters:
        obj df = Df with strategies
        
    Returns:
        obj new_strat_list = List of strategies after arithatic reduction
    '''
    # Create the holder for the output
    new_strat_list = []
    # Create the holder for strategies to remove
    remove_strats = []

    # Test each strategy to find cases where the elements are 1 vote off, and can therefore be removed.
    for strategy in strategy_list:
        # Break swith breaks out of loop
        mod_strategy = []
        for element in strategy:
            new_element = element - 1
            mod_strategy.append(new_element)
        
        remove_strats.append(mod_strategy)
        
    for strategy in strategy_list:
        if strategy in remove_strats:
            continue
        else:
            new_strat_list.append(strategy)
        
    return new_strat_list


# 2. "Multiplicative" reduction
def remove_similar(strategy_list):
    '''
    This is the multiplicative reduction.  It analyzes each as a vector to see if they have the same unit vector, 
    meaning they are scalar multiples of one another.  It removes the duplicate vector with the lower magnitude.
    
    Parameters:
        obj strategy_list = List of strategies
        
    Returns:
        obj unique_vectors = List of multiplicatively reduced strategies
    '''
    unique_vectors = []

    for v in strategy_list:
        unit_vector = v / LA.norm(v)
        is_unique = True
        for uv in unique_vectors:
            uv_unit_vector = uv / LA.norm(uv)
            if all(unit_vector == uv_unit_vector):
                if LA.norm(uv) < LA.norm(v):
                    unique_vectors.remove(uv)
                    unique_vectors.append(v) 
                    unique_vectors.remove(v)

                is_unique = False
                break
        if is_unique:
            unique_vectors.append(v)
            
    return unique_vectors

 
 # 3. Cutoff Strategy identify
def get_cutoff(strategies):
    '''
    Finds the highest vote total strategy which is inefficient, meaning there is a better voting strategy.
    For example, if a voter has both [5,1,0] and [4,1,0] available to them, the would always rationally go with
    the [5,1,0], as one of the principles to voting is its less about ratios since all are voting items are 
    independant except for the decision itself.  Thus, the case for this is where all but the max vote total
    are the same between two strategies, and the max vote of one is higher.
    
    Parameters:
        obj strategies = List of strategies
        
    Returns:
        str = List in string form of cutoff strategy
    '''
    strategies.reverse()
    for item_a in strategies:
        for item_b in strategies:
            if [a - b for a, b in zip(item_a, item_b)] == [1] + [0] * (len(item_a)-1):
                if item_b[0] != item_b[1]:
                    return str(item_b)


# ----------------------------------------------------------------------------------------------------------------------------------------
# III. GENERATE VOTE DIFFERENCE METRICS
# ----------------------------------------------------------------------------------------------------------------------------------------

def vote_difference_analysis(strategy):
    '''
    Outputs some comparison metrics about the individual voting items within a strategy.
    
    Parameters:
        obj strategy = List of integers
        
    Returns:
        int max_diff = Minimum difference of votes across voting elements
        int min_diff = Maximum difference of votes across voting elements
        int max_min_diff = Difference between max and min vote differences
        float avg_diff = Average of differences of votes across voting elements
    '''
    # Create a list to hold differences between votes for strategy
    res = []
    
    for i in range(len(strategy)):
        for j in range(len(strategy)):
            if i != j:
                res.append(abs(strategy[i] - strategy[j]))
    
    # Get vote differences
    max_diff = max(res)
    min_diff = min(res)
    max_min_diff = max_diff - min_diff
    avg_diff = round(sum(res)/len(res), 4)
    
    return max_diff, min_diff, max_min_diff, avg_diff


# ----------------------------------------------------------------------------------------------------------------------------------------
# IV. DETERMINE KEY METRICS FOR STRATEGY SETS OF DIFFERENT VALUES OF C, GIVEN N
# ----------------------------------------------------------------------------------------------------------------------------------------

def strategies_for_n(N, max_credits, display_details = False):
    '''
    Given a number of max credits, determine all viable strategy options for N.
    Basically: set max_credits, run the function, then pick lowest C that optimizes the voting pattern.
    
    Parameters:
        int N = Number of Issues
        int max_credits = Cmax = Maximum amount of vote credits to get N-length strategies for
        
    Returns:
        obj credit_outcomes = Dataframe with details on strategy sets for each C in N up to Cmax
    '''
    strategies = build_strategy_set(N, max_credits).sort_values(by = 'vote_credits')
    
    credit_outcomes = []
    
    for i in range(1, max_credits + 1):
        
        if display_details == True:
            print('Voice Credits:',i)
        
        # Vote max for the given i
        vmax = floor(i**.5)
        
        # Create list object to hold dicts ceated from strategy details
        result = []        
        
        # Extract the strategies that work with this number of votes
        strats_in_i = strategies[strategies['vote_credits'] <= i]
        strat_list = list(strats_in_i.strategy)
    
        strat_list = arithmatic_reduction(strat_list)
        
        # You can now remove the similar strategies having removed any ones higher that would skew this output
        strat_list = remove_similar(strat_list)

        strats_in_i = strats_in_i[strats_in_i.strategy.isin(strat_list)]
        
        # Create the metrics for each strategy
        for row in strats_in_i.strategy:
            # Basic info
            output = {}
            output['strategy'] = row
            row_votes = sum(row)
            row_vc = sum([x**2 for x in row])
            output['votes per credit'] = row_votes/row_vc

            output['wasted_credits'] = i - row_vc
            
            # Difference metrics
            max_diff, min_diff, max_min_diff, avg_diff = vote_difference_analysis(row)
                
            output['min_vote_diff'] = min_diff
            output['max_vote_diff'] = max_diff
            output['max_min_diff'] = max_min_diff
            output['avg_diff'] = avg_diff
            
            # Scoring metrics
            output['votes_over_vmax'] = row_votes - vmax
            output['credits_over_vmax'] = row_vc - vmax**2
            
            # Create a weight factor for each strategy 
            # This will weight each strategy's effect on the strategy set of a given C
            # !- vary wf below 
            wf = .5
            output['WF1'] = wf**(output['wasted_credits'])
            result.append(output)
        
        result = pd.DataFrame(result)
        
        # Make a list of strategies available for N
        n_strategies = list(result.strategy)

        # Identify a 'cutoff' strategy = last filter
        cutoff_strategy = get_cutoff(n_strategies)
        
        if display_details == True:
            print('cutoff_strategy',cutoff_strategy)

        # Need to change strategy back into a string
        result = result.astype({'strategy':str})

        # Find cutoff strategy and how many credits it wastes
        wasted_credits_cutoff = result[result['strategy'] == cutoff_strategy]['wasted_credits'].values
        
        if display_details == True:
            print('wasted_credits_cutoff',wasted_credits_cutoff)

        # Remove all strategies which waste more credits than the cutoff (including the cutoff)
        if len(wasted_credits_cutoff) > 0:
            result = result[result['wasted_credits'] < wasted_credits_cutoff[0]]
            
        # Create additional score columns using weight factor
        result['weighted_score'] = (result['votes_over_vmax'] + result['credits_over_vmax']) * result['WF1']
        
        # Weighted Differences
        result['min_vote_diff'] = result['min_vote_diff'] * result['WF1']
        result['max_vote_diff'] = result['max_vote_diff'] * result['WF1']
        result['max_min_diff'] = result['max_min_diff'] * result['WF1']
        result['avg_diff'] = result['avg_diff'] * result['WF1']
         
        # Create Strategy Set metrics for a given C
        sc = len(result) # Count of strategies
        v_c = result['votes per credit'].mean() 
        wc_s = result['wasted_credits'].sum() # Sum of wasted votes
        wc_m = result['wasted_credits'].mean() # Average of wasted credits for all strategies
        min_vd_m = result['min_vote_diff'].mean()
        max_vd_m = result['max_vote_diff'].mean()
        d_vd_m = result['max_min_diff'].mean()
        avg_diff_m = result['avg_diff'].mean()
        score_s = result['weighted_score'].sum()
        score_m = result['weighted_score'].mean()
        wf_s = result['WF1'].sum()
        wf_m = result['WF1'].mean()
        
        credit_outcomes.append({'Credits':i,
                                'Strategies':sc,
                                'Votes per Credit':v_c,
                                'Total Wasted Credits':wc_s,
                                'Avg. Wasted Credits per Strategy':wc_m,
                                'Avg. Min Vote Diff':min_vd_m,
                                'Avg. Max Vote Diff':max_vd_m,
                                'Avg. MinMax Vote Diff':d_vd_m,
                                'Avg. Diff':avg_diff_m,
                                'Avg. Score':score_m,
                                'Sum of Scores':score_s,
                                'Avg. Weight':wf_m,
                                'Sum of Weights':wf_s})
      
        if display_details == True:
            display(result)
                    
    return credit_outcomes


'''
You can run the acquire_shows function from the command line like a pro!
While in the directory in the terminal, type >> python qv_credit_optimization.py strategies_for_n {N} {max_credits} {display_details = True/False}
'''
if __name__ == '__main__':
    strategies_for_n(sys.argv[1], sys.argv[2], sys.argv[3])