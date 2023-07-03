import pandas as pd
import numpy as np
import math
from itertools import combinations_with_replacement, groupby
import re
import random

# ----------------------------------------------------------------------------------------------------------------------------------------
# MAIN ANALYTICAL FUNCTIONS - EMIT OPTIMAL C GIVEN N
# ----------------------------------------------------------------------------------------------------------------------------------------

def best_C_for_N_polar(N):
    '''
    Give me an integer N representing the number of voting items.

    I return the best C according to the analytical method (N+3)//4 * N for polar.
    '''
    return ((N+3)//4) * N


def best_C_for_N_priority(N):
    '''
    Give me an integer N representing the number of voting items.
    
    I return the best C according to the analytical method (N//2) * N for priority.
    '''
    return (N//2) * N


# ----------------------------------------------------------------------------------------------------------------------------------------
# MAIN COMPUTATIONAL FUNCTION - 'SCORE' C GIVEN N
# ----------------------------------------------------------------------------------------------------------------------------------------

def score(df):
    '''
    Give me a dataframe with the Max, Min, Weighted and Simulated Mean Voting Powers based on an N and a range of C.
    
    I return the dataframe where the maximum proportional distance is scored as a 100, and all other C are
    scored in relation to that 100.
    '''
    df['wtd/sim factor'] = df['VP_wtd']**2 / df['VP_sim_mean']**2
    max_factor = max(df['wtd/sim factor'])
    df['Score'] = round((df['wtd/sim factor'] / max_factor) * 100,2)   
    
    df.drop(columns = ['wtd/sim factor'], inplace = True)
    
    df = df.sort_values(by = 'Score', ascending = False)
    
    return df


# ----------------------------------------------------------------------------------------------------------------------------------------
# SUPPORT FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------------------------

def base_strategy_set_of(N, C):
    '''
    Give me an integer N to define an array length, and an integer C to tell me what the sum of
    the magnitudes of those N-integers MUST be.
    
    I return a list of base strategy sets: N-length arrays whose elements sum to C.  
    These base strategies are all possible combinations of integers, conventionally expressed as magnitudes
    (to account for negative votes) from the largest integer to the smallest or zero.  Duplicates are removed.
    '''
    # Generate all combinations of N integers that equal C, to include zeros.
    dp = [[[] for _ in range(C+1)] for _ in range(N+1)]
    for i in range(C+1):
        dp[1][i] = [[i]]
    for n in range(2, N+1):
        for c in range(1, C+1):
            for i in range(c+1):
                for subgroup in dp[n-1][c-i]:
                    group = [i] + subgroup
                    group.sort(reverse=True)
                    if group not in dp[n][c]:
                        dp[n][c].append(group)
    strategy_set = dp[N][C]

    return strategy_set


def element_frequencies(strategy):
    '''
    Give me a base strategy (array of integers).
    
    I return a list of the frequencies of each of the non-zero integers along with the count 
    of zeros, if any.  The returns are used to determine the number of combinations for the strategy.
    '''
    # Initialize data holders
    strat_dict = {}
    zeros = 0
    
    # Loop through each integer in the base strategy
    for element in strategy:
        # if its a zero, increase zero count holder
        if element == 0:
            zeros += 1
            continue
        # If the element has already been seen, increase its count, otherwise add it to dictionary
        if element in strat_dict.keys():
            strat_dict[element] += 1
        else:
            strat_dict[element] = 1
    
    frequencies = list(strat_dict.values())
    
    return frequencies, zeros


def count_of_all_possible_permutations_of(strategy, p=1):
    '''
    Give me a base strategy (array of integers).  Also give p=1 (default) for polar and p=0 for priority QV.
    
    That base strategy can then be used to find polar permutations (where one or more signs
    are flipped on positive integers in the base strategy) as well as all permutations of other numbers.
    When summed for all base strategies of a given N and C, the total number of possible voting
    options are given.
    
    Thus, I return a count of all NON-NULL polar permutations.
    '''
    # Get the element group details 
    frequencies, zeros = element_frequencies(strategy)
    
    # Calculate the total permutations of the numbers of the strategy
    # N! / (a! * b! * ... * x!...) where a,b,... are pulled from frequency list, and x is the count of zeros
    total_perms = math.factorial(len(strategy))//(math.prod([math.factorial(x) for x in frequencies])*math.factorial(zeros))  
    
    # If polar (p=1) each number can be represented as positive or negative, so count = 2^total number of non-zeros
    # If priority this term zeros out
    polar_strat_count = 2**(sum(frequencies)*p)

    # Multiply the two together
    total_count_of_sets = total_perms * polar_strat_count
    
    # Account for null strategies, if the base has any:
    if len(set(strategy)) == 1:
        # For polar there are two null strategies [n,n,...,n] and [-n,-n,...,-n], for priority just 1
        if p == 1:
            total_count_of_sets -= 2
        else:
            total_count_of_sets -= 1

    return total_count_of_sets


def get_strategy_details_for(N,C,p):
    '''
    Give me an integer N for number of voting items and integer C for voice credits available.  
    Also give me the parameter p for polar (p=1) or priority (p=0)
    
    I return a dataframe listing each strategy and it's voting power, along with the total number of 
    occurances of each strategy in the entire set of possibilities (total number of polar permutations).
    '''
    # Initilize a data holder - list that will eventually become a dataframe
    result = []
    
    # Get the base strategies
    base_strategies = base_strategy_set_of(N,C)
    
    # Loop through each strategy, get some key info on it, and add it to future dataframe
    for strategy in base_strategies:
        voting_power = sum([x**.5 for x in strategy])

        total_strat_count = count_of_all_possible_permutations_of(strategy,p)
            
        row = {'N':N,
               'C':C,
               'strategy':strategy,
               'VP':voting_power,
               'perms':total_strat_count}

        result.append(row)
        
    result = pd.DataFrame(result)
    
    # Now sort by maximum concentration to maximum spread:
    result = result.sort_values(by = 'VP')
    
    return result


def integrate_strategy_density(df):
    '''
    Give me a dataframe with strategy details for a given N and C.
    
    I return the dataframe with each Voting Power-related column weighted based
    on the number of polar permutations of the base strategy compared to the total number
    of all polar permuations of every strategy in (N,C).
    '''
    # Get the total number of strategy options for a given N and C
    total_perms = sum(df['perms'])

    df['VP_wtd'] = df['VP'] * df['perms'] / total_perms
    
    return df


def get_simple_median(df):
    '''
    Give me a dataframe with strategy details for a given N and C.
    
    I return the Voting Power of the base strategy that represents median voting power for that N and C using
    the index of the median strategy going from min to max voting power.
    '''
    middle = int(len(df) / 2) - 1
    VP_s_med = df.VP.iloc[middle]

    return VP_s_med

def get_weighted_median(df):
    '''
    Give me a dataframe with strategy details for a given N and C.
    
    I return the Voting Power of the base strategy that represents median voting power for that N and C using
    the index of the median permutation going from min to max voting power.
    '''
    # Find the median permutation
    median_perm = int(sum(df.perms) / 2)
    
    # Now loop through until you hit the base strategy the median permutation occurs in
    for row in df.index:
        # Get count of perms and subtract it from the running count of half the total permutations
        row_perms = df['perms'].iloc[row]
        median_perm -= row_perms
    
        if median_perm <= 0:
            strategy = df.strategy.iloc[row]
            #print(f'Weighted median for N,C = ({df.N.iloc[row]},{df.C.iloc[row]}) occurs at {strategy}')
            VP_w_med = sum([x**.5 for x in strategy])
            break
    
    return VP_w_med


def C_details(df):
    '''
    Give me a weighted N,C strategy details dataframe.
    
    In return I give you a condensed outcomes row for N and C, aggregating for Vp_wtd.  Vp_wtd = the average voting
    power if every possible strategy is voted on once and only once.
    '''
    VP_s_med = get_simple_median(df)
    VP_w_med = get_weighted_median(df)
    VP_max = df.VP.max()
    
    # Simplify
    df = df.drop(columns = ['VP'])
    
    # Extract N and C
    N = df.N.max()
        
    C_for_N = df.groupby('C').sum()
    
    C_for_N['N'] = N
    
    C_for_N['VP_s_med'] = VP_s_med
    
    C_for_N['VP_w_med'] = VP_w_med
    
    C_for_N['VP_max'] = VP_max
    
    return C_for_N


def C_for_N_sets(N, Cmin, Cmax,p=1):
    '''
    Give me N, the length of the Decision Space/Number of ballot items, as well as the minimum and maximum C which to analyze 
    up to.  Note: Cmin usually equals N for complete analysis.
    
    I return a dataframe with each C analyzed (from C = Cmin to C = Cmax).
    '''
    if p == 1:
        print('Calculating Polar Combinations')
    elif p == 0:
        print('Calculating Priority Combinations')
    else:
        print("Invalid parameter: p needs to be 0 or 1.")
        return
    # Initialize a return dataframe
    df = pd.DataFrame()

    # Start with C == N, loop through to Cmax
    for c in range(Cmin, Cmax + 1):
        # Generate Base Strategies and Info for N and c
        df_c = get_strategy_details_for(N,c,p)
        df_c = integrate_strategy_density(df_c)
        df_c = C_details(df_c)
        
        df = pd.concat([df,df_c])
    
    df['VP_min'] = df.index**.5
    
    # Clean up before return
    df = df[['N','perms','VP_min','VP_s_med','VP_wtd','VP_w_med','VP_max']]
    df = round(df,3)

    return df 


def simulated_vote(N,C,w=2):
    '''
    Give me integers N and C and optionally integer 'w' which is the ratio of positive to negative votes.
    
    I return a simulated voting strategy (array). 
    '''
    # Output holder
    strategy = []

    # For all but last element pick a random number from credits left
    for n in range(N-1):
        credits_used = random.randint(0,C)
        C -= credits_used
        strategy.append(credits_used)
    
    # If there are any credits left use them up, otherwise append 0
    if C > 0:
        strategy.append(C)
    else:
        strategy.append(0)

    # Now randomly sort the strategy
    np.random.shuffle(strategy)
        
    # Now randomly negate a certain percentage of the strats
    negatizer = random.choices([-1, 1], weights=[1, w], k=N)
    
    # Apply negatizer
    strategy = np.multiply(strategy, negatizer)
    
    return strategy  


def simulated_voting_power(N, Cmin, Cmax, voters, w = 2):
    '''
    Give me integers N, the minimum and maximum C to analyze, the number of voters to simulate and 
    optionally the integer 'w', which is the ratio of positive to negative votes.

    I return a dictionary of the mean voting power based on the simulated votes.
    '''
    # Create a holder for the simulated strategies for each C
    ballots = {}
    for C in range(Cmin,Cmax+1):
        votes = []
        for _ in range(voters):
            # Simulate a vote
            p = simulated_vote(N,C,w=w)
            
            # Reset to a base strategy
            p = [abs(x) for x in p]
            p.sort(reverse = True)
            
            #Now add this representative base strategy
            votes.append(p)
            
        ballots[C] = votes
        
    VP_per_C = {}

    # Convert the strategies into voting power and return as a dictionary
    for i in ballots.keys():
        v = ballots[i]
        VP = [sum(x**.5 for x in inner_lst) for inner_lst in v]
        mean_VP = sum(VP)/voters
        VP_per_C[i] = round(mean_VP ,5)
        
    return VP_per_C


def add_simulations(df, VP_per_C):
    '''
    Give me a dataframe of (N,C) results as well as the simulated voting power dictionary.

    I return the df with the simulated voting power.
    '''
    df['VP_sim_mean'] = df.index.map(VP_per_C)
    
    print(f'N = {df.N.max()}')

    # Reorder from smallest VP to largest
    df = df[['perms','VP_min', 'VP_sim_mean','VP_s_med', 'VP_wtd', 'VP_w_med', 'VP_max']]
    
    return df


# ----------------------------------------------------------------------------------------------------------------------------------------
# ALTERNATE APPROACH (GINI) FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------------------------

def gini(x):
    '''
    Give me a distribution.
    
    I return the GINI coefficient.
    '''
    x = np.array(x)
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))


def gini_for_C_N_sim(C,N,voters = 50000):
    '''
    Give me integers C, N and the number of voters to simulate.
    
    I return the GINI coefficient of the simulated voting for that C and N.
    '''
    # Initialize the list that will hold all Voting Powers for GINI analysis
    VP_list = []
    for _ in range(voters):
        # Simulate a vote
        p = simulated_vote(N,C)
        # Calculate Vp
        p = sum([abs(x)**.5 for x in p]) 
        
        VP_list.append(p)
    # Get and return the GINI for a given N and C
    Gf = gini(VP_list)
    
    return Gf


def gini_for_C_N_wtd(C,N):
    '''
    Give me integers C and N.
    
    I return the GINI coefficient of the complete set of unique voting power outcomes for that C and N.
    '''
    # Initialize the list that will hold all Voting Powers for GINI analysis
    VP_list = []
    # Get the voting power and the count each occurs
    df = get_strategy_details_for(N,C)
    # For each row, get the voting power and then add it to the Vp_list 
    # accoridng to the count of permutations for that base strategy
    for i in df.index:
        j = [df.iloc[i].VP] * df.iloc[i].perms
        VP_list.extend(j)
    # Get and return the GINI for a given N and C
    Gf = gini(VP_list)
    
    return Gf