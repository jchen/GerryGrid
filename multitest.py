import pickle
import numpy as np
import pandas as pd
from numpy import random
import math
import time
import statistics
from bitarray import bitarray
from tqdm import tqdm, trange, tnrange, tqdm_notebook

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('paper')
sns.set(style="whitegrid")
from functools import partial
import pandas

clen = 5
rlen = 5
num_blocks = clen * rlen

mcmc_steps = 1

def n_bitarray(n):
    """
    Creates a bit array of length num_blocks with n 'true' values.
    """
    ar = bitarray()
    for i in range(n):
        ar.append(True)
    for i in range(num_blocks - n):
        ar.append(False)
    return ar

def sq_array(ar):
    """
    Converts a bitarray into
    a rectangular array
    """
    sq_ar = [bitarray() for i in range(rlen)]
    for i in range(rlen):
        sq_ar[i] = ar[(clen * i):(clen * (i + 1))]
    return sq_ar

def get_bitarray(delta):
    """
    Converts a sq/rectangular array
    into a bitarray
    """
    output = bitarray()
    for row in delta:
        for entry in row:
            output.append(entry)
    return output

def print_ar(ar):
    """
    Prints a rectangular array
    """
    print("Grid: ")
    for row in ar:
        rowtext = ""
        for box in row:
            rowtext += "X" if box else "O"
            rowtext += " "
        print(rowtext)
    print()

def to_N(delta):
    """
    Stores a rectangular array as some index
    (written in base two is the original bitarray)
    """
    i = 0
    for r in reversed(range(rlen)):
        for c in reversed(range(clen)):
            i = (i << 1) | delta[r][c]
    return i

def bitarray_to_N(bar):
    """
    The same as above but for a bitarray and not
    a rectangular array
    """
    i = 0
    for dig in bar:
        i = (i << 1) | dig
    return i

def random_step(n):
    """
    Gives a random delta with NumH = n
    """
    a = n_bitarray(n)
    random.shuffle(a)
    return sq_array(a)

def random_sample(times, n):
    """
    Evaluates a times number of random_steps
    (random deltas) and returns a dictionary
    formatted {score: N(index)}
    """
    samp = {}
    for i in tnrange(times, desc = 'Random Sample', leave = False, disable = disable_tqdm):
        step = random_step(n)
        samp[evaluate(step, mcmc_steps)] = to_N(step)
    return samp

def mult_random_sample(times, k_max, n):
    """
    Does mult_random_sample times times and
    gives a list of all the maximums of the
    random_sample (since that is what we're
    interested in)
    """
    maxs = []
    for i in tnrange(times, desc = 'Multiple Random Sample', disable = disable_tqdm):
        run = random_sample(k_max, n)
        maxs.append((max(run), run[max(run)]))
    return maxs

# Threshold for Cellular Automata:
threshold = 0.6

def unhappy(delta):
    """
    returns a tuple of
    (list of coords of unhappy tiles,
    list of values of unhappy tiles,
    Clus, ClusH)
    """
    unhappy_tiles = []
    vals = bitarray()
    total_con = [0, 0]
    con = [0, 0]
    for row in range(rlen):
        for col in range(clen):
            box = delta[row][col]
            total_box = 0
            same_box = 0
            for dx, dy in [(1, 0), (-1, 0), (0, -1), (0, 1)]:
                r = row
                c = col
                nr = r + dx
                nc = c + dy
                if 0 <= nr < rlen and 0 <= nc < clen:
                    total_con[delta[r][c]] += 1
                    total_box += 1
                    samity = 0 if box ^ delta[nr][nc] else 1
                    same_box += samity
                    con[delta[r][c]] += samity
            if same_box / total_box < threshold:
                unhappy_tiles.append((r, c))
                vals.append(delta[r][c])
    return {'coords': unhappy_tiles,
            'vals': vals,
            'Clus': (con[0] + con[1]) / (total_con[0] + total_con[1]),
            'ClusH': con[1] / total_con[1]}

def step(unhappy_coords_shuffled, unhappy_list, delta):
    """
    Makes a step with delta and given values of delta
    and returns a new delta
    """
    idx = 0
    ndelta = delta
    for r, c in unhappy_coords_shuffled:
        ndelta[r][c] = unhappy_list[idx]
        idx += 1
    return ndelta

def greedy_step(delta):
    """
    Runs one evolutionary step, gets rid of the
    other parameters in step
    """
    unh = unhappy(delta)
    random.shuffle(unh['coords'])
    return step(unh['coords'], unh['vals'], delta)

def greedy_seq(n, mcmc_steps):
    """
    Runs a greedy sequence multiple times, until
    no more meaningful evolutions can be done.
    """
    dt = {}
    seed = n_bitarray(n)
    random.shuffle(seed)
    delta = sq_array(seed)
    dt[evaluate(delta, mcmc_steps)] = to_N(delta)
    laststep = to_N(delta)
    k = 0
    while True:
        delta = greedy_step(delta)
        k += 1
        delta_idx = to_N(delta)
        score = evaluate(delta, mcmc_steps)
        if delta_idx == laststep:
            break
        dt[score] = delta_idx
        laststep = delta_idx
    return (dt, k)

def shotgun_greedy(k_max, n, mcmc_steps):
    """
    Runs a greedy sequence multiple times, until
    it has been evaluated k_max times.
    Returns the full dictionary of
    {score: N(index)}
    """
    sample = {}
    times_run = 0
    pbar = tqdm_notebook(total=k_max, desc = 'Shotgun Greedy', leave = False, disable = disable_tqdm)
    while times_run < k_max:
        run = greedy_seq(n, mcmc_steps)
        sample.update(run[0])
        times_run += run[1]
        pbar.update(run[1])
    pbar.close()
    return sample

def mult_shotgun_greedy(times, k_max, n, mcmc_steps):
    """
    Runs shotgun_greedy multiple times,
    and returns a list of
    (max, N that gives the max)
    """
    maxs = []
    for i in tnrange(times, desc = 'Multiple Shotgun Greedy', disable = disable_tqdm):
        run = shotgun_greedy(k_max, n, mcmc_steps)
        maxs.append((max(run), run[max(run)]))
    return maxs

def prob_accept(change_e, temp):
    try:
        return 1 / (1 + math.exp(-change_e / temp))
    except OverflowError:
        return 0

random_accept_thresh = 0.4
temp_initial = 8
temp_ratio = 0.96
heating_ratio = 1.1

# Simulated Annealing algorithm with greedy chance
def simulated_anneal(k_max, n):
    """
    Runs a simulated annealing step
    as described in the paper/documentation
    """
    delta = random_step(n)
    samp = {}
    temp = temp_initial
    eval_now = evaluate(delta, mcmc_steps)
    ra_thresh = random_accept_thresh
    k = 0
    pbar = tqdm_notebook(total=k_max, desc = 'Simulated Anneal', leave = False, disable = disable_tqdm)
    while k < k_max:
        samp[eval_now] = to_N(delta)
        delta_n = greedy_step(delta)
        eval_new = evaluate(delta_n, mcmc_steps)
        k += 1
        pbar.update(1)
        change_e = eval_new - eval_now
        # we can also give the change as a proportion, which makes hyperparameter optimization
        # better for generalization:
        prop_change_e = change_e / clen
        if change_e > 0 or prob_accept(prop_change_e, temp) > random.uniform(0, 1):
#             print('Accept')
            delta = delta_n
            eval_now = eval_new
            temp = temp * temp_ratio
        else:
            delta_rand_n = random_step(n)
            eval_rand_new = evaluate(delta_rand_n, mcmc_steps)
            k += 1
            pbar.update(1)
            change_rand_e = eval_rand_new - eval_now
            prop_random_change_e = change_rand_e / clen
            if change_e > 0 or prob_accept(prop_random_change_e, temp) > random.uniform(0, 1):
#                 print('Accept')
                delta = delta_rand_n
                eval_now = eval_rand_new
                temp = temp * temp_ratio
            else:
                temp = temp * heating_ratio
    pbar.close()
    return samp

def random_swap_step(delta, n):
    delta_ba = get_bitarray(delta)
    delta_n = delta_ba
    change_ar = n_bitarray(n)
    random.shuffle(change_ar)
    change_vals = bitarray()
    idx = 0
    for i in change_ar:
        if i:
            change_vals.append(delta_ba[idx])
        idx += 1
    random.shuffle(change_vals)
    in_idx = 0
    idx = 0
    for i in change_ar:
        if i:
            delta_n[in_idx] = change_vals[idx]
            idx += 1
        in_idx += 1
    return sq_array(delta_n)

step_swap_size = 4

# After testing for various hyperparameters, this is what we found to be the best experimentally:
temp_initial_random = 1
temp_ratio_random = 0.8

# Simulated Annealing algorithm with random neighbour step
def simulated_anneal_random(k_max, n):
    """
    Runs a simulated annealing step
    as described in the paper/documentation
    """
    delta = random_step(n)
    samp = {}
    temp = temp_initial_random
    eval_now = evaluate(delta, mcmc_steps)
    ra_thresh = random_accept_thresh
    k = 0
    pbar = tqdm_notebook(total=k_max, desc = 'Simulated Anneal', leave = False, disable = disable_tqdm)
    while k < k_max:
        samp[eval_now] = to_N(delta)
        delta_n = random_swap_step(delta, step_swap_size)
        eval_new = evaluate(delta_n, mcmc_steps)
        k += 1
        pbar.update(1)
        change_e = eval_new - eval_now
        # we can also give the change as a proportion, which makes hyperparameter optimization
        # better for generalization (grid size does not affect what hyperparams do):
        prop_change_e = change_e / clen
        if change_e > 0 or prob_accept(prop_change_e, temp) > random.uniform(0, 1):
#             Testing:
#             print(eval_new)
#             print('Accept')
            delta = delta_n
            eval_now = eval_new
            temp = temp * temp_ratio_random
    pbar.close()
    return samp

single_run = True

def mult_simulated_anneal(times, k_max, n):
    """
    Runs simulated annealing multiple times
    """
    maxs = []
    for i in tnrange(times, desc = 'Multiple S.A.', leave = single_run, disable = disable_tqdm):
        run = simulated_anneal(k_max, n)
        maxs.append((max(run), run[max(run)]))
    return maxs

def mult_simulated_anneal_random(times, k_max, n):
    """
    Runs random simulated annealing multiple times
    """
    maxs = []
    for i in tnrange(times, desc = 'Multiple S.A. Random', leave = single_run, disable = disable_tqdm):
        run = simulated_anneal_random(k_max, n)
        maxs.append((max(run), run[max(run)]))
    return maxs

df = pickle.load(open("df.p", "rb"))

def evaluate(delta, times):
    """
    Replacement for the evaulative function, which
    relies on known exhaustive searches.
    """
    output = df.loc[to_N(delta)]['Avg']
#     print(output)
    return output

def average_runs(samp):
    """
    Finds the average of a lot of runs.
    """
    sum = 0
    for run in samp:
        sum += run[0]
    return sum / len(samp)

# Disables tqdm prompt (progress bar, we only want one main one)
disable_tqdm = True
# Sample size, runs each algorithm for each k_max samp_size times
samp_size = 2000
# NumH to optimize for
numH = 8

# Range of k's to test
k_max_range = [1]

# Inputs k_max_range

total_k_max = 0

for i in range(1,30):
    k_max_val = i * 4
    k_max_range.append(k_max_val)
    total_k_max += k_max_val

for i in range(10):
    k_max_val = 120 + i * 8
    k_max_range.append(k_max_val)
    total_k_max += k_max_val

for i in range(20):
    k_max_val = 200 + i * 10
    k_max_range.append(k_max_val)
    total_k_max += k_max_val

for i in range(10):
    k_max_val = 400 + i * 20
    k_max_range.append(k_max_val)
    total_k_max += k_max_val

for i in range(10):
    k_max_val = 600 + i * 25
    k_max_range.append(k_max_val)
    total_k_max += k_max_val

print(k_max_range)
print(total_k_max)

# Datatables for various algorithms
rand_dt = {}
sg_dt = {}
sa_dt = {}
sa_random_dt = {}

# Calculates how many steps need to be run (for timing)
total_tqdm_steps = total_k_max
pbar = tqdm(total = total_tqdm_steps, desc = 'Run')

for k_max in k_max_range:
    mult_rand_sample = mult_random_sample(samp_size, k_max, numH)
    rand_dt[k_max] = mult_rand_sample
    mult_sg_sample = mult_shotgun_greedy(samp_size, k_max, numH, mcmc_steps)
    sg_dt[k_max] = mult_sg_sample
    mult_sa = mult_simulated_anneal(samp_size, k_max, numH)
    sa_dt[k_max] = mult_sa
    mult_sa_random = mult_simulated_anneal_random(samp_size, k_max, numH)
    sa_random_dt[k_max] = mult_sa_random
    pbar.update(k_max)
pbar.close()

def pivot(dt):
    table = []
    for k in dt:
        for (value, max) in dt[k]:
            table.append([k, value])
    return table

rand_df = pd.DataFrame(pivot(rand_dt), columns = ['k_max', 'runmax'])
sg_df = pd.DataFrame(pivot(sg_dt), columns = ['k_max', 'runmax'])
sa_df = pd.DataFrame(pivot(sa_dt), columns = ['k_max', 'runmax'])
sa_random_df = pd.DataFrame(pivot(sa_random_dt), columns = ['k_max', 'runmax'])

rand_df['Method'] = 'Random'
sg_df['Method'] = 'RRILS'
sa_df['Method'] = 'S.A.'
sa_random_df['Method'] = 'S.A. (Random)'

full_df = pd.concat([rand_df, sg_df, sa_df, sa_random_df])

full_df_with_max = full_df.append([{'k_max': 0, 'runmax': 1.751373, 'Method': 'Maximum'}, {'k_max': 900, 'runmax': 1.751373, 'Method': 'Maximum'}], ignore_index = True)

full_df_with_max.to_csv('full_df_with_max.csv')

fig = plt.figure(figsize=(8.5, 6.5), dpi=300)
ax = sns.lineplot(x="k_max", y="runmax", style="Method", hue="Method", ci=100, dashes = {'Maximum': (1, 2), 'Random': '', 'S.A.': '', 'S.A. (Random)': '', 'RRILS': ''}, data = full_df_with_max)
ax.set(xlabel='$k_\mathrm{max}$', ylabel='Outcome')
ax.set_title('$k_\mathrm{max}$ versus Outcome for Varying Algorithms, $\mathsf{NumH} = 8$')

plt.savefig("algorithm-comparison-" + str(samp_size) + ".pdf")
plt.show()
