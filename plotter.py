import os
import csv

import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

# %%
log_dir = '/home/antonio/anaconda3/envs/mujoco131-gym/handful-of-trials-mio/log/'  # Directory specified in script, not including date+time
min_num_trials = 50  # Plots up to this many trials

returns = []
for subdir in os.listdir(log_dir):
    data = loadmat(os.path.join(log_dir, subdir, "logs.mat"))
    if data["returns"].shape[1] >= min_num_trials:
        returns.append(data["returns"][0][:min_num_trials])

returns = np.array(returns)
returns = np.maximum.accumulate(returns, axis=-1)
mean = np.mean(returns, axis=0)

# Plot result
plt.figure()
plt.plot(np.arange(1, min_num_trials + 1), mean)
plt.title("Performance")
plt.xlabel("Iteration number")
plt.ylabel("Return")
plt.show()

# %%

log_dir = '/home/antonio/anaconda3/envs/mujoco131-gym/handful-of-trials-mio/log/'

os.listdir(log_dir)
subdir = '2021-08-05--23:23:09'
data = loadmat(os.path.join(log_dir, subdir, "logs.mat"))

returns = np.array(data["returns"])
cum_returns = np.maximum.accumulate(returns, axis=-1)
mean = np.mean(cum_returns, axis=0)

# Plot result
plt.figure()
plt.plot(returns)
plt.title("Performance")
plt.xlabel("Iteration number")
plt.ylabel("Return")
plt.show()

plt.figure()
plt.plot(mean)
plt.title("Performance")
plt.xlabel("Iteration number")
plt.ylabel("Cumulative Return")
plt.show()

