import json
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import re

TASK = 'gsm8k'
base_dir = f'/home/alfred/Data-Mixing/results/{TASK}/'
folders = ['data', 'model', 'all']
colors = ['tab:blue', 'tab:orange', 'tab:green']

plt.figure()
for folder, color in zip(folders, colors):
    json_files = sorted(glob.glob(f'{base_dir}/{folder}/trial_*.json'))
    all_max_perf = []
    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                max_perf = [entry['max_performance_so_far'] for entry in data]
            elif isinstance(data, dict) and 'max_performance_so_far' in data:
                max_perf = [data['max_performance_so_far']]
            else:
                max_perf = []
            all_max_perf.append(max_perf)
    if not all_max_perf:
        continue
    max_len = max(len(lst) for lst in all_max_perf)
    for lst in all_max_perf:
        if len(lst) < max_len:
            lst.extend([np.nan] * (max_len - len(lst)))
    avg_max_perf = np.nanmean(all_max_perf, axis=0)
    std_max_perf = np.nanstd(all_max_perf, axis=0)

    iterations = np.arange(1, len(avg_max_perf) + 1)
    plt.plot(iterations, avg_max_perf, label=f'{folder}', color=color, alpha=0.7)
    plt.fill_between(iterations, avg_max_perf - std_max_perf, avg_max_perf + std_max_perf, alpha=0.15, color=color)

# --- Read additional trial performance values from text file ---
text_file_path = 'vae_bo_best_arrays.txt'  # update with actual path if needed
with open(text_file_path, 'r') as f:
    lines = f.readlines()

additional_perfs = []
for line in lines:
    if line.strip().startswith('Trial'):
        continue
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
    if numbers:
        values = list(map(float, numbers))
        additional_perfs.append(values)

# Compute average across all trials from text
max_len = max(len(lst) for lst in additional_perfs)
for lst in additional_perfs:
    if len(lst) < max_len:
        lst.extend([np.nan] * (max_len - len(lst)))

avg_perfs_from_txt = np.nanmean(additional_perfs, axis=0)
std_perfs_from_txt = np.nanstd(additional_perfs, axis=0)
iterations = np.arange(1, len(avg_perfs_from_txt) + 1)
plt.plot(iterations, avg_perfs_from_txt, label='vae', color='tab:red', alpha=0.7)
plt.fill_between(iterations, avg_perfs_from_txt - std_perfs_from_txt, avg_perfs_from_txt + std_perfs_from_txt, alpha=0.15, color='tab:red')


plt.xlabel('Iteration')
plt.ylabel('Average max_performance_so_far')
plt.title(f'Average max_performance_so_far vs Iteration for {TASK}')
plt.grid(True)
plt.legend()
os.makedirs('plots', exist_ok=True)
plt.savefig(f'plots/{TASK}_with_vae.png')
plt.close()
