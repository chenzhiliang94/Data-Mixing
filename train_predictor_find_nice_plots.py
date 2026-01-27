import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import argparse 
from typing import List, Tuple, Dict, Any
from scipy.ndimage import gaussian_filter1d

# --- Fixed Configuration ---
TRAIN_SIZE = 20
EPOCHS = 500
BATCH_SIZE = 8
LEARNING_RATE = 0.01
RANDOM_STATE = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_PATH = "trained_predictor_find_nice_plots"

# Hardcoded requirements based on prompt
REQUIRED_HISTORY_STEPS = [25, 50, 75, 100]
TARGET_PREDICTION_STEPS = [625]

# --- Classes & Functions ---

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DataProcessor:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.raw_data = self._load_json() 
        
    def _load_json(self) -> List[Dict[str, Any]]:
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            print(f"Error: File not found at {self.filepath}")
            return []

    def get_dataset_for_metric(self, raw_data_subset: List[Dict[str, Any]], metric_name: str, static_dim: int):
        X_data = []
        y_data = []
        trajectories = []
        valid_indices = [] 

        for i, experiment in enumerate(raw_data_subset):
            input_x = experiment.get('input_X', [])
            if len(input_x) != static_dim:
                continue

            evals_list = experiment.get('evaluations', [])
            step_map = {item['step']: item for item in evals_list if metric_name in item}

            # 1. Check history
            history_features = []
            found_all_history = True
            for req_step in REQUIRED_HISTORY_STEPS:
                if req_step in step_map:
                    history_features.append(float(req_step))
                    history_features.append(float(step_map[req_step][metric_name]))
                else:
                    found_all_history = False
                    break
            
            if not found_all_history:
                continue

            # 2. Check target
            target_step = TARGET_PREDICTION_STEPS[0]
            if target_step in step_map:
                valid_target_point = step_map[target_step]
                
                features = list(input_x)
                features.extend(history_features)
                features.append(float(target_step)) 
                
                X_data.append(features)
                y_data.append(valid_target_point[metric_name])
                valid_indices.append(i)

                traj = []
                sorted_steps = sorted([k for k in step_map.keys()])
                for s in sorted_steps:
                    traj.append((s, step_map[s][metric_name]))
                trajectories.append(traj)

        return np.array(X_data), np.array(y_data), trajectories, valid_indices

class MetricPredictorMLP(nn.Module):
    def __init__(self, input_dim: int):
        super(MetricPredictorMLP, self).__init__()
        layers = [
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        ]
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ModelTrainer:
    def __init__(self, model: nn.Module, learning_rate: float):
        self.model = model.to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)

    def train(self, X_train, y_train, epochs, batch_size):
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            self.model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32).to(device)
            return self.model(inputs).cpu().numpy().flatten()

def plot_combined_top_samples(best_samples, save_dir, task_name):
    """
    Plots the top 3 samples on a SINGLE graph with dual axes.
    3 colors (one per run/sample).
    2 line styles (Solid for Loss, Dashed for Performance).
    """

    # --- Helper for Smoothing ---
    def get_smooth_curve(steps, vals):
        from scipy.optimize import curve_fit

        # 1. Prepare data (ensure numpy arrays)
        x_data = np.array(steps, dtype=float)
        y_data = np.array(vals, dtype=float)

        # Power law fails if x=0. Offset slightly if necessary.
        x_data_fixed = x_data + 1e-6 if np.any(x_data == 0) else x_data

        # Define the models with a 'c' offset (asymptote)
        def exponential_decay(x, a, b, c):
            return a * np.exp(-b * x) + c

        def power_law(x, a, b, c):
            return a * np.power(x, -b) + c

        # Initial Guesses (Crucial for convergence)
        # a: initial value, b: decay rate, c: floor/asymptote
        a_guess = y_data[0]
        c_guess = np.min(y_data)
        b_guess = 0.01 

        # --- OPTION A: EXPONENTIAL DECAY ---
        # popt, _ = curve_fit(exponential_decay, x_data, y_data, p0=[a_guess, b_guess, c_guess], maxfev=10000)
        # smooth_vals = exponential_decay(x_data, *popt)

        # --- OPTION B: POWER LAW ---
        # Uncomment below and comment out Option A to switch
        popt, _ = curve_fit(power_law, x_data_fixed, y_data, p0=[a_guess, 0.5, c_guess], maxfev=10000)
        x_smooth_plot = np.linspace(x_data_fixed.min(), x_data_fixed.max(), 500)
        y_smooth_plot = power_law(x_smooth_plot, *popt) # Using the fitted params

        return x_smooth_plot, y_smooth_plot

    visuals_dir = os.path.join(save_dir, "best_plots_combined")
    os.makedirs(visuals_dir, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Define distinctive colors for the 3 distinct runs
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green (Standard Matplotlib/Seaborn tab10)
    
    # --- Axis 1: Eval Loss (Left) ---
    ax1.set_xlabel('Training Steps', fontsize=20)
    ax1.set_ylabel('Evaluation Loss', fontsize=20)
    # ax1.set_ylim(0.8, 3) 
    ax1.set_ylim(0, 5)
    
    # # --- Axis 2: Performance (Right) ---
    # ax2 = ax1.twinx() 
    # ax2.set_ylabel('Performance (Dashed Line)', fontsize=12, fontweight='bold')
    # ax2.set_ylim(0.2, 0.8) 

    # Loop through the top 3 samples and plot them
    for i, sample in enumerate(best_samples):
        c = colors[i % len(colors)]
        run_label = f"Run {i+1} (Err: {sample['total_err']:.2f})"
        
        # 1. Plot Eval Loss Trajectory (Left Axis, Solid)
        raw_steps_loss, raw_vals_loss = zip(*sample['traj_loss'])
        x_loss, y_loss = get_smooth_curve(raw_steps_loss, raw_vals_loss)
        ax1.plot(x_loss, y_loss, color=c, linestyle='-', linewidth=5, label=f"{run_label} - Loss")
        
        # # 2. Plot Performance Trajectory (Right Axis, Dashed)
        # steps_perf, vals_perf = zip(*sample['traj_perf'])
        # ax2.plot(steps_perf, vals_perf, color=c, linestyle='--', linewidth=2.5, label=f"{run_label} - Perf")

        # 3. Plot Markers for Target Predictions (Solid=Loss, Empty/Dashed=Perf visual metaphor)
        target_step = TARGET_PREDICTION_STEPS[0]
        
        # Loss Markers (Filled Star = Actual, X = Pred)
        ax1.scatter(raw_steps_loss[:4], raw_vals_loss[:4], color=c, marker='.', s=150, linewidth=5, zorder=10) # Actual loss at first few steps
        ax1.scatter([target_step], [sample['pred_loss']], color=c, marker='x', s=150, linewidth=5, zorder=10) # Pred Loss

        # # Perf Markers (Same color, same shapes, mapped to Right Axis)
        # ax2.scatter(steps_perf[1:3], vals_perf[1:3], color=c, marker='.', s=150, linewidth=2.5, zorder=10) # Actual perf at first two steps
        # ax2.scatter([target_step], [sample['pred_perf']], color=c, marker='x', s=150, linewidth=2.5, zorder=10) # Pred Perf

    # Combine Legends
    # We create a custom legend to explain the encoding clearly
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D([0], [0], color='black', lw=3, linestyle='-', label='Actual Evaluation Loss'),
        # Line2D([0], [0], color='black', lw=2.5, linestyle='--', label='Performance'),
        Line2D([0], [0], marker='.', color='black', label='Inputs to Neural Network', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='x', color='black', label='Neural Network Prediction', markersize=10, linestyle='None')
    ]
    
    # Add color indicators for runs
    # for i, sample in enumerate(best_samples):
    #     legend_elements.append(Line2D([0], [0], color=colors[i], lw=2.5, label=f"Run {i+1}"))

    ax1.legend(handles=legend_elements, loc='upper right', fontsize=16, frameon=True)
    
    plt.title(f"Actual vs Predicted Evaluation Loss", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(visuals_dir, f"combined_chosen3_runs_eval_loss_only" + str(random.randint(0, 10000)) + ".png")
    print(save_path)
    plt.savefig(save_path)

def run_experiment(task, dist_type):
    print(f"\n==================================================")
    print(f"Running Experiment: Task={task}")
    
    if dist_type == 'ood':
        STATIC_DIM = 18
    elif dist_type == 'in_dist':
        STATIC_DIM = 19
    else:
        raise ValueError("Invalid dist type")

    # Dimensions
    NUM_HISTORY_POINTS = len(REQUIRED_HISTORY_STEPS) # 2
    HISTORY_DIM = NUM_HISTORY_POINTS * 2 
    TARGET_STEP_DIM = 1 
    INPUT_DIM = STATIC_DIM + HISTORY_DIM + TARGET_STEP_DIM 
    
    FILE_PATH = f"results_eval_random_{dist_type}/['{task}']_eval_results.json" 
    SAVE_DIR = f'{BASE_PATH}/Dual_Optimization/{task}'
    os.makedirs(SAVE_DIR, exist_ok=True)

    set_all_seeds(RANDOM_STATE) 

    processor = DataProcessor(FILE_PATH)
    if not processor.raw_data: 
        print(f"Skipping {task}: No data found.")
        return

    # 1. Split Raw Data
    raw_train = processor.raw_data[:20]
    raw_test = processor.raw_data[20:100]

    if len(raw_train) < 20:
        print(f"Warning: Only {len(raw_train)} training samples available.")
    if len(raw_test) == 0:
        print("Error: No test samples available.")
        return

    # 2. Prepare Training Data for BOTH metrics
    X_train_loss, y_train_loss, _, _ = processor.get_dataset_for_metric(raw_train, 'eval_loss', STATIC_DIM)
    X_train_perf, y_train_perf, _, _ = processor.get_dataset_for_metric(raw_train, 'performance', STATIC_DIM)

    # 3. Train Models
    scaler_loss = StandardScaler()
    scaler_perf = StandardScaler()
    
    if len(X_train_loss) == 0 or len(X_train_perf) == 0:
        print("Missing training data.")
        return

    X_train_loss_scaled = scaler_loss.fit_transform(X_train_loss)
    # X_train_perf_scaled = scaler_perf.fit_transform(X_train_perf)

    print("Training Eval Loss Model...")
    model_loss = MetricPredictorMLP(input_dim=INPUT_DIM)
    trainer_loss = ModelTrainer(model_loss, LEARNING_RATE)
    trainer_loss.train(X_train_loss_scaled, y_train_loss.reshape(-1, 1), EPOCHS, BATCH_SIZE)

    # print("Training Performance Model...")
    # model_perf = MetricPredictorMLP(input_dim=INPUT_DIM)
    # trainer_perf = ModelTrainer(model_perf, LEARNING_RATE)
    # trainer_perf.train(X_train_perf_scaled, y_train_perf.reshape(-1, 1), EPOCHS, BATCH_SIZE)

    # 4. Evaluation & Ranking
    X_test_loss, y_test_loss, traj_loss, idx_test_loss = processor.get_dataset_for_metric(raw_test, 'eval_loss', STATIC_DIM)
    # X_test_perf, y_test_perf, traj_perf, idx_test_perf = processor.get_dataset_for_metric(raw_test, 'performance', STATIC_DIM)

    eval_candidates = {}

    # Map Loss Data
    X_test_loss_scaled = scaler_loss.transform(X_test_loss)
    preds_loss = trainer_loss.predict(X_test_loss_scaled)
    
    for i, raw_idx in enumerate(idx_test_loss):
        if raw_idx not in eval_candidates: eval_candidates[raw_idx] = {}
        eval_candidates[raw_idx]['loss'] = {
            'pred': preds_loss[i],
            'act': y_test_loss[i],
            'traj': traj_loss[i]
        }

    # # Map Perf Data
    # X_test_perf_scaled = scaler_perf.transform(X_test_perf)
    # preds_perf = trainer_perf.predict(X_test_perf_scaled)

    # for i, raw_idx in enumerate(idx_test_perf):
    #     if raw_idx not in eval_candidates: eval_candidates[raw_idx] = {}
    #     eval_candidates[raw_idx]['perf'] = {
    #         'pred': preds_perf[i],
    #         'act': y_test_perf[i],
    #         'traj': traj_perf[i]
    #     }

    valid_samples = []
    for raw_idx, data in eval_candidates.items():
        if 'loss' in data:
            err_loss = abs(data['loss']['pred'] - data['loss']['act'])
            total_err = err_loss
            
            valid_samples.append({
                'idx': raw_idx,
                'total_err': total_err,
                'pred_loss': data['loss']['pred'],
                'act_loss': data['loss']['act'],
                'traj_loss': data['loss']['traj']
            })
    for s in valid_samples:
        s['start_val'] = s['traj_loss'][0][1]
        s['end_val'] = s['traj_loss'][-1][1]
        
    crossover_pairs = []

    for i in range(len(valid_samples)):
        for j in range(len(valid_samples)):
            if i == j:
                continue

            si = valid_samples[i]
            sj = valid_samples[j]

            # Start: A > B
            if si['start_val'] > sj['start_val']:
                # End: B > A
                if sj['end_val'] > si['end_val']:
                    crossover_pairs.append((si, sj))
    print(len(crossover_pairs), " crossover pairs found.")
    for pair in crossover_pairs:
        A, B = pair
        # add a random sample at the end
        chosen_3 = [A, B, valid_samples[random.randint(0, len(valid_samples) - 1)]]
        print(A)
        print(B)
        # valid_samples.sort(key=lambda x: x['total_err'])
        # # 1. Highest eval loss at step 25
        # highest_start_sample = max(valid_samples, key=lambda x: x['traj_loss'][0][1])
        
        # # 2. Lowest eval loss at step 25
        # lowest_start_sample = min(valid_samples, key=lambda x: x['traj_loss'][0][1])
        
        # # 3. "Any other" sample (e.g., the one with the median starting value or just the first in the list that isn't the other two)
        # remaining_samples = [s for s in valid_samples if s['idx'] not in [highest_start_sample['idx'], lowest_start_sample['idx']]]
#       other_sample = remaining_samples[len(remaining_samples) // 2] # Picks the middle one

        # chosen_3 = [highest_start_sample, lowest_start_sample, other_sample]

        print(f"Chosen errors: {['{:.4f}'.format(s['total_err']) for s in chosen_3]}")

        # 5. Combined Plotting
        plot_combined_top_samples(chosen_3, SAVE_DIR, task)
    print(f"Completed: {task}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='Comma-separated list of tasks')
    parser.add_argument('--dist', type=str, required=True, choices=['ood', 'in_dist'])
    
    args = parser.parse_args()
    TASKS_LIST = [t.strip() for t in args.task.split(',')]
    
    for task in TASKS_LIST:
        run_experiment(task, args.dist)