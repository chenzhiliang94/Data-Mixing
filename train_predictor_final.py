import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import make_interp_spline
import joblib
import os
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import argparse 
from typing import List, Tuple, Dict, Any

# --- Fixed Configuration ---
# Modified: Only run for 20 samples as requested
SAMPLE_SIZES = [20] 
EPOCHS = 500
BATCH_SIZE = 8
LEARNING_RATE = 0.01
# Validation split is now manual (indices 20-30), so this constant is removed/unused
RANDOM_STATE = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_PATH = f"trained_predictor_fixed_20train_10val"

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
        self.X_perf: List[List[float]] = []
        self.y_perf: List[float] = []
        self.trajectories: List[List[Tuple[int, float]]] = [] 

    def _load_json(self) -> List[Dict[str, Any]]:
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            print(f"Error: File not found at {self.filepath}")
            return []

    def set_raw_data(self, data: List[Dict[str, Any]]):
        self.raw_data = data

    def preprocess(self, required_history_steps, target_prediction_steps, static_dim):
        self.X_perf, self.y_perf, self.trajectories = [], [], []

        for experiment in self.raw_data:
            input_x = experiment.get('input_X', [])
            if len(input_x) != static_dim:
                continue

            evals_list = experiment.get('evaluations', [])
            step_map = {item['step']: item for item in evals_list if EVAL_METHOD in item}

            # 1. Check if ALL required history points exist
            history_features = []
            found_all_history = True
            for req_step in required_history_steps:
                if req_step in step_map:
                    history_features.append(float(req_step))
                    history_features.append(float(step_map[req_step][EVAL_METHOD]))
                else:
                    found_all_history = False
                    break
            
            if not found_all_history:
                continue

            # 2. Iterate through requested TARGET steps
            for target_step in target_prediction_steps:
                if target_step in step_map:
                    valid_target_point = step_map[target_step]
                    
                    features_perf = list(input_x)
                    features_perf.extend(history_features)
                    features_perf.append(float(valid_target_point['step']))
                    
                    self.X_perf.append(features_perf)
                    self.y_perf.append(valid_target_point[EVAL_METHOD])

                    # Extract Full Trajectory up to this target step
                    traj = []
                    sorted_steps = sorted([k for k in step_map.keys() if k <= target_step])
                    for s in sorted_steps:
                        traj.append((s, step_map[s][EVAL_METHOD]))
                    self.trajectories.append(traj)

    def get_data_arrays(self):
        return np.array(self.X_perf), np.array(self.y_perf)
    
    def get_trajectories(self):
        return self.trajectories


class MetricPredictorMLP(nn.Module):
    def __init__(self, input_dim: int, use_sigmoid: bool = False):
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
        if use_sigmoid:
             layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ModelTrainer:
    def __init__(self, model: nn.Module, learning_rate: float):
        self.model = model.to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size, plot_path):
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        epoch_losses = [] 

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            
            avg_loss = running_loss / len(train_loader)
            epoch_losses.append(avg_loss)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs + 1), epoch_losses, label='Training Loss', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(plot_path)
        plt.close()

    def save_model(self, filepath: str):
        torch.save(self.model.state_dict(), filepath)


def visualize_individual_predictions(X_val_original, trajectories, y_val, predictions, save_dir, num_samples):
    visuals_dir = os.path.join(save_dir, f"visuals_N{num_samples}")
    os.makedirs(visuals_dir, exist_ok=True)
    
    # --- Grouping Logic ---
    grouped_data = {}
    
    for i in range(len(X_val_original)):
        # Signature: All features EXCEPT the last one (target_step)
        input_signature = tuple(X_val_original[i, :-1])
        
        target_step_val = int(X_val_original[i, -1])
        actual_loss = y_val[i]
        predicted_loss = predictions[i]
        traj = trajectories[i]
        
        if input_signature not in grouped_data:
            grouped_data[input_signature] = {
                'traj': traj,
                'predictions': []
            }
        
        grouped_data[input_signature]['predictions'].append({
            'step': target_step_val,
            'actual': actual_loss,
            'predicted': predicted_loss
        })
        
        if len(traj) > len(grouped_data[input_signature]['traj']):
            grouped_data[input_signature]['traj'] = traj

    # --- Plotting Logic ---
    exp_counter = 1
    
    for signature, data in grouped_data.items():
        traj = data['traj']
        preds_list = data['predictions']
        
        plt.figure(figsize=(10, 6))
        
        # 1. Plot Full Actual Curve (Smoothed)
        steps, losses = zip(*traj)
        steps = np.array(steps)
        losses = np.array(losses)

        plt.plot(steps, losses, color='grey', alpha=0.6, linestyle='-', linewidth=1.5, label='Actual Trajectory')
        
        # 2. Plot Input History Points (Blue Dots)
        input_steps = []
        input_losses = []
        for s, l in traj:
            if s in REQUIRED_HISTORY_STEPS:
                input_steps.append(s)
                input_losses.append(l)
        
        plt.scatter(input_steps, input_losses, color='blue', s=80, zorder=5, label='Input Data Points')
        
        # 3. Plot Continuous Predicted Line (Red Dashed)
        # Sort predictions by step to ensure the line connects them in order
        preds_list.sort(key=lambda x: x['step'])
        
        if input_steps:
            # Start line at the last known history point
            pred_line_x = [input_steps[-1]]
            pred_line_y = [input_losses[-1]]
            
            # Append all predicted points in sequence
            for p in preds_list:
                pred_line_x.append(p['step'])
                pred_line_y.append(p['predicted'])
                
            plt.plot(pred_line_x, pred_line_y, color='red', linestyle='--', alpha=0.6, linewidth=1.5)

        # 4. Plot Markers (Stars and Crosses)
        legend_actual_added = False
        legend_pred_added = False
        
        for p in preds_list:
            t_step = p['step']
            t_actual = p['actual']
            t_pred = p['predicted']
            
            # Actual (Green Star)
            label_act = 'Actual Target' if not legend_actual_added else ""
            plt.scatter([t_step], [t_actual], color='green', marker='*', s=150, zorder=6, label=label_act)
            if label_act: legend_actual_added = True
            
            # Predicted (Red Cross)
            label_pred = 'Predicted Target' if not legend_pred_added else ""
            plt.scatter([t_step], [t_pred], color='red', marker='x', s=100, linewidth=2, zorder=6, label=label_pred)
            if label_pred: legend_pred_added = True

        target_steps_str = ", ".join([str(p['step']) for p in preds_list])
        task = save_dir.split("/")[-1]
        plt.title(f'Task: {task} | Train Size: {num_samples} | Targets: {target_steps_str}')
        plt.xlabel('Step')
        plt.ylabel(EVAL_METHOD)
        
        if EVAL_METHOD == "eval_loss":
            plt.ylim(0, 4)
            plt.gca().yaxis.set_major_locator(MultipleLocator(0.5))
        elif EVAL_METHOD == "performance":
            plt.ylim(0, 1)
            plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
        
        plt.grid(True, alpha=0.3, which='both')
        plt.legend()
        
        plt.savefig(os.path.join(visuals_dir, f"experiment_{exp_counter}.png"))
        plt.close()
        
        exp_counter += 1


def plot_error_comparison(X_val, y_val, trajectories, save_dir, input_dim, sample_sizes):
    # Adjusted figsize since we are likely only plotting 1 subplot now
    plt.figure(figsize=(6 * len(sample_sizes), 5))
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    rmse_list = []

    for i, num_samples in enumerate(sample_sizes):
        model_path = f'{save_dir}/performance_mlp_{num_samples}samples.pth'
        scaler_path = f'{save_dir}/scaler_{num_samples}samples.joblib'
        
        if not os.path.exists(model_path):
            continue
            
        model = MetricPredictorMLP(input_dim=input_dim, use_sigmoid=False).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        scaler = joblib.load(scaler_path)
        
        X_scaled = scaler.transform(X_val)
        inputs = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            predictions = model(inputs).cpu().numpy().flatten()
            
        # Generate Visualization
        visualize_individual_predictions(X_val, trajectories, y_val, predictions, save_dir, num_samples)

        errors = predictions - y_val
        rmse = np.sqrt(np.mean(np.square(errors)))
        rmse_list.append(rmse)
        
        plt.subplot(1, len(sample_sizes), i+1)
        plt.hist(errors, bins=20, alpha=0.7, color=colors[i % len(colors)], edgecolor='white')
        plt.axvline(0, color='black', linestyle='--')
        plt.title(f'N={num_samples} | RMSE: {rmse:.4f}')
        plt.xlabel('Error')
        plt.xlim(-0.5, 0.5) 

    plt.tight_layout()
    plt.savefig(f"{save_dir}/error_comparison.png")
    plt.close()

    return rmse_list


def run_experiment(task):
    print(f"\n==================================================")
    print(f"Running Experiment: Task={task}")
    print(f"History Steps: {REQUIRED_HISTORY_STEPS}")
    print(f"Target Steps: {TARGET_PREDICTION_STEPS}")
    
    if DIST_TYPE == 'ood':
        STATIC_DIM = 18
    elif DIST_TYPE == 'in_dist':
        STATIC_DIM = 19
    else:
        raise ValueError("Invalid dist type")

    NUM_HISTORY_POINTS = len(REQUIRED_HISTORY_STEPS)
    HISTORY_DIM = NUM_HISTORY_POINTS * 2 
    TARGET_STEP_DIM = 1 
    INPUT_DIM = STATIC_DIM + HISTORY_DIM + TARGET_STEP_DIM
    
    FILE_PATH = f"results_eval_random_{DIST_TYPE}/['{task}']_eval_results.json" 
    
    hist_str = "_".join(map(str, REQUIRED_HISTORY_STEPS))
    tgt_str = "_".join(map(str, TARGET_PREDICTION_STEPS))
    SAVE_DIR = f'{BASE_PATH}/{EVAL_METHOD}_H{hist_str}_T{tgt_str}_curve/{task}'
    print(SAVE_DIR)
    os.makedirs(SAVE_DIR, exist_ok=True)

    set_all_seeds(RANDOM_STATE) 

    processor = DataProcessor(FILE_PATH)
    if not processor.raw_data: 
        print(f"Skipping {task}: No data found.")
        return

    # --- UPDATED SPLITTING LOGIC ---
    # Strictly slicing: 0-20 for training, 20-30 for validation
    
    num_train = 50
    if len(processor.raw_data) < num_train:
        print(f"Warning: Not enough data for strict 20/10 split. Found {len(processor.raw_data)} samples.")
        print("Proceeding with available data, but splits may be smaller than requested.")
    
    train_raw_exps = processor.raw_data[:num_train]
    val_raw_exps = processor.raw_data[20:]

    # Process Validation Set
    processor.set_raw_data(val_raw_exps)
    processor.preprocess(REQUIRED_HISTORY_STEPS, TARGET_PREDICTION_STEPS, STATIC_DIM)
    X_val_global, y_val_global = processor.get_data_arrays()
    val_trajectories = processor.get_trajectories()
    
    if len(X_val_global) == 0:
        print(f"Skipping {task}: No validation data extracted from samples 20-30.")
        return

    # Process Training Set
    processor.set_raw_data(train_raw_exps)
    processor.preprocess(REQUIRED_HISTORY_STEPS, TARGET_PREDICTION_STEPS, STATIC_DIM)
    X_train_final, y_train_final = processor.get_data_arrays()

    if len(X_train_final) == 0:
        print(f"Skipping {task}: No training data extracted from samples 0-20.")
        return

    # We use the loop structure for compatibility with plotting functions, 
    # but SAMPLE_SIZES is now just [20]
    for target_sample_size in SAMPLE_SIZES:
        
        print(f"Training with N={len(train_raw_exps)} experiments ({len(X_train_final)} datapoints)...")

        y_train_reshaped = y_train_final.reshape(-1, 1)
        y_val_reshaped = y_val_global.reshape(-1, 1)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_final)
        X_val_scaled = scaler.transform(X_val_global)

        model = MetricPredictorMLP(input_dim=INPUT_DIM, use_sigmoid=False)
        trainer = ModelTrainer(model, learning_rate=LEARNING_RATE)
        
        save_plot_path = f"{SAVE_DIR}/training_loss_{target_sample_size}.png"
        trainer.train(X_train_scaled, y_train_reshaped, X_val_scaled, y_val_reshaped, 
                      epochs=EPOCHS, batch_size=BATCH_SIZE, plot_path=save_plot_path)

        model_path = f'{SAVE_DIR}/performance_mlp_{target_sample_size}samples.pth'
        scaler_path = f'{SAVE_DIR}/scaler_{target_sample_size}samples.joblib'
        
        trainer.save_model(model_path)
        joblib.dump(scaler, scaler_path)

    rmse_list = plot_error_comparison(X_val_global, y_val_global, val_trajectories, SAVE_DIR, INPUT_DIM, SAMPLE_SIZES)
    print(f"Completed: {task}")
    
    return {"task": task, "rmse": rmse_list}

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train Metric Predictor with Variable History and Targets")

    parser.add_argument('--task', type=str, required=True, 
                        help='Comma-separated list of tasks (e.g., "commonsense_qa,arc_challenge")')

    parser.add_argument('--target_history_steps', type=str, required=True,
                        help='Comma-separated list of history steps to use as input (e.g., "50,100,200")')

    parser.add_argument('--target_prediction_steps', type=str, required=True,
                        help='Comma-separated list of target steps to predict (e.g., "625,1000")')

    parser.add_argument('--dist', type=str, required=True, 
                        choices=['ood', 'in_dist'],
                        help='Distribution type: determines STATIC_DIM and save path')

    parser.add_argument('--eval_method', type=str, required=True,
                    choices=['eval_loss', 'performance'],
                    help='Use eval_loss or performance for prediction')

    args = parser.parse_args()

    TASKS_LIST = [t.strip() for t in args.task.split(',')]
    REQUIRED_HISTORY_STEPS = [int(s.strip()) for s in args.target_history_steps.split(',')]
    TARGET_PREDICTION_STEPS = [int(s.strip()) for s in args.target_prediction_steps.split(',')]
    DIST_TYPE = args.dist
    EVAL_METHOD = args.eval_method

    print(f"Tasks to run: {TASKS_LIST}")

    good_settings_list = []
    
    for task in TASKS_LIST:
        res = run_experiment(task)
        if res:
            good_settings_list.append(res)