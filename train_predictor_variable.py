import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import make_interp_spline
import joblib
import os
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import argparse 
from typing import List, Tuple, Dict, Any

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

args = parser.parse_args()

TASKS_LIST = [t.strip() for t in args.task.split(',')]
REQUIRED_HISTORY_STEPS = [int(s.strip()) for s in args.target_history_steps.split(',')]
TARGET_PREDICTION_STEPS = [int(s.strip()) for s in args.target_prediction_steps.split(',')]
DIST_TYPE = args.dist

# --- Fixed Configuration ---
SAMPLE_SIZES = [30, 50, 80]
EPOCHS = 500
BATCH_SIZE = 8
LEARNING_RATE = 0.01
VALIDATION_SPLIT = 0.2  
RANDOM_STATE = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_PATH = f"trained_predictor_epochs{EPOCHS}_batchsize{BATCH_SIZE}_optimal_1"

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
            step_map = {item['step']: item for item in evals_list if 'eval_loss' in item}

            # 1. Check if ALL required history points exist
            history_features = []
            found_all_history = True
            for req_step in required_history_steps:
                if req_step in step_map:
                    history_features.append(float(req_step))
                    history_features.append(float(step_map[req_step]['eval_loss']))
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
                    self.y_perf.append(valid_target_point['eval_loss'])

                    # Extract Full Trajectory up to this target step
                    traj = []
                    sorted_steps = sorted([k for k in step_map.keys() if k <= target_step])
                    for s in sorted_steps:
                        traj.append((s, step_map[s]['eval_loss']))
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

        # # Apply smoothing only if we have enough points (k=3 requires at least 4 points)
        # if len(steps) > 3:
        #     try:
        #         # Create a B-spline representation of the curve
        #         spline = make_interp_spline(steps, losses, k=3) 
        #         steps_smooth = np.linspace(steps.min(), steps.max(), 500)
        #         losses_smooth = spline(steps_smooth)
                
        #         plt.plot(steps_smooth, losses_smooth, color='grey', alpha=0.6, linestyle='-', linewidth=1.5, label='Actual Trajectory')
        #     except Exception as e:
        #         # Fallback to straight lines if smoothing fails
        #         plt.plot(steps, losses, color='grey', alpha=0.6, linestyle='-', linewidth=1.5, label='Actual Trajectory')
        # else:
        #     plt.plot(steps, losses, color='grey', alpha=0.6, linestyle='-', linewidth=1.5, label='Actual Trajectory')

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
        plt.ylabel('Eval Loss')
        
        # --- Fixed Y-Axis to 0-4 with 0.5 intervals ---
        plt.ylim(0, 4)
        plt.gca().yaxis.set_major_locator(MultipleLocator(0.5))
        
        plt.grid(True, alpha=0.3, which='both')
        plt.legend()
        
        plt.savefig(os.path.join(visuals_dir, f"experiment_{exp_counter}.png"))
        plt.close()
        
        exp_counter += 1


def plot_error_comparison(X_val, y_val, trajectories, save_dir, input_dim, sample_sizes):
    plt.figure(figsize=(15, 5))
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
        
        plt.subplot(1, 3, i+1)
        plt.hist(errors, bins=20, alpha=0.7, color=colors[i], edgecolor='white')
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
    SAVE_DIR = f'{BASE_PATH}/{DIST_TYPE}_H{hist_str}_T{tgt_str}_curve/{task}'
    os.makedirs(SAVE_DIR, exist_ok=True)

    set_all_seeds(RANDOM_STATE) 

    processor = DataProcessor(FILE_PATH)
    if not processor.raw_data: 
        print(f"Skipping {task}: No data found.")
        return

    train_raw_exps, val_raw_exps = train_test_split(
        processor.raw_data, 
        test_size=VALIDATION_SPLIT, 
        random_state=RANDOM_STATE
    )

    # Preprocess Validation Set
    processor.set_raw_data(val_raw_exps)
    processor.preprocess(REQUIRED_HISTORY_STEPS, TARGET_PREDICTION_STEPS, STATIC_DIM)
    X_val_global, y_val_global = processor.get_data_arrays()
    val_trajectories = processor.get_trajectories()
    
    if len(X_val_global) == 0:
        print(f"Skipping {task}: No validation data found matching required steps.")
        return

    for target_sample_size in SAMPLE_SIZES:
        
        accumulated_samples_count = 0
        current_train_raw_exps = []
        shuffled_train_exps = train_raw_exps[:] 
        random.shuffle(shuffled_train_exps)

        for experiment in shuffled_train_exps:
            temp_exps = current_train_raw_exps + [experiment]
            processor.set_raw_data(temp_exps)
            processor.preprocess(REQUIRED_HISTORY_STEPS, TARGET_PREDICTION_STEPS, STATIC_DIM)
            X_train_subset, y_train_subset = processor.get_data_arrays()
            
            new_count = len(X_train_subset)
            if new_count >= target_sample_size:
                current_train_raw_exps = temp_exps
                X_train_final = X_train_subset[:target_sample_size]
                y_train_final = y_train_subset[:target_sample_size]
                accumulated_samples_count = new_count
                break
            elif new_count > accumulated_samples_count:
                current_train_raw_exps = temp_exps
                accumulated_samples_count = new_count
        
        if len(X_train_final) == 0:
            print(f"    Skipping sample size {target_sample_size}: No valid training data.")
            continue

        y_train_final = y_train_final.reshape(-1, 1)
        y_val_reshaped = y_val_global.reshape(-1, 1)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_final)
        X_val_scaled = scaler.transform(X_val_global)

        model = MetricPredictorMLP(input_dim=INPUT_DIM, use_sigmoid=False)
        trainer = ModelTrainer(model, learning_rate=LEARNING_RATE)
        
        save_plot_path = f"{SAVE_DIR}/training_loss_{target_sample_size}.png"
        trainer.train(X_train_scaled, y_train_final, X_val_scaled, y_val_reshaped, 
                      epochs=EPOCHS, batch_size=BATCH_SIZE, plot_path=save_plot_path)

        model_path = f'{SAVE_DIR}/performance_mlp_{target_sample_size}samples.pth'
        scaler_path = f'{SAVE_DIR}/scaler_{target_sample_size}samples.joblib'
        
        trainer.save_model(model_path)
        joblib.dump(scaler, scaler_path)

    rmse_list = plot_error_comparison(X_val_global, y_val_global, val_trajectories, SAVE_DIR, INPUT_DIM, SAMPLE_SIZES)
    print(f"Completed: {task}")
    
    return {"task": task, "rmse": rmse_list}

if __name__ == "__main__":
    print(f"Tasks to run: {TASKS_LIST}")

    good_settings_list = []
    
    for task in TASKS_LIST:
        res = run_experiment(task)
        if res:
            good_settings_list.append(res)