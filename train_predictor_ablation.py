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
parser = argparse.ArgumentParser(description="Train Metric Predictor with Ablation Studies")

parser.add_argument('--task', type=str, required=True, 
                    help='Comma-separated list of tasks (e.g., "commonsense_qa,arc_challenge")')

parser.add_argument('--default_history_steps', type=str, required=True,
                    help='Default history steps (comma-separated) to use as baseline (e.g., "50,100,200")')

parser.add_argument('--default_prediction_steps', type=str, required=True,
                    help='Default target steps (comma-separated) to use as baseline (e.g., "625,1000")')

parser.add_argument('--dist', type=str, required=True, 
                    choices=['ood', 'in_dist'],
                    help='Distribution type: determines STATIC_DIM and save path')

parser.add_argument('--eval_method', type=str, required=True,
                    choices=['eval_loss', 'performance'],
                    help='Use eval_loss or performance for prediction')

args = parser.parse_args()

TASKS_LIST = [t.strip() for t in args.task.split(',')]
DEFAULT_HISTORY_STEPS_LIST = [int(s.strip()) for s in args.default_history_steps.split(',')]
DEFAULT_PREDICTION_STEPS_LIST = [int(s.strip()) for s in args.default_prediction_steps.split(',')]
DIST_TYPE = args.dist
EVAL_METHOD = args.eval_method

# --- Ablation Configuration ---
SAMPLE_SIZES = [10,20,30] 
VALIDATION_SPLIT = 0.2  
RANDOM_STATE = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.01

# Default Baseline Configuration
DEFAULT_CONFIG = {
    'dropout': 0.05,
    'neurons': [32, 16], 
    'epochs': 200,
    'batch_size': 16,
    'history_steps': DEFAULT_HISTORY_STEPS_LIST,      # NEW: Now part of config
    'prediction_steps': DEFAULT_PREDICTION_STEPS_LIST # NEW: Now part of config
}

# Ablation Search Space
ABLATION_STUDIES = {
    'dropout': [0.0, 0.05, 0.2, 0.5],
    'neurons': [
        [16, 8],
        [32, 16],    
        [64, 32]   
    ],
    'layers': [
        [32, 16],       
        [32, 16, 8],    
        [64, 32, 16]    
    ],
    'epochs': [100, 200, 500],
    'batch_size': [8, 16, 32],
    # NEW: Data Ablations
    'history_steps': [
        [25, 50],
        [50], 
        [50, 100],
        [25, 50, 75, 100], 
        [50, 100, 150]
    ],
    'prediction_steps': [
        [625],
        [575, 625],
        [525, 575, 625]
    ]
}

BASE_OUTPUT_DIR = "train_predictor_ablation_study_results"

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

                    traj = []
                    sorted_steps = sorted([k for k in step_map.keys() if k <= target_step])
                    for s in sorted_steps:
                        traj.append((s, step_map[s][EVAL_METHOD]))
                    self.trajectories.append(traj)

    def get_data_arrays(self):
        return np.array(self.X_perf), np.array(self.y_perf)
    
    def get_trajectories(self):
        return self.trajectories

# --- Dynamic MLP Class ---
class MetricPredictorMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: List[int], dropout_rate: float):
        super(MetricPredictorMLP, self).__init__()
        
        layers = []
        in_dim = input_dim
        
        # Dynamically build hidden layers
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
            
        # Final Output Layer
        layers.append(nn.Linear(in_dim, 1))
        
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
        
        actual_batch_size = min(batch_size, len(X_train))
        if actual_batch_size == 0: actual_batch_size = 1
        
        train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True)
        
        epoch_losses = [] 

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            
            if len(train_loader) == 0: break 

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
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Training Loss', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(plot_path)
        plt.close()

    def save_model(self, filepath: str):
        torch.save(self.model.state_dict(), filepath)


def visualize_individual_predictions(X_val_original, trajectories, y_val, predictions, save_dir, num_samples, history_steps_used):
    visuals_dir = os.path.join(save_dir, f"visuals_N{num_samples}")
    os.makedirs(visuals_dir, exist_ok=True)
    
    grouped_data = {}
    
    for i in range(len(X_val_original)):
        input_signature = tuple(X_val_original[i, :-1])
        target_step_val = int(X_val_original[i, -1])
        actual_loss = y_val[i]
        predicted_loss = predictions[i]
        traj = trajectories[i]
        
        if input_signature not in grouped_data:
            grouped_data[input_signature] = {'traj': traj, 'predictions': []}
        
        grouped_data[input_signature]['predictions'].append({
            'step': target_step_val, 'actual': actual_loss, 'predicted': predicted_loss
        })
        
        if len(traj) > len(grouped_data[input_signature]['traj']):
            grouped_data[input_signature]['traj'] = traj

    exp_counter = 1
    
    for signature, data in grouped_data.items():
        traj = data['traj']
        preds_list = data['predictions']
        
        plt.figure(figsize=(10, 6))
        
        steps, losses = zip(*traj)
        steps = np.array(steps)
        losses = np.array(losses)

        plt.plot(steps, losses, color='grey', alpha=0.6, linestyle='-', linewidth=1.5, label='Actual Trajectory')
        
        input_steps = []
        input_losses = []
        for s, l in traj:
            if s in history_steps_used: # Check against the CONFIG specific history steps
                input_steps.append(s)
                input_losses.append(l)
        
        plt.scatter(input_steps, input_losses, color='blue', s=80, zorder=5, label='Input Data Points')
        
        preds_list.sort(key=lambda x: x['step'])
        
        if input_steps:
            pred_line_x = [input_steps[-1]]
            pred_line_y = [input_losses[-1]]
            for p in preds_list:
                pred_line_x.append(p['step'])
                pred_line_y.append(p['predicted'])
            plt.plot(pred_line_x, pred_line_y, color='red', linestyle='--', alpha=0.6, linewidth=1.5)

        legend_actual_added = False
        legend_pred_added = False
        
        for p in preds_list:
            t_step = p['step']
            t_actual = p['actual']
            t_pred = p['predicted']
            
            label_act = 'Actual Target' if not legend_actual_added else ""
            plt.scatter([t_step], [t_actual], color='green', marker='*', s=150, zorder=6, label=label_act)
            if label_act: legend_actual_added = True
            
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


# --- UPDATED: Return Full Errors ---
def plot_error_comparison(X_val, y_val, trajectories, save_dir, input_dim, sample_sizes, config):
    plt.figure(figsize=(15, 5))
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    # Store both RMSE and raw errors
    results_data = {} 

    for i, num_samples in enumerate(sample_sizes):
        model_path = f'{save_dir}/performance_mlp_{num_samples}samples.pth'
        scaler_path = f'{save_dir}/scaler_{num_samples}samples.joblib'
        
        if not os.path.exists(model_path):
            continue
            
        model = MetricPredictorMLP(
            input_dim=input_dim, 
            hidden_layers=config['neurons'], 
            dropout_rate=config['dropout']
        ).to(device)
        
        model.load_state_dict(torch.load(model_path))
        model.eval()
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X_val)
        inputs = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            predictions = model(inputs).cpu().numpy().flatten()
            
        visualize_individual_predictions(X_val, trajectories, y_val, predictions, save_dir, num_samples, config['history_steps'])

        errors = predictions - y_val
        rmse = np.sqrt(np.mean(np.square(errors)))
        
        # Save detailed data for the aggregate plot later
        results_data[f"N{num_samples}"] = {
            "rmse": float(rmse),
            "errors": errors.tolist() # Convert to list for easier handling/serialization if needed
        }
        
        plt.subplot(1, 3, i+1)
        plt.hist(errors, bins=20, alpha=0.7, color=colors[i], edgecolor='white')
        plt.axvline(0, color='black', linestyle='--')
        plt.title(f'N={num_samples} | RMSE: {rmse:.4f}')
        plt.xlabel('Error'); plt.xlim(-0.5, 0.5) 

    plt.tight_layout()
    plt.savefig(f"{save_dir}/error_comparison.png")
    plt.close()

    return results_data

# --- NEW: Function to generate the massive summary grid ---
def generate_ablation_summary_plot(task, ablation_type, collected_results, values_list, sample_sizes, save_dir):
    """
    Plots a grid of histograms.
    Rows: Ablation Values (e.g. Dropout 0.0, 0.05...)
    Cols: Sample Sizes (N=30, 50, 80)
    """
    nrows = len(values_list)
    ncols = len(sample_sizes)
    
    # Dynamic Figure Size: Height grows with number of variations
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 3.5 * nrows), sharex=True, sharey=True)
    
    # Handle case where there is only 1 row (axes is 1D array)
    if nrows == 1: axes = axes.reshape(1, -1)
    
    colors = ['#ff9999', '#66b3ff', '#99ff99'] # Colors for N30, N50, N80

    for r, val in enumerate(values_list):
        val_key = get_config_name(ablation_type, val)
        
        # Get data for this specific configuration (Row)
        config_data = collected_results.get(val_key, {})
        
        for c, num_samples in enumerate(sample_sizes):
            ax = axes[r, c]
            sample_key = f"N{num_samples}"
            
            # Check if we have data for this N
            if sample_key in config_data:
                errors = config_data[sample_key]['errors']
                rmse = config_data[sample_key]['rmse']
                
                ax.hist(errors, bins=20, alpha=0.7, color=colors[c], edgecolor='white')
                ax.axvline(0, color='black', linestyle='--', linewidth=1)
                
                # Annotation
                ax.text(0.95, 0.95, f"RMSE: {rmse:.4f}", 
                        transform=ax.transAxes, ha='right', va='top', 
                        bbox=dict(boxstyle="round", fc="white", alpha=0.8))
            else:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')

            # Labels
            if r == 0:
                ax.set_title(f"Train Size: {num_samples}", fontsize=12, fontweight='bold')
            
            if c == 0:
                # Format label for list types (neurons, layers) vs simple types
                label_val = str(val)
                if len(label_val) > 20: label_val = label_val[:17] + "..." # Truncate long labels
                ax.set_ylabel(f"{ablation_type}\n{label_val}", fontsize=10, rotation=0, labelpad=40, va='center')

            ax.set_xlim(-0.5, 0.5)
            if r == nrows - 1:
                ax.set_xlabel("Prediction Error")

    plt.suptitle(f"Ablation Study: {task} - {ablation_type}", fontsize=16, y=1.00)
    plt.tight_layout()
    
    # Save in summary folder
    summary_dir = os.path.join(save_dir, f"{EVAL_METHOD}_summaries", task)
    os.makedirs(summary_dir, exist_ok=True)
    save_path = os.path.join(summary_dir, f"{ablation_type}_summary_grid.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"    Saved summary plot to: {save_path}")

# --- Main Experiment Logic ---

def get_config_name(ablation_type, value):
    if isinstance(value, list):
        return f"{ablation_type}_" + "_".join(map(str, value))
    return f"{ablation_type}_{value}"

def run_single_config(task, config, config_name):
    # Determine Input Dim
    if DIST_TYPE == 'ood':
        STATIC_DIM = 18
    elif DIST_TYPE == 'in_dist':
        STATIC_DIM = 19
    else:
        raise ValueError("Invalid dist type")

    # --- DYNAMIC INPUT CALCULATION BASED ON CONFIG ---
    current_history_steps = config['history_steps']
    current_prediction_steps = config['prediction_steps']
    
    NUM_HISTORY_POINTS = len(current_history_steps)
    HISTORY_DIM = NUM_HISTORY_POINTS * 2 
    TARGET_STEP_DIM = 1 
    INPUT_DIM = STATIC_DIM + HISTORY_DIM + TARGET_STEP_DIM
    
    FILE_PATH = f"results_eval_random_{DIST_TYPE}/['{task}']_eval_results.json" 
    
    hist_str = "_".join(map(str, current_history_steps))
    tgt_str = "_".join(map(str, current_prediction_steps))
    
    # Updated Save Directory
    SAVE_DIR = f'{BASE_OUTPUT_DIR}/{EVAL_METHOD}_H{hist_str}_T{tgt_str}/{task}/{config_name}'
    os.makedirs(SAVE_DIR, exist_ok=True)

    set_all_seeds(RANDOM_STATE) 

    processor = DataProcessor(FILE_PATH)
    if not processor.raw_data: 
        print(f"Skipping {task}: No data found.")
        return None

    train_raw_exps, val_raw_exps = train_test_split(processor.raw_data, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE)

    # Preprocess Validation Set (USING CONFIG SPECIFIC STEPS)
    processor.set_raw_data(val_raw_exps)
    processor.preprocess(current_history_steps, current_prediction_steps, STATIC_DIM)
    X_val_global, y_val_global = processor.get_data_arrays()
    val_trajectories = processor.get_trajectories()
    
    if len(X_val_global) == 0: return None

    for target_sample_size in SAMPLE_SIZES:
        
        accumulated_samples_count = 0
        current_train_raw_exps = []
        shuffled_train_exps = train_raw_exps[:] 
        random.shuffle(shuffled_train_exps)

        for experiment in shuffled_train_exps:
            temp_exps = current_train_raw_exps + [experiment]
            processor.set_raw_data(temp_exps)
            
            # Preprocess Train Set (USING CONFIG SPECIFIC STEPS)
            processor.preprocess(current_history_steps, current_prediction_steps, STATIC_DIM)
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
        
        if len(X_train_final) == 0: continue

        y_train_final = y_train_final.reshape(-1, 1)
        y_val_reshaped = y_val_global.reshape(-1, 1)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_final)
        X_val_scaled = scaler.transform(X_val_global)

        # Dynamic Model Instantiation
        model = MetricPredictorMLP(
            input_dim=INPUT_DIM, 
            hidden_layers=config['neurons'], 
            dropout_rate=config['dropout']
        )
        trainer = ModelTrainer(model, learning_rate=LEARNING_RATE) 
        
        save_plot_path = f"{SAVE_DIR}/training_loss_{target_sample_size}.png"
        trainer.train(X_train_scaled, y_train_final, X_val_scaled, y_val_reshaped, 
                      epochs=config['epochs'], 
                      batch_size=config['batch_size'], 
                      plot_path=save_plot_path)

        model_path = f'{SAVE_DIR}/performance_mlp_{target_sample_size}samples.pth'
        scaler_path = f'{SAVE_DIR}/scaler_{target_sample_size}samples.joblib'
        
        trainer.save_model(model_path)
        joblib.dump(scaler, scaler_path)

    # Calculate and Plot Errors using the specific config
    rmse_map = plot_error_comparison(X_val_global, y_val_global, val_trajectories, SAVE_DIR, INPUT_DIM, SAMPLE_SIZES, config)
    return rmse_map

def run_full_ablation():
    global_results = {}

    for task in TASKS_LIST:
        print(f"\n>>> Starting Ablation for Task: {task} <<<")
        task_results = {}

        for ablation_type, values_list in ABLATION_STUDIES.items():
            print(f"  --- Ablation: {ablation_type} ---")
            
            # Helper to collect results for the summary plot
            current_ablation_collection = {} 
            
            for val in values_list:
                current_config = DEFAULT_CONFIG.copy()
                if ablation_type == 'layers': current_config['neurons'] = val 
                elif ablation_type == 'neurons': current_config['neurons'] = val
                else: current_config[ablation_type] = val
                
                config_name = get_config_name(ablation_type, val)
                print(f"    Running Config: {config_name}")
                
                # run_single_config now returns the DICT with errors
                run_res = run_single_config(task, current_config, config_name)
                
                if run_res:
                    # Store for JSON summary (just keep RMSE to keep JSON small)
                    rmse_only_map = {k: v['rmse'] for k, v in run_res.items()}
                    task_results[config_name] = rmse_only_map
                    
                    # Store full data (with errors) for the Plot
                    current_ablation_collection[config_name] = run_res

            # --- GENERATE SUMMARY PLOT HERE ---
            # After finishing the loop for ONE ablation type, generate the massive grid plot
            if current_ablation_collection:
                generate_ablation_summary_plot(
                    task=task,
                    ablation_type=ablation_type,
                    collected_results=current_ablation_collection,
                    values_list=values_list,
                    sample_sizes=SAMPLE_SIZES,
                    save_dir=BASE_OUTPUT_DIR
                )

        global_results[task] = task_results

    summary_path = f"{BASE_OUTPUT_DIR}/{EVAL_METHOD}_ablation_results_summary.json"
    with open(summary_path, 'w') as f: json.dump(global_results, f, indent=4)
    print(f"\nAll Ablation Studies Completed. Summary saved to {summary_path}")

if __name__ == "__main__":
    print(f"Tasks to run: {TASKS_LIST}")
    run_full_ablation()