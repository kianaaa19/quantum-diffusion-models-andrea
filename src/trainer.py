# src.trainer.py
import os
import itertools
from datetime import datetime
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from src.utils.circuit_training import training

# Define the list of possible values for each hyperparameter to search over
hyperparameters = {
    "model_type":        ["PQC"],  # choose between 'NNCPQC' and 'PQC'
    "num_layers":        [20],  # unitary+activation blocks for NNCPQC
    "PQC_LR":            [1e-4, 5e-4],  # CHANGED: increased from 5e-5
    "MLP_LR":            [1e-4, 5e-4],  # CHANGED: increased from 5e-5
    "batch_size":        [64],
    "num_epochs":        [2000],
    "scheduler_patience":[10],  # after how many epochs to reduce the learning rate
    "scheduler_gamma":   [0.97],  # factor by which to reduce the learning rate
    "T":                 [10],  # number of time steps in the diffusion process
    "num_qubits":        [8],  # 2^8 = 256 amplitudes (16x16 Fashion-MNIST patches)
    "beta_0":            [1e-2],  # DDPM parameter
    "beta_T":            [1e-2],  # DDPM parameter
    "schedule":          ['linear'], # noise schedule type: 'linear' or 'cosine'
    "schedule_exponent": [0.5],  # exponent for the cosine schedule
    "init_variance":     [0.05, 0.1, 0.2],  # CHANGED: reduced from 0.7 to avoid barren plateaus
    "wd_PQC":            [1e-5],  # L2 regularization
    "wd_MLP":            [1e-4],
    "digits":            [[0, 1]],  # Fashion-MNIST classes: 0=T-shirt, 1=Trouser
    "inference_noise":   [0.005],  # noise to add to the parameters during inference
    "load_epoch":        [None],  # checkpoint epoch (PQC only supports epoch checkpoints)
    "activation":        [True],  # whether to use activation function
    "MLP_width":         [64],  # Width of the MLP (NNCPQC only)
    "MLP_depth":         [5],  # Depth of the MLP (NNCPQC only)
    "PQC_depth":         [4], # Depth of the PQC (NNCPQC only)
    "ACT_depth":         [4], # Depth of the activation block (NNCPQC only)
    "num_ancilla":       [1],  # Number of ancilla qubits (NNCPQC only)
    "checkpoint":        [None], # specify a path to Params as '/path/Params/' to use that checkpoint
    "pqc_layers":        [[2, 2, 2]], # depths for the three PQC blocks when model_type='PQC'
    
    # ==================== QEM PARAMETERS ====================
    "use_qem":           [True],  # Enable Quantum Error Mitigation
    "qem_method":        ['readout'],  # 'readout', 'zne', or 'both'
    "qem_calibration_shots": [1000],  # Number of shots for calibration (if applicable)
    
    # ==================== QNGD PARAMETERS ====================
    "use_qngd":          [True],  # Enable Quantum Natural Gradient Descent
    "qngd_mode":         ['hybrid'],  # 'full', 'hybrid', or 'off'
    "qngd_regularization": [1e-4],  # Regularization for QFIM inversion (numerical stability)
    "qngd_update_frequency": [3, 5],  # CHANGED: more frequent QFIM updates
    "qngd_block_diag":   [True],  # Use block-diagonal approximation for QFIM (faster)
    
    # ==================== EARLY STOPPING (NEW) ====================
    "use_early_stopping": [True],  # Enable early stopping
    "early_stop_patience": [100],  # Stop after 100 epochs without improvement
    "early_stop_min_delta": [1e-4],  # Minimum improvement threshold
    
    # ==================== LOCAL LOSS (NEW - CRITICAL) ====================
    "use_local_loss":    [True],  # Use local Pauli-Z loss to avoid barren plateaus
}

DATA_LENGTH = 8096  # number of samples in the dataset
NUM_TRIALS = 2  # number of trials for each hyperparameter combination
results_dir = 'results'

# ================> DON'T MODIFY BELOW THIS LINE<================
# create path to save weights and results
timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(os.getcwd(), results_dir, timestr)

# Generate all combinations of hyperparameters
all_combinations = list(itertools.product(*hyperparameters.values()))

# Loop over all possible combinations of hyperparameters
for i, values in enumerate(all_combinations, start=1):
    for trial in range(1, NUM_TRIALS+1):
        
        # Construct the path
        path = os.path.join(log_dir, f"run_{i}", f"trial_{trial}")
        
        # Ensure the path exists
        os.makedirs(path, exist_ok=True)
        
        # Execute training
        training(path, values, DATA_LENGTH)
