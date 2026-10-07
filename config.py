"""
=====================================================================
config.py

Physics-Informed Constitutive Neural Network (PICNN)
for Hyperelastic Arterial Tissue

Standalone Neural Helmholtz Free Energy Model

Author : Rahul Kumar
Institute : IIT Madras

=====================================================================
"""

import os
import random
import numpy as np
import torch

# =====================================================================
# Random Seed
# =====================================================================

SEED = 12345

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# =====================================================================
# Device
# =====================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DTYPE = torch.float64

torch.set_default_dtype(DTYPE)

# =====================================================================
# Directories
# =====================================================================

PROJECT_DIR = os.getcwd()

DATA_DIR = os.path.join(PROJECT_DIR, "data")

MODEL_DIR = os.path.join(PROJECT_DIR, "saved_models")

CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")

RESULT_DIR = os.path.join(PROJECT_DIR, "results")

FIGURE_DIR = os.path.join(PROJECT_DIR, "figures")

LOG_DIR = os.path.join(PROJECT_DIR, "logs")

for folder in [
    DATA_DIR,
    MODEL_DIR,
    CHECKPOINT_DIR,
    RESULT_DIR,
    FIGURE_DIR,
    LOG_DIR,
]:
    os.makedirs(folder, exist_ok=True)

# =====================================================================
# Experimental Data
# =====================================================================

DATA_FILE = os.path.join(
    DATA_DIR,
    "abdominal_aorta.csv"
)

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# =====================================================================
# Material Information
# =====================================================================

DIM = 3

N_FIBERS = 2

INCOMPRESSIBLE = True

# Fiber angle (degrees)

BETA = 29.0

# Fiber dispersion parameter
# Used only for computing generalized structural tensors

KAPPA = 0.226

# =====================================================================
# Network Architecture
# =====================================================================

# Inputs:
#
# I1_bar
# I2_bar
# J
# I41_bar
# I42_bar

INPUT_DIM = 5

OUTPUT_DIM = 1

HIDDEN_LAYERS = [

    128,

    128,

    128,

    128,

    64,

    64,

    32

]

ACTIVATION = "Softplus"

SOFTPLUS_BETA = 40.0

DROPOUT = 0.0

BIAS = True

# =====================================================================
# Optimizer
# =====================================================================

OPTIMIZER = "Adam"

LEARNING_RATE = 1.0e-3

WEIGHT_DECAY = 1.0e-8

BETAS = (0.9, 0.999)

EPS = 1e-8

# =====================================================================
# Learning Rate Scheduler
# =====================================================================

USE_SCHEDULER = True

LR_DECAY = 0.95

LR_STEP = 500

MIN_LR = 1e-6

# =====================================================================
# LBFGS Fine Tuning
# =====================================================================

USE_LBFGS = True

LBFGS_MAX_ITER = 500

LBFGS_HISTORY = 100

# =====================================================================
# Training
# =====================================================================

EPOCHS = 5000

BATCH_SIZE = 64

PRINT_EVERY = 10

SAVE_EVERY = 100

VALIDATE_EVERY = 20

# =====================================================================
# Physics Loss Weights
# =====================================================================

#
# Total loss
#
# L =
# w_data * L_data
# + w_momentum * L_momentum
# + w_angular * L_angular
# + w_reference * L_reference
# + w_energy * L_energy
# + w_BE * L_BE
# + w_poly * L_poly
# + w_stability * L_stability
# + w_regularization * L_reg
#

W_DATA = 1.0

W_MOMENTUM = 1.0

W_ANGULAR = 1.0

W_REFERENCE = 10.0

W_ENERGY = 1.0

W_BE = 1.0

W_POLY = 2.0

W_STABILITY = 2.0

W_REGULARIZATION = 1.0e-8

# =====================================================================
# Numerical Tolerances
# =====================================================================

EPSILON = 1e-12

TOL = 1e-8

# =====================================================================
# Automatic Differentiation
# =====================================================================

CREATE_GRAPH = True

RETAIN_GRAPH = True

# =====================================================================
# Plotting
# =====================================================================

FIG_DPI = 300

FIG_FORMAT = "png"

FONT_SIZE = 14

LINE_WIDTH = 2.5

MARKER_SIZE = 7

# =====================================================================
# Checkpoints
# =====================================================================

BEST_MODEL_NAME = "best_model.pt"

LAST_MODEL_NAME = "last_model.pt"

CHECKPOINT_NAME = "checkpoint.pt"

# =====================================================================
# Validation Metrics
# =====================================================================

VALIDATION_METRICS = [

    "RMSE",

    "MAE",

    "MAPE",

    "R2",

]

# =====================================================================
# Reference Configuration
# =====================================================================

REFERENCE_STRETCH = 1.0

REFERENCE_J = 1.0

REFERENCE_ENERGY = 0.0

REFERENCE_STRESS = 0.0

# =====================================================================
# Physics Flags
# =====================================================================

ENFORCE_OBJECTIVITY = True

ENFORCE_MOMENTUM = True

ENFORCE_ANGULAR_MOMENTUM = True

ENFORCE_BAKER_ERICKSEN = True

ENFORCE_POLYCONVEXITY = True

ENFORCE_REFERENCE_STATE = True

ENFORCE_ENERGY_POSITIVITY = True

ENFORCE_STABILITY = True

ENFORCE_MONOTONICITY = True

# =====================================================================
# End
# =====================================================================