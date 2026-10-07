"""
=====================================================================

utils.py

Utility functions for the
Physics-Informed Constitutive Neural Network (PICNN)

Standalone Neural Helmholtz Free Energy Model

Author : Rahul Kumar
Institute : IIT Madras

=====================================================================
"""

import os
import random
import numpy as np
import torch

import config

###############################################################
# Random Seed
###############################################################

def set_seed(seed=config.SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

###############################################################
# Device
###############################################################

def get_device():

    return config.DEVICE

###############################################################
# Count Parameters
###############################################################

def count_parameters(model):

    return sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )

###############################################################
# Gradient Norm
###############################################################

def gradient_norm(model):

    total = 0.0

    for p in model.parameters():

        if p.grad is not None:

            total += p.grad.data.norm(2).item()**2

    return np.sqrt(total)

###############################################################
# Save Model
###############################################################

def save_model(

        model,

        optimizer,

        epoch,

        loss,

        filename

):

    checkpoint = {

        "epoch":epoch,

        "model_state_dict":

        model.state_dict(),

        "optimizer_state_dict":

        optimizer.state_dict(),

        "loss":loss

    }

    torch.save(

        checkpoint,

        filename

    )

###############################################################
# Load Model
###############################################################

def load_model(

        model,

        optimizer,

        filename

):

    checkpoint = torch.load(

        filename,

        map_location=config.DEVICE

    )

    model.load_state_dict(

        checkpoint["model_state_dict"]

    )

    optimizer.load_state_dict(

        checkpoint["optimizer_state_dict"]

    )

    epoch = checkpoint["epoch"]

    loss = checkpoint["loss"]

    return model, optimizer, epoch, loss

###############################################################
# Save Best Model
###############################################################

def save_best_model(

        model,

        optimizer,

        epoch,

        loss

):

    filename = os.path.join(

        config.MODEL_DIR,

        config.BEST_MODEL_NAME

    )

    save_model(

        model,

        optimizer,

        epoch,

        loss,

        filename

    )

###############################################################
# Save Last Model
###############################################################

def save_last_model(

        model,

        optimizer,

        epoch,

        loss

):

    filename = os.path.join(

        config.MODEL_DIR,

        config.LAST_MODEL_NAME

    )

    save_model(

        model,

        optimizer,

        epoch,

        loss,

        filename

    )

###############################################################
# Tensor to NumPy
###############################################################

def tensor_to_numpy(x):

    return (

        x

        .detach()

        .cpu()

        .numpy()

    )

###############################################################
# NumPy to Tensor
###############################################################

def numpy_to_tensor(x):

    return torch.tensor(

        x,

        dtype=config.DTYPE,

        device=config.DEVICE

    )

###############################################################
# RMSE
###############################################################

def rmse(

        prediction,

        target

):

    return torch.sqrt(

        torch.mean(

            (prediction-target)**2

        )

    )

###############################################################
# MAE
###############################################################

def mae(

        prediction,

        target

):

    return torch.mean(

        torch.abs(

            prediction-target

        )

    )

###############################################################
# Relative Error
###############################################################

def relative_error(

        prediction,

        target

):

    return torch.mean(

        torch.abs(

            prediction-target

        )

        /

        (

            torch.abs(target)

            +

            config.EPSILON

        )

    )

###############################################################
# R2 Score
###############################################################

def r2_score(

        prediction,

        target

):

    ss_res = torch.sum(

        (target-prediction)**2

    )

    ss_tot = torch.sum(

        (

            target

            -

            torch.mean(target)

        )**2

    )

    return 1.0-ss_res/ss_tot

###############################################################
# Learning Rate
###############################################################

def current_lr(

        optimizer

):

    return optimizer.param_groups[0]["lr"]

###############################################################
# Print Header
###############################################################

def print_header(title):

    print()

    print("="*70)

    print(title)

    print("="*70)

###############################################################
# Training Information
###############################################################

def print_epoch(

        epoch,

        train_loss,

        valid_loss,

        lr

):

    print(

        f"Epoch : {epoch:6d}"

        f" | Train : {train_loss:.6e}"

        f" | Valid : {valid_loss:.6e}"

        f" | LR : {lr:.3e}"

    )

###############################################################
# Physics Loss Report
###############################################################

def print_losses(

        losses

):

    print()

    print("-"*60)

    print("Loss Components")

    print("-"*60)

    for key,value in losses.items():

        print(

            f"{key:25s}"

            f"{value:.6e}"

        )

###############################################################
# Create Directory
###############################################################

def create_directories():

    folders = [

        config.DATA_DIR,

        config.MODEL_DIR,

        config.CHECKPOINT_DIR,

        config.RESULT_DIR,

        config.FIGURE_DIR,

        config.LOG_DIR

    ]

    for folder in folders:

        os.makedirs(

            folder,

            exist_ok=True

        )

###############################################################
# Save Training History
###############################################################

def save_history(

        history,

        filename

):

    np.save(

        filename,

        history

    )

###############################################################
# Load Training History
###############################################################

def load_history(

        filename

):

    return np.load(

        filename,

        allow_pickle=True

    ).item()

###############################################################
# Check NaN
###############################################################

def check_nan(tensor):

    if torch.isnan(tensor).any():

        raise RuntimeError(

            "NaN detected."

        )

###############################################################
# Check Inf
###############################################################

def check_inf(tensor):

    if torch.isinf(tensor).any():

        raise RuntimeError(

            "Inf detected."

        )

###############################################################
# Safe Clamp
###############################################################

def safe_clamp(

        x,

        minimum=1e-12

):

    return torch.clamp(

        x,

        min=minimum

    )

###############################################################
# Unit Test
###############################################################

if __name__ == "__main__":

    print_header(

        "Utility Module"

    )

    create_directories()

    set_seed()

    print("Device :", get_device())

    print("Utilities Loaded Successfully.")