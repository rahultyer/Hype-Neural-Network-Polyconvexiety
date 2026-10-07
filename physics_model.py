"""
==============================================================

main.py

Main driver for the Physics-Informed Constitutive Neural Network

Author : Rahul Kumar

==============================================================
"""

import torch

import config

from dataset import load_data

from hgo_energy import HGOMaterial

from neural_energy import EnergyNetwork

from constitutive import ConstitutiveModel

from losses import ConstitutiveLoss

from trainer import Trainer

from validation import Validator


###############################################################
# Main Program
###############################################################

def main():

    print("="*60)
    print("Physics-Informed Constitutive Neural Network")
    print("="*60)

    ###########################################################
    # Load Experimental Data
    ###########################################################

    train_loader, valid_loader, test_loader = load_data(
        "data/abdominal_aorta.csv"
    )

    print("Dataset Loaded")

    ###########################################################
    # Material Model
    ###########################################################

    material = HGOMaterial()

    print("Analytical constitutive model created")

    ###########################################################
    # Neural Energy Model
    ###########################################################

    neural_model = EnergyNetwork().to(config.DEVICE)

    print("Neural energy model created")

    ###########################################################
    # Constitutive Model
    ###########################################################

    constitutive = ConstitutiveModel(
        material,
        neural_model
    )

    ###########################################################
    # Loss Function
    ###########################################################

    loss_function = ConstitutiveLoss()

    ###########################################################
    # Trainer
    ###########################################################

    trainer = Trainer(
        model=neural_model,
        constitutive=constitutive,
        loss_function=loss_function,
        train_loader=train_loader,
        validation_loader=valid_loader
    )

    ###########################################################
    # Training
    ###########################################################

    trainer.fit()

    ###########################################################
    # Validation
    ###########################################################

    validator = Validator(constitutive)

    print("\nRunning validation...\n")

    # Example prediction loop
    for batch in test_loader:

        result = validator.predict(
            batch["lambda_theta"],
            batch["lambda_z"]
        )

        sigma_pred = result["sigma"]

        break

    ###########################################################
    # Save Network
    ###########################################################

    torch.save(
        neural_model.state_dict(),
        "saved_models/picnn_model.pt"
    )

    print("Model Saved")

    ###########################################################
    print("Finished Successfully")
    ###########################################################


###############################################################

if __name__ == "__main__":

    main()