"""
=========================================================================

dataset.py

Dataset loader for the Physics-Informed Constitutive Neural Network
(PICNN)

Standalone Neural Helmholtz Free Energy Model

Author : Rahul Kumar
Institute : IIT Madras

=========================================================================

Experimental CSV format
-----------------------

lambda_theta
lambda_z
sigma_theta
sigma_z

(Optional)

pressure

=========================================================================

"""

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import config


##########################################################################
# Dataset
##########################################################################

class BiaxialDataset(Dataset):

    """
    Dataset for planar biaxial tensile test.
    """

    def __init__(self, dataframe):

        self.lambda_theta = torch.tensor(
            dataframe["lambda_theta"].values,
            dtype=config.DTYPE,
            device=config.DEVICE
        ).view(-1,1)

        self.lambda_z = torch.tensor(
            dataframe["lambda_z"].values,
            dtype=config.DTYPE,
            device=config.DEVICE
        ).view(-1,1)

        self.sigma_theta = torch.tensor(
            dataframe["sigma_theta"].values,
            dtype=config.DTYPE,
            device=config.DEVICE
        ).view(-1,1)

        self.sigma_z = torch.tensor(
            dataframe["sigma_z"].values,
            dtype=config.DTYPE,
            device=config.DEVICE
        ).view(-1,1)

        if "pressure" in dataframe.columns:

            self.pressure = torch.tensor(
                dataframe["pressure"].values,
                dtype=config.DTYPE,
                device=config.DEVICE
            ).view(-1,1)

        else:

            self.pressure = torch.zeros_like(self.lambda_theta)

    ######################################################################

    def __len__(self):

        return len(self.lambda_theta)

    ######################################################################

    def __getitem__(self,index):

        sample = {

            "lambda_theta": self.lambda_theta[index],

            "lambda_z": self.lambda_z[index],

            "sigma_theta": self.sigma_theta[index],

            "sigma_z": self.sigma_z[index],

            "pressure": self.pressure[index]

        }

        return sample


##########################################################################
# Read csv
##########################################################################

def read_csv(file_name):

    data = pd.read_csv(file_name)

    required = [

        "lambda_theta",

        "lambda_z",

        "sigma_theta",

        "sigma_z"

    ]

    for item in required:

        if item not in data.columns:

            raise ValueError(f"Missing column : {item}")

    return data


##########################################################################
# Normalize (optional)
##########################################################################

def normalize_dataframe(df):

    return df.copy()


##########################################################################
# Split
##########################################################################

def split_dataframe(df):

    train_df, temp_df = train_test_split(

        df,

        train_size=config.TRAIN_RATIO,

        shuffle=True,

        random_state=config.SEED

    )

    validation_fraction = (

        config.VALID_RATIO

        /

        (

            config.VALID_RATIO

            +

            config.TEST_RATIO

        )

    )

    valid_df, test_df = train_test_split(

        temp_df,

        train_size=validation_fraction,

        shuffle=True,

        random_state=config.SEED

    )

    return train_df, valid_df, test_df


##########################################################################
# Build dataset
##########################################################################

def build_dataset(df):

    return BiaxialDataset(df)


##########################################################################
# DataLoader
##########################################################################

def build_loader(dataset, shuffle):

    return DataLoader(

        dataset,

        batch_size=config.BATCH_SIZE,

        shuffle=shuffle,

        drop_last=False

    )


##########################################################################
# Complete pipeline
##########################################################################

def load_data(file_name=None):

    if file_name is None:

        file_name = config.DATA_FILE

    dataframe = read_csv(file_name)

    dataframe = normalize_dataframe(dataframe)

    train_df, valid_df, test_df = split_dataframe(dataframe)

    train_dataset = build_dataset(train_df)

    valid_dataset = build_dataset(valid_df)

    test_dataset = build_dataset(test_df)

    train_loader = build_loader(train_dataset, True)

    valid_loader = build_loader(valid_dataset, False)

    test_loader = build_loader(test_dataset, False)

    return (

        train_loader,

        valid_loader,

        test_loader

    )


##########################################################################
# Dataset information
##########################################################################

def dataset_summary(file_name=None):

    if file_name is None:

        file_name = config.DATA_FILE

    df = read_csv(file_name)

    print("="*60)

    print("Dataset Summary")

    print("="*60)

    print("Samples :", len(df))

    print(df.describe())

    print("="*60)


##########################################################################
# Quick test
##########################################################################

if __name__ == "__main__":

    dataset_summary()

    train_loader, valid_loader, test_loader = load_data()

    batch = next(iter(train_loader))

    print(batch.keys())

    print(batch["lambda_theta"].shape)