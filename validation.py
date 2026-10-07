"""
===============================================================

validation.py

Validation utilities for the Physics-Informed Constitutive Neural Network

===============================================================
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
class Validator:

    def __init__(self,
                 constitutive):

        self.constitutive = constitutive
            def predict(self,
                lambda_theta,
                lambda_z):

        inv = compute_invariants(
            lambda_theta,
            lambda_z
        )

        F = inv["F"]

        F.requires_grad_(True)

        inv["F"] = F

        result = self.constitutive.evaluate(
            F,
            inv
        )

        return result
            def rmse(self,
             prediction,
             target):

        return np.sqrt(

            mean_squared_error(

                target,

                prediction

            )

        )
            def mae(self,
            prediction,
            target):

        return mean_absolute_error(

            target,

            prediction

        )
            def r2(self,
           prediction,
           target):

        return r2_score(

            target,

            prediction

        )
            def plot_stress_theta(
            self,
            stretch,
            sigma_exp,
            sigma_pred):
                        plt.figure(figsize=(6,5))

        plt.plot(

            stretch,

            sigma_exp,

            'ko',

            label="Experiment"

        )

        plt.plot(

            stretch,

            sigma_pred,

            'r-',

            linewidth=2,

            label="PICNN"

        )

        plt.xlabel(r'$\lambda_\theta$')

        plt.ylabel(r'$\sigma_\theta$')

        plt.legend()

        plt.grid(True)

        plt.tight_layout()
            def parity_plot(
            self,
            target,
            prediction):
                        plt.figure(figsize=(5,5))

        plt.scatter(

            target,

            prediction

        )

        minimum = min(

            target.min(),

            prediction.min()

        )

        maximum = max(

            target.max(),

            prediction.max()

        )

        plt.plot(

            [minimum,maximum],

            [minimum,maximum],

            'k--'

        )

        plt.xlabel("Experimental")

        plt.ylabel("Prediction")

        plt.axis("equal")

        plt.grid(True)
            def energy_plot(
            self,
            stretch,
            psi):
                        plt.figure()

        plt.plot(

            stretch,

            psi,

            linewidth=2

        )

        plt.xlabel("Stretch")

        plt.ylabel("Energy")

        plt.grid(True)
            def summary(
            self,
            sigma_exp,
            sigma_pred):
                        rmse = self.rmse(

            sigma_pred,

            sigma_exp

        )

        mae = self.mae(

            sigma_pred,

            sigma_exp

        )

        r2 = self.r2(

            sigma_pred,

            sigma_exp

        )

        print("--------------------------------")

        print("Validation Summary")

        print("--------------------------------")

        print("RMSE :",rmse)

        print("MAE  :",mae)

        print("R²   :",r2)

        print("--------------------------------")