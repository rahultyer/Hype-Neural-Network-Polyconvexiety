"""
===============================================================

losses.py

Loss functions for constitutive neural network

===============================================================
"""

import torch

import torch.nn as nn

import config
###############################################################

class ConstitutiveLoss:

    def __init__(self):

        self.mse = nn.MSELoss()
        ###############################################################

    def stress_loss(

            self,

            sigma_pred,

            sigma_exp

    ):
                loss = self.mse(

            sigma_pred,

            sigma_exp

        )

        return loss
        ###############################################################

    def energy_loss(

            self,

            psi

    ):
                penalty = torch.relu(

            -psi

        )

        return torch.mean(

            penalty**2

        )
        ###############################################################

    def reference_loss(

            self,

            psi_ref

    ):
                return torch.mean(

            psi_ref**2

        )
        ###############################################################

    def regularization(

            self,

            network

    ):
                reg = 0.0

        for parameter in network.parameters():

            reg += torch.sum(

                parameter**2

            )

        return reg
        ###############################################################

    def fiber_penalty(

            self,

            I41,

            I42,

            psi_fib

    ):
                compression = (

            (I41<1.0)

            |

            (I42<1.0)

        )

        return torch.mean(

            compression*psi_fib**2

        )
        ###############################################################

    def total_loss(

            self,

            sigma_pred,

            sigma_exp,

            psi,

            psi_ref,

            network,

            I41,

            I42,

            psi_fib

    ):
                loss_sigma = self.stress_loss(

            sigma_pred,

            sigma_exp

        )

        loss_energy = self.energy_loss(

            psi

        )

        loss_reference = self.reference_loss(

            psi_ref

        )

        loss_reg = self.regularization(

            network

        )

        loss_fiber = self.fiber_penalty(

            I41,

            I42,

            psi_fib

        )

        total = (

            config.W_DATA

            *loss_sigma

            +

            loss_energy

            +

            loss_reference

            +

            config.W_REG

            *loss_reg

            +

            loss_fiber

        )

        return total