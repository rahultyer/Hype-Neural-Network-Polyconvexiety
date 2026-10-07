"""
===============================================================

hgo_energy.py

Holzapfel-Gasser-Ogden constitutive model

Reference:

Holzapfel, Gasser & Ogden (2000)

===============================================================
"""

import torch

import config
###############################################################

class HGOMaterial:
    ###############################################################

    def __init__(self):

        self.mu = config.MU

        self.k1 = config.K1

        self.k2 = config.K2

        self.bulk = config.BULK
        ###############################################################

    def isotropic_energy(

            self,

            I1

    ):

        psi = (

            self.mu/2.0

        )*(

            I1-3.0

        )

        return psi
        ###############################################################

    def volumetric_energy(

            self,

            J

    ):

        psi = (

            self.bulk/2.0

        )*(

            J-1.0

        )**2

        return psi
        ###############################################################

    def fiber_energy(

            self,

            I41,

            I42

    ):

        E1 = torch.relu(

            I41-1.0

        )

        E2 = torch.relu(

            I42-1.0

        )

        psi1 = (

            self.k1/

            (2.0*self.k2)

        )*(

            torch.exp(

                self.k2*

                E1**2

            )-1.0

        )

        psi2 = (

            self.k1/

            (2.0*self.k2)

        )*(

            torch.exp(

                self.k2*

                E2**2

            )-1.0

        )

        return psi1+psi2
        ###############################################################

    def energy(

            self,

            invariants

    ):

        psi_iso = self.isotropic_energy(

            invariants["I1"]

        )

        psi_vol = self.volumetric_energy(

            invariants["J"]

        )

        psi_fib = self.fiber_energy(

            invariants["I41"],

            invariants["I42"]

        )

        psi = (

            psi_iso

            +

            psi_vol

            +

            psi_fib

        )

        return psi