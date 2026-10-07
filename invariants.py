"""
===============================================================

invariants.py

Kinematics and invariant computation

This module computes

1. Deformation Gradient
2. Right Cauchy Green tensor
3. Left Cauchy Green tensor
4. Isochoric tensors
5. HGO pseudo invariants
6. Fiber directions
7. Dispersion tensor
8. Jacobian

===============================================================
"""

import torch

import config
###############################################################

I = torch.eye(3,dtype=config.DTYPE,device=config.DEVICE)
###############################################################

def fiber_directions():

    """
    Two symmetric collagen families

    Returns

    a01

    a02

    """

    beta = torch.deg2rad(

        torch.tensor(

            config.BETA,

            dtype=config.DTYPE,

            device=config.DEVICE

        )

    )

    a01 = torch.tensor([

        torch.cos(beta),

        torch.sin(beta),

        0.0

    ],

    dtype=config.DTYPE,

    device=config.DEVICE)

    a02 = torch.tensor([

        torch.cos(beta),

        -torch.sin(beta),

        0.0

    ],

    dtype=config.DTYPE,

    device=config.DEVICE)

    return a01,a02
###############################################################

from kinematics import compute_kinematics
kin = compute_kinematics(
    lambda_theta,
    lambda_z
)

F = kin["F"]

C = kin["C"]

B = kin["B"]

J = kin["J"]

def isochoric_C(

        C,

        J

):

    factor = (

        J**(-2.0/3.0)

    ).view(-1,1,1)

    return factor*C
###############################################################

def I1(C):

    return torch.einsum(

        "bii->b",

        C

    ).view(-1,1)
###############################################################

def dispersion_tensor(

        a0

):

    H = (

        config.KAPPA*I

        +

        (1.0-3.0*config.KAPPA)

        *

        torch.outer(

            a0,

            a0

        )

    )

    return H
###############################################################

def pseudo_invariant(

        C_bar,

        H

):

    HC = torch.matmul(

        C_bar,

        H

    )

    I4 = torch.einsum(

        "bii->b",

        HC

    )

    return I4.view(-1,1)
###############################################################

def compute_invariants(

        lambda_theta,

        lambda_z

):

    F = deformation_gradient(

        lambda_theta,

        lambda_z

    )

    C = right_cauchy_green(F)

    B = left_cauchy_green(F)

    J = jacobian(F).view(-1,1)

    C_bar = isochoric_C(

        C,

        J

    )

    invariant1 = I1(C_bar)

    a01,a02 = fiber_directions()

    H1 = dispersion_tensor(a01)

    H2 = dispersion_tensor(a02)

    I41 = pseudo_invariant(

        C_bar,

        H1

    )

    I42 = pseudo_invariant(

        C_bar,

        H2

    )

    return {

        "F":F,

        "C":C,

        "B":B,

        "J":J,

        "C_bar":C_bar,

        "I1":invariant1,

        "I41":I41,

        "I42":I42,

        "H1":H1,

        "H2":H2

    }
    def I2(C):

    I1_value = I1(C)

    traceCC = torch.einsum(
        "bij,bji->b",
        C,
        C
    ).view(-1,1)

    return 0.5 * (

        I1_value**2

        -

        traceCC

    )
    def I3(C):

    return torch.det(C).view(-1,1)
    Cof = J * torch.inverse(F).transpose(1,2)
    eigvals = torch.linalg.eigvalsh(C)

lambda_i = torch.sqrt(eigvals)