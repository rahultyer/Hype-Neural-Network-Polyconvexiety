"""
=====================================================================

constitutive.py

Constitutive equations for the
Physics-Informed Constitutive Neural Network

This module computes

1. Helmholtz free energy
2. PK1 stress
3. PK2 stress
4. Kirchhoff stress
5. Cauchy stress

Author : Rahul Kumar
IIT Madras

=====================================================================
"""

import torch

import config

from invariants import compute_invariants

#####################################################################
# Helmholtz Energy
#####################################################################

def helmholtz_energy(

        model,

        lambda_theta,

        lambda_z

):

    """
    Returns

    Psi

    """

    inv = compute_invariants(

        lambda_theta,

        lambda_z

    )

    psi = model(

        inv["I1"],

        inv["I2"],

        inv["J"],

        inv["I41"],

        inv["I42"]

    )

    return psi,inv


#####################################################################
# PK1 Stress
#####################################################################

def first_piola(

        model,

        lambda_theta,

        lambda_z

):

    """
    First Piola-Kirchhoff stress

    P = dPsi/dF

    """

    psi,inv = helmholtz_energy(

        model,

        lambda_theta,

        lambda_z

    )

    F = inv["F"]

    P = torch.autograd.grad(

        outputs=psi.sum(),

        inputs=F,

        create_graph=True,

        retain_graph=True

    )[0]

    return P,psi,inv


#####################################################################
# PK2 Stress
#####################################################################

def second_piola(

        model,

        lambda_theta,

        lambda_z

):

    """
    Second Piola-Kirchhoff Stress

    S = 2 dPsi/dC

    """

    psi,inv = helmholtz_energy(

        model,

        lambda_theta,

        lambda_z

    )

    C = inv["C"]

    grad = torch.autograd.grad(

        outputs=psi.sum(),

        inputs=C,

        create_graph=True,

        retain_graph=True

    )[0]

    S = 2.0*grad

    return S,psi,inv


#####################################################################
# Kirchhoff Stress
#####################################################################

def kirchhoff_stress(

        model,

        lambda_theta,

        lambda_z

):

    """
    tau = P F^T

    """

    P,psi,inv = first_piola(

        model,

        lambda_theta,

        lambda_z

    )

    F = inv["F"]

    tau = torch.matmul(

        P,

        F.transpose(1,2)

    )

    return tau,psi,inv


#####################################################################
# Cauchy Stress
#####################################################################

def cauchy_stress(

        model,

        lambda_theta,

        lambda_z

):

    """
    sigma = (1/J) P F^T

    """

    tau,psi,inv = kirchhoff_stress(

        model,

        lambda_theta,

        lambda_z

    )

    sigma = tau / inv["J"].view(-1,1,1)

    return sigma,psi,inv


#####################################################################
# Stress Components
#####################################################################

def stress_components(

        sigma

):

    """
    Returns

    sigma11

    sigma22

    sigma33

    sigma12

    sigma13

    sigma23

    """

    return {

        "sigma11":sigma[:,0,0].view(-1,1),

        "sigma22":sigma[:,1,1].view(-1,1),

        "sigma33":sigma[:,2,2].view(-1,1),

        "sigma12":sigma[:,0,1].view(-1,1),

        "sigma13":sigma[:,0,2].view(-1,1),

        "sigma23":sigma[:,1,2].view(-1,1)

    }


#####################################################################
# Hydrostatic Pressure
#####################################################################

def pressure(

        sigma

):

    """

    p = -tr(sigma)/3

    """

    tr = (

        sigma[:,0,0]

        +

        sigma[:,1,1]

        +

        sigma[:,2,2]

    ).view(-1,1)

    return -tr/3.0


#####################################################################
# Deviatoric Stress
#####################################################################

def deviatoric_stress(

        sigma

):

    """

    s = sigma + pI

    """

    p = pressure(

        sigma

    )

    I = torch.eye(

        3,

        dtype=config.DTYPE,

        device=config.DEVICE

    )

    return sigma + p.view(-1,1,1)*I


#####################################################################
# Von Mises Stress
#####################################################################

def von_mises(

        sigma

):

    """

    Von Mises equivalent stress

    """

    s = deviatoric_stress(

        sigma

    )

    J2 = 0.5*torch.sum(

        s*s,

        dim=(1,2)

    ).view(-1,1)

    return torch.sqrt(

        3.0*J2

    )


#####################################################################
# Unit Test
#####################################################################

if __name__ == "__main__":

    from neural_energy import build_model

    model = build_model()

    lam1 = torch.tensor(

        [[1.2]],

        dtype=config.DTYPE,

        device=config.DEVICE,

        requires_grad=True

    )

    lam2 = torch.tensor(

        [[1.1]],

        dtype=config.DTYPE,

        device=config.DEVICE,

        requires_grad=True

    )

    sigma,psi,inv = cauchy_stress(

        model,

        lam1,

        lam2

    )

    print("="*60)

    print("Helmholtz Energy")

    print("="*60)

    print(psi)

    print()

    print("="*60)

    print("Cauchy Stress")

    print("="*60)

    print(sigma)

    print()

    print("="*60)

    print("Von Mises")

    print("="*60)

    print(von_mises(sigma))