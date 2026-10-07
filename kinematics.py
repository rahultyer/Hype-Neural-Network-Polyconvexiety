"""
=========================================================================
kinematics.py

Continuum kinematics for the Physics-Informed Constitutive Neural Network

This module computes

1. Deformation Gradient
2. Right Cauchy-Green tensor
3. Left Cauchy-Green tensor
4. Jacobian
5. Cofactor of deformation gradient
6. Isochoric deformation gradient
7. Isochoric C and B tensors
8. Green-Lagrange strain
9. Almansi strain
10. Principal stretches

Author : Rahul Kumar
IIT Madras
=========================================================================

"""

import torch

import config

# ==============================================================
# Identity Tensor
# ==============================================================

I = torch.eye(
    3,
    dtype=config.DTYPE,
    device=config.DEVICE
)

# ==============================================================
# Deformation Gradient
# ==============================================================

def deformation_gradient(
        lambda_theta,
        lambda_z,
        incompressible=True
):

    """
    Construct deformation gradient for planar biaxial loading.

    Parameters
    ----------
    lambda_theta
    lambda_z

    Returns
    -------
    F
    """

    batch = lambda_theta.shape[0]

    if incompressible:

        lambda_r = 1.0 / (lambda_theta * lambda_z)

    else:

        lambda_r = torch.ones_like(lambda_theta)

    F = torch.zeros(

        batch,

        3,

        3,

        dtype=config.DTYPE,

        device=config.DEVICE

    )

    F[:,0,0] = lambda_theta.squeeze()

    F[:,1,1] = lambda_z.squeeze()

    F[:,2,2] = lambda_r.squeeze()

    return F


# ==============================================================
# Determinant
# ==============================================================

def jacobian(F):

    return torch.det(F).view(-1,1)


# ==============================================================
# Cofactor
# ==============================================================

def cofactor(F):

    """
    Cof(F)=J F^{-T}

    """

    J = jacobian(F)

    FinvT = torch.inverse(F).transpose(1,2)

    Cof = J.view(-1,1,1) * FinvT

    return Cof


# ==============================================================
# Right Cauchy Green Tensor
# ==============================================================

def right_cauchy_green(F):

    return torch.matmul(

        F.transpose(1,2),

        F

    )


# ==============================================================
# Left Cauchy Green Tensor
# ==============================================================

def left_cauchy_green(F):

    return torch.matmul(

        F,

        F.transpose(1,2)

    )


# ==============================================================
# Isochoric deformation gradient
# ==============================================================

def isochoric_F(F):

    J = jacobian(F)

    factor = J.pow(-1.0/3.0)

    return factor.view(-1,1,1) * F


# ==============================================================
# Isochoric Right Cauchy Green
# ==============================================================

def isochoric_C(C,J):

    factor = J.pow(-2.0/3.0)

    return factor.view(-1,1,1) * C


# ==============================================================
# Isochoric Left Cauchy Green
# ==============================================================

def isochoric_B(B,J):

    factor = J.pow(-2.0/3.0)

    return factor.view(-1,1,1) * B


# ==============================================================
# Green-Lagrange strain
# ==============================================================

def green_lagrange(C):

    return 0.5 * (

        C -

        I.unsqueeze(0)

    )


# ==============================================================
# Almansi strain
# ==============================================================

def almansi(B):

    Binv = torch.inverse(B)

    return 0.5 * (

        I.unsqueeze(0)

        -

        Binv

    )


# ==============================================================
# Principal stretches
# ==============================================================

def principal_stretches(C):

    """

    λ_i = sqrt(eigenvalues(C))

    """

    eigvals = torch.linalg.eigvalsh(C)

    stretches = torch.sqrt(

        torch.clamp(

            eigvals,

            min=0.0

        )

    )

    return stretches


# ==============================================================
# Principal directions
# ==============================================================

def principal_directions(C):

    """

    Returns eigenvectors of C.

    """

    eigvals, eigvecs = torch.linalg.eigh(C)

    return eigvecs


# ==============================================================
# Polar decomposition
# ==============================================================

def right_stretch_tensor(C):

    eigvals, eigvecs = torch.linalg.eigh(C)

    Lambda = torch.diag_embed(

        torch.sqrt(

            eigvals

        )

    )

    U = eigvecs @ Lambda @ eigvecs.transpose(1,2)

    return U


# ==============================================================
# Rotation Tensor
# ==============================================================

def rotation_tensor(F):

    """

    F = R U

    """

    U = right_stretch_tensor(

        right_cauchy_green(F)

    )

    R = F @ torch.inverse(U)

    return R


# ==============================================================
# Complete kinematics
# ==============================================================

def compute_kinematics(
        lambda_theta,
        lambda_z,
        incompressible=True
):

    F = deformation_gradient(

        lambda_theta,

        lambda_z,

        incompressible

    )

    J = jacobian(F)

    Cof = cofactor(F)

    C = right_cauchy_green(F)

    B = left_cauchy_green(F)

    Fbar = isochoric_F(F)

    Cbar = isochoric_C(C,J)

    Bbar = isochoric_B(B,J)

    E = green_lagrange(C)

    e = almansi(B)

    stretches = principal_stretches(C)

    directions = principal_directions(C)

    R = rotation_tensor(F)

    return {

        "F":F,

        "J":J,

        "Cof":Cof,

        "C":C,

        "B":B,

        "Fbar":Fbar,

        "Cbar":Cbar,

        "Bbar":Bbar,

        "E":E,

        "e":e,

        "PrincipalStretches":stretches,

        "PrincipalDirections":directions,

        "Rotation":R

    }


# ==============================================================
# Unit test
# ==============================================================

if __name__ == "__main__":

    l1 = torch.tensor(
        [[1.2],[1.4]],
        dtype=config.DTYPE,
        device=config.DEVICE
    )

    l2 = torch.tensor(
        [[1.1],[1.3]],
        dtype=config.DTYPE,
        device=config.DEVICE
    )

    kin = compute_kinematics(l1,l2)

    print("="*60)

    for key,value in kin.items():

        print(key)

        print(value)

        print()