Closed-form constitutive models remain the dominant tool for
describing the mechanical behavior of collagenous soft tissues, yet
the most widely used of them, the Gasser-Ogden-Holzapfel (GOH)
model, pre-averages the fiber distribution into an algebraic
structure tensor and applies a Heaviside tension-compression switch
to suppress the contribution of compressed fibers. This is known to
introduce stress discontinuities, non-physical duplicated
interpretations of fiber stretch, and a spurious "perversion point"
under combined loading. In this work we build a physics-informed
constitutive neural network that removes this discontinuity at its
root by replacing the pre-averaged structure-tensor invariants with
the vanishing matched invariant E(N), evaluated individually
for every fiber direction N on the unit sphere and only
averaged after the (nonlinear) fiber energy has been applied
(a microsphere, or angular-integration, formulation). The individual
fiber energy law, classically an analytical two-parameter
exponential, is replaced by a small monotonic, non-negative neural
network, trained jointly with a separate polyconvex input-convex
neural network (ICNN) that corrects the isotropic ground-matrix
response. The full model - an analytical volumetric term, an
analytical neo-Hookean isotropic term, the (partly neural) VanAngI
anisotropic fiber term, and the polyconvex ICNN correction - is
trained end-to-end against planar biaxial stress-stretch data using
a physics-informed loss that enforces the stress data fit together
with linear momentum balance, near-incompressibility, the
Baker-Ericksen inequality, and Coleman-Noll thermodynamic
consistency; angular momentum balance (stress symmetry) holds
identically by construction because the energy is built exclusively
from objective invariants of the right Cauchy-Green tensor. We
report the resulting fit quality (RMSE, MAE, R^2) against
biaxial data. To make the trained model usable in a commercial
finite element package without a custom user-material subroutine, we
further develop a symbolic unrolling procedure that expresses both
trained neural networks - weights, biases, and activation functions
included - as closed-form algebraic expressions in the components of
the deformation gradient, and use this to deploy the full trained
free-energy functional as a native "User Defined" hyperelastic
material in COMSOL Multiphysics. Three-dimensional finite element
simulations of uniaxial extension, simple shear, torsion, and
\todo{tissue expansion / your target boundary value problem}
reproduce the trained constitutive behavior to within numerical
precision, demonstrating a complete pipeline from sparse experimental
biaxial data to a deployable, physically admissible, three-dimensional
finite element material model.
