# GCpack --- Galaxy Cluster Analysis Toolkit

## Overview

GCpack is a Python-based computational framework for the analysis of
galaxy clusters, with a primary focus on X-ray surface brightness
modeling and statistical inference of intracluster medium (ICM)
properties. The package provides a unified environment for forward
modeling, parameter estimation, mock data generation, and observational
data analysis.

It is designed to support both simulation-driven studies and
observational pipelines, emphasizing transparency, modularity, and
physically motivated modeling.

------------------------------------------------------------------------

## Scientific Scope

Galaxy cluster studies rely critically on accurate characterization of
X-ray surface brightness profiles, which trace the thermodynamic
structure of the ICM. GCpack implements widely used parametric models
and analysis tools to:

-   Infer structural parameters of galaxy clusters
-   Quantify observational uncertainties
-   Generate realistic mock datasets
-   Evaluate systematic effects in profile fitting
-   Derive physically meaningful quantities such as mass profiles

------------------------------------------------------------------------

## Core Methodology

### Surface Brightness Modeling

GCpack implements standard parametric descriptions of X-ray surface
brightness:

-   β-model
-   Double β-model
-   Vikhlinin (2006) profile

------------------------------------------------------------------------

### Mock Data Generation

-   Physically motivated parameter sampling
-   Radial binning consistent with observational analyses
-   Poisson noise injection
-   Inclusion of cosmic X-ray background (CXB)

------------------------------------------------------------------------

### Parameter Estimation

Model fitting is performed via non-linear least squares using
scipy.optimize.curve_fit with covariance estimation.

------------------------------------------------------------------------

### Uncertainty Propagation

Uncertainties are propagated using the uncertainties framework.

------------------------------------------------------------------------

### Derived Physical Quantities

Mass profiles are estimated under hydrostatic equilibrium and isothermal
assumptions.

------------------------------------------------------------------------

### Redshift Distribution Analysis

-   Iterative σ-clipping
-   Adaptive clipping
-   MAD-based statistics
-   3D visualization in (RA, Dec, z)

------------------------------------------------------------------------

### Image-Level Modeling

-   Substructured cluster simulations
-   2D surface brightness maps

------------------------------------------------------------------------

## Dependencies

-   numpy
-   scipy
-   matplotlib
-   uncertainties
-   pandas
-   astropy
-   plotly

------------------------------------------------------------------------

## Example Usage

``` python
from gcpack import mock_profiles, fitter
import numpy as np

profiles, r, bgsub = mock_profiles(10, exp_t=20, num_bins=30)

bestfit, params, cov = fitter(
    r[0],
    profiles[0],
    yerr=np.sqrt(profiles[0]),
    P0=[1e-5, 200, 0.6],
    profile_type="beta"
)
```

------------------------------------------------------------------------

## Author

Alpish Srivastava\
PhD Candidate in Astrophysics\
University of Rome Tor Vergata
