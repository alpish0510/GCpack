# Galaxy Cluster Analysis Toolkit

## Overview

This repository provides a computational framework for the analysis of
galaxy clusters through X-ray surface brightness modeling, statistical
inference, and redshift distribution studies. The library integrates
physically motivated parametric models with observational data
processing utilities, enabling both forward modeling and inverse
parameter estimation within a consistent and reproducible workflow.

The toolkit is designed to support high-energy astrophysical analyses,
particularly in the context of intracluster medium (ICM) studies, where
accurate characterization of surface brightness profiles and associated
uncertainties is critical.

------------------------------------------------------------------------

## Methodology

### Surface Brightness Modeling

The library implements commonly used parametric descriptions of X-ray
surface brightness profiles, including:

-   The single β-model\
-   The double β-model\
-   The Vikhlinin (2006) profile

These models capture both core structure and outer slope behavior,
allowing flexible modeling of relaxed and disturbed clusters.

### Mock Data Generation

Synthetic surface brightness profiles are generated using physically
motivated parameter distributions. Poisson noise and background
contributions (e.g., CXB) are incorporated to emulate realistic
observational conditions, enabling validation of fitting procedures and
statistical pipelines.

### Parameter Estimation

Model parameters are inferred via non-linear least squares optimization
using `scipy.optimize.curve_fit`. Full uncertainty propagation is
supported through the `uncertainties` framework, ensuring consistent
treatment of measurement errors.

### Derived Quantities

Under the assumption of hydrostatic equilibrium and isothermality, the
toolkit provides routines for estimating cluster mass profiles from
fitted surface brightness parameters.

### Redshift Analysis

Statistical tools are included for the analysis of galaxy redshift
distributions, including iterative σ-clipping and adaptive filtering
techniques. Three-dimensional visualization in (RA, Dec, z) space is
supported for structural analysis.

### Image-Level Modeling

The framework includes utilities for generating two-dimensional model
realizations and synthetic cluster images with substructure,
facilitating comparisons with observational data products.

------------------------------------------------------------------------

## Software Architecture

The codebase is organized into modular components:

-   **Model functions**: Analytical surface brightness profiles\
-   **Simulation utilities**: Mock data generation\
-   **Fitting routines**: Parameter estimation and covariance
    evaluation\
-   **Analysis classes**: Surface brightness, mass, and redshift
    analysis\
-   **Visualization tools**: Publication-quality plots and 3D
    representations

This modular design ensures extensibility and transparency, with clear
separation between modeling, inference, and diagnostics.

------------------------------------------------------------------------

## Dependencies

The following Python packages are required:

-   numpy\
-   scipy\
-   matplotlib\
-   uncertainties\
-   pandas\
-   astropy\
-   plotly

------------------------------------------------------------------------

## Applications

This toolkit is suitable for:

-   X-ray analysis of galaxy clusters\
-   Validation of surface brightness models\
-   Simulation-based inference studies\
-   Cluster mass estimation under hydrostatic assumptions\
-   Redshift-based structure identification

------------------------------------------------------------------------

## License

This project is distributed under the MIT License.
