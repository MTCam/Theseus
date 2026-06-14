# Theseus

[![CI Quick](https://github.com/MTCam/Theseus/actions/workflows/ci-quick.yml/badge.svg)](https://github.com/MTCam/Theseus/actions/workflows/ci-quick.yml)
[![CI Nightly](https://github.com/MTCam/Theseus/actions/workflows/ci-nightly.yml/badge.svg)](https://github.com/MTCam/Theseus/actions/workflows/ci-nightly.yml)
[![Doxygen Docs](https://github.com/MTCam/Theseus/actions/workflows/doxygen.yml/badge.svg)](https://github.com/MTCam/Theseus/actions/workflows/doxygen.yml)

Theseus is a compressible flow solver developed at the Center for Hypersonics and Entry Systems Studies (CHESS) at the University of Illinois [1]. Its formulation and numerics are based on the work of Hasanli [2]. The code uses the MFEM finite element library [3], and targets modern and emerging HPC architectures for execution.

Theseus solves the compressible Navier-Stokes equations, written in conservative form as

$$
\frac{\partial \mathbf{Q}}{\partial t} + \nabla \cdot \left(\mathbf{F}^I - \mathbf{F}^V\right) = \mathbf{S}
$$

For a single component fluid, the state vector $\mathbf{Q}$, inviscid flux $\mathbf{F}^I$, and viscous flux $\mathbf{F}^V$ are

$$
\begin{bmatrix}
\rho\\
\rho E\\
\rho \vec{v}
\end{bmatrix},
\quad
\begin{bmatrix}
\rho\vec{v}\\
(\rho E + p)\vec{v}\\
\rho(\vec{v} \otimes \vec{v}) + p\delta_{ij}
\end{bmatrix},
\quad
\begin{bmatrix}
0\\
(\boldsymbol{\tau} \cdot \vec{v} - \mathbf{q})\\
\boldsymbol{\tau}
\end{bmatrix}.
$$

where $(\rho, \vec{v}, E)$ are the density, velocity, and total energy of the fluid, respectively. The viscous stress tensor $\boldsymbol{\tau}$ and heat flux $\mathbf{q}$ are

$$
\boldsymbol{\tau} = \mu\left[\left(\nabla\vec{v} + (\nabla\vec{v})^T\right) - \frac{2}{3}(\nabla \cdot \vec{v})\mathbf{I}\right],
\quad
\mathbf{q} = -\kappa\nabla T,
$$

where $T$ is the fluid temperature and $(\mu, \kappa)$ are viscosity and thermal conductivity, respectively. Source terms are supported through $\mathbf{S}$.

A modular gas model in Theseus implements the transport model and equation of state (EOS), providing transport and thermal properties as a function of the conserved state $\mathbf{Q}$. Theseus currently implements a single-component calorically perfect gas model and is being extended to local thermal equilibrium (LTE) mixtures.

Theseus employs a nodal DGSEM formulation on tensor-product elements for spatial discretization [4, 5]. The solution strategy follows an entropy-conservative BR1 form [6], and uses explicit $s$-stage Runge-Kutta methods for time advancement.

## Documentation

Doxygen infrastructure is included:

- `Doxyfile` for local documentation generation
- `.github/workflows/doxygen.yml` for CI validation and documentation artifact publishing

To generate docs locally (after installing `doxygen`):

```bash
doxygen Doxyfile
```

Generated HTML output defaults to `docs/doxygen/html/index.html`.

## References

1. Center for Hypersonics & Entry Systems Studies (CHESS), University of Illinois Urbana-Champaign. https://chess.grainger.illinois.edu/
2. Hasanli, F. (2025). *Implementation of a High Order Discontinuous Galerkin Spectral Element Method for the Euler and Navier--Stokes Equations on Unstructured Grids*. M.S. thesis, UIUC. https://hdl.handle.net/2142/129734
3. Anderson, R. et al. (2021). *MFEM: A modular finite element methods library*. Computers & Mathematics with Applications, 81, 42--74. https://doi.org/10.1016/j.camwa.2020.06.009
4. Fisher, T. C., & Carpenter, M. H. (2013). *High-order entropy stable finite difference schemes for nonlinear conservation laws: Finite domains*. Journal of Computational Physics, 252, 518--557.
5. Gassner, G. J., Winters, A. R., & Kopriva, D. A. (2016). *Split form nodal discontinuous Galerkin schemes with summation-by-parts property for the compressible Euler equations*. Journal of Computational Physics, 327, 39--66.
6. Gassner, G. J., Winters, A. R., Hindenlang, F. J., & Kopriva, D. A. (2018). *The BR1 scheme is stable for the compressible Navier--Stokes equations*. Journal of Scientific Computing, 77(1), 154--200. https://doi.org/10.1007/s10915-018-0702-1

## AI Assistance Disclaimer

AI-enhanced tools (including GitHub Copilot and ChatGPT) are used extensively in this project to support code review, testing, and documentation efforts. The numerics, mathematical formulation, and core intellectual content are original work by the human authors and the cited references. AI tools are used to improve project implementation and infrastructure, and all contributions are extensively reviewed by human authors.
