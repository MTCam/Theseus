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
