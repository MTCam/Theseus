# Theseus

[![CI Quick](https://github.com/MTCam/Theseus/actions/workflows/ci-quick.yml/badge.svg)](https://github.com/MTCam/Theseus/actions/workflows/ci-quick.yml)
[![CI Nightly](https://github.com/MTCam/Theseus/actions/workflows/ci-nightly.yml/badge.svg)](https://github.com/MTCam/Theseus/actions/workflows/ci-nightly.yml)
[![Doxygen Docs](https://github.com/MTCam/Theseus/actions/workflows/doxygen.yml/badge.svg)](https://github.com/MTCam/Theseus/actions/workflows/doxygen.yml)

Theseus is a compressible flow solver developed at the Center for Hypersonics and Entry Systems Studies (CHESS) at the University of Illinois [1]. Its formulation and numerics are based on the work of Hasanli [2], and the code uses the MFEM finite element library [3], targeting modern and emerging HPC architectures. Theseus started out as a refactor of [Prandtl](https://github.com/chess-uiuc/Prandtl) [4], the code produced by Hasanli, to enable GPU execution. The implementation diverged substantially into a new codebase that is lighter and faster, while currently less feature-complete than Prandtl.

## Documentation

Most technical solver details are maintained in the Doxygen front page (`docs/mainpage.dox`).

Doxygen infrastructure includes:

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
4. Prandtl repository: https://github.com/chess-uiuc/Prandtl

## AI Assistance Disclaimer

AI-enhanced tools (including GitHub Copilot and ChatGPT) are used extensively in this project to support code review, testing, and documentation efforts. The numerics, mathematical formulation, and core intellectual content are original work by the human authors and the cited references. AI tools are used to improve project implementation and infrastructure, and all contributions are extensively reviewed by human authors.
