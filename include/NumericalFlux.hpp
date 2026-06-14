#pragma once

#include "mfem.hpp"
#include "NavierStokesFlux.hpp"

// namespace Theseus
// {

// using namespace mfem;

// class NumericalFlux
// {
// protected:
//     const NavierStokesFlux fluxFunction;
//     int dim, num_equations;
// public:

//     NumericalFlux(const NavierStokesFlux &fluxFunction)
//         : fluxFunction(fluxFunction), dim(fluxFunction.dim), num_equations(fluxFunction.num_equations) {}

//     virtual real_t ComputeVolumeFlux(const Vector &state1, const Vector &state2,
//                                      const Vector &metric1, const Vector &metric2,
//                                      Vector &F_tilde) = 0;
//     virtual real_t ComputeFaceFlux(const Vector &state1, const Vector &state2, const Vector &nor,
//                                    Vector &flux) const = 0;
//     const NavierStokesFlux &GetFluxFunction() const
//     {
//         return fluxFunction;
//     }
    
//     virtual ~NumericalFlux() = default;

// };

// }
