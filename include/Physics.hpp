#pragma once

#include "theseus_kernels.hpp"

namespace Theseus
{

  struct PhysicsConstants
  {
    real_t gamma;
    real_t gammaInverse;
    real_t gammaP1;
    real_t gammaM1;
    real_t gammaP1Inverse;
    real_t gammaM1Inverse;
    real_t gamma_gammaM1Inverse; // gamma * gammaM1Inverse;
    real_t gammaM1_gammaInverse; // gammaM1 * gammaInverse;

    real_t Pr;
    real_t PrInverse;
    real_t R_gas;
    real_t cp;
    real_t mu;

    MFEM_HOST_DEVICE PhysicsConstants() = default;

    PhysicsConstants(real_t gamma, real_t Pr, real_t R_gas, real_t mu)
      : gamma(gamma), Pr(Pr), R_gas(R_gas), mu(mu),
        gammaInverse(1.0 / gamma), gammaM1(gamma - 1.0), gammaP1(gamma + 1.0),
        gammaM1Inverse(1.0 / gammaM1), gammaP1Inverse(1.0 / gammaP1),
        gammaM1_gammaInverse(gammaM1 * gammaInverse), gamma_gammaM1Inverse(gamma * gammaM1Inverse),
        PrInverse(1.0 / Pr), cp(gamma_gammaM1Inverse * R_gas) {}

    real_t mu_bulk = 2.0 / 3.0;
    real_t mu0 = 1.716e-5;
    real_t T0 = 273.15;
    real_t Ts = 110.4;
  };

}
