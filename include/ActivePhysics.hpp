#include "GasModel.hpp"
#include "LaxFriedrichsFlux.hpp"
#include "ChandrashekarFlux.hpp"
#include "HLLFlux.hpp"

namespace Theseus
{
  struct ActivePhysics {
    using GasModel = IdealGasModel;
    using InviscidFlux = ChandrashekarFlux::InviscidFlux;
  };

}
