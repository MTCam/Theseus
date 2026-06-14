#pragma once

#include "mfem.hpp"

namespace Prandtl
{

  class Indicator
  {
  protected:
    std::shared_ptr<mfem::ParFiniteElementSpace> vfes;
    std::shared_ptr<mfem::ParFiniteElementSpace> fes0;
    std::shared_ptr<mfem::ParGridFunction> eta;

    mfem::Array<int> vdof_indices, ind_indx;
    mfem::Vector el_vdofs, ind_dof;
    int num_equations, ndofs, order, dim;
    mfem::Vector state;
  public:
    Indicator(std::shared_ptr<mfem::ParFiniteElementSpace> vfes,
	      std::shared_ptr<mfem::ParFiniteElementSpace> fes0,
	      std::shared_ptr<mfem::ParGridFunction> eta);
    virtual void CheckIndicatorSmoothness(const mfem::Vector &indicator) = 0;
    virtual ~Indicator() = default;
  };
}
