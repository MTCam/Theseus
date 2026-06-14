#pragma once

#include "mfem.hpp"
#include "DGSEMIntegrator.hpp"
//#include "BdrFaceIntegrator.hpp"
#include "timer.hpp"
#include "general/forall.hpp"
#include "dgsem_cache_utilities.hpp"
#include "bc_kernels.hpp"

namespace Theseus
{
  
  class DGSEMNonlinearForm : public mfem::ParNonlinearForm
  {
  private:
    mutable mfem::Vector aux2_x, aux2_y, aux2_z;
    mutable mfem::Vector rhs_aux_;
    mutable std::vector<mfem::Vector> grad_aux_;
    mutable std::vector<mfem::Vector> vol_grad_prim;
    mutable std::vector<mfem::Vector> int_grad_prim;
    mutable std::vector<mfem::Vector> bnd_grad_prim;
    mutable mfem::Vector vol_u;
    mutable mfem::Vector int_u;
    mutable mfem::Vector bnd_u;
    mutable mfem::ParGridFunction GRAD_X, GRAD_Y, GRAD_Z;
    Theseus::DGSEMOperatorCache *cache = nullptr;
    Theseus::DGSEMDeviceCache device_cache;

  public:

    DGSEMNonlinearForm(mfem::ParFiniteElementSpace *pfes);

    // Device path Interfaces
    real_t MultEuler(const mfem::Vector &pu, mfem::Vector &pdudt) const;
    real_t MultCNS(const mfem::Vector &u, const std::vector<mfem::Vector *> &grad_prim, mfem::Vector &dudt) const;
    void GradOperator(const mfem::Vector &u, std::vector<mfem::Vector *> &grad_u) const;

    // Device path helpers
    real_t MultEuler_Volume(const mfem::Vector &pu, mfem::Vector &pdudt) const;
    real_t MultEuler_InteriorFaces(const mfem::Vector &pu, mfem::Vector &pdudt) const;
    real_t MultEuler_BoundaryFaces(const mfem::Vector &pu, mfem::Vector &pdudt) const;
    real_t MultCNS_Volume(const mfem::Vector &pu, const std::vector<mfem::Vector *> &p_grad_prim,
                          mfem::Vector &pdudt) const;
    real_t MultCNS_InteriorFaces(const mfem::Vector &pu,
                                 const std::vector<mfem::Vector *> &p_grad_prim,
                                 mfem::Vector &pdudt) const;
    real_t MultCNS_BoundaryFaces(const mfem::Vector &pu, const std::vector<mfem::Vector *> &p_grad_prim,
                                 mfem::Vector &pdudt) const;
    void GradOperator_Volume(const mfem::Vector &pu, std::vector<mfem::Vector *> &p_grad_u) const;
    void GradOperator_InteriorFaces(const mfem::Vector &pu, std::vector<mfem::Vector *> &p_grad_u) const;
    void GradOperator_BoundaryFaces(const mfem::Vector &pu, std::vector<mfem::Vector *> &p_grad_u) const;

    // Setup utilities
    void SetOperatorCache(DGSEMOperatorCache *cache_){
      cache = cache_;
      GetDeviceCache(*cache, device_cache);
    }

  };
}
