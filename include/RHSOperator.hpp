#pragma once

#include "DGSEMIntegrator.hpp"
#include "DGSEMNonlinearForm.hpp"
#include "ModalBasis.hpp"
#include "Indicator.hpp"
#include "BasicOperations.hpp"
#include "GasModel.hpp"
#include "dgsem_cache_utilities.hpp"
#include "bc_cache_utilities.hpp"

namespace Theseus
{
  class RHSOperatorBase : public mfem::TimeDependentOperator,
                          public mfem::ParNonlinearForm
  {
  protected:
    mutable real_t max_char_speed;
    std::shared_ptr<mfem::ParFiniteElementSpace> vfes;
    std::shared_ptr<mfem::ParFiniteElementSpace> fes0;
    std::shared_ptr<mfem::ParMesh> pmesh;
    std::shared_ptr<mfem::ParGridFunction> eta;
    std::shared_ptr<mfem::ParGridFunction> r_gf;
    std::shared_ptr<Prandtl::Indicator> indicator;

    const int num_equations, dim, order, num_elements;
    const int num_dofs_scalar;
    const int Ndofs;

    const real_t sharpness_fac = 9.21024;
    const real_t modalThreshold;
    const real_t alpha_min;
    const real_t alpha_max;

    std::vector<mfem::Array<int>> bdr_marker;
    mfem::Array<Theseus::BCDescriptor> bc_descriptors;
    mfem::Vector bc_vector_data;
    mfem::Vector bc_scalar_data;

    mutable real_t alpha_dof;
    mutable Theseus::IntegralMeasures diag0;
  public:
    RHSOperatorBase(std::shared_ptr<mfem::ParFiniteElementSpace> vfes_,
                    std::shared_ptr<mfem::ParFiniteElementSpace> fes0_,
                    std::shared_ptr<mfem::ParMesh> pmesh_,
                    std::shared_ptr<mfem::ParGridFunction> eta_,
                    std::shared_ptr<mfem::ParGridFunction> alpha_,
                    std::shared_ptr<Prandtl::Indicator> indicator_,
                    std::shared_ptr<mfem::ParGridFunction> r_gf_ = nullptr,
                    const real_t alpha_max = 0.5, const real_t alpha_min = 0.001)
    : mfem::TimeDependentOperator(vfes_->GetTrueVSize()),
      mfem::ParNonlinearForm(vfes_.get()),
      vfes(vfes_), fes0(fes0_), pmesh(pmesh_),
      eta(eta_), r_gf(r_gf_), indicator(indicator_),
      num_equations(vfes->GetVDim()),
      dim(pmesh->SpaceDimension()),
      order(vfes->GetElementOrder(0)),
      num_elements(pmesh->GetNE()),
      num_dofs_scalar(vfes_->GetTrueVSize()/vfes_->GetVDim()),
      Ndofs(vfes->GetFE(0)->GetDof()),
      modalThreshold(0.5 * std::pow(10.0, -1.8 * std::pow(order, 0.25))),
      alpha_min(alpha_min), alpha_max(alpha_max)
    {
      diag0.mass = 0.0;
      diag0.ke = 0.0;
      diag0.en = 0.0;
    }
    virtual ~RHSOperatorBase() = default;
    void SetBCDescriptorData(const mfem::Array<Theseus::BCDescriptor> &bc_descr, const mfem::Vector &bc_scalar_dat,
                             const mfem::Vector &bc_vector_dat)
    {
      bc_descriptors = bc_descr;
      bc_scalar_data = bc_scalar_dat;
      bc_vector_data = bc_vector_dat;
    }

    void AddBdrFaceMarker(mfem::Array<int> &bdr_marker_)
    {
      bdr_marker.push_back(bdr_marker_);
    }

    virtual void Finalize(real_t time=0)
    {
      SetTime(time);
    }

    IntegralMeasures GetIntegralMeasuresBaseline() const { return diag0; }

    inline real_t GetMaxCharSpeed()
    {
      return max_char_speed;
    }

    inline real_t& GetTimeRef()
    {
      return t;
    }
    virtual void ComputeIntegralMeasures(const mfem::Vector &u, Theseus::IntegralMeasures &diag) const
    { std::cout << "RHSOperatorBase::ComputeIntegralMeasures empty." << std::endl; }
    virtual void Mult(const mfem::Vector &u, mfem::Vector &dudt) const override
    { std::cout << "RHSOperatorBase::Mult empty." << std::endl; }
  };

  template<typename PhysicsT>
  class RHSOperator : public RHSOperatorBase
  {
  public:
    using Physics = PhysicsT;
    using OperatorCache = DGSEMOperatorCacheT<Physics>;
    using DeviceCache = DGSEMDeviceCacheT<Physics>;
    using Gas = typename Physics::GasModel;
    using InviscidFlux = typename Physics::InviscidFlux;
  protected:
    mutable OperatorCache operator_cache;
    mutable DeviceCache device_cache;
  public:
    RHSOperator(std::shared_ptr<mfem::ParFiniteElementSpace> vfes_,
		std::shared_ptr<mfem::ParFiniteElementSpace> fes0_,
		std::shared_ptr<mfem::ParMesh> pmesh_,
		std::shared_ptr<mfem::ParGridFunction> eta_,
		std::shared_ptr<mfem::ParGridFunction> alpha_,
		std::shared_ptr<Prandtl::Indicator> indicator_,
		const Gas &gasModel_,
		std::shared_ptr<mfem::ParGridFunction> r_gf_ = nullptr,
		const real_t alpha_max = 0.5, const real_t alpha_min = 0.001)
      : RHSOperatorBase(vfes_, fes0_, pmesh_, eta_, alpha_,
			indicator_, r_gf_, alpha_max, alpha_min)
    {
      operator_cache.gas = gasModel_;
      operator_cache.alpha = alpha_;
    }

    virtual ~RHSOperator() = default;
    void Finalize(real_t time = 0) override;

#ifdef SUBCELL_FV_BLENDING
    void ComputeBlendingCoefficient(const mfem::Vector &u) const;
    void ComputeBlendingCoefficientFromIndicator(const mfem::Vector &indicator_field) const;
    void ComputeIndicatorField(const mfem::Vector &u, mfem::Vector &indicator_field) const;
#endif

    void Mult(const mfem::Vector &u, mfem::Vector &dudt) const override;
    void ComputeIntegralMeasures(const mfem::Vector &u, Theseus::IntegralMeasures &diag) const override;
    virtual real_t FlowMult(const mfem::Vector &pu, mfem::Vector &pdudt) const = 0;
  };

}

#include "RHSOperator_impl.hpp"
