#pragma once
#include "parse_helpers.hpp"
#include "SimFactory.hpp"
#include "EulerOperator.hpp"
#include "NSOperator.hpp"
#include "LaxFriedrichsFlux.hpp"
#include "ChandrashekarFlux.hpp"
#include "HLLFlux.hpp"

namespace Theseus {

  // TODO: Update for LTE Table Data, and LTE Gas Model setup
  template <typename Physics>
  std::unique_ptr<Theseus::RHSOperatorBase>
  MakeTypedRHSOperator(
		       bool inviscid,
		       const nlohmann::json &runtime,
		       std::shared_ptr<mfem::ParFiniteElementSpace> vfes,
		       std::shared_ptr<mfem::ParFiniteElementSpace> fes0,
		       std::shared_ptr<mfem::ParMesh> pmesh,
		       std::shared_ptr<mfem::ParGridFunction> eta,
		       std::shared_ptr<mfem::ParGridFunction> alpha,
		       std::vector<std::shared_ptr<mfem::ParGridFunction> > &grad_u,
		       std::shared_ptr<Prandtl::Indicator> indicator,
		       std::shared_ptr<mfem::ParGridFunction> r_gf,
		       mfem::real_t alpha_max,
		       const std::string &gasModelName,
		       const std::string &numFluxName)
  {
    using Gas = typename Physics::GasModel;
    const int dim = pmesh->Dimension();
    const int num_dofs_scalar = vfes->GetNDofs();

    Theseus::PhysicsConstants physics_constants(runtime.value("gamma", 1.4),
						runtime.value("Pr", 0.72),
						runtime.value("R_gas", 287.05),
						runtime.value("mu", 0.02));
    Theseus::StateLayout layout(dim, num_dofs_scalar);

    auto gas_model =
      std::make_shared<Gas>(physics_constants, layout);
    
    if (inviscid) {
      return std::make_unique<Theseus::EulerOperator<Physics>>(vfes, fes0, pmesh, eta, alpha,
							       indicator,
							       gas_model,
							       gasModelName,
							       numFluxName,
							       r_gf,
							       alpha_max);
    } else {
      return std::make_unique<Theseus::NSOperator<Physics>>(vfes, fes0, pmesh, eta, alpha, grad_u,
							    indicator,
							    gas_model,
							    gasModelName,
							    numFluxName,
							    r_gf,
							    alpha_max);
    }
  }

  std::unique_ptr<Theseus::RHSOperatorBase>
  MakeRHSOperator(
		  const nlohmann::json& runtime,
		  std::shared_ptr<mfem::ParFiniteElementSpace> vfes,
		  std::shared_ptr<mfem::ParFiniteElementSpace> fes0,
		  std::shared_ptr<mfem::ParMesh> pmesh,
		  std::shared_ptr<mfem::ParGridFunction> eta,
		  std::shared_ptr<mfem::ParGridFunction> alpha,
		  std::vector<std::shared_ptr<mfem::ParGridFunction> > &grad_u,
		  std::shared_ptr<Prandtl::Indicator> indicator,
		  std::shared_ptr<mfem::ParGridFunction> r_gf,
		  mfem::real_t alpha_max)
  {
    
    std::string gas_model_string =
      to_lower(runtime.value("gas_model", std::string{}));
    
    std::string inv_flux_string =
      to_lower(runtime.value("numerical_flux", std::string{}));
    
    std::string flow_model_string =
      to_lower(runtime.value("flow_model", std::string{}));

    const bool use_cpg =
      gas_model_string.empty() ||
      gas_model_string == "cpg" ||
      gas_model_string == "ideal" ||
      gas_model_string == "ideal_gas";
    
    const bool use_chan =
      inv_flux_string.empty() ||
      inv_flux_string == "chandrashekar" ||
      starts_with(inv_flux_string, "chan");
    
    const bool use_hll =
      inv_flux_string == "hll";
    
    const bool use_llf =
      inv_flux_string == "llf" ||
      inv_flux_string == "lfr" ||
      inv_flux_string == "laxfriedrichs" ||
      inv_flux_string == "lax_friedrichs" ||
      starts_with(inv_flux_string, "lax");

    const bool viscous =
      starts_with(flow_model_string, "visc") ||
      starts_with(flow_model_string, "nav")  ||
      starts_with(flow_model_string, "cns")  ||
      starts_with(flow_model_string, "ns");
    const bool inviscid = !viscous;

  
    if (use_cpg)
      {
	std::string gasModelName("CPG1");
	if (use_chan)
	  {
	    std::string numFluxName("Chandrashekar");
	    using Physics =
	      Theseus::PhysicsTraits<Theseus::IdealGasModel,
				     Theseus::ChandrashekarFlux::InviscidFlux>;
	    return MakeTypedRHSOperator<Physics>(inviscid, runtime,
						 vfes, fes0, pmesh, eta, alpha, grad_u,
						 indicator, r_gf, alpha_max, gasModelName,
						 numFluxName);
	  }
	else if (use_hll)
	  {
	    std::string numFluxName("HLL");
	    using Physics =
	      Theseus::PhysicsTraits<Theseus::IdealGasModel,
				     Theseus::HLLFlux::InviscidFlux>;
	    
	    return MakeTypedRHSOperator<Physics>(inviscid, runtime,
						 vfes, fes0, pmesh, eta, alpha, grad_u,
						 indicator, r_gf, alpha_max, gasModelName,
						 numFluxName);
	  }
	else if (use_llf)
	  {
	    std::string numFluxName("LLF");
	    using Physics =
	      Theseus::PhysicsTraits<Theseus::IdealGasModel,
				     Theseus::LaxFriedrichsFlux::InviscidFlux>;

	    return MakeTypedRHSOperator<Physics>(inviscid, runtime,
						 vfes, fes0, pmesh, eta, alpha, grad_u,
						 indicator, r_gf, alpha_max, gasModelName,
						 numFluxName);
	  }
	else {
	  std::cerr << "Error: Invalid Numerical Flux Type specified: "
		    << inv_flux_string << "\n"
		    << "Supported: Chandrashekar, LLF/LFR, HLL"
		    << std::endl;
	  return nullptr;
	}
      }
    else {
      std::cerr << "Error: Invalid Gas Model specified: "
		<< gas_model_string << "\n"
		<< "Supported: CPG, ideal, ideal_gas"
		<< std::endl;
      return nullptr;
    }

  }
}
