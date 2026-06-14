#pragma once

#include "mfem.hpp"
#include "GasState.hpp"

namespace Theseus
{

  template<typename GasModelT>
  inline void Conserv2Entropy(const GasModelT &gasModel,
			      const mfem::Vector &state,
			      mfem::Vector &ent_state)
  {
    Theseus::PointStateView S{state.GetData()};
    Theseus::PointStateViewRW E{ent_state.GetData()};
    gasModel.entropy_state(S, E);
  }

  template<typename GasModelT>
  inline void Conserv2Entropy(const GasModelT &gasModel,
			      const mfem::DenseMatrix &vdof_mat,
			      mfem::DenseMatrix &ent_mat)
  {
    ent_mat = 0.0;
    mfem::Vector state, ent_state(vdof_mat.Width());
    for (int d = 0; d < vdof_mat.Height(); d++)
      {
        vdof_mat.GetRow(d, state);
        Conserv2Entropy(gasModel, state, ent_state);
        ent_mat.SetRow(d, ent_state);
      }
  }

  template<typename GasModelT>
  inline void EntropyGrad2PrimGrad(const GasModelT &gasModel,
				   const mfem::DenseMatrix &vdof_mat,
				   mfem::DenseMatrix &grad)
  {
    mfem::Vector state, grad_state;
    int numeq = gasModel.num_equations();
    mfem::Vector gradPrim(numeq);    
    Theseus::PointStateViewRW dPrim{gradPrim.GetData()};

    for (int d = 0; d < vdof_mat.Height(); d++)
      {
        vdof_mat.GetRow(d, state);
        grad.GetRow(d, grad_state);
        Theseus::PointStateView S{state.GetData()};
        Theseus::PointStateView dS{grad_state.GetData()};
        gasModel.grad_entropy_to_grad_prim(S, dS, dPrim);
        grad.SetRow(d, gradPrim);
      }
  }

  template<typename GasModelT> 
  inline void Entropy2Conserv(const GasModelT &gasModel,
			      const mfem::Vector &ent_state,
			      mfem::Vector &state)
  {
    Theseus::PointStateView Se{ent_state.GetData()};
    Theseus::PointStateViewRW Sc{state.GetData()};
    gasModel.entropy_to_conserved(Se, Sc);
  }

}
