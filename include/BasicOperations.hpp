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

  MFEM_HOST_DEVICE inline
  int hunt(const mfem::real_t *arr, int n, mfem::real_t x, int ind_lo)
  {
      int ind_hi, ind_mid;
      int incr = 1;
      bool ascend = (arr[n-1] >= arr[0]);

      if (ind_lo < 0 || ind_lo >= n)
      {
          ind_lo = -1;
          ind_hi =  n;
      }
      else
      {
          // Right or Left Hunt
          if ( (x >= arr[ind_lo]) == ascend)
          {
              // Hunt right
              if (ind_lo == n-1) return ind_lo;
              ind_hi = ind_lo + incr;

              while (ind_hi < n && ((x >= arr[ind_hi]) == ascend))
              {
                  ind_lo = ind_hi;
                  incr *= 2;
                  ind_hi = ind_lo + incr;
                  if (ind_hi > n-1)
                  {
                      ind_hi = n;
                      break;
                  }
              }
          }
          // Hunt left
          else
          {
              if (ind_lo == 0)
              {
                  ind_lo = -1;
                  return ind_lo;
              }
              ind_hi = ind_lo;
              ind_lo = ind_lo-1;
              while (ind_lo >= 0 && ((x < arr[ind_lo]) == ascend))
              {
                  ind_hi = ind_lo;
                  incr *= 2;
                  if (incr >= ind_hi)
                  {
                      ind_lo = -1;
                      break;
                  }
                  else ind_lo = ind_hi - incr;
              }
          }
      }

      // Binary Search in the estimated bracket
      while(ind_hi - ind_lo != 1)
      {
          ind_mid = ind_lo + (ind_hi - ind_lo)/2;
          if( (x >= arr[ind_mid]) == ascend)
          {
              ind_lo = ind_mid;
          }
          else
          {
              ind_hi = ind_mid;
          }
      }

      if(x == arr[n-1]) ind_lo = n-2;
      if(x == arr[0]) ind_lo = 0;
      return ind_lo;
  }

}
