#include "DGSEMOperator.hpp"

namespace Theseus
{

  void EulerOperator::Finalize(real_t time)
  {    
    this->SetTime(time);
    GetOperatorCache(vfes.get(), &operator_cache);
    AssembleBoundaryFaceGeometryTerms(vfes.get(), bdr_marker, &operator_cache);
#ifdef SUBCELL_FV_BLENDING
    ComputeSubcellMetrics(vfes.get(), &operator_cache);
#endif

    // operator_cache.gas = gasModel; // gasModel *must* be POD
    // operator_cache.alpha = alpha; // Alpha is a driver-owned gridfunc
    operator_cache.bc_descriptors = bc_descriptors;
    operator_cache.bc_scalar_data = bc_scalar_data;
    operator_cache.bc_vector_data = bc_vector_data;

    GetDeviceCache(operator_cache, device_cache);
  }


  EulerOperator::~EulerOperator()
  {}

#ifdef SUBCELL_FV_BLENDING  
  void EulerOperator::ComputeIndicatorField(const mfem::Vector &u,
                                            mfem::Vector &indicator_field) const
  {
    ScopedTimer timer("ComputeIndicator");

    // This block is executed by the host
    const int nval_restr = operator_cache.restr_v->Height();
    // Copy the device cache so that it is not member data
    auto dc = device_cache;

    // Device cache parameters
    const int dim = dc.dim;
    const int ne = dc.num_elements;
    const int ndof = dc.ndof_scalar_el;
    const int neq = dc.num_equations;
    const int Np_x = dc.Np_x;
    const int Np_y = dc.Np_y;
    const int Np_z = dc.Np_z;

    MFEM_ASSERT(nval_restr == ne*ndof*neq, "Unexpected size for volume restriction in indicator calc.");
    const int nval_ind = nval_restr / neq;
    mfem::Vector Ue(nval_restr);
    indicator_field.SetSize(nval_ind);
    Ue.UseDevice();
    indicator_field.UseDevice();

    real_t *ifield_d = indicator_field.Write();

    operator_cache.restr_v->Mult(u, Ue);
    const real_t *Ue_d = Ue.Read();
    const int estride = ndof*neq;
    
    // Inside the FORALL below, executed on device
    mfem::forall(nval_ind, [=] MFEM_HOST_DEVICE (int vind)
    {
      const int e = vind / ndof;
      const int evind = vind - e * ndof;
      const real_t *u_el = Ue_d + e * estride;
      real_t elstate[Theseus::MAXEQ];
      Kernels::el_gather_state(u_el, ndof, neq, evind, elstate);
      Theseus::PointStateView S{elstate};
      ifield_d[vind] = dc.gas.pressure(S) * dc.gas.density(S);
    });

  }

  void EulerOperator::ComputeBlendingCoefficientFromIndicator(const mfem::Vector &indicator_field) const
  {
    {
      Theseus::ScopedTimer timer("CheckIndicatorSmoothness_Host");
      // This is a HOST-only routine.  Make sure the input is host-readable
      indicator->CheckIndicatorSmoothness(indicator_field);
    }
    Theseus::ScopedTimer timer("ComputeAlpha_Host");
    real_t *alpha_h = operator_cache.alpha->HostWrite();
    const real_t *eta_h = eta->HostRead();
    for (int el = 0; el < num_elements; el++)
      {
        alpha_dof = 1.0 / (1.0 + std::exp(-sharpness_fac * (eta_h[el] - modalThreshold) / modalThreshold));
        if (alpha_dof < alpha_min)
          {
            alpha_dof = 0.0;
          }
        else if (alpha_dof > (1.0 - alpha_min))
          {
            alpha_dof = 1.0;
          }
        alpha_h[el] = std::min(alpha_dof, alpha_max);
      }
  }
#endif

  void EulerOperator::ComputeIntegralMeasures(const Vector &u, IntegralMeasures &diag) const
  {
    Theseus::ScopedTimer timer("ComputeIntegralMeasures");

    // This block is executed by the host
    const int nval_restr = operator_cache.restr_v->Height();

    // Copy the device cache so that it is not member data
    auto dc = device_cache;

    // Device cache parameters
    const int ne = dc.num_elements;
    const int ndof = dc.ndof_scalar_el;
    const int neq = dc.num_equations;
    const real_t *qWts_d = dc.elQWgts_d;
    auto gas = dc.gas;

    mfem::Vector Ue(nval_restr);
    Ue.UseDevice();

    operator_cache.restr_v->Mult(u, Ue);
    const real_t *Ue_d = Ue.Read();
    const int estride = ndof*neq;

    mfem::Vector elMass_integral(ne);
    mfem::Vector elKE_integral(ne);
    mfem::Vector elEnergy_integral(ne);
    mfem::Vector elMaxPressure(ne);
    mfem::Vector elMaxTemperature(ne);
    mfem::Vector elMaxDensity(ne);
    mfem::Vector elMinPressure(ne);
    mfem::Vector elMinTemperature(ne);
    mfem::Vector elMinDensity(ne);

    elMass_integral.UseDevice();
    elKE_integral.UseDevice();
    elEnergy_integral.UseDevice();
    elMaxPressure.UseDevice();
    elMaxTemperature.UseDevice();
    elMaxDensity.UseDevice();
    elMinPressure.UseDevice();
    elMinTemperature.UseDevice();
    elMinDensity.UseDevice();

    real_t *elMass_int_d = elMass_integral.Write();
    real_t *elKE_int_d = elKE_integral.Write();
    real_t *elEnergy_int_d = elEnergy_integral.Write();

    real_t *elPress_max_d = elMaxPressure.Write();
    real_t *elTemp_max_d = elMaxTemperature.Write();
    real_t *elDens_max_d = elMaxDensity.Write();
    real_t *elPress_min_d = elMinPressure.Write();
    real_t *elTemp_min_d = elMinTemperature.Write();
    real_t *elDens_min_d = elMinDensity.Write();

    // Inside the FORALL below, executed on device
    mfem::forall(ne, [=] MFEM_HOST_DEVICE (int e)
    {
      const real_t *u_el = Ue_d + e * estride;
      const real_t *qWgt = qWts_d + e * ndof;
   
      real_t mass_int = 0.0;
      real_t ke_int = 0.0;
      real_t en_int = 0.0;
      real_t min_dens = 1e32;
      real_t max_dens = 0.0;
      real_t min_temp = 1e32;
      real_t max_temp = 0.0;
      real_t min_press = 1e32;
      real_t max_press = 0.0;

      for(int ep = 0;ep < ndof;ep++){
        real_t elstate[Theseus::MAXEQ];
        Kernels::el_gather_state(u_el, ndof, neq, ep, elstate);
        Theseus::PointStateView S{elstate};

        real_t rho = gas.density(S);
        real_t ke = gas.kinetic_energy_density(S);
        real_t rhoE = gas.energy(S); // energy density
        real_t press = gas.pressure(S);
        real_t temper = gas.temperature(S);

        mass_int += rho * qWgt[ep];
        ke_int += ke * qWgt[ep];
        en_int += rhoE * qWgt[ep];

        min_temp = Kernels::rmin(min_temp, temper);
        max_temp = Kernels::rmax(max_temp, temper);
        min_dens = Kernels::rmin(min_dens, rho);
        max_dens = Kernels::rmax(max_dens, rho);
        min_press = Kernels::rmin(min_press, press);
        max_press = Kernels::rmax(max_press, press);
      }

      elMass_int_d[e]   = mass_int;
      elKE_int_d[e]     = ke_int;
      elEnergy_int_d[e] = en_int;
      elPress_max_d[e]  = max_press;
      elPress_min_d[e]  = min_press;
      elDens_max_d[e]   = max_dens;
      elDens_min_d[e]   = min_dens;
      elTemp_min_d[e]   = min_temp;
      elTemp_max_d[e]   = max_temp;

    });

    // diag.mass = mfem::Sum(elMass_integral);
    // diag.ke = mfem::Sum(elKE_integral);
    // diag.en = mfem::Sum(elEnergy_integral);
    diag.mass = 0.0;
    diag.ke   = 0.0;
    diag.en   = 0.0;
    diag.min_press = 1e32;
    diag.max_press = 0.0;
    diag.min_dens = 1e32;
    diag.max_dens = 0.0;
    diag.min_temp = 1e32;
    diag.max_temp = 0.0;

    const real_t *mass_h = elMass_integral.HostRead();
    const real_t *ke_h   = elKE_integral.HostRead();
    const real_t *en_h   = elEnergy_integral.HostRead();
    const real_t *minpress_h = elMinPressure.HostRead();
    const real_t *maxpress_h = elMaxPressure.HostRead();
    const real_t *mindens_h = elMinDensity.HostRead();
    const real_t *maxdens_h = elMaxDensity.HostRead();
    const real_t *mintemp_h = elMinTemperature.HostRead();
    const real_t *maxtemp_h = elMaxTemperature.HostRead();

    for (int e = 0; e < ne; ++e) {
      diag.mass += mass_h[e];
      diag.ke   += ke_h[e];
      diag.en   += en_h[e];
      diag.min_press = Kernels::rmin(diag.min_press, minpress_h[e]);
      diag.max_press = Kernels::rmax(diag.max_press, maxpress_h[e]);
      diag.min_temp = Kernels::rmin(diag.min_temp, mintemp_h[e]);
      diag.max_temp = Kernels::rmax(diag.max_temp, maxtemp_h[e]);
      diag.min_dens = Kernels::rmin(diag.min_dens, mindens_h[e]);
      diag.max_dens = Kernels::rmax(diag.max_dens, maxdens_h[e]);
    }

    real_t sendbuf[3] = {diag.mass, diag.ke, diag.en};
    real_t recvbuf[3] = {0.0, 0.0, 0.0};

    MPI_Allreduce(sendbuf, recvbuf, 3, MPITypeMap<real_t>::mpi_type, MPI_SUM, pmesh->GetComm());

    diag.mass = recvbuf[0];
    diag.ke = recvbuf[1];
    diag.en = recvbuf[2];

    sendbuf[0] = diag.min_press;
    sendbuf[1] = diag.min_temp;
    sendbuf[2] = diag.min_dens;

    MPI_Allreduce(sendbuf, recvbuf, 3, MPITypeMap<real_t>::mpi_type, MPI_MIN, pmesh->GetComm());

    diag.min_press = recvbuf[0];
    diag.min_temp = recvbuf[1];
    diag.min_dens = recvbuf[2];

    sendbuf[0] = diag.max_press;
    sendbuf[1] = diag.max_temp;
    sendbuf[2] = diag.max_dens;

    MPI_Allreduce(sendbuf, recvbuf, 3, MPITypeMap<real_t>::mpi_type, MPI_MAX, pmesh->GetComm());

    diag.max_press = recvbuf[0];
    diag.max_temp = recvbuf[1];
    diag.max_dens = recvbuf[2];

    if(diag0.mass == 0.0){
      diag0 = diag;
    }

  }
  
  void EulerOperator::Mult(const mfem::Vector &u, mfem::Vector &dudt) const
  {
    Theseus::ScopedTimer timer("EulerRHS");
    
    const Vector &Ustate = u;
    
    
#ifdef SUBCELL_FV_BLENDING
    {
      Theseus::ScopedTimer timer("SubcellBlendingStep");
      mfem::Vector indicator_field;
      // Since the CV are on-device, and computing
      // the indicator requires the CV, we compute
      // the indicator on-device and xfer only it and
      // alpha (the blending coeff) from host/device.
      ComputeIndicatorField(Ustate, indicator_field);
      ComputeBlendingCoefficientFromIndicator(indicator_field);
    }
#endif

    // max_char_speed is consumed by external components
    // between steps
    max_char_speed = MultEuler(Ustate, dudt);
    
  }

  // Assemble volume part of RHS for all elements
  // NOTE:
  //  - No axisymmetry (temporarily disabled in device version of MULT)
  real_t EulerOperator::MultEuler_Volume(const Vector &pu, Vector &pdudt) const
  {
    Theseus::ScopedTimer timer("MultEuler_Volume");
    
    // This block is executed by the host
    mfem::Vector Ue(operator_cache.restr_v->Height());
    mfem::Vector dUe(operator_cache.restr_v->Height());

    Ue.UseDevice();
    dUe.UseDevice();

    operator_cache.restr_v->Mult(pu, Ue);
  
    // Zero the array on-device
    {
      real_t *d = dUe.Write();
      mfem::forall(dUe.Size(), [=] MFEM_HOST_DEVICE (int i) { d[i] = real_t(0); });
    }

    const real_t *Ue_d = Ue.Read();
    real_t *dUe_d = dUe.Write();

    // Copy the device cache so that it is not member data
    auto dc = device_cache;

    // Device cache parameters
    const int dim = dc.dim;
    const int ne = dc.num_elements;
    const int ndof = dc.ndof_scalar_el;
    const int neq = dc.num_equations;

#ifdef SUBCELL_FV_BLENDING
    const int Np_x = dc.Np_x;
    const int Np_y = dc.Np_y;
    const int Np_z = dc.Np_z;
    const int npe = Np_x * Np_y * Np_z;
    const int ndofe = npe * neq;
    const int npe_metric_xi = (Np_x + 1)*Np_y*Np_z;
    const int npe_metric_eta = Np_x*(Np_y + 1)*Np_z;
    const int npe_metric_zeta = Np_x * Np_y * (Np_z + 1);
    const real_t *metric_xi_d = dc.subcell_metric_xi_d;
    const real_t *metric_eta_d = (dim > 1 ? dc.subcell_metric_eta_d : nullptr);
    const real_t *metric_zeta_d = (dim > 2 ? dc.subcell_metric_zeta_d : nullptr);

    mfem::Vector dUfv(operator_cache.restr_v->Height());
    dUfv.UseDevice();
    real_t *dUfv_d = dUfv.Write();
    // zero the array on-device
    {
      real_t *d = dUfv_d;
      mfem::forall(dUfv.Size(), [=] MFEM_HOST_DEVICE (int i) { d[i] = real_t(0); });
    }

    const real_t *alpha_d = operator_cache.alpha->Read();
#endif

    // Derived parameters
    const int metric_stride = ndof * dim * dim;
    const int jac_stride    = ndof;
    const int estride = ndof*neq;
  
    // Device cache data/arrays
    const int *elem_attr_d = dc.elem_attr_d;
    const int *attr_marker_d = dc.attr_marker_d;
    const real_t *elJac_d = dc.elJac_d;
    const real_t *elMetric_d = dc.elMetric_d;

    real_t *ws_d = dc.elWaveSpeed_d;

    // Inside the FORALL below, executed on device
    mfem::forall(ne, [=] MFEM_HOST_DEVICE (int e)
    {
    
      const real_t *jac_el    = elJac_d    + e * jac_stride;
      const real_t *metric_el = elMetric_d + e * metric_stride;

      const int attr = elem_attr_d[e];
      if (attr_marker_d[attr-1] == 0) {
        ws_d[e] = 0.0;
        return;
      }

      const int eoff = e * estride;
      const real_t *u_el = Ue_d + eoff;
      real_t *du_el = dUe_d + eoff;

      real_t cs_el = \
        DGSEMIntegrator::AssembleElementVolumeKernel(dc, u_el,
                                                     jac_el, metric_el, du_el);
#ifdef SUBCELL_FV_BLENDING
      real_t alpha_fv = alpha_d[e];
      if(alpha_fv > 1e-16){
        real_t alpha_inv = (1.0 - alpha_fv);
        real_t *du_fv = dUfv_d + eoff;
        const real_t *el_metric_xi = metric_xi_d + e * npe_metric_xi * dim;
        const real_t *el_metric_eta = (dim > 1 ? metric_eta_d + e * npe_metric_eta * dim :
                                       nullptr);
        const real_t *el_metric_zeta = (dim > 2 ? metric_zeta_d + e * npe_metric_zeta * dim :
                                        nullptr);
        const real_t cs_fv =                                              \
          DGSEMIntegrator::ComputeFVFluxesKernel(dc, u_el, jac_el, el_metric_xi, el_metric_eta, el_metric_zeta, du_fv);
      
        for(int ipt = 0;ipt < estride;ipt++){
          du_el[ipt] = alpha_inv * du_el[ipt] + alpha_fv * du_fv[ipt];
        }

        cs_el = Kernels::rmax(cs_el, cs_fv);
      }
#endif

      ws_d[e] = cs_el;

    });

    // Scatter RHS back to storage
    operator_cache.restr_v->AddMultTranspose(dUe, pdudt);

    // Finish up on the host:
    //  - Reduce for rank-local max_char_speed
    const real_t *ws = operator_cache.elWaveSpeed.HostRead();
    real_t max_char_speed = 0.0;
    for(int e = 0;e < operator_cache.num_elements;e++)
      {
        max_char_speed = std::max(max_char_speed, ws[e]);
      }

    return max_char_speed;
  }

  real_t EulerOperator::MultEuler_InteriorFaces(const Vector &pu, Vector &pdudt) const
  {
    Theseus::ScopedTimer timer("MultEuler_InteriorFaces");
    auto dc = device_cache;
    const int dim = dc.dim;
    const int neq = dc.num_equations;
    const int nfp = dc.num_face_points;
    const int nfaces = operator_cache.restr_f->Height() / (nfp * neq * 2); // (+/-)
    const int face_stride = 2 * nfp * neq;
    const int side_stride = nfp * neq;
    const int face_size = 2*nfp*neq;
    const int norm_size = nfp*dim;
  
    mfem::Vector u_faces(operator_cache.restr_f->Height());
    mfem::Vector rhs_faces(operator_cache.restr_f->Height());
    mfem::Vector faces_dudt(pdudt);
    faces_dudt.UseDevice();
    rhs_faces.UseDevice();
    u_faces.UseDevice();

    // If zeroed before accumulation, do it explicitly on device:
    // Potentially, this is not needed at all since I think we overwrite everything
    {
      real_t *d = rhs_faces.Write();
      mfem::forall(rhs_faces.Size(), [=] MFEM_HOST_DEVICE (int i) { d[i] = real_t(0); });
    }

    operator_cache.restr_f->Mult(pu, u_faces);

    const real_t *u_d = u_faces.Read();
    real_t *rhs_d = rhs_faces.Write();

    const real_t *nor_d   = dc.nor_d;      // size nfaces*nfp*dim
    const real_t *inv1_d  = dc.fw_minus_d; // size nfaces*nfp
    const real_t *inv2_d  = dc.fw_plus_d;  // size nfaces*nfp

    real_t *ws_d = dc.ifWaveSpeed_d;

    mfem::forall(nfaces, [=] MFEM_HOST_DEVICE (int i)
    {
      const int face_offset = i*face_size;
      const int n_offset = i*norm_size;
      const int w_offset = i*nfp;

      const real_t *u_face_d = u_d + face_offset;
      real_t *rhs_face_d = rhs_d + face_offset;
      const real_t *nor_face_d = nor_d + n_offset;
      const real_t *w_minus_d = inv1_d + w_offset;
      const real_t *w_plus_d = inv2_d + w_offset;
    
      real_t ws = DGSEMIntegrator::AssembleElementFaceKernel(dc, u_face_d, nor_face_d,
                                                             w_minus_d, w_plus_d, rhs_face_d);
      ws_d[i] = ws;
    
    });

    operator_cache.restr_f->MultTranspose(rhs_faces, faces_dudt);
    pdudt += faces_dudt; // on device? 

    // Finish up on the host:
    //  - Reduce for rank-local max_char_speed
    const real_t *ws = operator_cache.ifWaveSpeed.HostRead();
    real_t max_char_speed_facial = 0.0;
    for(int f = 0;f < operator_cache.num_interior_faces;f++)
      {
        max_char_speed_facial = std::max(max_char_speed_facial, ws[f]);
      }

    return max_char_speed_facial;
  }


  real_t EulerOperator::MultEuler_BoundaryFaces(const Vector &pu, Vector &pdudt) const
  {
    Theseus::ScopedTimer timer("MultEuler_BoundaryFaces");

    auto dc = device_cache;
    const int dim = dc.dim;
    const int neq = dc.num_equations;
    const int nfp = dc.num_face_points;
    const int face_size = nfp * neq;
    const int restr_size = operator_cache.restr_b->Height();
    const int nfaces_restr = restr_size / face_size;
    const int norm_size = nfp * dc.dim;
    const int npoints_bnd = nfaces_restr * nfp;

    if(operator_cache.u_bnd.Size() != restr_size)
      {
        operator_cache.u_bnd.SetSize(restr_size);
        operator_cache.u_bnd.UseDevice();
        operator_cache.rhs_bnd.SetSize(restr_size);
        operator_cache.rhs_bnd.UseDevice();
        operator_cache.dudt_bnd.SetSize(pdudt.Size());
        operator_cache.dudt_bnd.UseDevice();
      }

    mfem::Vector &u_faces(operator_cache.u_bnd);
    mfem::Vector &rhs_faces(operator_cache.rhs_bnd);
    mfem::Vector &faces_dudt(operator_cache.dudt_bnd);

    // If zeroed before accumulation, do it explicitly on device:
    // Potentially, this is not needed at all since I think we overwrite everything
    {
      real_t *rd = rhs_faces.Write();
      mfem::forall(rhs_faces.Size(), [=] MFEM_HOST_DEVICE (int i)
      { rd[i] = real_t(0);});
      real_t *fd = faces_dudt.Write();
      mfem::forall(faces_dudt.Size(), [=] MFEM_HOST_DEVICE (int i)
      { fd[i] = real_t(0);});
    }

    operator_cache.restr_b->Mult(pu, u_faces);

    const real_t *u_d = u_faces.Read();
    real_t *rhs_d = rhs_faces.Write();

    const real_t *nor_d   = dc.bnd_nor_d;      // size nfaces*nfp*dim
    const real_t *inv1_d  = dc.bnd_wt_d; // size nfaces*nfp
    const int *bnd_marker_index_d = dc.bnd_marker_index_d;
    real_t *ws_d = dc.bndWaveSpeed_d;

    mfem::forall(npoints_bnd, [=] MFEM_HOST_DEVICE (int p)
    {
      const int f = p / nfp;
      const int fp = p % nfp;

      int bnd_face_marker_index = bnd_marker_index_d[f];
      if(bnd_face_marker_index < 0){
        ws_d[p] = 0.0;
        return;
      }
      //    int bc_index = bnd_marker_to_bc_descr_d[bnd_face_marker_index];
      int bc_index = bnd_face_marker_index; // no mapping atm
      if(bc_index < 0){
        ws_d[p] = 0.0;
        return;
      }
      const Theseus::BCDescriptor &bc = dc.bc_descr_d[bc_index];
      if (bc.type == int(Theseus::BCType::Invalid))
        {
          ws_d[p] = 0.0;
          return;
        }

      const int face_offset = f * face_size;
      const int n_offset = f * norm_size;
      const int w_offset = f * nfp;

      const real_t *u_face_d = u_d + face_offset;
      real_t *rhs_face_d = rhs_d + face_offset;
      const real_t *nor_face_d = nor_d + n_offset;
      const real_t *w_minus_d = inv1_d + w_offset;
      const real_t *nor_point = nor_face_d + fp*dim;
      real_t scale = -w_minus_d[fp];
      // #ifdef AXISYMMETRIC
      // NOTE: axisymmetric not ready for device yet
      // scale *= rad_face[fp];
      // #else
      // #error "Axisymmetric boundary device path not implemented yet."
      // #endif
      real_t state1[Theseus::MAXEQ];
      real_t fluxN[Theseus::MAXEQ];

      Theseus::Kernels::el_gather_state(u_face_d, nfp, neq, fp, state1);
      const real_t ws = \
        Theseus::BC::ApplyBoundaryConditionInviscid(dc, bc, state1,
                                                    nor_point, fluxN);
      Theseus::Kernels::el_scatter_add(fluxN, nfp, neq, fp, scale, rhs_face_d);
      ws_d[p] = ws;

    });

    operator_cache.restr_b->MultTranspose(rhs_faces, faces_dudt);
    pdudt += faces_dudt; // on device? (likely yes) 

    // Finish up on the host:
    //  - Reduce for rank-local max_char_speed
    const real_t *ws = operator_cache.bndWaveSpeed.HostRead();
    real_t max_char_speed_facial = 0.0;
    for(int p = 0;p < npoints_bnd;p++)
      {
        max_char_speed_facial = std::max(max_char_speed_facial, ws[p]);
      }

    return max_char_speed_facial;
  }


  // Top level MULT for inviscid cases, called from DGSEMOperator
  real_t EulerOperator::MultEuler(const Vector &u, Vector &dudt) const
  {
    Theseus::ScopedTimer timer("MultEuler");

    auto report_bad = [&](const char *name, const mfem::Vector &v)
    {
      int nbad = CountBadEntries(v);
      if (nbad)
        {
          mfem::out << "BAD VALUES IN: (" << name << "), count=" << nbad << std::endl;
        }
    };

    const mfem::Vector &pu = Prolongate(u);
    if (P)
      {
        operator_cache.pdudt.SetSize(P->Height());
      }

    mfem::Vector &pdudt = P ? operator_cache.pdudt : dudt;
    int psize = pdudt.Size();

    // Zero on-device
    real_t *pdudt_d = pdudt.Write();
    mfem::forall(psize, pdudt_d[i]=0);

    real_t max_char_speed = 0.0;
    // This step overwrites contents of pdudt
    max_char_speed = MultEuler_Volume(pu, pdudt);

    real_t max_char_speed_facial = 0.0;
    max_char_speed_facial = MultEuler_InteriorFaces(pu, pdudt);
    // report_bad("int rhs", pdudt);

    // std::cout << "Facial wavespeed: " << max_char_speed_facial << std::endl;
    max_char_speed = std::max(max_char_speed, max_char_speed_facial);
    real_t max_char_speed_bnd = 0.0;
    max_char_speed_bnd = MultEuler_BoundaryFaces(pu, pdudt);
    // report_bad("bnd rhs", pdudt);

    max_char_speed = std::max(max_char_speed, max_char_speed_bnd);

    if (Serial())
      {
        if(cP) cP->MultTranspose(pdudt, dudt);
      }
    else
      {
        if(P) P->MultTranspose(pdudt, dudt);
      }

    const int N = ess_tdof_list.Size();
    const auto idx = ess_tdof_list.Read();
    auto DU_RW = dudt.ReadWrite();
    mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) { DU_RW[idx[i]] = 0.0; });

    return max_char_speed;
  }

  // void EulerOperator::AddBdrFaceIntegrator(BdrFaceIntegrator *bfi, Array<int> &bdr_marker_)
  // {
  //   bdr_marker.push_back(bdr_marker_);
  // }

}
