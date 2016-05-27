#include "acoustic_wave.hpp"
#include "parameters.hpp"
#include "utilities.hpp"

#include <float.h>

using namespace std;
using namespace mfem;

//#define BASIS_DG



void AcousticWave::run_GMsFEM() const
{
#if defined(MFEM_USE_MPI)
  run_GMsFEM_parallel();
#else
  run_GMsFEM_serial();
#endif
}



void AcousticWave::run_GMsFEM_parallel() const
{
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size == 1)
    run_GMsFEM_serial();
  else
    MFEM_ABORT("NOT implemented");
}



void fill_up_n_fine_cells_per_coarse(int n_fine, int n_coarse,
                                     std::vector<int> &result)
{
  const int k = n_fine / n_coarse;
  for (size_t i = 0; i < result.size(); ++i)
    result[i] = k;
  const int p = n_fine % n_coarse;
  for (int i = 0; i < p; ++i)
    ++result[i];
}



void AcousticWave::run_GMsFEM_serial() const
{
  MFEM_VERIFY(param.mesh, "The mesh is not initialized");

  StopWatch chrono;

  chrono.Start();

  const int dim = param.dimension;
  const int n_elements = param.mesh->GetNE();

  cout << "FE space generation..." << flush;
  FiniteElementCollection *fec = new DG_FECollection(param.method.order, dim);
  FiniteElementSpace fespace(param.mesh, fec);
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
  chrono.Clear();

  cout << "Number of unknowns: " << fespace.GetVSize() << endl;

  double *one_over_rho = new double[n_elements]; // one over density
  double *one_over_K   = new double[n_elements]; // one over bulk modulus

  double Rho[] = { DBL_MAX, DBL_MIN };
  double Vp[]  = { DBL_MAX, DBL_MIN };
  double Kap[] = { DBL_MAX, DBL_MIN };

  for (int i = 0; i < n_elements; ++i)
  {
    const double rho = param.media.rho_array[i];
    const double vp  = param.media.vp_array[i];
    const double K   = rho*vp*vp;

    MFEM_VERIFY(rho > 0. && vp > 0., "Incorrect media properties arrays");

    Rho[0] = std::min(Rho[0], rho);
    Rho[1] = std::max(Rho[1], rho);
    Vp[0]  = std::min(Vp[0], vp);
    Vp[1]  = std::max(Vp[1], vp);
    Kap[0] = std::min(Kap[0], K);
    Kap[1] = std::max(Kap[1], K);

    one_over_rho[i] = 1. / rho;
    one_over_K[i]   = 1. / K;
  }

  std::cout << "Rho: min " << Rho[0] << " max " << Rho[1] << "\n";
  std::cout << "Vp:  min " << Vp[0]  << " max " << Vp[1] << "\n";
  std::cout << "Kap: min " << Kap[0] << " max " << Kap[1] << "\n";

  const bool own_array = true;
  CWConstCoefficient one_over_rho_coef(one_over_rho, own_array);
  CWConstCoefficient one_over_K_coef(one_over_K, own_array);

  cout << "Fine scale stif matrix..." << flush;
  BilinearForm stif_fine(&fespace);
  stif_fine.AddDomainIntegrator(new DiffusionIntegrator(one_over_rho_coef));
  stif_fine.AddInteriorFaceIntegrator(
        new DGDiffusionIntegrator(one_over_rho_coef,
                                  param.method.dg_sigma,
                                  param.method.dg_kappa));
  stif_fine.AddBdrFaceIntegrator(
        new DGDiffusionIntegrator(one_over_rho_coef,
                                  param.method.dg_sigma,
                                  param.method.dg_kappa));
  stif_fine.Assemble();
  stif_fine.Finalize();
  const SparseMatrix& S_fine = stif_fine.SpMat();
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
  chrono.Clear();

  cout << "Fine scale mass matrix..." << flush;
  BilinearForm mass_fine(&fespace);
  mass_fine.AddDomainIntegrator(new MassIntegrator(one_over_K_coef));
  mass_fine.Assemble();
  mass_fine.Finalize();
  const SparseMatrix& M_fine = mass_fine.SpMat();
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
  chrono.Clear();

#if defined(OUTPUT_MASS_MATRIX)
  {
    cout << "Output mass matrix..." << flush;
    ofstream mout("mass_mat.dat");
    mass_fine.PrintMatlab(mout);
    cout << "M.nnz = " << M.NumNonZeroElems() << endl;
    cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
    chrono.Clear();
  }
#endif

//  cout << "Damp matrix..." << flush;
//  VectorMassIntegrator *damp_int = new VectorMassIntegrator(rho_damp_coef);
//  damp_int->SetIntRule(GLL_rule);
//  BilinearForm dampM(&fespace);
//  dampM.AddDomainIntegrator(damp_int);
//  dampM.Assemble();
//  dampM.Finalize();
//  SparseMatrix& D = dampM.SpMat();
//  double omega = 2.0*M_PI*param.source.frequency; // angular frequency
//  D *= 0.5*param.dt*omega;
//  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
//  chrono.Clear();

  cout << "Fine scale RHS vector... " << flush;
  LinearForm b_fine(&fespace);
  ConstantCoefficient zero(0.0); // homogeneous Dirichlet bc
  if (param.source.plane_wave)
  {
    PlaneWaveSource plane_wave_source(param, one_over_K_coef);
    b_fine.AddDomainIntegrator(new DomainLFIntegrator(plane_wave_source));
    b_fine.AddBdrFaceIntegrator(
          new DGDirichletLFIntegrator(zero, one_over_rho_coef,
                                      param.method.dg_sigma,
                                      param.method.dg_kappa));
    b_fine.Assemble();
  }
  else
  {
    ScalarPointForce scalar_point_force(param, one_over_K_coef);
    b_fine.AddDomainIntegrator(new DomainLFIntegrator(scalar_point_force));
    b_fine.AddBdrFaceIntegrator(
          new DGDirichletLFIntegrator(zero, one_over_rho_coef,
                                      param.method.dg_sigma,
                                      param.method.dg_kappa));
    b_fine.Assemble();
  }
  cout << "||b_h||_L2 = " << b_fine.Norml2() << endl;
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
  chrono.Clear();


  std::vector<int> n_fine_cell_per_coarse_x(param.method.gms_Nx);
  fill_up_n_fine_cells_per_coarse(param.grid.nx, param.method.gms_Nx,
                                  n_fine_cell_per_coarse_x);

  std::vector<int> n_fine_cell_per_coarse_y(param.method.gms_Ny);
  fill_up_n_fine_cells_per_coarse(param.grid.ny, param.method.gms_Ny,
                                  n_fine_cell_per_coarse_y);

  const double hx = param.grid.get_hx();
  const double hy = param.grid.get_hy();

  const int gen_edges = 1;

  std::vector<DenseMatrix> R;

  if (param.dimension == 2)
  {
    R.resize(param.method.gms_Nx * param.method.gms_Ny);

    int offset_x = 0, offset_y = 0;

    for (int iy = 0; iy < param.method.gms_Ny; ++iy)
    {
      const int n_fine_y = n_fine_cell_per_coarse_y[iy];
      const double SY = n_fine_y * hy;

      for (int ix = 0; ix < param.method.gms_Nx; ++ix)
      {
        const int n_fine_x = n_fine_cell_per_coarse_x[ix];
        const double SX = n_fine_x * hx;
        Mesh *ccell =
            new Mesh(n_fine_x, n_fine_y, Element::QUADRILATERAL, gen_edges, SX, SY);

        double *local_one_over_rho = new double[n_fine_x * n_fine_y];
        double *local_one_over_K   = new double[n_fine_x * n_fine_y];
        for (int fiy = 0; fiy < n_fine_y; ++fiy)
        {
          for (int fix = 0; fix < n_fine_x; ++fix)
          {
            const int loc_cell = fiy*n_fine_x + fix;
            const int glob_cell = (offset_y + fiy) * param.grid.nx +
                                  (offset_x + fix);

            local_one_over_rho[loc_cell] = one_over_rho[glob_cell];
            local_one_over_K[loc_cell]   = one_over_K[glob_cell];
          }
        }
        const bool own_array = true;
        CWConstCoefficient local_one_over_rho_coef(local_one_over_rho, own_array);
        CWConstCoefficient local_one_over_K_coef(local_one_over_K, own_array);

#ifdef BASIS_DG
        compute_basis_DG(ccell, param.method.gms_nb, param.method.gms_ni,
                         local_one_over_rho_coef, local_one_over_K_coef,
                         R[iy*param.method.gms_Nx + ix]);
#else
        compute_basis_CG(ccell, param.method.gms_nb, param.method.gms_ni,
                         local_one_over_rho_coef, local_one_over_K_coef,
                         R[iy*param.method.gms_Nx + ix]);
#endif
        delete ccell;

        offset_x += n_fine_x;
      }
      offset_y += n_fine_y;
    }
  }
  else // 3D
  {
    std::vector<int> n_fine_cell_per_coarse_z(param.method.gms_Nz);
    fill_up_n_fine_cells_per_coarse(param.grid.nz, param.method.gms_Nz,
                                    n_fine_cell_per_coarse_z);

    const double hz = param.grid.get_hz();

    R.resize(param.method.gms_Nx * param.method.gms_Ny * param.method.gms_Nz);

    int offset_x = 0, offset_y = 0, offset_z = 0;

    for (int iz = 0; iz < param.method.gms_Nz; ++iz)
    {
      const int n_fine_z = n_fine_cell_per_coarse_z[iz];
      const double SZ = n_fine_z * hz;
      for (int iy = 0; iy < param.method.gms_Ny; ++iy)
      {
        const int n_fine_y = n_fine_cell_per_coarse_y[iy];
        const double SY = n_fine_y * hy;
        for (int ix = 0; ix < param.method.gms_Nx; ++ix)
        {
          const int n_fine_x = n_fine_cell_per_coarse_x[ix];
          const double SX = n_fine_x * hx;
          Mesh *ccell =
              new Mesh(n_fine_cell_per_coarse_x[ix],
                       n_fine_cell_per_coarse_y[iy],
                       n_fine_cell_per_coarse_z[iz],
                       Element::HEXAHEDRON, gen_edges, SX, SY, SZ);

          double *local_one_over_rho = new double[n_fine_x * n_fine_y * n_fine_z];
          double *local_one_over_K   = new double[n_fine_x * n_fine_y * n_fine_z];
          for (int fiz = 0; fiz < n_fine_z; ++fiz)
          {
            for (int fiy = 0; fiy < n_fine_y; ++fiy)
            {
              for (int fix = 0; fix < n_fine_x; ++fix)
              {
                const int loc_cell = fiz*n_fine_x*n_fine_y + fiy*n_fine_x + fix;
                const int glob_cell = (offset_z + fiz) * param.grid.nx * param.grid.ny +
                                      (offset_y + fiy) * param.grid.nx +
                                      (offset_x + fix);

                local_one_over_rho[loc_cell] = one_over_rho[glob_cell];
                local_one_over_K[loc_cell]   = one_over_K[glob_cell];
              }
            }
          }
          const bool own_array = true;
          CWConstCoefficient local_one_over_rho_coef(local_one_over_rho, own_array);
          CWConstCoefficient local_one_over_K_coef(local_one_over_K, own_array);

#ifdef BASIS_DG
          compute_basis_DG(ccell, param.method.gms_nb, param.method.gms_ni,
                           local_one_over_rho_coef, local_one_over_K_coef,
                           R[iz*param.method.gms_Nx*param.method.gms_Ny +
                             iy*param.method.gms_Nx + ix]);
#else
          compute_basis_CG(ccell, param.method.gms_nb, param.method.gms_ni,
                           local_one_over_rho_coef, local_one_over_K_coef,
                           R[iz*param.method.gms_Nx*param.method.gms_Ny +
                             iy*param.method.gms_Nx + ix]);
#endif
          delete ccell;

          offset_x += n_fine_x;
        }
        offset_y += n_fine_y;
      }
      offset_z += n_fine_z;
    }
  }

  // global sparse R matrix
  int n_rows = 0;
  int n_cols = 0;
  int n_non_zero = 0;
  for (size_t i = 0; i < R.size(); ++i)
  {
    const int h = R[i].Height();
    const int w = R[i].Width();
    n_rows += w; // transpose
    n_cols += h; // transpose
    n_non_zero += h * w;
  }
  MFEM_VERIFY(n_cols == S_fine.Height(), "Dimensions mismatch");
  int *Ri = new int[n_rows + 1];
  int *Rj = new int[n_non_zero];
  double *Rdata = new double[n_non_zero];

  Ri[0] = 0;
  int k = 0;
  int p = 0;
  int offset = 0;
  for (size_t r = 0; r < R.size(); ++r)
  {
    const int h = R[r].Height();
    const int w = R[r].Width();
    for (int i = 0; i < w; ++i)
    {
      Ri[k+1] = Ri[k] + h;
      ++k;

      for (int j = 0; j < h; ++j)
      {
        Rj[p] = offset + j;
        Rdata[p] = R[r](j, i);
        ++p;
      }
    }
    offset += h;
  }

  SparseMatrix R_global(Ri, Rj, Rdata, n_rows, n_cols);
  SparseMatrix *R_global_T = Transpose(R_global);

  SparseMatrix *M_coarse = RAP(M_fine, R_global);
  SparseMatrix *S_coarse = RAP(S_fine, R_global);

  const int N = M_coarse->Height();

  Vector b_coarse(N);
  R_global.Mult(b_fine, b_coarse);

  const SparseMatrix& CopyFrom = *M_coarse;
  const int nnz = CopyFrom.NumNonZeroElems();
  const bool ownij  = false;
  const bool ownval = true;
  SparseMatrix Sys(CopyFrom.GetI(), CopyFrom.GetJ(), new double[nnz],
                   CopyFrom.Height(), CopyFrom.Width(), ownij, ownval,
                   CopyFrom.areColumnsSorted());
  Sys = 0.0;
  //Sys += D;
  Sys += *M_coarse;
  GSSmoother prec(Sys);


  const string method_name = "GMsFEM_";

  cout << "Open seismograms files..." << flush;
  ofstream *seisU; // for pressure
  open_seismo_outs(seisU, param, method_name);
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
  chrono.Clear();

  GridFunction u_0(&fespace); // fine scale pressure
//  GridFunction u_2(&fespace);

  Vector U_0(N); // coarse scale pressure
  U_0 = 0.0;
  Vector U_1 = U_0;
  Vector U_2 = U_0;

  const int n_time_steps = param.T / param.dt + 0.5; // nearest integer
  const int tenth = 0.1 * n_time_steps;

  cout << "N time steps = " << n_time_steps
       << "\nTime loop..." << endl;

  // the values of the time-dependent part of the source
  vector<double> time_values(n_time_steps);
  for (int time_step = 1; time_step <= n_time_steps; ++time_step)
  {
    const double cur_time = time_step * param.dt;
    time_values[time_step-1] = RickerWavelet(param.source,
                                             cur_time - param.dt);
  }

  const string name = method_name + param.extra_string;
  const string pref_path = (string)param.output_dir + "/" + SNAPSHOTS_DIR;
  VisItDataCollection visit_dc(name.c_str(), param.mesh);
  visit_dc.SetPrefixPath(pref_path.c_str());
  visit_dc.RegisterField("pressure", &u_0);

  StopWatch time_loop_timer;
  time_loop_timer.Start();
  double time_of_snapshots = 0.;
  double time_of_seismograms = 0.;
  for (int time_step = 1; time_step <= n_time_steps; ++time_step)
  {
    Vector y = U_1; y *= 2.0; y -= U_2;        // y = 2*u_1 - u_2

    Vector z0; z0.SetSize(N);                  // z0 = M * (2*u_1 - u_2)
    M_coarse->Mult(y, z0);

    Vector z1; z1.SetSize(N); S_coarse->Mult(U_1, z1);     // z1 = S * u_1
    Vector z2 = b_coarse; z2 *= time_values[time_step-1]; // z2 = timeval*source

    // y = dt^2 * (S*u_1 - timeval*source), where it can be
    // y = dt^2 * (S*u_1 - ricker*pointforce) OR
    // y = dt^2 * (S*u_1 - gaussfirstderivative*momenttensor)
    y = z1; y -= z2; y *= param.dt*param.dt;

    // RHS = M*(2*u_1-u_2) - dt^2*(S*u_1-timeval*source)
    Vector RHS = z0; RHS -= y;

//    for (int i = 0; i < N; ++i) y[i] = diagD[i] * u_2[i]; // y = D * u_2

    // RHS = M*(2*u_1-u_2) - dt^2*(S*u_1-timeval*source) + D*u_2
//    RHS += y;

    // (M+D)*x_0 = M*(2*x_1-x_2) - dt^2*(S*x_1-r*b) + D*x_2
    PCG(Sys, prec, RHS, U_0, 0, 200, 1e-12, 0.0);

    // Compute and print the L^2 norm of the error
    if (time_step % tenth == 0) {
      cout << "step " << time_step << " / " << n_time_steps
           << " ||solution||_{L^2} = " << U_0.Norml2() << endl;
    }

    if (time_step % param.step_snap == 0) {
      StopWatch timer;
      timer.Start();
      visit_dc.SetCycle(time_step);
      visit_dc.SetTime(time_step*param.dt);
      R_global_T->Mult(U_0, u_0);
      visit_dc.Save();
      timer.Stop();
      time_of_snapshots += timer.UserTime();
    }

//    if (time_step % param.step_seis == 0) {
//      StopWatch timer;
//      timer.Start();
//      R_global_T.Mult(U_0, u_0);
//      output_seismograms(param, *param.mesh, u_0, seisU);
//      timer.Stop();
//      time_of_seismograms += timer.UserTime();
//    }

    U_2 = U_1;
    U_1 = U_0;
  }

  time_loop_timer.Stop();

  delete[] seisU;

  cout << "Time loop is over\n\tpure time = " << time_loop_timer.UserTime()
       << "\n\ttime of snapshots = " << time_of_snapshots
       << "\n\ttime of seismograms = " << time_of_seismograms << endl;

  delete S_coarse;
  delete M_coarse;
  delete R_global_T;

  delete fec;
}

