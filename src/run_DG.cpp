#include "acoustic_wave.hpp"
#include "parameters.hpp"
#include "utilities.hpp"

#include <float.h>

using namespace std;
using namespace mfem;



void AcousticWave::run_DG() const
{
#if defined(MFEM_USE_MPI)
  run_DG_parallel();
#else
  run_DG_serial();
#endif
}



void AcousticWave::run_DG_parallel() const
{
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size == 1)
    run_DG_serial();
  else
    MFEM_ABORT("NOT implemented");
}



void AcousticWave::run_DG_serial() const
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

    MFEM_VERIFY(rho > 1.0 && vp > 1.0, "Incorrect media properties arrays");

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

  const bool own_array = false;
  CWConstCoefficient one_over_rho_coef(one_over_rho, own_array);
  CWConstCoefficient one_over_K_coef(one_over_K, own_array);

  cout << "Stif matrix..." << flush;
  BilinearForm stif(&fespace);
  stif.AddDomainIntegrator(new DiffusionIntegrator(one_over_rho_coef));
  stif.AddInteriorFaceIntegrator(
        new DGDiffusionIntegrator(one_over_rho_coef,
                                  param.method.dg_sigma,
                                  param.method.dg_kappa));
  stif.AddBdrFaceIntegrator(
        new DGDiffusionIntegrator(one_over_rho_coef,
                                  param.method.dg_sigma,
                                  param.method.dg_kappa));
  stif.Assemble();
  stif.Finalize();
  const SparseMatrix& S = stif.SpMat();
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
  chrono.Clear();

  cout << "Mass matrix..." << flush;
  BilinearForm mass(&fespace);
  mass.AddDomainIntegrator(new MassIntegrator(one_over_K_coef));
  mass.Assemble();
  mass.Finalize();
  const SparseMatrix& M = mass.SpMat();
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
  chrono.Clear();

#if defined(OUTPUT_MASS_MATRIX)
  {
    cout << "Output mass matrix..." << flush;
    ofstream mout("mass_mat.dat");
    mass.PrintMatlab(mout);
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

  cout << "Sys matrix..." << flush;
  const SparseMatrix& CopyFrom = M;
  const int nnz = CopyFrom.NumNonZeroElems();
  const bool ownij  = false;
  const bool ownval = true;
  SparseMatrix Sys(CopyFrom.GetI(), CopyFrom.GetJ(), new double[nnz],
                   CopyFrom.Height(), CopyFrom.Width(), ownij, ownval,
                   CopyFrom.areColumnsSorted());
  Sys = 0.0;
  //Sys += D;
  Sys += M;
  GSSmoother prec(Sys);
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
  chrono.Clear();

  cout << "RHS vector... " << flush;
  LinearForm b(&fespace);
  ConstantCoefficient zero(0.0); // homogeneous Dirichlet bc
  if (param.source.plane_wave)
  {
    PlaneWaveSource plane_wave_source(param, one_over_K_coef);
    b.AddDomainIntegrator(new DomainLFIntegrator(plane_wave_source));
    b.AddBdrFaceIntegrator(
          new DGDirichletLFIntegrator(zero, one_over_rho_coef,
                                      param.method.dg_sigma,
                                      param.method.dg_kappa));
    b.Assemble();
  }
  else
  {
    ScalarPointForce scalar_point_force(param, one_over_K_coef);
    b.AddDomainIntegrator(new DomainLFIntegrator(scalar_point_force));
    b.AddBdrFaceIntegrator(
          new DGDirichletLFIntegrator(zero, one_over_rho_coef,
                                      param.method.dg_sigma,
                                      param.method.dg_kappa));
    b.Assemble();
  }
  cout << "||b||_L2 = " << b.Norml2() << endl;
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
  chrono.Clear();

  const string method_name = "DG_";

  cout << "Open seismograms files..." << flush;
  ofstream *seisU; // for pressure
  open_seismo_outs(seisU, param, method_name);
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
  chrono.Clear();

  GridFunction u_0(&fespace); // pressure
  GridFunction u_1(&fespace);
  GridFunction u_2(&fespace);
  u_0 = 0.0;
  u_1 = 0.0;
  u_2 = 0.0;

  const int n_time_steps = param.T / param.dt + 0.5; // nearest integer
  const int tenth = 0.1 * n_time_steps;

  const int N = u_0.Size();

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

  const string name = method_name + param.output.extra_string;
  const string pref_path = (string)param.output.directory + "/" + SNAPSHOTS_DIR;
  VisItDataCollection visit_dc(name.c_str(), param.mesh);
  visit_dc.SetPrefixPath(pref_path.c_str());
  visit_dc.RegisterField("pressure", &u_0);

  StopWatch time_loop_timer;
  time_loop_timer.Start();
  double time_of_snapshots = 0.;
  double time_of_seismograms = 0.;
  for (int time_step = 1; time_step <= n_time_steps; ++time_step)
  {
    Vector y = u_1; y *= 2.0; y -= u_2;        // y = 2*u_1 - u_2

    Vector z0; z0.SetSize(N);                  // z0 = M * (2*u_1 - u_2)
    M.Mult(y, z0);

    Vector z1; z1.SetSize(N); S.Mult(u_1, z1);     // z1 = S * u_1
    Vector z2 = b; z2 *= time_values[time_step-1]; // z2 = timeval*source

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
    PCG(Sys, prec, RHS, u_0, 0, 200, 1e-12, 0.0);

    // Compute and print the L^2 norm of the error
    if (time_step % tenth == 0) {
      cout << "step " << time_step << " / " << n_time_steps
           << " ||solution||_{L^2} = " << u_0.Norml2() << endl;
    }

    if (time_step % param.step_snap == 0) {
      StopWatch timer;
      timer.Start();
      visit_dc.SetCycle(time_step);
      visit_dc.SetTime(time_step*param.dt);
      visit_dc.Save();
      timer.Stop();
      time_of_snapshots += timer.UserTime();
    }

    if (time_step % param.step_seis == 0) {
      StopWatch timer;
      timer.Start();
      output_seismograms(param, *param.mesh, u_0, seisU);
      timer.Stop();
      time_of_seismograms += timer.UserTime();
    }

    u_2 = u_1;
    u_1 = u_0;
  }

  time_loop_timer.Stop();

  delete[] seisU;

  cout << "Time loop is over\n\tpure time = " << time_loop_timer.UserTime()
       << "\n\ttime of snapshots = " << time_of_snapshots
       << "\n\ttime of seismograms = " << time_of_seismograms << endl;

  delete fec;
}


