#include "acoustic_wave.hpp"
#include "parameters.hpp"
#include "utilities.hpp"

using namespace std;
using namespace mfem;



static
void compute_boundary_basis_CG(const Parameters &param,
                               Mesh *fine_mesh, int n_boundary_bf, int n_interior_bf,
                               Coefficient &one_over_rho_coef,
                               Coefficient &one_over_K_coef,
                               DenseMatrix &R)
{
  StopWatch chrono;
  chrono.Start();

  cout << "FE space generation..." << flush;
  H1_FECollection fec(param.method.order, param.dimension);
  FiniteElementSpace fespace(fine_mesh, &fec);
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;

  chrono.Clear();
  cout << "Stif matrix..." << flush;
  BilinearForm stif(&fespace);
  stif.AddDomainIntegrator(new DiffusionIntegrator(one_over_rho_coef));
  stif.Assemble();
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;

  Array<int> ess_bdr;
  Array<int> ess_tdof_list;
  if (fine_mesh->bdr_attributes.Size())
  {
    ess_bdr.SetSize(fine_mesh->bdr_attributes.Max());
    ess_bdr = 1;
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
  }

  chrono.Clear();
  cout << "Snapshot matrix..." << flush;
  DenseMatrix W(fespace.GetVSize(), ess_tdof_list.Size());
  {

    Vector b(fespace.GetVSize()); // RHS (it's always 0 in the loop)

    const int maxiter = 1000;
    const double rtol = 1e-12;
    const double atol = 1e-24;
    for (int bd = 0; bd < ess_tdof_list.Size(); ++bd)
    {
      Vector x;
      W.GetColumnReference(bd, x);
      x = 0.;
      x(ess_tdof_list[bd]) = 1.;
      b = 0.;

      mfem::SparseMatrix A; // mat after eliminating b.c.
      mfem::Vector X, B;
      stif.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      mfem::GSSmoother precond(A);
      mfem::PCG(A, precond, B, X, 0, maxiter, rtol, atol);

      stif.RecoverFEMSolution(X, b, x);
    }

//    {
//      char vishost[] = "localhost";
//      int  visport   = 19916;
//      socketstream mode_sock(vishost, visport);
//      mode_sock.precision(8);
//      for (int bd = 0; bd < ess_tdof_list.Size(); ++bd) {
//        Vector x;
//        W.GetColumn(bd, x);
//        GridFunction X;
//        X.Update(&fespace, x, 0);
//        mode_sock << "solution\n" << *fine_mesh << X
//                  << "window_title 'Snapshot " << bd+1 << '/' << ess_tdof_list.Size()
//                  << "'" << std::endl;

//        char c;
//        std::cout << "press (q)uit or (c)ontinue --> " << std::flush;
//        std::cin >> c;
//        if (c != 'c')
//          break;
//      }
//      mode_sock.close();
//    }
  }
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;

  chrono.Clear();
  cout << "Sfull matrix..." << flush;
  const SparseMatrix &S = stif.SpMat();
  const SparseMatrix &Se= stif.SpMatElim();
  SparseMatrix *Sfull = Add(S, Se);
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;

  chrono.Clear();
  cout << "WTSW matrix..." << flush;
  const DenseMatrix *WTSW = RAP(*Sfull, W);
  delete Sfull;
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;

  chrono.Clear();
  cout << "Edge mass matrix..." << flush;
  BilinearForm edge_mass(&fespace);
  edge_mass.AddBoundaryIntegrator(new MassIntegrator(one_over_K_coef));
  edge_mass.Assemble();
  edge_mass.Finalize();
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;

  chrono.Clear();
  cout << "WTEMW matrix..." << flush;
  const SparseMatrix &EM = edge_mass.SpMat();
  const DenseMatrix *WTEMW = RAP(EM, W);
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;

  DenseMatrix eigenvectors(WTSW->Height(), WTSW->Width());
  solve_dsygvd(*WTSW, *WTEMW, eigenvectors);

  DenseMatrix selected_eigenvectors(eigenvectors.Height(), n_boundary_bf);
  selected_eigenvectors.CopyCols(eigenvectors, 0, n_boundary_bf-1);

  DenseMatrix boundary_basis(fespace.GetVSize(), n_boundary_bf);
  Mult(W, selected_eigenvectors, boundary_basis);

  R.SetSize(fespace.GetVSize(), n_boundary_bf + n_interior_bf);
  R.CopyCols(boundary_basis, 0, n_boundary_bf-1);

//  for (int i = 0; i < n_boundary_bf; ++i)
//  {
//    for (int j = 0; j < fespace.GetVSize(); ++j)
//      R(j, i) = boundary_basis(j, i);
//  }

  delete WTEMW;
  delete WTSW;

  {
    char vishost[] = "localhost";
    int  visport   = 19916;
    socketstream mode_sock(vishost, visport);
    mode_sock.precision(8);
    for (int bf = 0; bf < n_boundary_bf; ++bf) {
      Vector x;
      R.GetColumn(bf, x);
      GridFunction X;
      X.Update(&fespace, x, 0);
      mode_sock << "solution\n" << *fine_mesh << X
                << "window_title 'CG boundary basis " << bf+1 << '/'
                << n_boundary_bf << "'" << std::endl;

      char c;
      std::cout << "press (q)uit or (c)ontinue --> " << std::flush;
      std::cin >> c;
      if (c != 'c')
        break;
    }
    mode_sock.close();
  }
}



static
void compute_interior_basis_CG(const Parameters &param,
                               Mesh *fine_mesh, int n_boundary_bf, int n_interior_bf,
                               Coefficient &one_over_rho_coef,
                               Coefficient &one_over_K_coef,
                               DenseMatrix &R)
{
  StopWatch chrono;
  chrono.Start();

  ParMesh par_fine_mesh(MPI_COMM_SELF, *fine_mesh);

  cout << "Parallel FE space generation..." << flush;
  H1_FECollection fec(param.method.order, param.dimension);
  ParFiniteElementSpace par_fespace(&par_fine_mesh, &fec);
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;

  Array<int> ess_bdr;
  if (fine_mesh->bdr_attributes.Size())
  {
    ess_bdr.SetSize(fine_mesh->bdr_attributes.Max());
    ess_bdr = 1;
  }

  chrono.Clear();
  cout << "Par Stif matrix..." << flush;
  ParBilinearForm par_stif(&par_fespace);
  par_stif.AddDomainIntegrator(new DiffusionIntegrator(one_over_rho_coef));
  par_stif.Assemble();
  par_stif.EliminateEssentialBCDiag(ess_bdr, 1.0);
  par_stif.Finalize();
  HypreParMatrix *par_S = par_stif.ParallelAssemble();
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;

  chrono.Clear();
  cout << "Par Mass matrix..." << flush;
  ParBilinearForm par_mass(&par_fespace);
  par_mass.AddDomainIntegrator(new MassIntegrator(one_over_K_coef));
  par_mass.Assemble();
  par_mass.EliminateEssentialBCDiag(ess_bdr, numeric_limits<double>::min());
  par_mass.Finalize();
  HypreParMatrix *par_M = par_mass.ParallelAssemble();
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;

  HypreBoomerAMG amg(*par_S);
  amg.SetPrintLevel(0);

  HypreLOBPCG lobpcg(MPI_COMM_SELF);
  lobpcg.SetNumModes(n_interior_bf);
  lobpcg.SetPreconditioner(amg);
  lobpcg.SetMaxIter(100);
  lobpcg.SetTol(1e-8);
  lobpcg.SetPrecondUsageMode(1);
  lobpcg.SetPrintLevel(1);
  lobpcg.SetMassMatrix(*par_M);
  lobpcg.SetOperator(*par_S);

//  Array<double> eigenvalues;
  lobpcg.Solve();
//  lobpcg.GetEigenvalues(eigenvalues);

  for (int i = 0; i < n_interior_bf; ++i)
  {
    Vector x;
    R.GetColumnReference(n_boundary_bf + i, x);
    x = lobpcg.GetEigenvector(i);
  }

  delete par_M;
  delete par_S;

  {
    char vishost[] = "localhost";
    int  visport   = 19916;
    socketstream mode_sock(vishost, visport);
    mode_sock.precision(8);
    for (int bf = 0; bf < n_interior_bf; ++bf) {
      Vector x;
      R.GetColumn(n_boundary_bf + bf, x);
      GridFunction X;
      X.Update(&par_fespace, x, 0);
      mode_sock << "solution\n" << par_fine_mesh << X
                << "window_title 'CG interior basis " << bf+1 << '/'
                << n_interior_bf << "'" << std::endl;

      char c;
      std::cout << "press (q)uit or (c)ontinue --> " << std::flush;
      std::cin >> c;
      if (c != 'c')
        break;
    }
    mode_sock.close();
  }
}



//static
//void compute_interior_basis_CG_()
//{
//  Mesh mesh(10, 10, Element::QUADRILATERAL, 1, 100., 100.);
//  int dim = mesh.Dimension();

//  ParMesh par_mesh(MPI_COMM_SELF, mesh);

//  int order = 1;
//  H1_FECollection fec(order, dim);
//  ParFiniteElementSpace par_fespace(&par_mesh, &fec);

//  Array<int> ess_bdr;
//  if (par_mesh.bdr_attributes.Size())
//  {
//    ess_bdr.SetSize(par_mesh.bdr_attributes.Max());
//    ess_bdr = 1;
//  }

//  ConstantCoefficient one(1.);

//  ParBilinearForm par_stif(&par_fespace);
//  par_stif.AddDomainIntegrator(new DiffusionIntegrator(one));
//  par_stif.Assemble();
//  par_stif.EliminateEssentialBCDiag(ess_bdr, 1.0);
//  par_stif.Finalize();
//  HypreParMatrix *par_S = par_stif.ParallelAssemble();

//  ParBilinearForm par_mass(&par_fespace);
//  par_mass.AddDomainIntegrator(new MassIntegrator(one));
//  par_mass.Assemble();
//  par_mass.EliminateEssentialBCDiag(ess_bdr, numeric_limits<double>::min());
//  par_mass.Finalize();
//  HypreParMatrix *par_M = par_mass.ParallelAssemble();

//  HypreBoomerAMG amg(*par_S);
//  amg.SetPrintLevel(0);

//  int n_modes = 10;

//  HypreLOBPCG lobpcg(MPI_COMM_SELF);
//  lobpcg.SetNumModes(n_modes);
//  lobpcg.SetPreconditioner(amg);
//  lobpcg.SetMaxIter(100);
//  lobpcg.SetTol(1e-8);
//  lobpcg.SetPrecondUsageMode(1);
//  lobpcg.SetPrintLevel(1);
//  lobpcg.SetMassMatrix(*par_M);
//  lobpcg.SetOperator(*par_S);

//  lobpcg.Solve();

//  DenseMatrix modes(par_fespace.GetVSize(), n_modes);

//  for (int i = 0; i < n_modes; ++i)
//  {
//    Vector x;
//    modes.GetColumnReference(i, x);
//    x = lobpcg.GetEigenvector(i);
//  }

//  delete par_M;
//  delete par_S;

//  {
//    char vishost[] = "localhost";
//    int  visport   = 19916;
//    socketstream mode_sock(vishost, visport);
//    mode_sock.precision(8);
//    for (int bf = 0; bf < n_modes; ++bf) {
//      Vector x;
//      modes.GetColumn(bf, x);
//      GridFunction X;
//      X.Update(&par_fespace, x, 0);
//      mode_sock << "solution\n" << par_mesh << X
//                << "window_title 'CG interior basis " << bf+1 << '/'
//                << n_modes << "'" << std::endl;

//      char c;
//      std::cout << "press (q)uit or (c)ontinue --> " << std::flush;
//      std::cin >> c;
//      if (c != 'c')
//        break;
//    }
//    mode_sock.close();
//  }
//}



void AcousticWave::compute_basis_CG(Mesh *fine_mesh, int n_boundary_bf, int n_interior_bf,
                                    Coefficient &one_over_rho_coef,
                                    Coefficient &one_over_K_coef,
                                    DenseMatrix &R) const
{
  compute_boundary_basis_CG(param, fine_mesh, n_boundary_bf, n_interior_bf,
                            one_over_rho_coef, one_over_K_coef, R);
  compute_interior_basis_CG(param, fine_mesh, n_boundary_bf, n_interior_bf,
                            one_over_rho_coef, one_over_K_coef, R);
}

