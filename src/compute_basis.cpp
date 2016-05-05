#include "acoustic_wave.hpp"
#include "parameters.hpp"
#include "utilities.hpp"

using namespace std;
using namespace mfem;



void solve_dsygvd(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &eigenvectors)
{
  int ITYPE = 1; // solve Ax = \lambda Bx
  char JOBZ = 'V'; // get eigenvectors
  char UPLO = 'L'; // upper or lower triangle is used

  int N = A.Height(); // matrix dimension
  double *a_data = new double[N*N];
  {
    const double *d = A.Data();
    for (int i = 0; i < N*N; ++i)
      a_data[i] = d[i];
  }

  int LDA = N; // leading dimension of A
  double *b_data = new double[N*N];
  {
    const double *d = B.Data();
    for (int i = 0; i < N*N; ++i)
      b_data[i] = d[i];
  }

  int LDB = N; // leading dimension of B

  double *eigenvalues = new double[N];

  int LWORK = 1 + 6*N + 2*N*N;
  double *WORK = new double[LWORK];

  int LIWORK = 3 + 5*N;
  int *IWORK = new int[LIWORK];

  int INFO;
  dsygvd_(&ITYPE,
          &JOBZ,
          &UPLO,
          &N,
          a_data,
          &LDA,
          b_data,
          &LDB,
          eigenvalues,
          WORK,
          &LWORK,
          IWORK,
          &LIWORK,
          &INFO);

  delete[] IWORK;
  delete[] WORK;
  delete[] eigenvalues;
  delete[] b_data;

  if (INFO != 0)
  {
    std::cerr << "\nINFO = " << INFO << "\nN = " << N << std::endl;
    MFEM_ABORT("The solution of the eigenproblem is not found");
  }

//  for (int i = 0; i < N; ++i)
//    math::normalize(N, &a_data[i*N]);

  for (int i = 0, k = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j, ++k)
      eigenvectors(j, i) = a_data[k];
  }

  delete[] a_data;
}



// helper function
void get_bdr_vdofs(const mfem::FiniteElementSpace& fespace,
                   const mfem::Array<int>& ess_bdr,
                   std::vector<int>& bdr_vdofs)
{
  mfem::Array<int> ess_vdofs;
  fespace.GetEssentialVDofs(ess_bdr, ess_vdofs); // bdr dofs are marked by -1
  bdr_vdofs.clear();
  bdr_vdofs.reserve(ess_vdofs.Size());
  for (int i = 0; i < ess_vdofs.Size(); ++i) {
    if (ess_vdofs[i] < 0)
      bdr_vdofs.push_back(i);
  }
}




void AcousticWave::compute_basis(Mesh *fine_mesh, int n_boundary_bf, int n_interior_bf,
                                 Coefficient &one_over_rho_coef,
                                 Coefficient &one_over_K_coef,
                                 DenseMatrix &R) const
{
  StopWatch chrono;
  chrono.Start();

  cout << "FE space generation..." << flush;
  DG_FECollection fec(param.method.order, param.dimension);
  FiniteElementSpace fespace(fine_mesh, &fec);
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;

  mfem::Array<int> ess_bdr(fine_mesh->bdr_attributes.Max());
  ess_bdr = 1;
  std::vector<int> bdr_vdofs;
  get_bdr_vdofs(fespace, ess_bdr, bdr_vdofs);

  Array<int> ess_tdof_list;
  if (fine_mesh->bdr_attributes.Size())
  {
    Array<int> ess_bdr(fine_mesh->bdr_attributes.Max());
    ess_bdr = 1;
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
  }
  if (ess_tdof_list.Size() == 0)
    ess_tdof_list.Append(0);

  chrono.Clear();
  cout << "Snapshot matrix..." << flush;
  DenseMatrix W(fespace.GetVSize(), ess_tdof_list.Size());
  {

//    DG_FECollection fec(param.method.order, param.dimension);
//    FiniteElementSpace fespace(fine_mesh, &fec);

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

    Vector b(fespace.GetVSize()); // RHS (it's always 0 in the loop)
    Vector x;

    const int maxiter = 1000;
    const double rtol = 1e-12;
    const double atol = 1e-24;
    for (int bd = 0; bd < ess_tdof_list.Size(); ++bd)
    {
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

    {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream mode_sock(vishost, visport);
      mode_sock.precision(8);
      for (int bd = 0; bd < ess_tdof_list.Size(); ++bd) {
        Vector x;
        W.GetColumn(bd, x);
        GridFunction X;
        X.Update(&fespace, x, 0);
        mode_sock << "solution\n" << *fine_mesh << X
                  << "window_title 'Snapshot " << bd+1 << '/' << ess_tdof_list.Size()
                  << "'" << std::endl;

        char c;
        std::cout << "press (q)uit or (c)ontinue --> " << std::flush;
        std::cin >> c;
        if (c != 'c')
          break;
      }
      mode_sock.close();
    }
  }
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;

  chrono.Clear();
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
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;

  chrono.Clear();
  cout << "WTSW matrix..." << flush;
  const SparseMatrix &S = stif.SpMat();
  const DenseMatrix *WTSW = RAP(S, W);
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

  for (int i = 0; i < n_boundary_bf; ++i)
  {
    for (int j = 0; j < fespace.GetVSize(); ++j)
      R(j, i) = boundary_basis(j, i);
  }

  {
    ParMesh par_fine_mesh(MPI_COMM_SELF, *fine_mesh);
    cout << "Parallel FE space generation..." << flush;
    ParFiniteElementSpace par_fespace(&par_fine_mesh, &fec);
    cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
    chrono.Clear();

    cout << "Stif matrix..." << flush;
    ParBilinearForm par_stif(&par_fespace);
    par_stif.AddDomainIntegrator(new DiffusionIntegrator(one_over_rho_coef));
    par_stif.AddInteriorFaceIntegrator(
          new DGDiffusionIntegrator(one_over_rho_coef,
                                    param.method.dg_sigma,
                                    param.method.dg_kappa));
    par_stif.AddBdrFaceIntegrator(
          new DGDiffusionIntegrator(one_over_rho_coef,
                                    param.method.dg_sigma,
                                    param.method.dg_kappa));
    par_stif.Assemble();
    HypreParMatrix *par_S = par_stif.ParallelAssemble();
    cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
    chrono.Clear();

    cout << "Mass matrix..." << flush;
    ParBilinearForm par_mass(&par_fespace);
    par_mass.AddDomainIntegrator(new MassIntegrator(one_over_K_coef));
    par_mass.Assemble();
    HypreParMatrix *par_M = par_mass.ParallelAssemble();
    cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
    chrono.Clear();

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

    Vector x;
    for (int i = 0; i < n_interior_bf; ++i)
    {
      R.GetColumnReference(n_boundary_bf + i, x);
      const Vector *y = lobpcg.GetEigenvector(i).GlobalVector();
      x = *y;
      delete y;
    }

    delete par_M;
    delete par_S;
  }

}

