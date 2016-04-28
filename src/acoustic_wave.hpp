#ifndef ACOUSTIC_WAVE_HPP
#define ACOUSTIC_WAVE_HPP

#include "config.hpp"
#include "mfem.hpp"

#include <fstream>
#include <vector>

class Parameters;



class AcousticWave
{
public:
  AcousticWave(const Parameters& p) : param(p) { }
  ~AcousticWave() { }

  void run();

private:
  const Parameters& param;

  void run_FEM();
  void run_SEM();
  void run_DG();

  void run_FEM_serial();
  void run_SEM_serial();
  void run_DG_serial();
#if defined(MFEM_USE_MPI)
  void run_FEM_parallel();
  void run_SEM_parallel();
  void run_DG_parallel();
#endif
};



/**
 * Cell-wise constant coefficient.
 */
class CWConstCoefficient : public mfem::Coefficient
{
public:
  CWConstCoefficient(const double *array, bool own = 1)
    : val_array(array), own_array(own)
  { }

  virtual ~CWConstCoefficient() { if (own_array) delete[] val_array; }

  virtual double Eval(mfem::ElementTransformation &T,
                      const mfem::IntegrationPoint &/*ip*/)
  {
    const int index = T.Attribute - 1; // use attribute as a cell number
    MFEM_VERIFY(index >= 0, "index is negative");
    return val_array[index];
  }

protected:
  const double *val_array;
  bool own_array;
};



/**
 * A coefficient obtained with multiplication of a cell-wise constant
 * coefficient and a function.
 */
class CWFunctionCoefficient : public CWConstCoefficient
{
public:
  CWFunctionCoefficient(double(*F)(const mfem::Vector&, const Parameters&),
                        const Parameters& p,
                        const double *array, bool own = 1)
    : CWConstCoefficient(array, own)
    , Function(F)
    , param(p)
  { }

  virtual ~CWFunctionCoefficient() { }

  virtual double Eval(mfem::ElementTransformation &T,
                      const mfem::IntegrationPoint &ip)
  {
    const int index = T.Attribute - 1; // use attribute as a cell number
    const double cw_coef = val_array[index];
    mfem::Vector transip;
    T.Transform(ip, transip);
    const double func_val = (*Function)(transip, param);
    return cw_coef * func_val;
  }

protected:
  double(*Function)(const mfem::Vector&, const Parameters&);
  const Parameters& param;
};



double compute_function_at_point(const mfem::Mesh& mesh,
                                 const mfem::Vertex& point, int cell,
                                 const mfem::GridFunction& U);

mfem::Vector compute_function_at_points(const mfem::Mesh& mesh, int n_points,
                                        const mfem::Vertex *points,
                                        const int *cells_containing_points,
                                        const mfem::GridFunction& U);

void open_seismo_outs(std::ofstream* &seisU, const Parameters &param,
                      const std::string &method_name);

void output_seismograms(const Parameters& param, const mfem::Mesh& mesh,
                        const mfem::GridFunction &U, std::ofstream* &seisU);

#endif // ACOUSTIC_WAVE_HPP
