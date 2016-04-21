#ifndef SOURCE_HPP
#define SOURCE_HPP

#include "config.hpp"
#include "mfem.hpp"

class Parameters;
class SourceParameters;



double RickerWavelet(const SourceParameters& source, double t);
double GaussFirstDerivative(const SourceParameters& source, double t);

double PointForce(const SourceParameters& source, const mfem::Vector& location,
                  const mfem::Vector& x, int dim);

double DeltaPointForce(const SourceParameters& source,
                       const mfem::Vector& location, const mfem::Vector& x,
                       int dim);
double GaussPointForce(const SourceParameters& source,
                       const mfem::Vector& location, const mfem::Vector& x,
                       int dim);


/**
 * Implementation of a scalar point force type of source.
 */
class ScalarPointForce: public mfem::Coefficient
{
public:
  ScalarPointForce(const Parameters& p, mfem::Coefficient& c);
  ~ScalarPointForce() { }

  double Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip);

private:
  const Parameters& param;
  mfem::Coefficient& coef;
  mfem::Vector location;
};



/**
 * Implementation of a plane wave type of source.
 */
class PlaneWaveSource: public mfem::Coefficient
{
public:
  PlaneWaveSource(const Parameters& p, mfem::Coefficient& c);
  ~PlaneWaveSource() { }

  double Eval(mfem::ElementTransformation &T,
              const mfem::IntegrationPoint &ip);

private:
  const Parameters& param;
  mfem::Coefficient& coef;
};

#endif // SOURCE_HPP
