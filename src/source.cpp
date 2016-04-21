#include "source.hpp"
#include "parameters.hpp"

using namespace std;
using namespace mfem;

double RickerWavelet(const SourceParameters& source, double t)
{
  const double f = source.frequency;
  const double a  = M_PI*f*(t-1.0/f);
  return source.scale*(1.-2.*a*a)*exp(-a*a);
}

double GaussFirstDerivative(const SourceParameters& source, double t)
{
  const double f = source.frequency;
  const double a = M_PI*f*(t-1.0/f);
  return source.scale*(t-1.0/f)*exp(-a*a);
}

double PointForce(const SourceParameters& source, const Vector& location,
                  const Vector& x, int dim)
{
  double value = 0.;
  if (!strcmp(source.spatial_function, "delta"))
    value = DeltaPointForce(source, location, x, dim);
  else if (!strcmp(source.spatial_function, "gauss"))
    value = GaussPointForce(source, location, x, dim);
  else
    MFEM_ABORT("Unknown spatial function: " + string(source.spatial_function));

  return value;
}

double DeltaPointForce(const SourceParameters& /*source*/,
                       const Vector& location, const Vector& x, int dim)
{
  const double tol = FLOAT_NUMBERS_EQUALITY_TOLERANCE;
  double value = 0.0;

  if (dim == 2)
  {
    const double loc[] = { location(0), location(1) };
    if (x.DistanceTo(loc) < tol)
      value = 1.0;
  }
  else // 3D
  {
    const double loc[] = { location(0), location(1), location(2) };
    if (x.DistanceTo(loc) < tol)
      value = 1.0;
  }

  return value;
}

double GaussPointForce(const SourceParameters& source, const Vector& location,
                       const Vector& x, int dim)
{
  const double xdiff  = x(0) - location(0);
  const double ydiff  = x(1) - location(1);
  const double zdiff  = (dim == 3 ? x(2) - location(2) : 0.);
  const double xdiff2 = xdiff*xdiff;
  const double ydiff2 = ydiff*ydiff;
  const double zdiff2 = zdiff*zdiff;
  const double h2 = source.gauss_support * source.gauss_support;
  const double G = exp(-(xdiff2 + ydiff2 + zdiff2) / h2);
  return G;
}



//------------------------------------------------------------------------------
//
// A source represented by a vector point force.
//
//------------------------------------------------------------------------------
ScalarPointForce::ScalarPointForce(const Parameters& p, Coefficient& c)
  : param(p), coef(c)
{
  location.SetSize(param.dimension);
  for (int i = 0; i < param.dimension; ++i)
    location(i) = param.source.location(i);
}

double ScalarPointForce::Eval(ElementTransformation &T,
                              const IntegrationPoint &ip)
{
  Vector transip;
  T.Transform(ip, transip);
  const double force = PointForce(param.source, location, transip,
                                  param.dimension);
  return force*coef.Eval(T, ip);
}



//------------------------------------------------------------------------------
//
// A set of source distributed along a plane.
//
//------------------------------------------------------------------------------
PlaneWaveSource::PlaneWaveSource(const Parameters& p, Coefficient& c)
  : param(p), coef(c)
{ }

double PlaneWaveSource::Eval(ElementTransformation &T,
                             const IntegrationPoint &ip)
{
  Vector transip;
  T.Transform(ip, transip);

  const double py = transip(1);
  const double tol = FLOAT_NUMBERS_EQUALITY_TOLERANCE;

  // if the point 'transip' is on the plane of the plane wave, we have a source
  // located at the exact same point
  if (fabs(py - param.source.location(1)) < tol)
  {
    const double force = PointForce(param.source, transip, transip,
                                    param.dimension);
    return force*coef.Eval(T, ip);
  }
  else
    return 0.0;
}
