#include "acoustic_wave.hpp"
#include "GLL_quadrature.hpp"
#include "parameters.hpp"
#include "receivers.hpp"
#include "utilities.hpp"

using namespace std;
using namespace mfem;



void AcousticWave::run()
{
  if (!strcmp(param.method.name, "fem") || !strcmp(param.method.name, "FEM"))
  {
    run_FEM();
  }
  else if (!strcmp(param.method.name, "sem") || !strcmp(param.method.name, "SEM"))
  {
    run_SEM();
  }
  else if (!strcmp(param.method.name, "dg") || !strcmp(param.method.name, "DG"))
  {
    run_DG();
  }
  else if (!strcmp(param.method.name, "gmsfem") || !strcmp(param.method.name, "GMsFEM"))
  {
    run_GMsFEM();
  }
  else
  {
    MFEM_ABORT("Unknown method to be used: " + string(param.method.name));
  }
}



//------------------------------------------------------------------------------
//
// Auxiliary useful functions
//
//------------------------------------------------------------------------------
double compute_function_at_point(const Mesh& mesh, const Vertex& point,
                                 int cell, const GridFunction& U)
{
  const int dim = mesh.Dimension();
  MFEM_VERIFY(dim == 2 || dim == 3, "Wrong dimension");

  const Element* element = mesh.GetElement(cell);
  if (dim == 2)
  {
    MFEM_VERIFY(dynamic_cast<const Quadrilateral*>(element), "The mesh "
                "element has to be a quadrilateral");
  }
  else // 3D
  {
    MFEM_VERIFY(dynamic_cast<const Hexahedron*>(element), "The mesh "
                "element has to be a hexahedron");
  }

  std::vector<double> limits(6);
  get_limits(mesh, *element, limits);

  const double x0 = limits[0];
  const double y0 = limits[1];
  const double z0 = limits[2];
  const double x1 = limits[3];
  const double y1 = limits[4];
  const double z1 = limits[5];

  const double hx = x1 - x0;
  const double hy = y1 - y0;
  const double hz = z1 - z0;

  if (dim == 2)
  {
    MFEM_VERIFY(hx > 0 && hy > 0, "Size of the quad is wrong");
  }
  else
  {
    MFEM_VERIFY(hx > 0 && hy > 0 && hz > 0, "Size of the hex is wrong");
  }

  IntegrationPoint ip;
  ip.x = (point(0) - x0) / hx; // transfer to the reference space [0,1]^d
  ip.y = (point(1) - y0) / hy;
  if (dim == 3)
    ip.z = (point(2) - z0) / hz;

  return U.GetValue(cell, ip);
}



Vector compute_function_at_points(const Mesh& mesh, int n_points,
                                  const Vertex *points,
                                  const int *cells_containing_points,
                                  const GridFunction& U)
{
  Vector U_at_points(n_points);

  for (int p = 0; p < n_points; ++p)
  {
    U_at_points(p) = compute_function_at_point(mesh, points[p],
                                               cells_containing_points[p], U);
  }
  return U_at_points;
}



void open_seismo_outs(ofstream* &seisU, const Parameters &param,
                      const string &method_name)
{
  const int n_rec_sets = param.sets_of_receivers.size();

  seisU = new ofstream[n_rec_sets];

  for (int r = 0; r < n_rec_sets; ++r)
  {
    const ReceiversSet *rec_set = param.sets_of_receivers[r];
    const string desc = rec_set->description();

    string seismofile = (string)param.output_dir + "/" + SEISMOGRAMS_DIR +
                        method_name + param.extra_string + desc + "_p.bin";
    seisU[r].open(seismofile.c_str(), ios::binary);
    MFEM_VERIFY(seisU[r], "File '" + seismofile + "' can't be opened");
  } // loop for sets of receivers
}



void output_seismograms(const Parameters& param, const Mesh& mesh,
                        const GridFunction &U, ofstream* &seisU)
{
  // for each set of receivers
  for (size_t rec = 0; rec < param.sets_of_receivers.size(); ++rec)
  {
    MFEM_VERIFY(seisU[rec].is_open(), "The stream for writing seismograms is "
                "not open");

    const ReceiversSet *rec_set = param.sets_of_receivers[rec];

    // pressure at the receivers
    const Vector u =
      compute_function_at_points(mesh, rec_set->n_receivers(),
                                 &(rec_set->get_receivers()[0]),
                                 &(rec_set->get_cells_containing_receivers()[0]), U);
    MFEM_ASSERT(u.Size() == rec_set->n_receivers(), "Sizes mismatch");
    for (int i = 0; i < u.Size(); ++i) {
      float val = u(i);
      seisU[rec].write(reinterpret_cast<char*>(&val), sizeof(val));
    }
  } // loop over receiver sets
}



