#include "Python.h"
#include "numpy/arrayobject.h"
#include "GCoptimization.h"
#include <algorithm>
#include <math.h>

#define INF 10000000
#define N 255

struct ForSmoothFn {
  int num_labels;
  PyArrayObject *adj;
  PyObject *func;
  int *sites;
  double sigma;
  int bias;
};

PyMODINIT_FUNC initgcot();
int smoothFn(int s1, int s2, int l1, int l2, void *extraData);
extern "C" {
static PyObject *graph_cut(PyObject *self, PyObject *args);
static PyObject *adjacent(PyObject *self, PyObject *args);
}
