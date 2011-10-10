#include "Python.h"
#include "numpy/arrayobject.h"
#include "GCoptimization.h"
#include <algorithm>

#define INF 10000000
#define N 255

struct ForSmoothFn {
	int num_labels;
	PyArrayObject *adj;
	int *sites;
};

PyMODINIT_FUNC initgcot();
unsigned int matref(PyArrayObject *mat,int i, int j);
void matset(PyArrayObject *mat,int i, int j, unsigned int val);
unsigned int matref(PyArrayObject *mat,int i, int j, int k);
void matset(PyArrayObject *mat,int i, int j, int k, unsigned int val);
int valid_matrix(PyArrayObject *mat);
extern "C" {
static PyObject *graph_cut(PyObject *self, PyObject *args);
}
int smoothFn(int s1, int s2, int l1, int l2, void *extraData);
