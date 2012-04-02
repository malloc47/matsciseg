#include "gcoc.h"

#define N 255
#define LTHRESH 10

static PyMethodDef gcocMethods[] = { 
  {"graph_cut", graph_cut, METH_VARARGS, "Graph Cut Optimization wrapper"},
  {NULL, NULL, 0, NULL}};

PyMODINIT_FUNC initgcoc() { 
      (void) Py_InitModule("gcoc", gcocMethods);
      import_array();
}

int smoothFnCb(int s1, int s2, int l1, int l2, void *extraData) {
	ForSmoothFn *extra = (ForSmoothFn *) extraData;
	// int num_labels = extra->num_labels;
	PyArrayObject *adj = extra->adj;
	PyObject *func = extra->func;
	int *sites = extra->sites;

	if(l1 == l2) { return 0; }

	if(!(*((npy_bool*)PyArray_GETPTR2(adj,l1,l2)))) { return INF; };
	
	PyObject *args;
	PyObject *result;

	args = Py_BuildValue("(iiiii)",sites[s1],sites[s2],l1,l2,
			     (*((npy_bool*)PyArray_GETPTR2(adj,l1,l2))));
	result = PyEval_CallObject(func, args);
	Py_DECREF(args);

	if (result == NULL)
	  return NULL;

	if(!PyInt_Check(result))
	  return NULL;

	int result_value = (int)PyLong_AsLong(result);

	Py_DECREF(result);

	return result_value;
}

int smoothFnI(int s1, int s2, int l1, int l2, void *extraData) {
	ForSmoothFn *extra = (ForSmoothFn *) extraData;
	// int num_labels = extra->num_labels;
	PyArrayObject *adj = extra->adj;
	int *sites = extra->sites;

	if(l1 == l2) { return 0; }

	if(!(*((npy_int16*)PyArray_GETPTR2(adj,l1,l2)))) { return INF; };

	// use this for intensity image
	return int((1.0/double((abs(sites[s1]-sites[s2]) < LTHRESH ? LTHRESH : abs(sites[s1]-sites[s2]))+1)) * N);
}

int smoothFnE(int s1, int s2, int l1, int l2, void *extraData) {
	ForSmoothFn *extra = (ForSmoothFn *) extraData;
	// int num_labels = extra->num_labels;
	PyArrayObject *adj = extra->adj;
	int *sites = extra->sites;

	if(l1 == l2) { return 0; }

	if(!(*((npy_int16*)PyArray_GETPTR2(adj,l1,l2)))) { return INF; };

	// use this for edge image
	return int( 1/(std::max(double(sites[s1]),double(sites[s2]))+1) * N );
}

void GridGraph_DArraySArray(int width,int height,int num_pixels,int num_labels);

static PyObject *graph_cut(PyObject *self, PyObject *args) {
  PyArrayObject *data_p, *img_p, *seedimg_p, *adj_p, *output;
  PyObject *func = NULL;
  int num_labels;
  int mode = 0;
  int d[3];
  bool has_func = false;
  // rediculous amount of typechecking, as it makes for fewer
  // headaches later
  if (!PyArg_ParseTuple(args, "O!O!O!O!i|iO:set_callback", 
			&PyArray_Type, &data_p,
			&PyArray_Type, &img_p, 
			&PyArray_Type, &seedimg_p, 
			&PyArray_Type, &adj_p, 
			&num_labels,
			&mode,
			&func)) {
    PyErr_SetString(PyExc_ValueError, "Parameters not right");
    return NULL;
  }

  printf("mode %i\n",mode);

  // check that the objects were successfully assigned
  if (NULL == data_p    ||
      NULL == img_p     ||
      NULL == seedimg_p ||
      NULL == adj_p ) {
    PyErr_SetString(PyExc_ValueError, "No data in matrix");
    return NULL; 
  }

  has_func = (NULL != func);

  if(has_func) {
    if (!PyCallable_Check(func)) {
      PyErr_SetString(PyExc_TypeError, "No function passed");
      return NULL;
    }
    // increment function reference
    Py_XINCREF(func);
  }

  // CORNER CASE HERE: if data term is dimension 2, we should not fail, but instead just do nothing
  if(data_p->nd == 2) {
    printf("Warning: No Graph Cut work to do! Data term is 2D.\n");
  	return PyArray_Return(seedimg_p);
  }
  
  if(data_p->nd != 3    ||
     img_p->nd != 2     ||
     seedimg_p->nd != 2 ||
     adj_p->nd != 2) {
    printf("%d, %d, %d, %d\n",data_p->nd,img_p->nd,seedimg_p->nd,adj_p->nd);
    PyErr_SetString(PyExc_ValueError, "Wrong input matrix depth");
    return NULL;
  }

  // double-check, for no particular reason
  if(!PyArray_ISUNSIGNED(img_p)    ||
     !PyArray_ISINTEGER(seedimg_p)) {
    PyErr_SetString(PyExc_ValueError, "Wrong intput matrix property");
    return NULL;
  }

  if(data_p->descr->type_num != NPY_BOOL ||
     img_p->descr->type_num != NPY_UINT8 ||
     seedimg_p->descr->type_num != NPY_INT16 ||
     adj_p->descr->type_num != NPY_BOOL) {
    PyErr_SetString(PyExc_ValueError, "Wrong intput matrix type");
    return NULL;
  }

  if(data_p->dimensions[0] != img_p->dimensions[0]     ||
     data_p->dimensions[1] != img_p->dimensions[1]     ||
     data_p->dimensions[0] != seedimg_p->dimensions[0] ||
     data_p->dimensions[1] != seedimg_p->dimensions[1] ||
     data_p->dimensions[2] != adj_p->dimensions[0]     ||
     data_p->dimensions[2] != adj_p->dimensions[1]) {
    PyErr_SetString(PyExc_ValueError, "Input sizes do not match");
    return NULL;
  }

  d[0] = data_p->dimensions[0];
  d[1] = data_p->dimensions[1];
  d[2] = data_p->dimensions[2];
  
  GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(d[1],d[0],num_labels);

  int *data = new int[d[0]*d[1]*num_labels];
  int *sites = new int[d[0]*d[1]];
  int *result = new int[d[0]*d[1]];

  int i=0,j=0,k=0;

  // load up data term
  for(i=0;i<d[0]; i++)
    for(j=0;j<d[1]; j++)
      for(k=0;k<num_labels; k++)
  	data[ ( j+i*d[1]) * num_labels + k ] = 
	  *((npy_bool*)PyArray_GETPTR3(data_p,i,j,k)) == 1 ? 0 : INF;

  // load up segmentation sites
  for(i=0;i<d[0]; i++)
    for(j=0;j<d[1]; j++)
      sites[j+(i*d[1])] = int(*((npy_uint8*)PyArray_GETPTR2(img_p,i,j)));

  // initialize
  for(i=0;i<d[0]; i++)
    for(j=0;j<d[1]; j++)
      gc->setLabel(j+(i*d[1]),
		   int(*((npy_int16*)PyArray_GETPTR2(seedimg_p,i,j))));

  // set data term
  gc->setDataCost(data);

  // Setup data to pass to the smooth function
  ForSmoothFn toFn;
  toFn.adj = adj_p;
  if(has_func)
    toFn.func = func;
  toFn.num_labels = num_labels;
  toFn.sites = sites;

  // set the smooth function pointer
  if(has_func)
    gc->setSmoothCost(&smoothFnCb,&toFn);
  else if(!mode)
    gc->setSmoothCost(&smoothFnI,&toFn);
  else
    gc->setSmoothCost(&smoothFnE,&toFn);

  // TODO: pointless
  // initialize labeling to previous slice 
  for(i=0;i<d[0]*d[1]; i++)
    result[i] = gc->whatLabel(i);

  // do the graph cut
  gc->swap(1);

  // retrieve labeling
  for(i=0;i<d[0]*d[1]; i++)
    result[i] = gc->whatLabel(i);

  if(gc->giveDataEnergy() > 0.1)
    printf("WARNING: NONZERO DATA COST %lld\n",gc->giveDataEnergy());

  npy_intp d_out[3];
  d_out[0] = d[0];
  d_out[1] = d[1];
  d_out[2] = d[2];

  // output = (PyArrayObject *) PyArray_FromDims(2,d,NPY_INT16);
  output = (PyArrayObject *) PyArray_SimpleNew(2,d_out,NPY_INT16);

  k=0;
  // watch the ordering of y/i and x/j
  for(i=0;i<d[0]; i++)
    for(j=0;j<d[1]; j++)
      *((npy_int16*)PyArray_GETPTR2(output,i,j)) = result[k++];

  delete [] data;
  delete [] sites;
  delete [] result;

  if(has_func)
    Py_XDECREF(func);

  return PyArray_Return(output);
}
