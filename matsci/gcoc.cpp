#include "gcoc.h"

#define N 255
#define LTHRESH 10
#define MODE_I 0
#define MODE_E 1
#define MODE_M 2
#define MODE_S 3
#define MODE_T 4
//#define DEBUG
#ifdef DEBUG
#define VPRINTF(...) printf(__VA_ARGS__)
#else
#define VPRINTF(...) 
#endif

static PyMethodDef gcocMethods[] = { 
  {"graph_cut", graph_cut, METH_VARARGS, "Graph Cut Optimization wrapper"},
  {"adjacent", adjacent, METH_VARARGS, "Calculate adjacency matrix"},
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
	  return 0;

	if(!PyInt_Check(result))
	  return 0;

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
	return int((1.0/double((abs(sites[s1]-sites[s2]) < LTHRESH ? LTHRESH : abs(sites[s1]-sites[s2]))+1)) * N + extra->bias);
}

int smoothFnE(int s1, int s2, int l1, int l2, void *extraData) {
	ForSmoothFn *extra = (ForSmoothFn *) extraData;
	// int num_labels = extra->num_labels;
	PyArrayObject *adj = extra->adj;
	int *sites = extra->sites;

	if(l1 == l2) { return 0; }

	if(!(*((npy_int16*)PyArray_GETPTR2(adj,l1,l2)))) { return INF; };

	// use this for edge image
	return int( 1/(std::max(double(sites[s1]),double(sites[s2]))+1) * N + extra->bias);
}

int smoothFnM(int s1, int s2, int l1, int l2, void *extraData) {
	ForSmoothFn *extra = (ForSmoothFn *) extraData;
	// int num_labels = extra->num_labels;
	PyArrayObject *adj = extra->adj;
	int *sites = extra->sites;

	if(l1 == l2) { return 0; }

	if(!(*((npy_int16*)PyArray_GETPTR2(adj,l1,l2)))) { return INF; };

	// use this for minimums in image
	return int( 1/(256-std::min(double(sites[s1]),double(sites[s2]))) * N + extra->bias);
}

int smoothFnS(int s1, int s2, int l1, int l2, void *extraData) {
	ForSmoothFn *extra = (ForSmoothFn *) extraData;
	double sigma = extra->sigma;
	PyArrayObject *adj = extra->adj;
	int *sites = extra->sites;
	if(l1 == l2) { return 0; }
	if(!(*((npy_int16*)PyArray_GETPTR2(adj,l1,l2)))) { return INF; };

	return int( exp( -1 * ((pow(double(sites[s1]) - double(sites[s2]),2.0))/(2.0*pow(sigma,2.0)))  )  * N + extra->bias);
}

int smoothFnSE(int s1, int s2, int l1, int l2, void *extraData) {
	ForSmoothFn *extra = (ForSmoothFn *) extraData;
	double sigma = extra->sigma;
	PyArrayObject *adj = extra->adj;
	int *sites = extra->sites;
	if(l1 == l2) { return 0; }
	if(!(*((npy_int16*)PyArray_GETPTR2(adj,l1,l2)))) { return INF; };

	return int( exp( -1 * ((pow(std::max(double(sites[s1]),double(sites[s2])),2.0))/(2.0*pow(sigma,2.0)))  )  * N + extra->bias);
}

void GridGraph_DArraySArray(int width,int height,int num_pixels,int num_labels);

static PyObject *graph_cut(PyObject *self, PyObject *args) {
  PyArrayObject *data_p, *img_p, *seedimg_p, *adj_p, *output;
  PyObject *func = NULL;
  int num_labels;
  int mode = MODE_I;
  double sigma = 10.0;
  int bias = 0;
  int replace = 0;
  int iter = 1;
  int d[3];
  bool has_func = false;
  // rediculous amount of typechecking, as it makes for fewer
  // headaches later
  if (!PyArg_ParseTuple(args, "O!O!O!O!i|idiiiO:set_callback", 
			&PyArray_Type, &data_p,
			&PyArray_Type, &img_p, 
			&PyArray_Type, &seedimg_p, 
			&PyArray_Type, &adj_p, 
			&num_labels,
			&mode,
			&sigma,
			&bias,
			&replace,
			&iter,
			&func)) {
    PyErr_SetString(PyExc_ValueError, "Parameters not right");
    return NULL;
  }

  VPRINTF("mode %i\n",mode);
  VPRINTF("sigma %f\n",sigma);
  VPRINTF("bias %i\n",bias);

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
    printf("WARNING: No Graph Cut work to do! Data term is 2D.\n");
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

  if( (data_p->descr->type_num != NPY_BOOL && 
       data_p->descr->type_num != NPY_INT16 )||
     img_p->descr->type_num != NPY_UINT8     ||
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
  
  // gc->setVerbosity(2);

  int *data = new int[d[0]*d[1]*num_labels];
  int *sites = new int[d[0]*d[1]];
  int *result = new int[d[0]*d[1]];
  bool *adj = new bool[adj_p->dimensions[0]*adj_p->dimensions[1]];

  int i=0,j=0,k=0;

  // load up data term
  if(data_p->descr->type_num == NPY_BOOL)
    for(i=0;i<d[0]; i++)
      for(j=0;j<d[1]; j++)
	for(k=0;k<num_labels; k++)
	  data[ ( j+i*d[1]) * num_labels + k ] = 
	    *((npy_bool*)PyArray_GETPTR3(data_p,i,j,k)) == replace ? INF : 0;
  else
    for(i=0;i<d[0]; i++)
      for(j=0;j<d[1]; j++)
	for(k=0;k<num_labels; k++)
	  data[ ( j+i*d[1]) * num_labels + k ] = 
	    *((npy_int16*)PyArray_GETPTR3(data_p,i,j,k)) == replace ? 
               INF : *((npy_int16*)PyArray_GETPTR3(data_p,i,j,k));

  // load up segmentation sites
  for(i=0;i<d[0]; i++)
    for(j=0;j<d[1]; j++)
      sites[j+(i*d[1])] = int(*((npy_uint8*)PyArray_GETPTR2(img_p,i,j)));

  // initialize
  for(i=0;i<d[0]; i++)
    for(j=0;j<d[1]; j++)
      gc->setLabel(j+(i*d[1]),
		   int(*((npy_int16*)PyArray_GETPTR2(seedimg_p,i,j))));

  // adj term
  for(i=0;i<adj_p->dimensions[0]; i++)
    for(j=0;j<adj_p->dimensions[1]; j++) {
      // adj[j+(i*adj_p->dimensions[1])] = bool(*((npy_bool*)PyArray_GETPTR2(adj_p,i,j)));
      adj[j+(i*adj_p->dimensions[1])] = bool(*((npy_bool*)PyArray_GETPTR2(adj_p,i,j)));
    }

  // set data term
  gc->setDataCost(data);

  // Setup data to pass to the smooth function
  ForSmoothFn toFn;
  toFn.adj = adj_p;
  if(has_func)
    toFn.func = func;
  toFn.num_labels = num_labels;
  toFn.sites = sites;
  toFn.bias = bias;

  // set the smooth function pointer
  if(has_func) {
    gc->setSmoothCost(&smoothFnCb,&toFn);
    VPRINTF("custom function\n");
  }
  else if(mode==MODE_I) {
    gc->setSmoothCost(&smoothFnI,&toFn);
    VPRINTF("intensity function\n");
  }
  else if(mode==MODE_E) {
    gc->setSmoothCost(&smoothFnE,&toFn);
    VPRINTF("edge function\n");
  }
  else if(mode==MODE_M) {
    gc->setSmoothCost(&smoothFnM,&toFn);
    VPRINTF("min function\n");
  }
  else if(mode==MODE_S) {
    toFn.sigma = sigma;
    gc->setSmoothCost(&smoothFnS,&toFn);
    VPRINTF("contrast-sensitive function\n");
  }
  else if(mode==MODE_T) {
    toFn.sigma = sigma;
    gc->setSmoothCost(&smoothFnSE,&toFn);
    VPRINTF("max contrast-sensitive function\n");
  }
  else { 
    PyErr_SetString(PyExc_ValueError, "Invalid mode specified");
    return NULL;
  }

  // TODO: pointless
  // initialize labeling to previous slice 
  for(i=0;i<d[0]*d[1]; i++)
    result[i] = gc->whatLabel(i);

  // printf("Energy before: %lld\n",(long long)gc->compute_energy());

  // do the graph cut
  if(iter>0) {
    gc->swap(iter,adj);
    // printf("Energy after: %lld\n",(long long)gc->compute_energy());
  }

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
  delete [] adj;

  if(has_func)
    Py_XDECREF(func);


  // Py_XDECREF(data_p);
  // Py_XDECREF(img_p);
  // Py_XDECREF(seedimg_p);
  // Py_XDECREF(adj_p);

  Py_INCREF(output);

  return Py_BuildValue("(O,L)",output,
                       (long long)gc->compute_energy());

  // return PyArray_Return(output);
}

inline int min(int a, int b) {return a>b ? b : a;}
inline int max(int a, int b) {return a<b ? b : a;}

// adjacency calculation
static PyObject *adjacent(PyObject *self, PyObject *args) {
  PyArrayObject *seedimg_p, *adj_p;
  PyObject *func = NULL;
  int adj_size;
  int d[2];

  if (!PyArg_ParseTuple(args, "O!i", 
			&PyArray_Type, &seedimg_p, 
			&adj_size)) {
    PyErr_SetString(PyExc_ValueError, "Parameters not right");
    return NULL;
  }

  // check that the objects were successfully assigned
  if (NULL == seedimg_p) {
    PyErr_SetString(PyExc_ValueError, "No data in matrix");
    return NULL; 
  }

  if(seedimg_p->nd != 2) {
    printf("%d\n",seedimg_p->nd);
    PyErr_SetString(PyExc_ValueError, "Wrong input matrix depth");
    return NULL;
  }

  // double-check, for no particular reason
  if(!PyArray_ISINTEGER(seedimg_p)) {
    PyErr_SetString(PyExc_ValueError, "Wrong intput matrix property");
    return NULL;
  }

  if(seedimg_p->descr->type_num != NPY_INT16) {
    PyErr_SetString(PyExc_ValueError, "Wrong intput matrix type");
    return NULL;
  }

  d[0] = seedimg_p->dimensions[0];
  d[1] = seedimg_p->dimensions[1];

  npy_intp d_out[2];
  d_out[0] = adj_size;
  d_out[1] = adj_size;

  adj_p = (PyArrayObject *) PyArray_SimpleNew(2,d_out,NPY_BOOL);
  PyArray_FILLWBYTE (adj_p,0);

#define get_int16(obj,i,j) int(*((npy_int16*)PyArray_GETPTR2(obj,i,j)))
#define set_bool(obj,i,j) *((npy_bool*)PyArray_GETPTR2(obj,i,j))
#define set_bool2(obj,i,j) set_bool(obj,i,j) = 1; set_bool(obj,j,i) = 1

#define EIGHT() int c = get_int16(seedimg_p,i,j);			\
  for(wi=max(0,i-DIST);wi<=min(d[0],i+DIST); wi++)			\
    for(wj=max(0,j-DIST);wj<=min(d[1],j+DIST); wj++) {			\
      int k = get_int16(seedimg_p,wi,wj);				\
      set_bool2(adj_p,k,c);						\
    }									\

#define FOUR() c = get_int16(seedimg_p,i,j);				\
  for(wi=max(0,i-DIST);wi<=min(d[0]-1,i+DIST); wi++) {			\
    k = get_int16(seedimg_p,wi,j);					\
    set_bool2(adj_p,c,k);						\
  }									\
  for(wj=max(0,j-DIST);wj<=min(d[1]-1,j+DIST); wj++) {			\
    k = get_int16(seedimg_p,i,wj);					\
    set_bool2(adj_p,c,k);						\
  }									\

#define FOUR_TWO() c = get_int16(seedimg_p,i,j);			\
  for(wi=max(0,i-2);wi<=min(d[0]-1,i+2); wi++) {			\
    k = get_int16(seedimg_p,wi,j);					\
    set_bool2(adj_p,c,k);						\
  }									\
  for(wj=max(0,j-2);wj<=min(d[1]-1,j+2); wj++) {			\
    k = get_int16(seedimg_p,i,wj);					\
    set_bool2(adj_p,c,k);						\
  }									\
  k = get_int16(seedimg_p,i-1,j);					\
  l = get_int16(seedimg_p,i+1,j);					\
  set_bool2(adj_p,k,l);							\
  k = get_int16(seedimg_p,i,j-1);					\
  l = get_int16(seedimg_p,i,j+1);					\
  set_bool2(adj_p,k,l);							\


#define EDGE(i,j,start,end)			\
  (i) = (start);				\
  for( (j) = 0; (j) < (end) ; (j) +=1 ) {	\
    FOUR();					\
  }						\

#define DIST 2

  int i=0,j=0,wi=0,wj=0,c,k,l;
  // old brute-force method
  for(i=0;i<d[0];i++) 
    for(j=0;j<d[1];j++) {
      FOUR()
    }

  // these two loops give us the checkerboard pattern
  // todo: generalize this
  /*  for(i=2;i<d[0]-2;i+=2) 
    for(j=2;j<d[1]-2;j+=2) {
      FOUR_TWO()
    }

  for(i=1;i<d[0]-1;i+=2) 
    for(j=1;j<d[1]-1;j+=2) {
      FOUR_TWO()
    }

  // handle borders
  // hack right now
  EDGE(i,j,0,d[1])
  EDGE(i,j,d[0]-1,d[1])
  EDGE(j,i,0,d[0])
  EDGE(j,i,d[0]-1,d[0])
  */
  // Py_XDECREF(seedimg_p);
  Py_INCREF(adj_p);

  return PyArray_Return(adj_p);
}
