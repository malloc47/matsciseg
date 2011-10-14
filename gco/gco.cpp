#include "gco.h"

static PyMethodDef gcoMethods[] = { 
  {"graph_cut", graph_cut, METH_VARARGS, "Graph Cut Optimization wrapper"},
  {"test", test, METH_VARARGS, "Test"},
  {NULL, NULL, 0, NULL}};

PyMODINIT_FUNC initgco() { 
      (void) Py_InitModule("gco", gcoMethods);
      import_array();
}

unsigned int matref2(PyArrayObject *mat,int i, int j) {
  return *(unsigned int*)PyArray_GETPTR2(mat,i,j);
  // return *(unsigned int *)(mat->data + i*mat->strides[0] + j*mat->strides[1]);
}

void matset2(PyArrayObject *mat,int i, int j, unsigned int val) {
  *(unsigned int *)(mat->data + 
		    i*mat->strides[0] + 
		    j*mat->strides[1]) = val;
}

int matref3(PyArrayObject *mat,int i, int j, int k) {
  printf("%d\n",*(int*)PyArray_GETPTR3(mat,i,j,k));
  return *(int*)PyArray_GETPTR3(mat,i,j,k);
  // return *(unsigned int *)(mat->data + i*mat->strides[0] + j*mat->strides[1] + k*mat->strides[2]);
}

void matset3(PyArrayObject *mat,int i, int j, int k, unsigned int val) {
  *(unsigned int *)(mat->data + 
		    i*mat->strides[0] + 
		    j*mat->strides[1] +
		    k*mat->strides[2]) = val;
}

int valid_matrix(PyArrayObject *mat, int dim) {
  /* printf("%d==%d, %d==2\n",mat->descr->type_num,NPY_UINT8,mat->nd); */
  if ((mat->descr->type_num != NPY_UINT8 && mat->descr->type_num != NPY_INT16) || mat->nd != dim) {
     PyErr_SetString(PyExc_ValueError,
		     "Wrong dimensions or type of input matrix");
     return 0; 
   }
   return 1;
}

int smoothFn(int s1, int s2, int l1, int l2, void *extraData) {
	ForSmoothFn *extra = (ForSmoothFn *) extraData;
	int num_labels = extra->num_labels;
	PyArrayObject *adj = extra->adj;
	int *sites = extra->sites;

	if(l1 == l2) { return 0; }

	// BAD: Don't do this
	if(!matref2(adj,l1,l2)) { return INF; }
	
	//return int((1.0/double((abs(sites[s1]-sites[s2]) < LTHRESH ? LTHRESH : abs(sites[s1]-sites[s2]))+1)) * N);
	//return int( 1/(double(sites[s1]+sites[s2])/2) * N );
	//return int( N - int(double(sites[s1]+sites[s2])/2));
	return int( 1/(std::max(double(sites[s1]),double(sites[s2]))+1) * N );
	//return int( 1/(min(double(sites[s1]),double(sites[s2]))+1) * N );
}

void GridGraph_DArraySArray(int width,int height,int num_pixels,int num_labels);

int smoothFn2(int p1, int p2, int l1, int l2) {
	if ( (l1-l2)*(l1-l2) <= 4 ) return((l1-l2)*(l1-l2));
	else return(4);
}

static PyObject *test(PyObject *self, PyObject *args) {
  PyArrayObject *data_p;

  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &data_p))
    return NULL;

  int t = *((npy_uint8*)PyArray_GETPTR3(data_p,0,0,0));

  printf("C:\n");
  // printf("%u\n",matref3(data_p,0,0,0));
  printf("%u\n",t);

  printf("%d==%d not %d",data_p->descr->type_num,NPY_UINT8,NPY_INT16);

  return Py_None;
}

static PyObject *graph_cut(PyObject *self, PyObject *args) {
  PyArrayObject *data_p, *img_p, *seedimg_p, *adj_p, *output;
  int num_labels;
  int d[2];
  if (!PyArg_ParseTuple(args, "O!O!O!O!i", 
			&PyArray_Type, &data_p,
			&PyArray_Type, &img_p, 
			&PyArray_Type, &seedimg_p, 
			&PyArray_Type, &adj_p, 
			&num_labels))
    return NULL;

  // check that the objects were successfully assigned
  if (NULL == data_p    ||
      NULL == img_p     ||
      NULL == seedimg_p ||
      NULL == adj_p )
    return NULL; 

  // check that the objects are valid matrices
  if(!valid_matrix(data_p,3)    || 
     !valid_matrix(img_p,2)     || 
     !valid_matrix(seedimg_p,2) || 
     !valid_matrix(adj_p,2))
    return NULL;

  d[0] = seedimg_p->dimensions[0];
  d[1] = seedimg_p->dimensions[1];

  int t = *((int*)PyArray_GETPTR3(data_p,0,0,1));

  printf("C:\n");
  // printf("%u\n",matref3(data_p,0,0,0));
  printf("%d\n",t);

  printf("%d==%d not %d",data_p->descr->type_num,NPY_UINT8,NPY_INT16);

  // For testing
  // GridGraph_DArraySArray(10,5,50,7);
  output = (PyArrayObject *) PyArray_FromDims(2,d,NPY_UINT8);
  return PyArray_Return(output);

  printf("initializing graph optimization engine\n");

  GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(d[1],d[0],num_labels);

  printf("initializing c arrays\n");

  int *data = new int[d[0]*d[1]*num_labels];
  int *sites = new int[d[0]*d[1]];
  int *result = new int[d[0]*d[1]];

  unsigned int i,j,k;

  printf("copying data term\n");
  printf("%d X %d X %d\n",data_p->strides[0], data_p->strides[1], data_p->strides[2]);
  printf("%d X %d X %d\n",data_p->dimensions[0], data_p->dimensions[1], data_p->dimensions[2]);

  // load up data term
  for(i=0;i<d[0]; i++)
    for(j=0;j<d[1]; j++)
      for(k=0;k<num_labels; k++)
  	data[ ( j+i*d[1]) * num_labels + k ] = 
	  int(matref3(data_p,i,j,k)) == 255 ? 0 : INF; 
  printf("copying segmentation sites\n");

  // load up segmentation sites
  for(i=0;i<d[0]; i++)
    for(j=0;j<d[1]; j++)
      sites[j+(i*d[1])] = matref2(img_p,i,j);

  printf("initializing labels\n");

  // initialize
  for(i=0;i<d[0]; i++)
    for(j=0;j<d[1]; j++)
      gc->setLabel(j+(i*d[1]),matref2(seedimg_p,i,j));

  printf("initializing data term\n");

  // set data term
  gc->setDataCost(data);

  // Setup data to pass to the smooth function
  ForSmoothFn toFn;
  toFn.adj = adj_p;
  toFn.num_labels = num_labels;
  toFn.sites = sites;

  printf("initializing smoothness funciton pointer\n");

  // set the smooth function pointer
  gc->setSmoothCost(&smoothFn,&toFn);
  // gc->setSmoothCost(&smoothFn2);

  // TODO: pointless
  // initialize labeling to previous slice 
  for(i=0;i<d[0]*d[1]; i++)
    result[i] = gc->whatLabel(i);

  printf("doing graph cut\n");

  // do the graph cut
  gc->swap(1);

  printf("retreiving results\n");

  // retrieve labeling
  for(i=0;i<d[0]*d[1]; i++)
    result[i] = gc->whatLabel(i);

  if(gc->giveDataEnergy() > 0)
    printf("WARNING: NONZERO DATA COST");

  printf("creating output array\n");

  output = (PyArrayObject *) PyArray_FromDims(2,d,NPY_UINT8);

  // watch the ordering of y/i and x/j
  for(i=0;i<d[0]; i++)
    for(j=0;j<d[1]; j++)
      matset2(output,i,j,result[k++]);

  delete [] data;
  delete [] sites;
  delete [] result;

  return PyArray_Return(output);
}
