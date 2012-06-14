//------------------------------------------------------------------------------
//Youjie Zhou @ Wed Jun 13 15:15:49 EDT 2012
//------------------------------------------------------------------------------
#include <Python.h>
#include <numpy/arrayobject.h>
#include "RegionGraph.h"
#include "array.h"
//------------------------------------------------------------------------------
#define Pixel(array,x,y,type) (*(type*)(array->data + (x)*array->strides[1] + (y)*array->strides[0]))
//------------------------------------------------------------------------------
static void AssignLabelMatrix(TRegionGraph &RegionGraph, PyArrayObject *npLabel)
{
    int w = npLabel->dimensions[1];
    int h = npLabel->dimensions[0];

    for(int y=0;y<h;y++)
    {
        for(int x=0;x<w;x++)
        {
            short cur = Pixel(npLabel,x,y,short);

#define AddToGraph(x,y,dir,opdir)\
    {\
        if ( (x)>=0 && (x)<w && (y)>=0 && (y)<h )\
        {\
            short now = Pixel(npLabel,x,y,short);\
            if ( now != cur )\
            {\
                RegionGraph.img[cur].dir.Append(&RegionGraph.img[now]);\
                RegionGraph.img[now].opdir.Append(&RegionGraph.img[cur]);\
            }\
        }\
    }

            AddToGraph(x  ,y-1,N ,S )
            AddToGraph(x+1,y-1,NE,SW)
            AddToGraph(x+1,y  , E, W)
            AddToGraph(x+1,y+1,SE,NW)
            AddToGraph(x  ,y+1,S ,N )
            AddToGraph(x-1,y+1,SW,NE)
            AddToGraph(x-1,y  , W, E)
            AddToGraph(x-1,y-1,NW,SE)
        }
    }
}
//------------------------------------------------------------------------------
static PyObject *adjacent(PyObject *self, PyObject *args)
{
    PyArrayObject *npLabel  = NULL;

    int nLabel   = 0;
    int nDepth   = 1;
    int surround = 0;

    if (PyArg_ParseTuple(args,"O!iii",
                &PyArray_Type,&npLabel,
                &nLabel,&nDepth,&surround)==false &&
        npLabel==NULL)
    {
        PyErr_SetString(PyExc_ValueError,"Invalid arguments.\n");
        return NULL;
    }

    if (npLabel->nd != 2)
    {
        PyErr_SetString(PyExc_ValueError,"Invalid number of dimensions.\n");
        return NULL;
    }
    if (npLabel->descr->type_num != NPY_INT16) //make sure it's short
    {
        PyErr_SetString(PyExc_ValueError,"Invalid data type.\n");
        return NULL;
    }

//    unsigned w = npLabel->dimensions[1];
//    unsigned h = npLabel->dimensions[0];
//
//    fprintf(stdout,"w=%d h=%d\n",w,h);
//
//    for(int y=0;y<h;y++)
//    {
//        for(int x=0;x<w;x++)
//            fprintf(stdout,"%d ",Pixel(npLabel,x,y,short));
//        fprintf(stdout,"\n");
//    }

    TRegionGraph RegionGraph;
    RegionGraph.Alloc(nLabel);
    
    AssignLabelMatrix(RegionGraph,npLabel);
//    RegionGraph.PrintOut();

    TAlcArray<bool> AdjMatrix;
    AdjMatrix.Alloc(nLabel*nLabel);
    AdjMatrix.InitZero();

    RegionGraph.AdjMatrix(AdjMatrix.img,nDepth,surround);

    npy_intp dims[2];
    dims[0] = nLabel;
    dims[1] = nLabel;

    return PyArray_SimpleNewFromData(2,dims,NPY_BOOL,AdjMatrix.img);
}
//------------------------------------------------------------------------------
static PyMethodDef adjcMethods[] =
{
    {"adjacent",adjacent,METH_VARARGS,"Build Adjacent Matrix On a Integer Label Matrix"},
    {NULL,NULL,0,NULL}
};
//------------------------------------------------------------------------------
PyMODINIT_FUNC initadjc()
{
    (void)Py_InitModule("adjc",adjcMethods);
    import_array(); //for numpy
}
//------------------------------------------------------------------------------
