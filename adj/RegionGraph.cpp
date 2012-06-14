//------------------------------------------------------------------------------
#include "RegionGraph.h"
//------------------------------------------------------------------------------
#define SetAdj(array,x,y)\
    {\
        *(array + y*Number + x) = true;\
        *(array + x*Number + y) = true;\
    }
//------------------------------------------------------------------------------
#define AdjStubFunc(dir)\
    static void AdjStub##dir(bool *buff, int x, TRegionGraph::TNode *img, int Number, TRegionGraph::TNode *node, int cnt, int nDepth)\
    {\
        if (cnt>=nDepth)\
            return;\
        \
        for(int n=0;n<node->dir.Number;n++)\
        {\
            SetAdj(buff,x,(node->dir.img[n]-img))\
            AdjStub##dir(buff,x,img,Number,node->dir.img[n],cnt+1,nDepth);\
        }\
    }
//------------------------------------------------------------------------------
AdjStubFunc(N)
AdjStubFunc(NE)
AdjStubFunc(E)
AdjStubFunc(SE)
AdjStubFunc(S)
AdjStubFunc(SW)
AdjStubFunc(W)
AdjStubFunc(NW)
//------------------------------------------------------------------------------
#define AdjStubRun(dir)\
    {\
        AdjStub##dir(buff,i,img,Number,img+i,0,nDepth);\
    }\
//------------------------------------------------------------------------------
void TRegionGraph::AdjMatrix(bool *buff, int nDepth, bool surround)
{
    if (nDepth<1) //at least one neighbour should be counted
        return;

    for(int i=0;i<Number;i++)
    {
        AdjStubRun(N)
        AdjStubRun(E)
        AdjStubRun(S)
        AdjStubRun(W)

        if (surround)
        {
            AdjStubRun(NE)
            AdjStubRun(SE)
            AdjStubRun(SW)
            AdjStubRun(NW)
        }
    }
}
//------------------------------------------------------------------------------
