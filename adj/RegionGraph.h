//------------------------------------------------------------------------------
#ifndef RegionGraphH
#define RegionGraphH
//------------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <assert.h>
#include "array.h"
//------------------------------------------------------------------------------
#pragma pack(push,1)
//------------------------------------------------------------------------------
class TRegionGraph
{
public:
    struct TNode
    {
        TAlcArray<TNode*> N ;
        TAlcArray<TNode*> NE;
        TAlcArray<TNode*>  E;
        TAlcArray<TNode*> SE;
        TAlcArray<TNode*> S ;
        TAlcArray<TNode*> SW;
        TAlcArray<TNode*>  W;
        TAlcArray<TNode*> NW;
    };

public:
    TNode   *img;
    unsigned Number;

public:
    TRegionGraph()
    :img(NULL),Number(0)
    {
    }

    void Alloc(int num)
    {
        Close();
        img = new TNode[num];
        assert(img!=NULL);
        Number = num;
    }

    void Close()
    {
        if (img)
        {
            delete []img;
            img    = NULL;
            Number = 0;
        }
    }

    int GetInx(TNode *node)
    {
        return node-img;
    }

    void AdjMatrix(bool *buff, int nDepth=1, bool surround=false);

    void PrintOut()
    {
        for(int i=0;i<Number;i++)
        {
            fprintf(stdout,"%d\n",i);

#define debugout(dir)\
            {\
                fprintf(stdout,"\t%02s ",#dir);\
                for(int k=0;k<img[i].dir.Number;k++)\
                    fprintf(stdout,"%d ",img[i].dir.img[k]-img);\
                fprintf(stdout,"\n");\
            }

            debugout(N)
            debugout(NE)
            debugout(E)
            debugout(SE)
            debugout(S)
            debugout(SW)
            debugout(W)
            debugout(NW)
        }
    }
};
//------------------------------------------------------------------------------
#pragma pack(pop)
//------------------------------------------------------------------------------
#endif //Exclusive Include
//------------------------------------------------------------------------------
