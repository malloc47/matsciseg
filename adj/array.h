//------------------------------------------------------------------------------
#ifndef ArrayH
#define ArrayH
//------------------------------------------------------------------------------
#include <memory.h>
#include <stdlib.h>
#include <assert.h>
//------------------------------------------------------------------------------
#pragma pack(push,1)
//------------------------------------------------------------------------------
template<class T>
class TAlcArray
{
public:
    T  *img;
    int len;
    int Number;

public:
    TAlcArray()
    :img(NULL),len(0),Number(0)
    {
    }

    void Alloc(int num)
    {
        Close();

        img = (T*)malloc(num*sizeof(T));
        assert(img);
        len    = num;
        Number = 0;
    }

    void Close()
    {
        if (img)
        {
            free(img);
            img    = NULL;
            len    = 0;
            Number = 0;
        }
    }

    void InitZero()
    {
        assert(img);
        memset(img,0,len*sizeof(T));
    }

    bool Grow(int Extend)
    {
        T *t = (T*)(realloc(img,(len+Extend)*sizeof(T)));
        if (t==NULL)
            return false;
        img  = t;
        len += Extend;
        return true;
    }
    
    T &New(int MaxCap=10)
    {
        if (Number>=len)
            Grow(Number-len+MaxCap);

        Number ++;
        return img[Number-1];
    }

    void Append(T o) //only for simple data structure
    {
        for(int i=0;i<Number;i++)
            if (img[i] == o)
                return;
        New() = o;
    }

    T &operator [] (int i)
    {
        assert(img);
        assert(i>=0&&i<len);
        return img[i];
    }
};
//------------------------------------------------------------------------------
#pragma pack(pop)
//------------------------------------------------------------------------------
#endif //Exclusive Include
//------------------------------------------------------------------------------
