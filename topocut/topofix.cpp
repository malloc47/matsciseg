//mex CC=gcc-4.1 CXX=g++-4.1 CXX=g++-4.1 LD=g++-4.1 -lm -output persistence persistence_original_2D_kill.cpp

#include "Python.h"
#include "numpy/arrayobject.h"

#include <exception>
#include <math.h>
#include <queue>
#include <stack>
#include <vector>
#include <map>
#include <set>
#include <list>
#include <assert.h>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
using namespace std;

static PyObject *topofix(PyObject *self, PyObject *args);

static PyMethodDef topofixMethods[] = { 
  {"topofix", topofix, METH_VARARGS, "Python TopoCut Wrapper"},
  {NULL, NULL, 0, NULL}};

PyMODINIT_FUNC inittopofix() { 
      (void) Py_InitModule("topofix", topofixMethods);
      import_array();
}

void myMessage (const string msg,bool showtime){
	time_t now;
	 time(&now);
	if (showtime){
	  cout << ctime(&now) << "-------" << msg.c_str() << endl;
	}else{
	  cout<<msg.c_str()<<endl;
	}
}

#define  OUTPUT_MSG(MSG)  {stringstream * ss=new stringstream(stringstream::in | stringstream::out); *ss << MSG << ends; myMessage(ss->str(),true); delete ss; }

#define  OUTPUT_NOTIME_MSG(MSG)  {stringstream * ss=new stringstream(stringstream::in | stringstream::out); *ss << MSG << ends; myMessage(ss->str(),false); delete ss; }

// Function definitions.
// -----------------------------------------------------------------

void myassert(bool a, int source=0){
	if (!a){
	       	int b=5;
		if(source==0){
			OUTPUT_MSG("ASSERTION FAILED!!!!!!!");
		}else if(source==2){
			b=6;
			OUTPUT_MSG("ASSERTION FAILED!!---- Due to hole--killing!!!!!!!!-----captured");
		}else{
			OUTPUT_MSG("ASSERTION FAILED!!---- Due to hole--killing!!!!!!!!");
		}
	}
//	assert(a);
	return;
}

class myDoubleMatrix {
	public:
		int nrow;
		int ncol;
		vector< vector< double > > data;
  myDoubleMatrix(int m,int n,double v=0.0) { 
	  nrow=m;
	  ncol=n;
	  int i,j;
	  for (i=0;i<nrow;i++)
		  data.push_back(vector< double >(ncol,v));
	  return;
  } 
  double get(int r, int c) { 
	 	myassert((0<=r)&&(r<nrow));
	 	myassert((0<=c)&&(c<ncol));
	     return data[r][c];
  }
  void put(int r, int c, double v) { 
	 	myassert((0<=r)&&(r<nrow));
	 	myassert((0<=c)&&(c<ncol));
	        data[r][c]=v;
  }
  void input1D(double *ptr){
	  int i,j;
	  for (i=0;i<nrow;i++)
		  for (j=0;j<ncol;j++)
			  data[i][j]=ptr[j*nrow+i];
  }
  void output1D(double *ptr){
	  int i,j;
	  for (i=0;i<nrow;i++)
		  for (j=0;j<ncol;j++)
			  ptr[j*nrow+i]=data[i][j];
  }
  void output1D_C(double *ptr){
	  int i,j;
      for (j=0;j<ncol;j++)
        for (i=0;i<nrow;i++)
          ptr[i*ncol+j]=data[i][j];
  }
};  

class Vertex{
	public:
	int xidx;
	int yidx;
	int mapXCoord;
	int mapYCoord;
	Vertex(int xi,int yi) : 
	xidx(xi),yidx(yi),mapXCoord(xi*2),mapYCoord(yi*2){}
	Vertex() : xidx(-1),yidx(-1),mapXCoord(-1),mapYCoord(-1){}
};

class Edge{
	public:
	int v1_order;
	int v2_order;
	int mapXCoord;
	int mapYCoord;
	Edge(int v1o,int v2o, int mx, int my) : 
	v1_order(min(v1o,v2o)),v2_order(max(v1o,v2o)),mapXCoord(mx),mapYCoord(my){}
	Edge() : 
	v1_order(-1),v2_order(-1),mapXCoord(-1),mapYCoord(-1){}
};

class Triangle{
	public:
	int v1_order;
	int v2_order;
	int v3_order;
	int v4_order;
	int e1_order;
	int e2_order;
	int e3_order;
	int e4_order;
	int mapXCoord;
	int mapYCoord;
	Triangle(int v1o,int v2o,int v3o,int v4o,int e1o,int e2o,int e3o,int e4o,int mx,int my):
	mapXCoord(mx),mapYCoord(my){
		vector< int >tmpvec (4,0);
		tmpvec[0]=v1o;
		tmpvec[1]=v2o;
		tmpvec[2]=v3o;
		tmpvec[3]=v4o;
		sort(tmpvec.begin(),tmpvec.end());
		v1_order=tmpvec[0];
		v2_order=tmpvec[1];
		v3_order=tmpvec[2];
		v4_order=tmpvec[3];

		vector< int >tmpEdgevec (4,0);
		tmpEdgevec[0]=e1o;
		tmpEdgevec[1]=e2o;
		tmpEdgevec[2]=e3o;
		tmpEdgevec[3]=e4o;
		sort(tmpEdgevec.begin(),tmpEdgevec.end());
		e1_order=tmpEdgevec[0];
		e2_order=tmpEdgevec[1];
		e3_order=tmpEdgevec[2];
		e4_order=tmpEdgevec[3];
	}
	Triangle() : 
	v1_order(-1),v2_order(-1),v3_order(-1),v4_order(-1),mapXCoord(-1),mapYCoord(-1){}
};

myDoubleMatrix * phi;

bool vCompVal(Vertex a, Vertex b){ return phi->data[a.xidx][a.yidx] < phi->data[b.xidx][b.yidx]; }

bool eComp(Edge a, Edge b){ 
	if(a.v2_order!=b.v2_order)
		return a.v2_order<b.v2_order;
	if(a.v1_order!=b.v1_order)
		return a.v1_order<b.v1_order;
}

bool trigComp(Triangle a, Triangle b){ 
	if(a.e4_order!=b.e4_order)
		return a.e4_order<b.e4_order;
	if(a.e3_order!=b.e3_order)
		return a.e3_order<b.e3_order;
	if(a.e2_order!=b.e2_order)
		return a.e2_order<b.e2_order;
	if(a.e1_order!=b.e1_order)
		return a.e1_order<b.e1_order;
}
// bool trigComp(Triangle a, Triangle b){ 
// 	if(a.v4_order!=b.v4_order)
// 		return a.v4_order<b.v4_order;
// 	if(a.v3_order!=b.v3_order)
// 		return a.v3_order<b.v3_order;
// 	if(a.v2_order!=b.v2_order)
// 		return a.v2_order<b.v2_order;
// 	if(a.v1_order!=b.v1_order)
// 		return a.v1_order<b.v1_order;
// }

#define BIG_INT 0xFFFFFFF
template <class Container>
struct Counter : public std::iterator <std::output_iterator_tag,
                         void, void, void, void>
{ 
	size_t &cnt;

    Counter(size_t &x) : cnt(x) {}	
 
	template<typename t>
    Counter& operator=(t)
	{        
        return *this;
    }
    
    Counter& operator* () 
	{
        return *this;
    }
    
    Counter& operator++(int) 
	{
		++cnt;
        return *this;
    }    

	Counter& operator++() 
	{
		++cnt;
        return *this;
    }    
};

// We avoid excessive allocations by calculating the size of the resulting list.
// Then we resize the result and populate it with the actual values.
vector< int > list_sym_diff(vector< int > &sa, vector< int > &sb){
	//assume inputs are both sorted increasingly	
	size_t count = 0;
	Counter< vector< int > > counter(count);
	set_symmetric_difference(sa.begin(), sa.end(), sb.begin(), sb.end(), counter);
	vector< int > out;	
	
	out.reserve(count);
	set_symmetric_difference(sa.begin(), sa.end(), sb.begin(), sb.end(), back_inserter(out));	

	return out;
}


//-----------------------------------------------------
//vertex-edge pair and persistence
//edge-trig pair and persistence
//-----------------------------------------------------
class VertEdgePair{
  public:
  int vbidx;
  int edidx;
  double robustness;
  double birth;
  double death;

  //initialize coordinates using the input vertices and persistence
  VertEdgePair(int vbi, int edi, double rob, double b, double d) : 
  vbidx(vbi),edidx(edi),
  robustness(rob),birth(b),death(d){}
  
  bool operator<(const VertEdgePair &rhs) const{
    return (this->robustness >= rhs.robustness);
  }
};

class EdgeTrigPair{
  public:
  int ebidx, tdidx;
  
  double robustness;
  double birth;
  double death;
  
  //initialize coordinates using the input vertices and persistence
  EdgeTrigPair( int ebi, int tdi, double rob,double b,double d) : 
  ebidx(ebi),tdidx(tdi),
  robustness(rob),birth(b),death(d){}

  bool operator<(const EdgeTrigPair &rhs) const{
    return (this->robustness >= rhs.robustness);
  }
};

//-----------------------------------------------------
//compute 2D persistence
// m,n: size of the two dimensions
// pers_thd: threshold of persistence (only bigger persistence would be recorded
// rob_thd: threshold of robustness
// levelset_val: the image value of the levelset (0 in image segmentation)
// persistenceM: persistence flow, +pers to creator and -pers to destroyer
// robustnessM: robustness flow, +pers to creator or -pers to destroyer, depending on which is closer to the levelset_val 
// veList: vertex-edge pair, together with corresponding persistence
// etrigList: edge-triangle pair, together with corresponding persistence
//
// assume the global variable phi is already available (which stores the height function)
//-----------------------------------------------------
#define MAX_PERS_PTS 1000	//maximum of numbers of persistence pts

enum CellTypeEnum {CT_UNDEFINED, VERTEX, EDGEVERT, EDGEHORI, TRIG};
enum EdgePersTypeEnum {EP_UNDEFINED, DESTROYER, CREATOR};
enum CellFixTypeEnum {CF_UNDEFINED, MOVEDOWN, MOVEUP};
class CellMap{
public:
	int vertNum,edgeNum,trigNum;
	int mapNRow, mapNCol;
	int currEOrder, currTOrder;
	vector< vector < int > > cellOrder;
	vector< vector < CellTypeEnum > > cellType;
	
	vector< vector < EdgePersTypeEnum > > edgePersType;
	
	vector< Vertex > * vList;
	vector< Edge > * eList;
	vector< Triangle > * trigList;

	vector< vector < CellFixTypeEnum > > cellFixType;

	void setVertOrder(int rid, int cid, int vorder){
		
		myassert( (rid >=0)&&(rid < mapNRow)&&(cid >= 0)&&(cid < mapNCol) );
		myassert( cellType[rid][cid] == VERTEX );
		
		myassert(cellOrder[rid][cid] == -1); //this vert has not been specified
		
		cellOrder[rid][cid] = vorder;

		vector< vector < int > > neighborOrder(5, vector< int >(5,-1));

		int i,j;
		
		// get the neighboring orders
		for (i=rid-2;i<=rid+2;i++){
			if ((i<0)||(i>=mapNRow)) continue;
			for(j=cid-2;j<=cid+2;j++){
				if ((j<0)||(j>=mapNCol)) continue;
				neighborOrder[i+2-rid][j+2-cid]=cellOrder[i][j];
			}
		}
		
		// update the neighboring orders
// 		int ue,de,le,re,ult,urt,dlt,drt;
// 		int uv,dv,lv,rv,ulv,urv,dlv,drv;
		int & uv = neighborOrder[0][2];
		int & dv = neighborOrder[4][2];
		int & lv = neighborOrder[2][0];
		int & rv = neighborOrder[2][4];
		int & ulv = neighborOrder[0][0];
		int & urv = neighborOrder[0][4];
		int & dlv = neighborOrder[4][0];
		int & drv = neighborOrder[4][4];

		int & ue = neighborOrder[1][2];
		int & de = neighborOrder[3][2];
		int & le = neighborOrder[2][1];
		int & re = neighborOrder[2][3];
		int & ult = neighborOrder[1][1];
		int & urt = neighborOrder[1][3];
		int & dlt = neighborOrder[3][1];
		int & drt = neighborOrder[3][3];

		if (uv>=0){
			ue = currEOrder;
			currEOrder++;
		}
		if (dv>=0){
			de = currEOrder;
			currEOrder++;
		}
		if (lv>=0){
			le = currEOrder;
			currEOrder++;
		}
		if (rv>=0){
			re = currEOrder;
			currEOrder++;
		}
		if ((uv>=0)&&(ulv>=0)&&(lv>=0)){
			ult = currTOrder;
			currTOrder++;
		}
		if ((uv>=0)&&(urv>=0)&&(rv>=0)){
			urt = currTOrder;
			currTOrder++;
		}
		if ((dv>=0)&&(dlv>=0)&&(lv>=0)){
			dlt = currTOrder;
			currTOrder++;
		}
		if ((dv>=0)&&(drv>=0)&&(rv>=0)){
			drt = currTOrder;
			currTOrder++;
		}

		// copy the neighboring orders back
		for (i=rid-2;i<=rid+2;i++){
			if ((i<0)||(i>=mapNRow)) continue;
			for(j=cid-2;j<=cid+2;j++){
				if ((j<0)||(j>=mapNCol)) continue;
				if (cellOrder[i][j]!=neighborOrder[i+2-rid][j+2-cid]){
					myassert( cellOrder[i][j]==-1 );
					myassert( cellType[i][j]!=VERTEX );
					cellOrder[i][j]=neighborOrder[i+2-rid][j+2-cid];
				}
			}
		}
		
	}


	CellMap(vector< Vertex > * vL, vector< Edge > * eL, vector< Triangle > * tL){

		int i,j,idx;

		vList=vL;
		eList=eL;
		trigList=tL;
		myassert(eList->size()==0);
		myassert(trigList->size()==0);
		
		vertNum = vList->size();
		edgeNum = phi->nrow *(phi->ncol - 1) + phi->ncol*(phi->nrow - 1);
		trigNum = (phi->ncol - 1) * (phi->nrow - 1);

		mapNRow = 2*(phi->nrow)-1;
		mapNCol = 2*(phi->ncol)-1;
		currEOrder=0;
		currTOrder=0;

//		int i, j;

		cellOrder.assign(mapNRow, vector< int >(mapNCol, -1));
		edgePersType.assign(mapNRow, vector< EdgePersTypeEnum >(mapNCol, EP_UNDEFINED));
		cellFixType.assign(mapNRow, vector< CellFixTypeEnum >(mapNCol, CF_UNDEFINED));		

		// create cellType, row by row
		vector< CellTypeEnum > 	zeroRow( mapNCol, VERTEX );
		for (i = 0; i < mapNCol; i++)
			if ( i % 2 == 1 ) zeroRow[i] = EDGEHORI;
		vector< CellTypeEnum > 	firstRow( mapNCol, EDGEVERT );
		for (i = 0; i < mapNCol; i++)
			if ( i % 2 == 1 ) firstRow[i] = TRIG;
		for ( i = 0; i < mapNRow; i++)
			if (i % 2 == 0)
				cellType.push_back(zeroRow);
			else
				cellType.push_back(firstRow);

		// flood 
		for( i = 0; i<vertNum; i++)
			setVertOrder((* vList)[i].xidx*2 ,(* vList)[i].yidx*2, i);
		myassert(currEOrder == edgeNum);
		myassert(currTOrder == trigNum);
		
		eList->assign(edgeNum,Edge());
		trigList->assign(trigNum,Triangle());

//		int idx;
		// build elist and tlist
		for (i=0;i<mapNRow;i++)
			for (j=0;j<mapNCol;j++){
				idx = cellOrder[i][j];
				myassert(idx>=0);
				if (cellType[i][j]==EDGEHORI){
					myassert(idx<edgeNum);
					(* eList)[idx]=Edge(cellOrder[i][j-1],cellOrder[i][j+1],i,j);
				}else if(cellType[i][j]==EDGEVERT){
					myassert(idx<edgeNum);
					(* eList)[idx]=Edge(cellOrder[i-1][j],cellOrder[i+1][j],i,j);
				}else if(cellType[i][j]==TRIG){
					myassert(idx<trigNum);
					(* trigList)[idx]=Triangle(cellOrder[i-1][j-1],cellOrder[i+1][j-1],cellOrder[i-1][j+1],cellOrder[i+1][j+1],
								cellOrder[i-1][j],cellOrder[i+1][j],cellOrder[i][j-1],cellOrder[i][j+1],
								i,j);
				}
			}

		//resort elist and triglist, so that they are alphabetically sorted
		sort(eList->begin(),eList->end(),eComp);
		for(i=0;i<edgeNum;i++)
			cellOrder[(*eList)[i].mapXCoord][(*eList)[i].mapYCoord]=i;
		for (i=0;i<mapNRow;i++)
			for (j=0;j<mapNCol;j++){
				idx = cellOrder[i][j];
				myassert(idx>=0);
				if(cellType[i][j]==TRIG){
					myassert(idx<trigNum);
					(* trigList)[idx]=Triangle(cellOrder[i-1][j-1],cellOrder[i+1][j-1],cellOrder[i-1][j+1],cellOrder[i+1][j+1],
								cellOrder[i-1][j],cellOrder[i+1][j],cellOrder[i][j-1],cellOrder[i][j+1],
								i,j);
				}
			}
		sort(trigList->begin(),trigList->end(),trigComp);
		for(i=0;i<trigNum;i++)
			cellOrder[(*trigList)[i].mapXCoord][(*trigList)[i].mapYCoord]=i;

	}
	
	void buildBoundary2D(vector<vector < int > > * boundary_2D){
		int i, j, idx;
		for (i=0; i<trigNum; i++){
			(* boundary_2D)[i].push_back( (* trigList)[i].e1_order );
			(* boundary_2D)[i].push_back( (* trigList)[i].e2_order );
			(* boundary_2D)[i].push_back( (* trigList)[i].e3_order );
			(* boundary_2D)[i].push_back( (* trigList)[i].e4_order );
		}
	}
					
	void buildBoundary1D(vector<vector < int > > * boundary_1D){
		int i, j, idx;
		for (i=0; i<edgeNum; i++){
			(* boundary_1D)[i].push_back( (* eList)[i].v1_order );
			(* boundary_1D)[i].push_back( (* eList)[i].v2_order );
		}
	}

	void setEPersType( vector< int > * low_2D_e2t ){
		myassert(low_2D_e2t->size() == edgeNum);
		int i,j,idx;
		for (i=0; i< mapNRow; i++)
			for (j=0; j< mapNCol; j++)
				if ((cellType[i][j]==EDGEVERT)||(cellType[i][j]==EDGEHORI)){
					idx=cellOrder[i][j];
					if ((* low_2D_e2t)[idx]!=-1)
						edgePersType[i][j] = CREATOR;
					else
						edgePersType[i][j] = DESTROYER;
				}
	}

	void setVertFixType(int rid, int cid, CellFixTypeEnum ftype){
		myassert(ftype != CF_UNDEFINED);

		myassert( (rid >=0)&&(rid < mapNRow)&&(cid >= 0)&&(cid < mapNCol) );

		myassert( cellType[rid][cid] == VERTEX );
		
		if(cellFixType[rid][cid] == ftype) return;	//alreay fixed
		
		myassert(cellFixType[rid][cid] == CF_UNDEFINED);	//cell was either unknown, or the same as ftype

		cellFixType[rid][cid] = ftype;	//fix the vertex first

		vector< vector < CellFixTypeEnum > > neighborFixType(5, vector< CellFixTypeEnum >(5,CF_UNDEFINED));

		int i,j;
		
		// get the neighboring orders
		for (i=rid-2;i<=rid+2;i++){
			if ((i<0)||(i>=mapNRow)) continue;
			for(j=cid-2;j<=cid+2;j++){
				if ((j<0)||(j>=mapNCol)) continue;
				neighborFixType[i+2-rid][j+2-cid]=cellFixType[i][j];
			}
		}
		
// 		CellFixTypeEnum ue,de,le,re,ult,urt,dlt,drt;
// 		CellFixTypeEnum uv,dv,lv,rv,ulv,urv,dlv,drv;
		CellFixTypeEnum & uv = neighborFixType[0][2];
		CellFixTypeEnum & dv = neighborFixType[4][2];
		CellFixTypeEnum & lv = neighborFixType[2][0];
		CellFixTypeEnum & rv = neighborFixType[2][4];
		CellFixTypeEnum & ulv = neighborFixType[0][0];
		CellFixTypeEnum & urv = neighborFixType[0][4];
		CellFixTypeEnum & dlv = neighborFixType[4][0];
		CellFixTypeEnum & drv = neighborFixType[4][4];

		CellFixTypeEnum & ue = neighborFixType[1][2];
		CellFixTypeEnum & de = neighborFixType[3][2];
		CellFixTypeEnum & le = neighborFixType[2][1];
		CellFixTypeEnum & re = neighborFixType[2][3];
		CellFixTypeEnum & ult = neighborFixType[1][1];
		CellFixTypeEnum & urt = neighborFixType[1][3];
		CellFixTypeEnum & dlt = neighborFixType[3][1];
		CellFixTypeEnum & drt = neighborFixType[3][3];

		if (uv==ftype){
			myassert(ue==CF_UNDEFINED);
			ue = ftype;
		}
		if (dv==ftype){
			myassert(de==CF_UNDEFINED);
			de = ftype;
		}
		if (lv==ftype){
			myassert(le==CF_UNDEFINED);
			le = ftype;
		}
		if (rv==ftype){
			myassert(re==CF_UNDEFINED);
			re = ftype;
		}
		if ((uv==ftype)&&(ulv==ftype)&&(lv==ftype)){
			myassert(ult == CF_UNDEFINED);
			ult = ftype;
		}
		if ((uv==ftype)&&(urv==ftype)&&(rv==ftype)){
			myassert(urt == CF_UNDEFINED);
			urt = ftype;
		}
		if ((dv==ftype)&&(dlv==ftype)&&(lv==ftype)){
			myassert(dlt == CF_UNDEFINED);
			dlt = ftype;
		}
		if ((dv==ftype)&&(drv==ftype)&&(rv==ftype)){
			myassert(drt == CF_UNDEFINED);
			drt = ftype;
		}

		// copy the neighboring orders back
		for (i=rid-2;i<=rid+2;i++){
			if ((i<0)||(i>=mapNRow)) continue;
			for(j=cid-2;j<=cid+2;j++){
				if ((j<0)||(j>=mapNCol)) continue;
				if (cellFixType[i][j]!=neighborFixType[i+2-rid][j+2-cid]){
					myassert( cellFixType[i][j]==CF_UNDEFINED );
					myassert( cellType[i][j]!=VERTEX );
					cellFixType[i][j]=neighborFixType[i+2-rid][j+2-cid];
				}
			}
		}
		
	}
	
	CellFixTypeEnum getFixType( int dim, int idx ){
		int mapX, mapY;
		if(dim == 0){
			mapX = (* vList)[idx].mapXCoord;
			mapY = (* vList)[idx].mapYCoord;
			return cellFixType[mapX][mapY];
		};
		if(dim == 1){
			mapX = (* eList)[idx].mapXCoord;
			mapY = (* eList)[idx].mapYCoord;
			return cellFixType[mapX][mapY];
		};
		if(dim == 2){
			mapX = (* trigList)[idx].mapXCoord;
			mapY = (* trigList)[idx].mapYCoord;
			return cellFixType[mapX][mapY];
		};
		myassert(false);
	}
	
	void flood_comp(int vBirth, int eDeath, double btime, double dtime, myDoubleMatrix * const perturbM, myDoubleMatrix * const critM ){
		//starting from vBirth, going through any adjacent edge with <= eDeath idx

		vector< bool > vert_visited(vertNum,false);
		queue< int > vert_queue;

		vert_queue.push(vBirth);

		int high_v_of_edeath = (*eList)[eDeath].v2_order;	//higher v of edeath should not be included, might break merge pathes
		myassert(high_v_of_edeath!=vBirth);

		vector< int > parent_v(vertNum,-1); 

		int xcoord,ycoord,tmpv,mapxcoord,mapycoord;
		while(! vert_queue.empty()){
			tmpv = vert_queue.front();
//			if ((vert_visited[tmpv])||(tmpv!=high_v_of_edeath)){
			if (vert_visited[tmpv]){
				vert_queue.pop();
				continue;
			}
			vert_visited[tmpv] = true;

			xcoord=(* vList)[tmpv].xidx;
			ycoord=(* vList)[tmpv].yidx;
			
			mapxcoord=(* vList)[tmpv].mapXCoord;
			mapycoord=(* vList)[tmpv].mapYCoord;
			myassert(cellType[mapxcoord][mapycoord]==VERTEX);
			vert_queue.pop();

			if((tmpv==high_v_of_edeath)&&(cellFixType[mapxcoord][mapycoord]==MOVEDOWN)){
				// special case, keep tmpv unchanged, but flood on the same side of vBirth;

				//skip case when tmpv is on the boundary of the image
				if((mapxcoord==0)||(mapycoord==0)||(mapxcoord==mapNRow)||(mapycoord==mapNCol)) continue;

				CellFixTypeEnum uvFix=cellFixType[mapxcoord-2][mapycoord];
				CellFixTypeEnum dvFix=cellFixType[mapxcoord+2][mapycoord];
				CellFixTypeEnum lvFix=cellFixType[mapxcoord][mapycoord-2];
				CellFixTypeEnum rvFix=cellFixType[mapxcoord][mapycoord+2];
				int uvO=cellOrder[mapxcoord-2][mapycoord];
				int dvO=cellOrder[mapxcoord+2][mapycoord];
				int lvO=cellOrder[mapxcoord][mapycoord-2];
				int rvO=cellOrder[mapxcoord][mapycoord+2];
				int pvO=parent_v[tmpv];
				myassert(pvO>=0);

				if(((pvO==uvO)||(pvO==dvO))&&(lvFix!=MOVEDOWN))
					if (! vert_visited[lvO]){
						parent_v[lvO]=tmpv;
						vert_queue.push(lvO);
					}
				else if(((pvO==uvO)||(pvO==dvO))&&(rvFix!=MOVEDOWN))
					if (! vert_visited[rvO]){
						parent_v[rvO]=tmpv;
						vert_queue.push(rvO);
					}
				else if(((pvO==lvO)||(pvO==rvO))&&(uvFix!=MOVEDOWN))
					if (! vert_visited[uvO]){
						parent_v[uvO]=tmpv;
						vert_queue.push(uvO);
					}
				else if(((pvO==lvO)||(pvO==rvO))&&(dvFix!=MOVEDOWN))
					if (! vert_visited[dvO]){
						parent_v[dvO]=tmpv;
						vert_queue.push(dvO);
					}				
			}else{
				myassert((cellFixType[mapxcoord][mapycoord]==CF_UNDEFINED)||(cellFixType[mapxcoord][mapycoord]==MOVEUP));
				perturbM->data[xcoord][ycoord]=min(btime,perturbM->data[xcoord][ycoord]);
				critM->data[xcoord][ycoord]=min(-1.0*vBirth,critM->data[xcoord][ycoord]);
				setVertFixType(mapxcoord,mapycoord,MOVEUP);

				//check all neighbor edges, push relevant vertices into the queue;
				//up vertex
				if ((mapxcoord>0)&&(edgePersType[mapxcoord-1][mapycoord]==DESTROYER)&&(cellOrder[mapxcoord-1][mapycoord]<eDeath))
					if (! vert_visited[cellOrder[mapxcoord-2][mapycoord]]){
						parent_v[cellOrder[mapxcoord-2][mapycoord]]=tmpv;
						vert_queue.push(cellOrder[mapxcoord-2][mapycoord]);
					}
				//low vertex
				if ((mapxcoord<mapNRow-1)&&(edgePersType[mapxcoord+1][mapycoord]==DESTROYER)&&(cellOrder[mapxcoord+1][mapycoord]<eDeath))
					if (! vert_visited[cellOrder[mapxcoord+2][mapycoord]]){
						parent_v[cellOrder[mapxcoord+2][mapycoord]]=tmpv;
						vert_queue.push(cellOrder[mapxcoord+2][mapycoord]);
					}
				//left vertex
				if ((mapycoord>0)&&(edgePersType[mapxcoord][mapycoord-1]==DESTROYER)&&(cellOrder[mapxcoord][mapycoord-1]<eDeath))
					if (! vert_visited[cellOrder[mapxcoord][mapycoord-2]]){
						parent_v[cellOrder[mapxcoord][mapycoord-2]]=tmpv;
						vert_queue.push(cellOrder[mapxcoord][mapycoord-2]);
					}
				//right vertex
				if ((mapycoord<mapNCol-1)&&(edgePersType[mapxcoord][mapycoord+1]==DESTROYER)&&(cellOrder[mapxcoord][mapycoord+1]<eDeath))
					if (! vert_visited[cellOrder[mapxcoord][mapycoord+2]]){
						parent_v[cellOrder[mapxcoord][mapycoord+2]]=tmpv;
						vert_queue.push(cellOrder[mapxcoord][mapycoord+2]);
					}
			}
		};	//end of while(! vert_queue.empty())
	}
	void merge_comp(int vBirth, int eDeath, double btime, double dtime, myDoubleMatrix * const perturbM, myDoubleMatrix * const critM /*, int vHigh*/ ){
		//starting from edeath, go through a path of destroyer edges,  connecting vBirth and an component born earlier, and only go with edge with lower order
		
		vector< bool > vert_visited(vertNum,false);
		stack< int > vert_stack;
		
		int tmpv1 = (* eList)[eDeath].v1_order;
		int tmpv2 = (* eList)[eDeath].v2_order;
		
		vert_stack.push(tmpv1);
		
		int xcoord,ycoord,tmpv,mapxcoord,mapycoord;

		vector< int > parent_v(vertNum,-1);

		while(! vert_stack.empty()){
			tmpv = vert_stack.top();
			if (vert_visited[tmpv]){
				vert_stack.pop();
			 	continue;
			};
			vert_visited[tmpv]=true;

			xcoord=(* vList)[tmpv].xidx;
			ycoord=(* vList)[tmpv].yidx;
			
			mapxcoord=(* vList)[tmpv].mapXCoord;
			mapycoord=(* vList)[tmpv].mapYCoord;
			myassert(cellType[mapxcoord][mapycoord]==VERTEX);
			myassert((cellFixType[mapxcoord][mapycoord]==CF_UNDEFINED)||(cellFixType[mapxcoord][mapycoord]==MOVEDOWN));

// 			if ((tmpv==vBirth)||(tmpv==vHigh)) break;	//found the first end
			if (tmpv<=vBirth) break;	//found the first end
			// TODO: note this need to be fix if to keep multiple components

			vert_stack.pop();
			
			//check all neighbor edges, push relevant vertices into the queue;
			//up vertex
			if ((mapxcoord>0)&&(edgePersType[mapxcoord-1][mapycoord]==DESTROYER)&&(cellOrder[mapxcoord-1][mapycoord]<eDeath))
				if (! vert_visited[cellOrder[mapxcoord-2][mapycoord]]){
					myassert(parent_v[cellOrder[mapxcoord-2][mapycoord]]<0);
					parent_v[cellOrder[mapxcoord-2][mapycoord]]=tmpv;
					vert_stack.push(cellOrder[mapxcoord-2][mapycoord]);
				}
			//down vertex
			if ((mapxcoord<mapNRow-1)&&(edgePersType[mapxcoord+1][mapycoord]==DESTROYER)&&(cellOrder[mapxcoord+1][mapycoord]<eDeath))
				if (! vert_visited[cellOrder[mapxcoord+2][mapycoord]]){
					myassert(parent_v[cellOrder[mapxcoord+2][mapycoord]]<0);
					parent_v[cellOrder[mapxcoord+2][mapycoord]]=tmpv;
					vert_stack.push(cellOrder[mapxcoord+2][mapycoord]);
				}
			//left vertex
			if ((mapycoord>0)&&(edgePersType[mapxcoord][mapycoord-1]==DESTROYER)&&(cellOrder[mapxcoord][mapycoord-1]<eDeath))
				if (! vert_visited[cellOrder[mapxcoord][mapycoord-2]]){
					myassert(parent_v[cellOrder[mapxcoord][mapycoord-2]]<0);
					parent_v[cellOrder[mapxcoord][mapycoord-2]]=tmpv;
					vert_stack.push(cellOrder[mapxcoord][mapycoord-2]);
				}
			//right vertex
			if ((mapycoord<mapNCol-1)&&(edgePersType[mapxcoord][mapycoord+1]==DESTROYER)&&(cellOrder[mapxcoord][mapycoord+1]<eDeath))
				if (! vert_visited[cellOrder[mapxcoord][mapycoord+2]]){
					myassert(parent_v[cellOrder[mapxcoord][mapycoord+2]]<0);
					parent_v[cellOrder[mapxcoord][mapycoord+2]]=tmpv;
					vert_stack.push(cellOrder[mapxcoord][mapycoord+2]);
				}
		};	//end of while(! vert_stack.empty())

		myassert(! vert_stack.empty());		//vert_stack stores the first path.
		vector< bool > pathv_visited(vertNum,false);
		
		int endV1 = vert_stack.top();	//first end v, connected to tmpv1

		while(! vert_stack.empty())
			vert_stack.pop();
	
		tmpv = endV1;
		int oldtmpv=tmpv;
		while(tmpv >= 0){
		//move down everything on the path
			myassert(! pathv_visited[tmpv]);
			pathv_visited[tmpv]=true;

			xcoord=(* vList)[tmpv].xidx;
			ycoord=(* vList)[tmpv].yidx;
			
			mapxcoord=(* vList)[tmpv].mapXCoord;
			mapycoord=(* vList)[tmpv].mapYCoord;
			myassert(cellType[mapxcoord][mapycoord]==VERTEX);
			myassert((cellFixType[mapxcoord][mapycoord]==CF_UNDEFINED)||(cellFixType[mapxcoord][mapycoord]==MOVEDOWN));

			perturbM->data[xcoord][ycoord]=max(dtime,perturbM->data[xcoord][ycoord]);
			critM->data[xcoord][ycoord]=max((double)eDeath,critM->data[xcoord][ycoord]);
			
			setVertFixType(mapxcoord,mapycoord,MOVEDOWN);
	
			oldtmpv=tmpv;
			tmpv = parent_v[tmpv];
		};	//end of while(! vert_stack.empty())
		myassert( oldtmpv == tmpv1 );

		myassert(! vert_visited[tmpv2]);
		vert_stack.push(tmpv2);	//search path connecting tmpv2 to someone
		while(! vert_stack.empty()){
			tmpv = vert_stack.top();
			if (vert_visited[tmpv]){
				vert_stack.pop();
			 	continue;
			};
			vert_visited[tmpv]=true;

			xcoord=(* vList)[tmpv].xidx;
			ycoord=(* vList)[tmpv].yidx;
			
			mapxcoord=(* vList)[tmpv].mapXCoord;
			mapycoord=(* vList)[tmpv].mapYCoord;
			myassert(cellType[mapxcoord][mapycoord]==VERTEX);
			myassert((cellFixType[mapxcoord][mapycoord]==CF_UNDEFINED)||(cellFixType[mapxcoord][mapycoord]==MOVEDOWN));

			if (tmpv<=vBirth) break;	//found the first end
// 			if ((tmpv==vBirth)||(tmpv==vHigh)) break;	//found the first end
			// TODO: note this need to be fix if to keep multiple components

			vert_stack.pop();
			
			//check all neighbor edges, push relevant vertices into the queue;
			//up vertex
			if ((mapxcoord>0)&&(edgePersType[mapxcoord-1][mapycoord]==DESTROYER)&&(cellOrder[mapxcoord-1][mapycoord]<eDeath))
				if (! vert_visited[cellOrder[mapxcoord-2][mapycoord]]){
					myassert(parent_v[cellOrder[mapxcoord-2][mapycoord]]<0);
					parent_v[cellOrder[mapxcoord-2][mapycoord]]=tmpv;
					vert_stack.push(cellOrder[mapxcoord-2][mapycoord]);
				}
			//down vertex
			if ((mapxcoord<mapNRow-1)&&(edgePersType[mapxcoord+1][mapycoord]==DESTROYER)&&(cellOrder[mapxcoord+1][mapycoord]<eDeath))
				if (! vert_visited[cellOrder[mapxcoord+2][mapycoord]]){
					myassert(parent_v[cellOrder[mapxcoord+2][mapycoord]]<0);
					parent_v[cellOrder[mapxcoord+2][mapycoord]]=tmpv;
					vert_stack.push(cellOrder[mapxcoord+2][mapycoord]);
				}
			//left vertex
			if ((mapycoord>0)&&(edgePersType[mapxcoord][mapycoord-1]==DESTROYER)&&(cellOrder[mapxcoord][mapycoord-1]<eDeath))
				if (! vert_visited[cellOrder[mapxcoord][mapycoord-2]]){
					myassert(parent_v[cellOrder[mapxcoord][mapycoord-2]]<0);
					parent_v[cellOrder[mapxcoord][mapycoord-2]]=tmpv;
					vert_stack.push(cellOrder[mapxcoord][mapycoord-2]);
				}
			//right vertex
			if ((mapycoord<mapNCol-1)&&(edgePersType[mapxcoord][mapycoord+1]==DESTROYER)&&(cellOrder[mapxcoord][mapycoord+1]<eDeath))
				if (! vert_visited[cellOrder[mapxcoord][mapycoord+2]]){
					myassert(parent_v[cellOrder[mapxcoord][mapycoord+2]]<0);
					parent_v[cellOrder[mapxcoord][mapycoord+2]]=tmpv;
					vert_stack.push(cellOrder[mapxcoord][mapycoord+2]);
				}

		};	//end of while(! vert_stack.empty())

		myassert(! vert_stack.empty());		//vert_stack stores the second path.

		int endV2 = vert_stack.top();	//second end v, connected to tmpv2
		myassert(endV2!=endV1);
		myassert((endV2==vBirth)||(endV1==vBirth));
// 		myassert((endV2==vHigh)||(endV1==vHigh));

		while(! vert_stack.empty())
			vert_stack.pop();
	
		tmpv = endV2;
		oldtmpv=tmpv;
		while(tmpv >= 0){
		//move down everything on the path
			myassert(! pathv_visited[tmpv]);
			pathv_visited[tmpv]=true;

			xcoord=(* vList)[tmpv].xidx;
			ycoord=(* vList)[tmpv].yidx;
			
			mapxcoord=(* vList)[tmpv].mapXCoord;
			mapycoord=(* vList)[tmpv].mapYCoord;
			myassert(cellType[mapxcoord][mapycoord]==VERTEX);
			myassert((cellFixType[mapxcoord][mapycoord]==CF_UNDEFINED)||(cellFixType[mapxcoord][mapycoord]==MOVEDOWN));

			perturbM->data[xcoord][ycoord]=max(dtime,perturbM->data[xcoord][ycoord]);
			critM->data[xcoord][ycoord]=max((double)eDeath,critM->data[xcoord][ycoord]);
			
			setVertFixType(mapxcoord,mapycoord,MOVEDOWN);
			
			oldtmpv=tmpv;
			tmpv=parent_v[tmpv];
		};	//end of while(! vert_stack.empty())
		
		myassert(oldtmpv==tmpv2);
	}


	int make_hole(vector< int > circle_elist, myDoubleMatrix * const perturbM, myDoubleMatrix * const critM /*, int vHigh*/ ){
		//take the last edge as the time, move the whole circle down according to the time

		int ecreator = * circle_elist.rbegin();
		int vcreator = (* eList)[ecreator].v2_order;

		double creation_time = phi->data[(* vList)[vcreator].xidx][(* vList)[vcreator].yidx];
		if (creation_time < 0)
			return 0;

		int curr_e, curr_v1, curr_v2;
		int xcoord,ycoord;
		int mapxcoord,mapycoord;

		for( vector< int >::iterator myiiter=circle_elist.begin(); myiiter!=circle_elist.end(); myiiter++ ){
			curr_e= * myiiter;
			curr_v1 = (* eList)[curr_e].v1_order;
			curr_v2 = (* eList)[curr_e].v2_order;

			xcoord=(* vList)[curr_v1].xidx;
			ycoord=(* vList)[curr_v1].yidx;
			mapxcoord=(* vList)[curr_v1].mapXCoord;
			mapycoord=(* vList)[curr_v1].mapYCoord;
			myassert(cellType[mapxcoord][mapycoord]==VERTEX, 2);
			myassert((cellFixType[mapxcoord][mapycoord]==CF_UNDEFINED)||(cellFixType[mapxcoord][mapycoord]==MOVEDOWN), 2);
			perturbM->data[xcoord][ycoord]=max(creation_time, perturbM->data[xcoord][ycoord]);
			critM->data[xcoord][ycoord]=max((double)ecreator,critM->data[xcoord][ycoord]);

			xcoord=(* vList)[curr_v2].xidx;
			ycoord=(* vList)[curr_v2].yidx;
			mapxcoord=(* vList)[curr_v2].mapXCoord;
			mapycoord=(* vList)[curr_v2].mapYCoord;
			myassert(cellType[mapxcoord][mapycoord]==VERTEX, 2);
			myassert((cellFixType[mapxcoord][mapycoord]==CF_UNDEFINED)||(cellFixType[mapxcoord][mapycoord]==MOVEDOWN), 2);
			perturbM->data[xcoord][ycoord]=max(creation_time, perturbM->data[xcoord][ycoord]);
			critM->data[xcoord][ycoord]=max((double)ecreator,critM->data[xcoord][ycoord]);
		}

		return 1;
	}
//////////////////////////////////////////////////////////
// fixing holes
	bool isAnyFixType( int dim, int idx , CellFixTypeEnum ftype){
	//check if any of the vertices of the cell has fixing type ftype
		int mapX, mapY;
		int mapv1x,mapv1y,mapv2x,mapv2y;
		int mape1x,mape1y,mape2x,mape2y;
		CellTypeEnum etype;
		if(dim == 0){
			myassert(idx<=vertNum, 2);
			mapX = (* vList)[idx].mapXCoord;
			mapY = (* vList)[idx].mapYCoord;
			myassert(cellType[mapX][mapY], VERTEX);
			return (cellFixType[mapX][mapY]==ftype);
		};
		if(dim == 1){
			myassert(idx<=edgeNum, 2);
			mapX = (* eList)[idx].mapXCoord;
			mapY = (* eList)[idx].mapYCoord;
			myassert((cellType[mapX][mapY]==EDGEVERT)||(cellType[mapX][mapY]==EDGEHORI),2);

			etype=cellType[mapX][mapY];
			if(etype==EDGEVERT){
				mapv1x=mapX-1;
				mapv1y=mapY;
				mapv2x=mapX+1;
				mapv2y=mapY;
			}else{
				mapv1x=mapX;
				mapv1y=mapY-1;
				mapv2x=mapX;
				mapv2y=mapY+1;
			}
			return (cellFixType[mapv1x][mapv1y]==ftype)||(cellFixType[mapv2x][mapv2y]==ftype);
		};
		if(dim == 2){
			myassert(idx<=trigNum, 2);
			mapX = (* trigList)[idx].mapXCoord;
			mapY = (* trigList)[idx].mapYCoord;
			myassert(cellType[mapX][mapY], TRIG);

			mape1x=mapX-1;
			mape1y=mapY;
			mape2x=mapX+1;
			mape2y=mapY;
			return isAnyFixType(1,cellOrder[mape1x][mape1y],ftype)||isAnyFixType(1,cellOrder[mape2x][mape2y],ftype);
		};
		myassert(false);
	}

	void flood_hole(int eBirth, int tDeath, double btime, double dtime, myDoubleMatrix * const perturbM, myDoubleMatrix * const critM ){
		//starting from tDeath, going through any adjacent edge with > eBirth idx

		vector< bool > trig_visited(trigNum,false);
		queue< int > trig_queue;

		trig_queue.push(tDeath);

// 		int high_v_of_edeath = (*eList)[eDeath].v2_order;	//higher v of edeath should not be included, might break merge pathes
// 		myassert(high_v_of_edeath!=vBirth);

		vector< int > parent_t(trigNum,-1); 

// 		int xcoord,ycoord;
		int tmpt,mapxcoord,mapycoord;
		while(! trig_queue.empty()){
			tmpt = trig_queue.front();
//			if ((trig_visited[tmpt])||(tmpt!=high_v_of_edeath)){
			if (trig_visited[tmpt]){
				trig_queue.pop();
				continue;
			}
			trig_visited[tmpt] = true;

			mapxcoord=(* trigList)[tmpt].mapXCoord;
			mapycoord=(* trigList)[tmpt].mapYCoord;
			myassert(cellType[mapxcoord][mapycoord]==TRIG, 2);
			trig_queue.pop();

/*			if((tmpt==high_v_of_edeath)&&(cellFixType[mapxcoord][mapycoord]==MOVEDOWN)){
				// special case, keep tmpt unchanged, but flood on the same side of vBirth;

				//skip case when tmpt is on the boundary of the image
				if((mapxcoord==0)||(mapycoord==0)||(mapxcoord==mapNRow)||(mapycoord==mapNCol)) continue;

				CellFixTypeEnum uvFix=cellFixType[mapxcoord-2][mapycoord];
				CellFixTypeEnum dvFix=cellFixType[mapxcoord+2][mapycoord];
				CellFixTypeEnum lvFix=cellFixType[mapxcoord][mapycoord-2];
				CellFixTypeEnum rvFix=cellFixType[mapxcoord][mapycoord+2];
				int uvO=cellOrder[mapxcoord-2][mapycoord];
				int dvO=cellOrder[mapxcoord+2][mapycoord];
				int lvO=cellOrder[mapxcoord][mapycoord-2];
				int rvO=cellOrder[mapxcoord][mapycoord+2];
				int pvO=parent_t[tmpt];
				myassert(pvO>=0);

				if(((pvO==uvO)||(pvO==dvO))&&(lvFix!=MOVEDOWN))
					if (! trig_visited[lvO]){
						parent_t[lvO]=tmpt;
						trig_queue.push(lvO);
					}
				else if(((pvO==uvO)||(pvO==dvO))&&(rvFix!=MOVEDOWN))
					if (! trig_visited[rvO]){
						parent_t[rvO]=tmpt;
						trig_queue.push(rvO);
					}
				else if(((pvO==lvO)||(pvO==rvO))&&(uvFix!=MOVEDOWN))
					if (! trig_visited[uvO]){
						parent_t[uvO]=tmpt;
						trig_queue.push(uvO);
					}
				else if(((pvO==lvO)||(pvO==rvO))&&(dvFix!=MOVEDOWN))
					if (! trig_visited[dvO]){
						parent_t[dvO]=tmpt;
						trig_queue.push(dvO);
					}				
			}else{*/
// 				myassert((cellFixType[mapxcoord][mapycoord]==CF_UNDEFINED)||(cellFixType[mapxcoord][mapycoord]==MOVEUP));

				if(isAnyFixType(2,tmpt,MOVEUP))	continue;	//if any relevant vertex is moving up, do not flood this trig
				
				//move down all 4 relevant vertices
				Vertex v1 = (*vList)[cellOrder[mapxcoord-1][mapycoord-1]];
				Vertex v2 = (*vList)[cellOrder[mapxcoord-1][mapycoord+1]];
				Vertex v3 = (*vList)[cellOrder[mapxcoord+1][mapycoord-1]];
				Vertex v4 = (*vList)[cellOrder[mapxcoord+1][mapycoord+1]];
				
				perturbM->data[v1.xidx][v1.yidx]=max((double)dtime,perturbM->data[v1.xidx][v1.yidx]);
				critM->data[v1.xidx][v1.yidx]=max((double)tDeath,critM->data[v1.xidx][v1.yidx]);
				setVertFixType(v1.mapXCoord,v1.mapYCoord,MOVEDOWN);
				
				perturbM->data[v2.xidx][v2.yidx]=max((double)dtime,perturbM->data[v2.xidx][v2.yidx]);
				critM->data[v2.xidx][v2.yidx]=max((double)tDeath,critM->data[v2.xidx][v2.yidx]);
				setVertFixType(v2.mapXCoord,v2.mapYCoord,MOVEDOWN);
				
				perturbM->data[v3.xidx][v3.yidx]=max((double)dtime,perturbM->data[v3.xidx][v3.yidx]);
				critM->data[v3.xidx][v3.yidx]=max((double)tDeath,critM->data[v3.xidx][v3.yidx]);
				setVertFixType(v3.mapXCoord,v3.mapYCoord,MOVEDOWN);
				
				perturbM->data[v4.xidx][v4.yidx]=max((double)dtime,perturbM->data[v4.xidx][v4.yidx]);
				critM->data[v4.xidx][v4.yidx]=max((double)tDeath,critM->data[v4.xidx][v4.yidx]);
				setVertFixType(v4.mapXCoord,v4.mapYCoord,MOVEDOWN);

				//check all neighbor edges, push relevant trigs into the queue;
				//up trig
				if ((mapxcoord>1)&&(edgePersType[mapxcoord-1][mapycoord]==CREATOR)&&(cellOrder[mapxcoord-1][mapycoord]>eBirth))
					if (! trig_visited[cellOrder[mapxcoord-2][mapycoord]]){
						parent_t[cellOrder[mapxcoord-2][mapycoord]]=tmpt;
						trig_queue.push(cellOrder[mapxcoord-2][mapycoord]);
					}
				//low trig
				if ((mapxcoord<mapNRow-2)&&(edgePersType[mapxcoord+1][mapycoord]==CREATOR)&&(cellOrder[mapxcoord+1][mapycoord]>eBirth))
					if (! trig_visited[cellOrder[mapxcoord+2][mapycoord]]){
						parent_t[cellOrder[mapxcoord+2][mapycoord]]=tmpt;
						trig_queue.push(cellOrder[mapxcoord+2][mapycoord]);
					}
				//left trig
				if ((mapycoord>1)&&(edgePersType[mapxcoord][mapycoord-1]==CREATOR)&&(cellOrder[mapxcoord][mapycoord-1]>eBirth))
					if (! trig_visited[cellOrder[mapxcoord][mapycoord-2]]){
						parent_t[cellOrder[mapxcoord][mapycoord-2]]=tmpt;
						trig_queue.push(cellOrder[mapxcoord][mapycoord-2]);
					}
				//right trig
				if ((mapycoord<mapNCol-2)&&(edgePersType[mapxcoord][mapycoord+1]==CREATOR)&&(cellOrder[mapxcoord][mapycoord+1]>eBirth))
					if (! trig_visited[cellOrder[mapxcoord][mapycoord+2]]){
						parent_t[cellOrder[mapxcoord][mapycoord+2]]=tmpt;
						trig_queue.push(cellOrder[mapxcoord][mapycoord+2]);
					}
// 			}
		};	//end of while(! trig_queue.empty())
	}

  	void merge_hole(int eBirth, int tDeath, double btime, double dtime, myDoubleMatrix * const perturbM, myDoubleMatrix * const critM /*, int vHigh*/ ){
		//starting from ebirth, go through a path of creator edges,  connecting tdeath and an hole dies later, and only go with edge with >ebirth order
		
		vector< bool > trig_visited(vertNum,false);
		stack< int > trig_stack;

		int xcoord,ycoord,tmpt,mapxcoord,mapycoord;
			
		mapxcoord = (* eList)[eBirth].mapXCoord;
		mapycoord = (* eList)[eBirth].mapYCoord;
		
		int tmpt1, tmpt2;
		if(cellType[mapxcoord][mapycoord]==EDGEVERT){
			tmpt1 = (mapycoord==0) ? -1 : cellOrder[mapxcoord][mapycoord-1];
			tmpt2 = (mapycoord==mapNCol-1) ? -1 : cellOrder[mapxcoord][mapycoord+1];
		}else if(cellType[mapxcoord][mapycoord]==EDGEHORI){
			tmpt1 = (mapxcoord==0) ? -1 : cellOrder[mapxcoord-1][mapycoord];
			tmpt2 = (mapxcoord==mapNRow-1) ? -1 : cellOrder[mapxcoord+1][mapycoord];
		}
		
		if (tmpt1>=0) trig_stack.push(tmpt1);

		vector< int > parent_t(trigNum,-1);

		while(! trig_stack.empty()){
			tmpt = trig_stack.top();
			if (trig_visited[tmpt]){
				trig_stack.pop();
			 	continue;
			};
			trig_visited[tmpt]=true;

			mapxcoord=(* trigList)[tmpt].mapXCoord;
			mapycoord=(* trigList)[tmpt].mapYCoord;
			myassert(cellType[mapxcoord][mapycoord]==TRIG);
			myassert((cellFixType[mapxcoord][mapycoord]==CF_UNDEFINED)||(cellFixType[mapxcoord][mapycoord]==MOVEUP), 2);
			

// 			if ((tmpt==tDeath)||(tmpt==tHigh)) break;	//found the first end
			if (tmpt>=tDeath) break;	//found the first end
			// TODO: note this need to be fix if to keep multiple components

			trig_stack.pop();
			
			//check all neighbor edges, push relevant trigs into the queue;
			//up trig
			if ((mapxcoord>1)&&(edgePersType[mapxcoord-1][mapycoord]==CREATOR)&&(cellOrder[mapxcoord-1][mapycoord]>eBirth))
				if (! trig_visited[cellOrder[mapxcoord-2][mapycoord]]){
					myassert(parent_t[cellOrder[mapxcoord-2][mapycoord]]<0, 2);
					parent_t[cellOrder[mapxcoord-2][mapycoord]]=tmpt;
					trig_stack.push(cellOrder[mapxcoord-2][mapycoord]);
				}
			//down trig
			if ((mapxcoord<mapNRow-2)&&(edgePersType[mapxcoord+1][mapycoord]==CREATOR)&&(cellOrder[mapxcoord+1][mapycoord]>eBirth))
				if (! trig_visited[cellOrder[mapxcoord+2][mapycoord]]){
					myassert(parent_t[cellOrder[mapxcoord+2][mapycoord]]<0, 2);
					parent_t[cellOrder[mapxcoord+2][mapycoord]]=tmpt;
					trig_stack.push(cellOrder[mapxcoord+2][mapycoord]);
				}
			//left trig
			if ((mapycoord>1)&&(edgePersType[mapxcoord][mapycoord-1]==CREATOR)&&(cellOrder[mapxcoord][mapycoord-1]>eBirth))
				if (! trig_visited[cellOrder[mapxcoord][mapycoord-2]]){
					myassert(parent_t[cellOrder[mapxcoord][mapycoord-2]]<0, 2);
					parent_t[cellOrder[mapxcoord][mapycoord-2]]=tmpt;
					trig_stack.push(cellOrder[mapxcoord][mapycoord-2]);
				}
			//right trig
			if ((mapycoord<mapNCol-2)&&(edgePersType[mapxcoord][mapycoord+1]==CREATOR)&&(cellOrder[mapxcoord][mapycoord+1]>eBirth))
				if (! trig_visited[cellOrder[mapxcoord][mapycoord+2]]){
					myassert(parent_t[cellOrder[mapxcoord][mapycoord+2]]<0, 2);
					parent_t[cellOrder[mapxcoord][mapycoord+2]]=tmpt;
					trig_stack.push(cellOrder[mapxcoord][mapycoord+2]);
				}

		};	//end of while(! trig_stack.empty())

		myassert(! trig_stack.empty());		//trig_stack stores the first path.
		vector< bool > patht_visited(trigNum,false);
		
		int endT1 = trig_stack.empty() ? -1 : trig_stack.top();	//first end v, connected to tmpt1

		while(! trig_stack.empty())
			trig_stack.pop();
	
		tmpt = endT1;
		int oldtmpt=tmpt;
		while(tmpt >= 0){
		//move up everything on the path
			myassert(! patht_visited[tmpt]);
			patht_visited[tmpt]=true;
			if(oldtmpt!=tmpt){		
				//get the edge between tmpt and oldtmpt, lift the maximum of its two vertices (lift only when the vertex is not MOVEDOWN)
	
				int mapv1x,mapv1y,mapv2x,mapv2y; 
				
				mapxcoord=(* trigList)[tmpt].mapXCoord;
				mapycoord=(* trigList)[tmpt].mapYCoord;
				int mapxcoord_old=(* trigList)[oldtmpt].mapXCoord;
				int mapycoord_old=(* trigList)[oldtmpt].mapYCoord;
				
				mapv1x = (mapxcoord == mapxcoord_old) ? mapxcoord+1 : (mapxcoord+mapxcoord_old)/2;
				mapv1y = (mapycoord == mapycoord_old) ? mapycoord+1 : (mapycoord+mapycoord_old)/2;
				myassert(cellType[mapv1x][mapv1y]==VERTEX,2);

				mapv2x = (mapxcoord == mapxcoord_old) ? mapxcoord-1 : (mapxcoord+mapxcoord_old)/2;
				mapv2y = (mapycoord == mapycoord_old) ? mapycoord-1 : (mapycoord+mapycoord_old)/2;
				myassert(cellType[mapv2x][mapv2y]==VERTEX,2);

				CellFixTypeEnum v1_ftype = cellFixType[mapv1x][mapv1y];
				CellFixTypeEnum v2_ftype = cellFixType[mapv2x][mapv2y];

				bool lift_v1=false;
				bool lift_v2=false;

				if((v1_ftype!=MOVEDOWN)&&(v2_ftype!=MOVEDOWN)){
					if(cellOrder[mapv1x][mapv1y]>cellOrder[mapv2x][mapv2y])
						lift_v1 = true;
					else
						lift_v2 = true;
				}else if(v1_ftype!=MOVEDOWN)
					lift_v1 = true;
				else if(v2_ftype!=MOVEDOWN)
					lift_v2 = true;
				myassert( !lift_v2 || !lift_v1 );

				if(lift_v1){
					perturbM->data[mapv1x/2][mapv1y/2]=min(btime,perturbM->data[mapv1x/2][mapv1y/2]);
					critM->data[mapv1x/2][mapv1y/2]=min(-1.0*(double)eBirth,critM->data[mapv1x/2][mapv1y/2]);
					setVertFixType(mapv1x,mapv1y,MOVEUP);
				}	
	
				if(lift_v2){
					perturbM->data[mapv2x/2][mapv2y/2]=min(btime,perturbM->data[mapv2x/2][mapv2y/2]);
					critM->data[mapv2x/2][mapv2y/2]=min(-1.0*(double)eBirth,critM->data[mapv2x/2][mapv2y/2]);
					setVertFixType(mapv2x,mapv2y,MOVEUP);
				}	
			}	
			oldtmpt=tmpt;
			tmpt = parent_t[tmpt];
		};	//end of while(! trig_stack.empty())
		myassert( oldtmpt == tmpt1 );
		
		if (tmpt2>=0){ 
			myassert(! trig_visited[tmpt2]);
			trig_stack.push(tmpt2);	//search path connecting tmpt2 to someone
		};
		while(! trig_stack.empty()){
			tmpt = trig_stack.top();
			if (trig_visited[tmpt]){
				trig_stack.pop();
			 	continue;
			};
			trig_visited[tmpt]=true;

			mapxcoord=(* trigList)[tmpt].mapXCoord;
			mapycoord=(* trigList)[tmpt].mapYCoord;
			myassert(cellType[mapxcoord][mapycoord]==TRIG);
			myassert((cellFixType[mapxcoord][mapycoord]==CF_UNDEFINED)||(cellFixType[mapxcoord][mapycoord]==MOVEUP), 2);
			

// 			if ((tmpt==tDeath)||(tmpt==tHigh)) break;	//found the first end
			if (tmpt>=tDeath) break;	//found the first end
			// TODO: note this need to be fix if to keep multiple components

			trig_stack.pop();
			
			//check all neighbor edges, push relevant trigs into the queue;
			//up trig
			if ((mapxcoord>1)&&(edgePersType[mapxcoord-1][mapycoord]==CREATOR)&&(cellOrder[mapxcoord-1][mapycoord]>eBirth))
				if (! trig_visited[cellOrder[mapxcoord-2][mapycoord]]){
					myassert(parent_t[cellOrder[mapxcoord-2][mapycoord]]<0, 2);
					parent_t[cellOrder[mapxcoord-2][mapycoord]]=tmpt;
					trig_stack.push(cellOrder[mapxcoord-2][mapycoord]);
				}
			//down trig
			if ((mapxcoord<mapNRow-2)&&(edgePersType[mapxcoord+1][mapycoord]==CREATOR)&&(cellOrder[mapxcoord+1][mapycoord]>eBirth))
				if (! trig_visited[cellOrder[mapxcoord+2][mapycoord]]){
					myassert(parent_t[cellOrder[mapxcoord+2][mapycoord]]<0, 2);
					parent_t[cellOrder[mapxcoord+2][mapycoord]]=tmpt;
					trig_stack.push(cellOrder[mapxcoord+2][mapycoord]);
				}
			//left trig
			if ((mapycoord>1)&&(edgePersType[mapxcoord][mapycoord-1]==CREATOR)&&(cellOrder[mapxcoord][mapycoord-1]>eBirth))
				if (! trig_visited[cellOrder[mapxcoord][mapycoord-2]]){
					myassert(parent_t[cellOrder[mapxcoord][mapycoord-2]]<0, 2);
					parent_t[cellOrder[mapxcoord][mapycoord-2]]=tmpt;
					trig_stack.push(cellOrder[mapxcoord][mapycoord-2]);
				}
			//right trig
			if ((mapycoord<mapNCol-2)&&(edgePersType[mapxcoord][mapycoord+1]==CREATOR)&&(cellOrder[mapxcoord][mapycoord+1]>eBirth))
				if (! trig_visited[cellOrder[mapxcoord][mapycoord+2]]){
					myassert(parent_t[cellOrder[mapxcoord][mapycoord+2]]<0, 2);
					parent_t[cellOrder[mapxcoord][mapycoord+2]]=tmpt;
					trig_stack.push(cellOrder[mapxcoord][mapycoord+2]);
				}
		};	//end of while(! trig_stack.empty())

		myassert(! trig_stack.empty());		//trig_stack stores the second path.

		int endT2 = trig_stack.empty() ? -1 : trig_stack.top();	//first end v, connected to tmpt1
		myassert(endT2!=endT1);
		myassert((endT2==tDeath)||(endT1==tDeath));
// 		myassert((endV2==vHigh)||(endV1==vHigh));

		while(! trig_stack.empty())
			trig_stack.pop();
	
		tmpt = endT2;
		oldtmpt=tmpt;
		while(tmpt >= 0){
		//move up everything on the path
			myassert(! patht_visited[tmpt]);
			patht_visited[tmpt]=true;
			if(oldtmpt!=tmpt){		
				//get the edge between tmpt and oldtmpt, lift the maximum of its two vertices (lift only when the vertex is not MOVEDOWN)
	
				int mapv1x,mapv1y,mapv2x,mapv2y; 
				
				mapxcoord=(* trigList)[tmpt].mapXCoord;
				mapycoord=(* trigList)[tmpt].mapYCoord;
				int mapxcoord_old=(* trigList)[oldtmpt].mapXCoord;
				int mapycoord_old=(* trigList)[oldtmpt].mapYCoord;
				
				mapv1x = (mapxcoord == mapxcoord_old) ? mapxcoord+1 : (mapxcoord+mapxcoord_old)/2;
				mapv1y = (mapycoord == mapycoord_old) ? mapycoord+1 : (mapycoord+mapycoord_old)/2;
				myassert(cellType[mapv1x][mapv1y]==VERTEX,2);

				mapv2x = (mapxcoord == mapxcoord_old) ? mapxcoord-1 : (mapxcoord+mapxcoord_old)/2;
				mapv2y = (mapycoord == mapycoord_old) ? mapycoord-1 : (mapycoord+mapycoord_old)/2;
				myassert(cellType[mapv2x][mapv2y]==VERTEX,2);

				CellFixTypeEnum v1_ftype = cellFixType[mapv1x][mapv1y];
				CellFixTypeEnum v2_ftype = cellFixType[mapv2x][mapv2y];

				bool lift_v1=false;
				bool lift_v2=false;

				if((v1_ftype!=MOVEDOWN)&&(v2_ftype!=MOVEDOWN)){
					if(cellOrder[mapv1x][mapv1y]>cellOrder[mapv2x][mapv2y])
						lift_v1 = true;
					else
						lift_v2 = true;
				}else if(v1_ftype!=MOVEDOWN)
					lift_v1 = true;
				else if(v2_ftype!=MOVEDOWN)
					lift_v2 = true;
				myassert( !lift_v2 || !lift_v1 );

				if(lift_v1){
					perturbM->data[mapv1x/2][mapv1y/2]=min(btime,perturbM->data[mapv1x/2][mapv1y/2]);
					critM->data[mapv1x/2][mapv1y/2]=min(-1.0*(double)eBirth,critM->data[mapv1x/2][mapv1y/2]);
					setVertFixType(mapv1x,mapv1y,MOVEUP);
				}	
	
				if(lift_v2){
					perturbM->data[mapv2x/2][mapv2y/2]=min(btime,perturbM->data[mapv2x/2][mapv2y/2]);
					critM->data[mapv2x/2][mapv2y/2]=min(-1.0*(double)eBirth,critM->data[mapv2x/2][mapv2y/2]);
					setVertFixType(mapv2x,mapv2y,MOVEUP);
				}	
			}	

			oldtmpt=tmpt;
			tmpt = parent_t[tmpt];
		};	//end of while(! trig_stack.empty())
		myassert(oldtmpt==tmpt2);

		if((tmpt1>=0)&&(tmpt1<trigNum)&&(tmpt2>=0)&&(tmpt2<trigNum)){
			int mapv1x,mapv1y,mapv2x,mapv2y; 
			mapxcoord=(* trigList)[tmpt1].mapXCoord;
			mapycoord=(* trigList)[tmpt1].mapYCoord;
			int mapxcoord_old=(* trigList)[tmpt2].mapXCoord;
			int mapycoord_old=(* trigList)[tmpt2].mapYCoord;
			
			mapv1x = (mapxcoord == mapxcoord_old) ? mapxcoord+1 : (mapxcoord+mapxcoord_old)/2;
			mapv1y = (mapycoord == mapycoord_old) ? mapycoord+1 : (mapycoord+mapycoord_old)/2;
			myassert(cellType[mapv1x][mapv1y]==VERTEX,2);
	
			mapv2x = (mapxcoord == mapxcoord_old) ? mapxcoord-1 : (mapxcoord+mapxcoord_old)/2;
			mapv2y = (mapycoord == mapycoord_old) ? mapycoord-1 : (mapycoord+mapycoord_old)/2;
			myassert(cellType[mapv2x][mapv2y]==VERTEX,2);
	
			CellFixTypeEnum v1_ftype = cellFixType[mapv1x][mapv1y];
			CellFixTypeEnum v2_ftype = cellFixType[mapv2x][mapv2y];
	
			bool lift_v1=false;
			bool lift_v2=false;
	
			if((v1_ftype!=MOVEDOWN)&&(v2_ftype!=MOVEDOWN)){
				if(cellOrder[mapv1x][mapv1y]>cellOrder[mapv2x][mapv2y])
					lift_v1 = true;
				else
					lift_v2 = true;
			}else if(v1_ftype!=MOVEDOWN)
				lift_v1 = true;
			else if(v2_ftype!=MOVEDOWN)
				lift_v2 = true;
			myassert( !lift_v2 || !lift_v1 );
	
			if(lift_v1){
				perturbM->data[mapv1x/2][mapv1y/2]=min(btime,perturbM->data[mapv1x/2][mapv1y/2]);
				critM->data[mapv1x/2][mapv1y/2]=min(-1.0*(double)eBirth,critM->data[mapv1x/2][mapv1y/2]);
				setVertFixType(mapv1x,mapv1y,MOVEUP);
			}	
	
			if(lift_v2){
				perturbM->data[mapv2x/2][mapv2y/2]=min(btime,perturbM->data[mapv2x/2][mapv2y/2]);
				critM->data[mapv2x/2][mapv2y/2]=min(-1.0*(double)eBirth,critM->data[mapv2x/2][mapv2y/2]);
				setVertFixType(mapv2x,mapv2y,MOVEUP);
			}	
		}
	}

};	// end of class CellMap


pair< int, int > calcPers(const int m,const int n, 
	      const double rob_thd, const double levelset_val,
		double perturb_thd,myDoubleMatrix * const perturbM, myDoubleMatrix * const critM,
		int remove_only, int ncomp_ub, int nhole_ub){
//		double dist_from_path_thd, int remove_only, int kill_holes, int big_holes_remaining, double big_holes_thd){

	OUTPUT_MSG("Begin computing persistence");

	//constructing vList
	int i,j;
	vector< Vertex > * vList=new vector< Vertex >;
	vector< Edge > * eList=new vector< Edge >;
	vector< Triangle > * trigList=new vector< Triangle >;
	for (i=0;i<m;i++)
		for (j=0;j<n;j++)
			vList->push_back(Vertex(i,j));
 
	//sort vList
	sort(vList->begin(), vList->end(), vCompVal);

	OUTPUT_MSG("--vList constructed and sorted");

	CellMap myCM(vList,eList,trigList);

	//construct and reduce 2d boundary matrix
	vector< vector< int > > * boundary_2D=new vector< vector< int > >(myCM.trigNum, vector< int >()); //first index is col index, each col init to empty
	myCM.buildBoundary2D(boundary_2D);
	vector< int > * low_2D_e2t=new vector< int >(myCM.edgeNum,-1);

	multiset< EdgeTrigPair > etQueue;	// robustness pairs

	int num_e_creator=0;
	int num_t_destroyer=0;
	int num_v_creator=0;// number of vertices creating non-essential class
	int num_e_destroyer=0;// number of edge destroyer (non-essential)

	//output edge-trig pairs whose persistence is bigger than pers_thd
	int vBirth,vDeath;
	int tmp_int;
	double tmp_pers,tmp_rob,tmp_double,tmp_death,tmp_birth;
	
	int low;

	list<int>::iterator myiter;
	int tmpe12,tmpe23,tmpe13;
	map<int, int>::iterator tmpit;
	for (i=0;i<myCM.trigNum;i++){

		//reduce column i
		low = * (* boundary_2D)[i].rbegin();

		while ( ( ! (* boundary_2D)[i].empty() ) && ( (* low_2D_e2t)[low]!=-1 ) ){
			(* boundary_2D)[i]=list_sym_diff((* boundary_2D)[i],(* boundary_2D)[(* low_2D_e2t)[low]]);

			if(! (* boundary_2D)[i].empty()){
				low = * (* boundary_2D)[i].rbegin();
			}
		}
		if (! (* boundary_2D)[i].empty()){
			myassert(low>=0);
			myassert((* low_2D_e2t)[low]==-1);
			(* low_2D_e2t)[low]=i;
			num_t_destroyer++;
			num_e_creator++;

			//record pair
			Edge edgeCreator=(* eList)[low];
			Triangle trigDestroyer=(* trigList)[i];
			vBirth= edgeCreator.v2_order;
			vDeath= trigDestroyer.v4_order;
			tmp_death=phi->data[(* vList)[vDeath].xidx][(* vList)[vDeath].yidx];
			tmp_birth=phi->data[(* vList)[vBirth].xidx][(* vList)[vBirth].yidx];
			tmp_rob=min(fabs(tmp_birth-levelset_val),fabs(tmp_death-levelset_val));
			if( (tmp_birth<levelset_val) && (tmp_death>levelset_val) && (tmp_rob>rob_thd) ){
				etQueue.insert(EdgeTrigPair(low,i,tmp_rob, tmp_birth, tmp_death));	
			};

		}

		if (i % 100000 == 0)
		  OUTPUT_MSG( "reducing boundary 2D: i=" << i <<", trig number=" << myCM.trigNum );
	}
	
	myCM.setEPersType( low_2D_e2t );

	set < int > bigt;
	int hole_creator;
	int hole_ct = 0;
	int new_holes_created=0;

	delete boundary_2D;

	OUTPUT_MSG( "boundary_2D all reduced" );

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//construct and reduce 1d boundary matrix

	vector< int > * low_1D_v2e= new vector< int >(myCM.vertNum,-1);
	// for each creator vertex, store the edge it is paired to
	
	vector< vector< int > > * boundary_1D=new vector< vector< int > >(myCM.edgeNum, vector< int >()); //first index is col index, each col init to empty
	myCM.buildBoundary1D(boundary_1D);

	multiset< VertEdgePair > veQueue;	// robustness pairs

	int tmptmp=0;
	int tmptmptmp=0;

	for (i=0;i<myCM.edgeNum;i++){

		if ( (* low_2D_e2t)[i] >= 0 ){ 
			(*boundary_1D)[i].clear();
			tmptmp++;
		  continue;
		}else{
			tmptmptmp++;
		  myassert((* low_2D_e2t)[i] == -1);
		  myassert((*boundary_1D)[i].size()==2);
		};
	  
		//reduce column i
		low = * (* boundary_1D)[i].rbegin();

		while ( ( ! (* boundary_1D)[i].empty() ) && ( (* low_1D_v2e)[low]!=-1 ) ){
			(* boundary_1D)[i]=list_sym_diff((* boundary_1D)[i],(* boundary_1D)[(* low_1D_v2e)[low]]);
			if(! (* boundary_1D)[i].empty()){
				low = * (* boundary_1D)[i].rbegin();
			}
		}
		if (! (* boundary_1D)[i].empty()){
			myassert(low>=0);
			myassert((* low_1D_v2e)[low]==-1);
			(* low_1D_v2e)[low]=i;
			num_e_destroyer++;
			num_v_creator++;

			myassert((* boundary_1D)[i].size()==2);
			
			int high =  * (* boundary_1D)[i].begin();
			//reduce high
			while (  (* low_1D_v2e)[high]!=-1 ){
				int edge_high=(*low_1D_v2e)[high];
				(* boundary_1D)[i]=list_sym_diff((* boundary_1D)[i],(* boundary_1D)[edge_high]);
				myassert((* boundary_1D)[i].size()==2);
				high = * (* boundary_1D)[i].begin();
			}

			//record pair
			vBirth= low;	//creator vertex
			Edge eDestroyer = (* eList)[i];
			vDeath=eDestroyer.v2_order;
			tmp_death=phi->data[(* vList)[vDeath].xidx][(* vList)[vDeath].yidx];
			tmp_birth=phi->data[(* vList)[vBirth].xidx][(* vList)[vBirth].yidx];
			tmp_rob=min(fabs(tmp_birth-levelset_val),fabs(tmp_death-levelset_val));
			if( (tmp_birth<levelset_val) && (tmp_death>levelset_val) && (tmp_rob>rob_thd) ){
				veQueue.insert(VertEdgePair(low,i,tmp_rob, tmp_birth, tmp_death));	
				//the component could be killed by either merge or remove
			};

		}else{
			myassert(false);
		}

		if (i % 100000 == 0)
		  OUTPUT_MSG( "reducing boundary 1D: i=" << i <<", edge number=" << myCM.edgeNum );
	}
	myassert(num_v_creator==myCM.vertNum-1);
	myassert(num_e_destroyer==num_v_creator);
// 	myassert(num_e_creator+num_e_destroyer==myCM.edgeNum);
// 	myassert(num_t_destroyer==myCM.trigNum);
	OUTPUT_MSG( "boundary 1D all reduced" );

//////////////////////////////////////////////////////////////////////////////////////////////////
// fixing topology

	double dtime, btime;
	int eDeath;

	int num_removed =0;
	int num_merged =0;

	int comp_skipped = 1;

if( ncomp_ub >= 1 ){
	// enumerate through big persistence
	for(multiset< VertEdgePair >::iterator myveiter=veQueue.begin(); myveiter!=veQueue.end(); myveiter++){	

		if( comp_skipped < ncomp_ub ){
			comp_skipped ++;
			continue;
		};

		dtime=myveiter->death;
		btime=myveiter->birth;
		vBirth=myveiter->vbidx;
		eDeath=myveiter->edidx;

		if ((myCM.getFixType(0,vBirth)==MOVEUP) || ( (myCM.getFixType(0,vBirth)==CF_UNDEFINED) && (-1.0*btime<dtime) )) {
			//remove the component, and everything associated
			
			myCM.flood_comp(vBirth, eDeath, btime, dtime, perturbM, critM ); 	// flood the component, and mark relavent comp/holes
			num_removed++;
		}else if(remove_only==0){
			
			myassert((myCM.getFixType(0,vBirth)==MOVEDOWN) || ( (myCM.getFixType(0,vBirth)==CF_UNDEFINED) && (-1.0*btime>dtime) ));
//			myassert((myCM.getFixType(1,eDeath)==MOVEDOWN) || (myCM.getFixType(1,eDeath)==CF_UNDEFINED));

			myCM.merge_comp(vBirth, eDeath, btime, dtime, perturbM, critM/*, (*boundary_1D)[eDeath][0]*/ );

			num_merged++;		

		}	//end of if ((fixing_method[vBirth]==FIX_REMOVE) || ( (fixing_method[vBirth]==FIX_EITHER) && (-1.0*tmp_birth<=tmp_death) ))

	} //end of for(multiset< VertEdgePair >::iterator myveiter=veQueue.begin(); myveiter!=veQueue.end(); myveiter++)
}

//  	OUTPUT_MSG("Fixed components ="<<num_fixed_comp);
// 	OUTPUT_MSG(num_v_creator);


//////////////////////////////////////////////////////////////////////////////////	
	// fixing holes

	OUTPUT_MSG("--------------------------------------------KILLING HOLES____________________________________");

	int num_hole_sealed=0;
	int num_hole_teared=0;

	int eBirth, tDeath;
	int hole_skipped = 0;

	if(nhole_ub >= 0){
	// enumerate through big persistence
	for(multiset< EdgeTrigPair >::iterator myetiter=etQueue.begin(); myetiter!=etQueue.end(); myetiter++){	

		if( hole_skipped < nhole_ub ){
			hole_skipped ++;
			continue;
		};

		dtime=myetiter->death;
		btime=myetiter->birth;
		eBirth=myetiter->ebidx;
		tDeath=myetiter->tdidx;

		OUTPUT_MSG("etpair: " << dtime <<" " << btime );

		if ((myCM.getFixType(2,tDeath)==MOVEDOWN) || ( (myCM.getFixType(2,tDeath)==CF_UNDEFINED) && (-1.0*btime>=dtime) )) {
			//seal the hole, and everything associated
			
			myCM.flood_hole(eBirth, tDeath, btime, dtime, perturbM, critM ); 	// flood the hole, and mark relavent comp/holes
			num_hole_sealed++;
		}else if(remove_only==0){
			myassert((myCM.getFixType(2,tDeath)==MOVEUP) || ( (myCM.getFixType(2,tDeath)==CF_UNDEFINED) && (-1.0*btime<dtime) ), 1);
//			myassert((myCM.getFixType(1,eDeath)==MOVEDOWN) || (myCM.getFixType(1,eDeath)==CF_UNDEFINED));

 			myCM.merge_hole(eBirth, tDeath, btime, dtime, perturbM, critM ); 	// flood the component, and mark relavent comp/holes

			num_hole_teared++;		

		}	//end of if ((fixing_method[vBirth]==FIX_REMOVE) || ( (fixing_method[vBirth]==FIX_EITHER) && (-1.0*tmp_birth<=tmp_death) ))

/*		if(eDeath==179248) break;
*/
	} //end of for(multiset< VertEdgePair >::iterator myveiter=veQueue.begin(); myveiter!=veQueue.end(); myveiter++)

	} //end of if( nhole_ub >= 0)

	int num_topo_noise = num_removed+num_merged+num_hole_sealed+num_hole_teared;

	delete low_2D_e2t;
	delete low_1D_v2e;
	delete vList;
	delete eList;
	delete trigList;
	delete boundary_1D;
	

	return pair< int, int >( veQueue.size()+1, etQueue.size() );

// 	return num_rob_pair+(big_holes_remaining-hole_ct);	// count how many topology noise
// 	return num_topo_noise;
}

static PyObject *topofix(PyObject *self, PyObject *args) {
  PyArrayObject *phi_p, *output_p;
  int ncomp_ub=0;
  int nhole_ub=0;

  if (!PyArg_ParseTuple(args, "O!ii", 
                        &PyArray_Type, &phi_p,
                        &ncomp_ub, &nhole_ub)) {
    PyErr_SetString(PyExc_ValueError, "Wrong parameters");
    return NULL;
  }

  double rob_thd = 0.0;
  double levelset_val = 0 ;
  double perturb_thd = 0.0;
  int remove_only = 0;

  int m = phi_p->dimensions[0];
  int n = phi_p->dimensions[1];

  // yes, this is despicable global variable
  phi=new myDoubleMatrix(m,n);

  // convert phi_p to guaranteed c-contiguous array
  phi_p = (PyArrayObject *) PyArray_FROM_OTF((PyObject *)phi_p, 
                                             NPY_DOUBLE,  
                                             NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ALIGNED);

  phi->input1D((double*)PyArray_DATA(phi_p));

  myDoubleMatrix * perturbM=new myDoubleMatrix(m,n);
  myDoubleMatrix * critM=new myDoubleMatrix(m,n);

  pair< int, int > betti_numbers=calcPers(m,n,rob_thd
                                          , levelset_val
                                          , perturb_thd
                                          , perturbM
                                          , critM
                                          , remove_only
                                          , ncomp_ub
                                          , nhole_ub);

  OUTPUT_MSG("calcPers is DONE");

  npy_intp d_out[2] = {m, n};
  double* output = new double[n*m];
  perturbM->output1D_C(output); // use C-style indexing

  output_p = (PyArrayObject *)PyArray_SimpleNewFromData(2,d_out,NPY_DOUBLE,output);

  Py_INCREF(output_p);

  delete phi;
  delete perturbM;
  delete critM;

  return PyArray_Return(output_p);
}
