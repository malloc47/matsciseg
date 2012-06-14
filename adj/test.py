#-------------------------------------------------------------------------------
import timeit;
import numpy as np;
import adjc;
import gcoc,gco;
#-------------------------------------------------------------------------------
label = np.genfromtxt("test.label",dtype="int16");
count = 5;
#-------------------------------------------------------------------------------
#t = timeit.Timer(stmt="adjGood = gco.adjacent(label)",
#                 setup="import gco; from __main__ import label");
#print t.timeit(count)/count;
#adjGood = gco.adjacent(label);
#np.savetxt("good.adj",adjGood,fmt="%1d");
#-------------------------------------------------------------------------------
#x = np.array([[1,2,3],[4,5,6]],dtype=np.int16);
#adjc.adjacent(x,1,1,0);
#-------------------------------------------------------------------------------
#t = timeit.Timer(stmt="adjExp = adjc.adjacent(label,label.max()+1,2,0)",
#                 setup="import adjc; from __main__ import label");
#print t.timeit(count)/count;
#adjExp = adjc.adjacent(label,label.max()+1,1,0);
#np.savetxt("exp1.adj",adjExp,fmt="%1d");
adjExp = adjc.adjacent(label,label.max()+1,2,0);
np.savetxt("exp2.adj",adjExp,fmt="%1d");
#adjExp = adjc.adjacent(label,label.max()+1,3,0);
#np.savetxt("exp3.adj",adjExp,fmt="%1d");
#adjExp = adjc.adjacent(label,label.max()+1,3,1);
#np.savetxt("surround.adj",adjExp,fmt="%1d");
#-------------------------------------------------------------------------------
