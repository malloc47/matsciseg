#!/usr/bin/env python
import numpy as np
import scipy.io
import topofix

phi = scipy.io.loadmat('TopoCut_Final_Aug/phi1b.mat')['phi']
output = topofix.topofix(phi,1,-1)
perturbM_0d = scipy.io.loadmat('TopoCut_Final_Aug/phi1a.mat')['perturbM_0d']
print(str(output))
print(str(output.max()))
print(str(output.dtype))
print(str(perturbM_0d.dtype))
print(str(np.abs(np.subtract(output,perturbM_0d)).sum()))

#subprocess.call([sys.executable, "test.py"])
