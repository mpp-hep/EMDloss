import sys
import numpy as np
import h5py
from sklearn.utils import shuffle
import energyflow as ef

def caldEMD(x1, x2):
    if x1.shape[0] != x2.shape[0]: return 0
    EMD = []
    for i in range(x1.shape[0]):
        EMD.append(ef.emd.emd(x1[i,:,:], x2[i,:,:], gdim=2, R=0.8))
    return np.array(EMD, dtype = np.float32)

f = h5py.File(sys.argv[1], "r")
X1 = np.array(f.get("X")[:,:,[5,8,11]], dtype=np.float32)
#X1 = np.array(f.get("X")[:10,:,[5,8,11]], dtype=np.float32)

# make pairs
X2 = np.array(shuffle(X1, random_state=0))
# compute EMD
EMD = caldEMD(X1, X2)

print("done training")

X3 = shuffle(X2, random_state=0)
X4 = shuffle(X3, random_state=0)
# compute EMD                                                                                                                                                                               
EMD_test = caldEMD(X3, X4)

print("done test")

X5 = shuffle(X4, random_state=0)
X6 = shuffle(X5, random_state=0)
EMD_val = caldEMD(X5, X6)

print("done validation")

outFile = h5py.File(sys.argv[1].replace(".h5", "_EMD.h5"), 'w')
outFile.create_dataset("J1_train", data = X1, compression='gzip')
outFile.create_dataset("J2_train", data = X2, compression='gzip')
outFile.create_dataset("EMD_train", data = EMD,compression='gzip')
outFile.create_dataset("J1_test", data = X3, compression='gzip')
outFile.create_dataset("J2_test", data = X4, compression='gzip')
outFile.create_dataset("EMD_test", data = EMD_test,compression='gzip')
outFile.create_dataset("J1_val", data = X5, compression='gzip')
outFile.create_dataset("J2_val", data = X6, compression='gzip')
outFile.create_dataset("EMD_val", data = EMD_val,compression='gzip')
outFile.close()
