import numpy as np

def Photomontage(P, B, a = 90., b = 5.):

    x, y, m = P.shape[1], P.shape[2], P.shape[0]
    K = np.abs(np.mgrid[0:m,0:m][0]).astype(int)

    X = np.mgrid[0:m,0:x,0:y][2].astype(float)
    X[:,:,0] = a*(1 - B[:, :, 0])
    for i in range(1, y):
        X[:,:,i] = a * (1 - B[:, :, i]) + np.min( X[:, :, i - 1] + b * (np.linalg.norm( P[:, :, i] - P[K, :, i],
                   axis = 3 ) + np.linalg.norm( P[:, :, i - 1] - P[K, :, i - 1], axis = 3 )), axis = 1 )
        X[:,:,i - 1] = np.argmin(X[:,:,i - 1] + b * (np.linalg.norm( P[:, :, i] - P[K, :, i],
                       axis = 3 ) + np.linalg.norm( P[:, :, i - 1] - P[K, :, i - 1], axis = 3 )), axis = 1 )

    D = np.zeros((x,y), dtype = int)
    D[:,y - 1] = np.argmin(X[:,:,y - 1], axis = 0)
    for i in reversed((range(1, y - 1))):
        D[:,i] = X[D[:,i + 1],np.arange(0,x),i + np.zeros((x,), dtype = int)].astype(int)
    return np.array( [[P[D[i, j], i, j] for j in range( y )] for i in range( x )] ).astype(float) / 255

if __name__ == "__main__":
    import doctest
    doctest.testmod()