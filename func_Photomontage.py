import numpy as np

def Photomontage(P, B, a = 90., b = 5.):
    """Implements the magic of photomontage.
       >>> Photomontage(np.zeros((5, 100, 100, 3), dtype = int), np.zeros((5, 100, 100), dtype = int))[0,0]
       array([0., 0., 0.])
       >>> Photomontage(0, 0)
       Traceback (most recent call last):
        ...
       TypeError: P and B must be numpy arrays
       >>> Photomontage(np.zeros((100,)), np.zeros((100,)))
       Traceback (most recent call last):
        ...
       ValueError: wrong dimensionality
       >>> Photomontage(np.zeros((1, 100, 100, 3)), np.zeros((5, 100, 100)))
       Traceback (most recent call last):
        ...
       ValueError: wrong shape of P or B
       >>> Photomontage(np.zeros((5, 100, 100, 3), dtype = float), np.zeros((5, 100, 100)))
       Traceback (most recent call last):
        ...
       TypeError: wrong type of elements
       >>> Photomontage(np.zeros((5, 100, 100, 3), dtype = int) - 1, np.zeros((5, 100, 100), dtype = int))
       Traceback (most recent call last):
        ...
       ValueError: elements of P must be between 0 and 255 inclusive
       >>> Photomontage(np.zeros((5, 100, 100, 3), dtype = int), np.zeros((5, 100, 100), dtype = int) - 1)
       Traceback (most recent call last):
        ...
       ValueError: elements of B must be equal to 0 or 1
       >>> Photomontage(np.zeros((5, 100, 100, 3), dtype = int), np.zeros((5, 100, 100), dtype = int), '0')
       Traceback (most recent call last):
        ...
       TypeError: wrong type of a or b
       >>> Photomontage(np.zeros((5, 100, 100, 3), dtype = int), np.zeros((5, 100, 100), dtype = int), -1.)
       Traceback (most recent call last):
        ...
       ValueError: a and b must be non-negative
    """
    if type( P ) != np.ndarray or type( B ) != np.ndarray:
        raise TypeError( "P and B must be numpy arrays" )
    if len( P.shape ) != 4 or not (len( B.shape ) == 3 or len( B.shape ) == 4):
        raise ValueError( "wrong dimensionality" )
    if P.shape[0] != B.shape[0] or P.shape[3] != 3:
        raise ValueError( "wrong shape of P or B" )
    if P.shape[1] != B.shape[1] or P.shape[2] != B.shape[2]:
        raise ValueError( "wrong shape of P or B" )
    if P.dtype != 'int' or B.dtype != 'int':
        raise TypeError( "wrong type of elements" )
    if np.min( P ) < 0 or np.max( P ) > 255:
        raise ValueError( "elements of P must be between 0 and 255 inclusive" )
    if np.min( B ) < 0 or np.max( B ) > 1:
        raise ValueError( "elements of B must be equal to 0 or 1" )
    if type( a ) != float or type( b ) != float:
        raise TypeError( "wrong type of a or b" )
    if a < 0 or b < 0:
        raise ValueError( "a and b must be non-negative" )

    x, y, m = P.shape[1], P.shape[2], P.shape[0]
    K = np.abs(np.mgrid[0:m,0:m][0]).astype(int)

    if len(B.shape) == 4:
        B = B[:,:,:,0]

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