import numpy as np
import matplotlib.pyplot as plt


def get_normalization_matrix(x):
    """
    get_normalization_matrix Returns the transformation matrix used to normalize
    the inputs x
    Normalization corresponds to subtracting mean-position and positions
    have a mean distance of sqrt(2) to the center
    """
    # Input: x 3*N
    # 
    # Output: T 3x3 transformation matrix of points

    # TODO TASK:
    # --------------------------------------------------------------
    # Estimate transformation matrix used to normalize
    # the inputs x
    # --------------------------------------------------------------

    # Get centroid and mean-distance to centroid
    c = np.mean(x, axis=1)
    md = np.sqrt(2) / np.sqrt(np.mean(np.abs(x - x.mean())**2))
    T = np.array([[md,  0,  md * -c[0]],
                  [0,  md,  md * -c[1]],
                  [0,   0,  1        ]])

    return T


def eight_points_algorithm(x1, x2, normalize=True):
    """
    Calculates the fundamental matrix between two views using the normalized 8 point algorithm
    Inputs:
                    x1      3xN     homogeneous coordinates of matched points in view 1
                    x2      3xN     homogeneous coordinates of matched points in view 2
    Outputs:
                    F       3x3     fundamental matrix
    """
    N = x1.shape[1]

    if normalize:
        # Construct transformation matrices to normalize the coordinates
        # TODO
        t1 = get_normalization_matrix(x1)
        t2 = get_normalization_matrix(x2)
        # Normalize inputs
        # TODO
        x1 = t1 @ x1
        x2 = t2 @ x2
        
        

    # Construct matrix A encoding the constraints on x1 and x2
    # TODO
    A = np.array([x1[0, :] * x2[0, :], 
                  x1[0, :] * x2[1, :], 
                  x1[0, :] * x2[2, :], 
                  
                  x1[1, :] * x2[0, :], 
                  x1[1, :] * x2[1, :], 
                  x1[1, :] * x2[2, :], 
                  
                  x1[2, :] * x2[0, :], 
                  x1[2, :] * x2[1, :], 
                  x1[2, :] * x2[2, :]])
    A = A.T

    # Solve for f using SVD
    # TODO
    u, s, vh =  np.linalg.svd(A)
    F = vh[8].reshape(3,3)
    
    # Enforce that rank(F)=2
    # TODO
    u, s, vh =  np.linalg.svd(F)
    s[2] = 0
    F =  u @ np.diag(s) @ vh

    if normalize:
        # Transform F back
        # TODO
        F = t1.T @ F @ t2

    return F


def right_epipole(F):
    """
    Computes the (right) epipole from a fundamental matrix F.
    (Use with F.T for left epipole.)
    """

    # The epipole is the null space of F (F * e = 0)
    # TODO
    u, s, vh = np.linalg.svd(F)
    e = vh[2]

    return e


def plot_epipolar_line(im, F, x, e):
    """
    Plot the epipole and epipolar line F*x=0 in an image. F is the fundamental matrix
    and x a point in the other image.
    """
    m, n = im.shape[:2]
    # TODO
    line = x @ F
    
    _x = np.array([0, n])
    
    _y = np.array([ -( line[0] * _x[0] + line[2]) / line[1],
                    -( line[0] * _x[1] + line[2]) / line[1]])

    plt.plot(_x, _y, linewidth=3, linestyle='solid')
    
    #here I'm not shure...
    plt.plot((e[2] / e[0]), (e[1] / e[2]), 'ro', markersize=5)