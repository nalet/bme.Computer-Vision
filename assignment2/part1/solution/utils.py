import numpy as np
import matplotlib.pyplot as plt


def get_normalization_matrix(x):
    """
    get_normalization_matrix Returns the transformation matrix used to normalize
    the inputs x
    Normalization corresponds to subtracting mean-position and positions
    have a mean distance of sqrt(2) to the center
    """
    x2d = x[:2, :]
    # Input: x 3*N
    # Output: T 3x3 transformation matrix of points

    center = np.mean(x2d, 1, keepdims=True)
    mean_dist = np.mean(np.sqrt(np.sum((x2d - center) ** 2, axis=0)))
    center = center.flatten()

    T = np.array([[np.sqrt(2) / mean_dist, 0, -center[0] * np.sqrt(2) / mean_dist],
                  [0, np.sqrt(2) / mean_dist, -center[1] * np.sqrt(2) / mean_dist],
                  [0, 0, 1]])

    return T


def eight_points_algorithm(x1, x2, normalize=True):
    """
    % function Fundamental_Matrix =Eight_Point_Algorithm(x1,x2)
    % Calculates the Fundamental matrix between two views from the normalized 8 point algorithm
    % Inputs:
    %               x1      3xN     homogeneous coordinates of matched points in view 1
    %               x2      3xN     homogeneous coordinates of matched points in view 2
    % Outputs:
    %               F       3x3     Fundamental matrix
    """
    N = x1.shape[1]

    if normalize:
        # Construct transformation matrices to normalize the coordinates
        T1 = get_normalization_matrix(x1)
        T2 = get_normalization_matrix(x2)

        # Normalize inputs
        x1 = T1 @ x1
        x2 = T2 @ x2

    # Construct matrix A encoding the constraints on x1 and x2
    A = np.stack((x2[0, :] * x1[0, :],
                  x2[0, :] * x1[1, :],
                  x2[0, :],
                  x2[1, :] * x1[0, :],
                  x2[1, :] * x1[1, :],
                  x2[1, :],
                  x1[0, :],
                  x1[1, :],
                  np.ones((N,))), 1)

    # Solve for f using SVD
    U, S, V = np.linalg.svd(A)
    F = V.T[:, 8].reshape(3, 3)

    # Enforce that rank(F)=2
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = (U[:, :len(S)] * S) @ V

    # Transform F back
    if normalize:
        F = T2.T @ F @ T1

    return F


def right_epipole(F):
    """ Computes the (right) epipole from a fundamental matrix F.
        (Use with F.T for left epipole.) """

    # The epipole is the null space of F (F * e = 0)
    _, _, V = np.linalg.svd(F)
    e = V[-1]
    return e / e[2]


def plot_epipolar_line(im, F, x, e, ax=None):
    """ Plot the epipole and epipolar line F*x=0
        in an image. F is the fundamental matrix
        and x a point in the other image.
    """
    m, n = im.shape[:2]
    line = np.dot(F, x)

    # epipolar line parameter and values
    t = np.linspace(0, n, 100)
    lt = np.array([(line[2] + line[0] * tt) / (-line[1]) for tt in t])

    # take only line points inside the image
    ndx = (lt >= 0) & (lt < m)
    if ax is None:
        ax = plt
    ax.plot(t[ndx], lt[ndx], linewidth=2)
    # axis.plot(e[0] / e[2], e[1] / e[2], 'r*')
