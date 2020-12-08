import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import lsmr

from tqdm.notebook import trange, tqdm


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

    # TODO
    # --------------------------------------------------------------
    # Estimate transformation matrix used to normalize
    # the inputs x
    # --------------------------------------------------------------
    x2d = x[:2, :]
    # Get centroid and mean-distance to centroid
    center = np.mean(x2d, 1, keepdims=True)
    mean_dist = np.mean(np.sqrt(np.sum((x2d - center) ** 2, axis=0)))
    center = center.flatten()

    T = np.array([[np.sqrt(2) / mean_dist, 0, -center[0] * np.sqrt(2) / mean_dist],
                  [0, np.sqrt(2) / mean_dist, -center[1] * np.sqrt(2) / mean_dist],
                  [0, 0, 1]])

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


def ransac(x1, x2, threshold, num_steps=10, random_seed=42):
    if random_seed is not None:
        np.random.seed(random_seed)  # we are using a random seed to make the results reproducible
    # TODO setup variables    
    
    #we stack the points to have the same pairs for idx    
    data = np.vstack((x1,x2)).T 
    n = 40 
    F = None
    best_err = np.inf
    inliers = None

    for _ in tqdm(range(num_steps)): #for debug purposes
        # TODO calculate initial inliers with with the best candidate points
        all_idxs = np.arange( data.shape[0] )
        np.random.shuffle(all_idxs)        
        maybe_idxs = all_idxs[:n]
        test_idxs = all_idxs[n:]
        maybeinliers = data[maybe_idxs,:]
        test_points = data[test_idxs]
        
        # TODO estimate F with all the inliers     
        maybe_F = eight_points_algorithm(data.T[:3,:n], data.T[3:,:n])
        test_err = np.diag(test_points.T[:3].T @ maybe_F @ test_points.T[3:]) ** 2
        
        # TODO find final inliers with F
        also_idxs = test_idxs[test_err < threshold]
        alsoinliers = data[also_idxs,:]
        betterdata = np.concatenate( [maybeinliers, alsoinliers] )
        
        better_F = eight_points_algorithm(betterdata.T[:3,:n], betterdata.T[3:,:n])
        better_err = np.diag(betterdata.T[:3].T @ better_F @ betterdata.T[3:]) ** 2

        err = better_err.mean()
        if err < best_err:
            F = better_F
            best_err = err
            inliers = np.concatenate( [maybe_idxs, also_idxs] )

    return F, inliers


def decompose_essential_matrix(E, x1, x2):
    """
    Decomposes E into a rotation and translation matrix using the
    normalized corresponding points x1 and x2.
    """

    # Fix left camera-matrix
    Rl = np.eye(3)
    tl = np.array([[0, 0, 0]]).T
    Pl = np.concatenate((Rl, tl), axis=1)

    # TODO: Compute possible rotations and translations
    
    # s must be [1, 1, 0]
    u, s, vh = np.linalg.svd(E)
    E = u @ np.diag([1, 1, 0]) @ vh
    u, s, vh = np.linalg.svd(E)

    w = np.array([[ 0,  1,  0], 
                  [-1,  0,  0], 
                  [ 0,  0,  1]]) 
    
    z = np.array([[ 0, -1,  0], 
                  [ 1,  0,  0],
                  [ 0,  0,  0]])
    
    R1 =  u @ w.T @ vh
    s1 = -u @ z   @ u.T
    R2 =  u @ w   @ vh
    s2 =  u @ z   @ u.T

    t1 = np.array([[s1[2, 1]], 
                   [s1[0, 2]],
                   [s1[1, 0]]])
    
    t2 = np.array([[s2[2, 1]], 
                   [s2[0, 2]], 
                   [s2[1, 0]]])  

    # Four possibilities
    Pr = [np.concatenate((R1, t1), axis=1),
          np.concatenate((R1, t2), axis=1),
          np.concatenate((R2, t1), axis=1),
          np.concatenate((R2, t2), axis=1)]

    # Compute reconstructions for all possible right camera-matrices
    X3Ds = [infer_3d(x1[:, 0:1], x2[:, 0:1], Pl, x) for x in Pr]

    # Compute projections on image-planes and find when both cameras see point
    test = [np.prod(np.hstack((Pl @ np.vstack((X3Ds[i], [[1]])), Pr[i] @ np.vstack((X3Ds[i], [[1]])))) > 0, 1) for i in
            range(4)]
    test = np.array(test)
    idx = np.where(np.hstack((test[0, 2], test[1, 2], test[2, 2], test[3, 2])) > 0.)[0][0]

    # Choose correct matrix
    Pr = Pr[idx]

    return Pl, Pr


def infer_3d(x1, x2, Pl, Pr):
    # INFER3D Infers 3d-positions of the point-correspondences x1 and x2, using
    # the rotation matrices Rl, Rr and translation vectors tl, tr. Using a
    # least-squares approach.

    M = x1.shape[1]
    # Extract rotation and translation
    Rl = Pl[:3, :3]
    tl = Pl[:3, 3]
    Rr = Pr[:3, :3]
    tr = Pr[:3, 3]

    # Construct matrix A with constraints on 3d points
    row_idx = np.tile(np.arange(4 * M), (3, 1)).T.reshape(-1)
    col_idx = np.tile(np.arange(3 * M), (1, 4)).reshape(-1)

    A = np.zeros((4 * M, 3))
    A[:M, :3] = x1[0:1, :].T @ Rl[2:3, :] - np.tile(Rl[0:1, :], (M, 1))
    A[M:2 * M, :3] = x1[1:2, :].T @ Rl[2:3, :] - np.tile(Rl[1:2, :], (M, 1))
    A[2 * M:3 * M, :3] = x2[0:1, :].T @ Rr[2:3, :] - np.tile(Rr[0:1, :], (M, 1))
    A[3 * M:4 * M, :3] = x2[1:2, :].T @ Rr[2:3, :] - np.tile(Rr[1:2, :], (M, 1))

    A = sparse.csr_matrix((A.reshape(-1), (row_idx, col_idx)), shape=(4 * M, 3 * M))

    # Construct vector b
    b = np.zeros((4 * M, 1))
    b[:M] = np.tile(tl[0], (M, 1)) - x1[0:1, :].T * tl[2]
    b[M:2 * M] = np.tile(tl[1], (M, 1)) - x1[1:2, :].T * tl[2]
    b[2 * M:3 * M] = np.tile(tr[0], (M, 1)) - x2[0:1, :].T * tr[2]
    b[3 * M:4 * M] = np.tile(tr[1], (M, 1)) - x2[1:2, :].T * tr[2]

    # Solve for 3d-positions in a least-squares way
    w = lsmr(A, b)[0]
    x3d = w.reshape(M, 3).T

    return x3d
