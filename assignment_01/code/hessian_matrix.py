from numpy import *
from scipy import sparse


def hessian_matrix(u, omega, lmbda):
    # let 'u' be the known image
    m, n, *_ = u.shape

    # define the coordinates of a neighborhood around each pixel
    # (center, north, south, west, east, north-west, north-east, south-west, south-east)
    # there are 9 possible neighbors
    rows = zeros((9, m * n))
    cols = zeros((9, m * n))
    vals = zeros((9, m * n))

    for i in range(n):
        for j in range(m):
           # center pixel (i, j)
            rows[0, i * m + j] = i * m + j
            cols[0, i * m + j] = i * m + j
            vals[0, i * m + j] = omega[0]  # here change the value (could depend on 'u')
            # northern pixel (i, j - 1)
            rows[1, i * m + j] = i * m + j
            cols[1, i * m + j] = i * m + max(j - 1, 0)
            vals[1, i * m + j] = omega[1]  # here change the value (could depend on 'u')
            # southern pixel (i, j + 1)
            rows[2, i * m + j] = i * m + j
            cols[2, i * m + j] = i * m + min(j + 1, m - 1)
            vals[2, i * m + j] = omega[2]  # here change the value (could depend on 'u')
            # western pixel  (i - 1, j)
            rows[3, i * m + j] = i * m + j
            cols[3, i * m + j] = max(i - 1, 0) * m + j
            vals[3, i * m + j] = omega[3]  # here change the value (could depend on 'u')
            # eastern pixel  (i + 1, j)
            rows[4, i * m + j] = i * m + j
            cols[4, i * m + j] = min(i + 1, n - 1) * m + j
            vals[4, i * m + j] = omega[4]  # here change the value (could depend on 'u')
            # north-western pixel (i - 1, j - 1)
            rows[5, i * m + j] = i * m + j
            cols[5, i * m + j] = max(i - 1, 0) * m + max(j - 1, 0)
            vals[5, i * m + j] = 0  # here change the value (could depend on 'u')
            # north-eastern pixel (i + 1, j - 1)
            rows[6, i * m + j] = i * m + j
            cols[6, i * m + j] = min(i + 1, n - 1) * m + max(j - 1, 0)
            vals[6, i * m + j] = 0  # here change the value (could depend on 'u')
            # south-western pixel (i - 1, j + 1)
            rows[7, i * m + j] = i * m + j
            cols[7, i * m + j] = max(i - 1, 0) * m + min(j + 1, m - 1)
            vals[7, i * m + j] = 0  # here change the value (could depend on 'u')
            # south-eastern pixel  (i + 1, j + 1)
            rows[8, i * m + j] = i * m + j
            cols[8, i * m + j] = min(i + 1, n - 1) * m + min(j + 1, m - 1)
            vals[8, i * m + j] = 0  # here change the value (could depend on 'u')

    # generate a sparse matrix
    rows = rows.reshape(-1)
    cols = cols.reshape(-1)
    vals = vals.reshape(-1)
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n * m, n * m))
    return A
