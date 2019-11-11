# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '..\\..\..\AppData\Local\Temp'))
	print(os.getcwd())
except:
	pass
# %% [markdown]
# # Assignment 1 - Blending
# 
# Name: Nalet Meinen<br>
# Matriculation number: 13-463-955

# %%
from numpy import *
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy import sparse
from scipy.signal import convolve2d
from hessian_matrix import hessian_matrix


# %%
b = array(Image.open('monalisa.png')) / 256
f = array(Image.open('putin.jpg')) / 256

# head location of Mona and Putin
m = [83, 45, 166, 139]
p = [285, 1, 585, 307]

# align foreground
f2 = zeros((256, 256, 3))
tmp = resize(f[p[1]:p[3], p[0]:p[2], :], (m[3] - m[1], m[2] - m[0]))
f2[m[1]:m[3], m[0]:m[2], :] = tmp

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(b)
plt.title('Background')
plt.subplot(1, 2, 2)
plt.imshow(f2)
plt.title('Foreground')


# %%
# apply mask
c = [118, 106]
r = 23
bb = [96, 76, 141, 109]
x, y = meshgrid(arange(256), arange(256))
omega = ((x - c[0]) ** 2 + (y - c[1]) ** 2) < r ** 2      
omega[bb[1]:bb[3], bb[0]:bb[2]] = True

plt.figure()
plt.imshow(omega)
plt.title('Mask')

# %% [markdown]
# ## (a) Gradient Descent

# %%
def GD_E(u, b, f, omega, lmbda):

    m,n,l = shape(b)
    grad = zeros((m,n,l))
    kernel = array([[0,-1,0], [-1,4,-1], [0,-1,0]])#  values of the coefficients from the derivative calculation

    for k in arange(0,l):
        grad[:,:,k] = 2 * (1-omega) * (u[:,:,k]-b[:,:,k]) + 2 * lmbda * omega * (convolve2d(u[:,:,k],kernel,'same')-convolve2d(f[:,:,k],kernel,'same'))
    
    return grad


def GD(f, b, omega, lmbda):
    """
    b: backgound color image of size (M, N, 3)
    f: foreground color image of size (M, N, 3)
    omega: foreground mask of size (M, N)
    lmbda: parameter

    :returns u: blended image of size (M, N, 3)
    """
    
    m,n,l = shape(b)
    u = b.copy()

    eps = 0.05 # step size
    err_E = iinfo(int32).max # error which determines when to stop
    grad_E = zeros((m,n,l))
    #E = [] # value of the energy function
    
    
    

    while err_E > 0.01: # until the gradient is close enough to zero
        grad_E = GD_E(u, b, f, omega, lmbda)

        err_E = max(grad_E.flatten('F'))
        #E = [E E_AB(u, b, f, omega, lambda)]

        for k in arange(0,l):
            u[:,:,k] = u[:,:,k] - eps * grad_E[:,:,k]

    clip(u,0,255,u)
    
    return u #return b


# # %%
# # blend
# lmbda = 2  # change
# u = GD(f2, b, omega, lmbda)

# # display
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(b)
# plt.subplot(1, 2, 2)
# plt.imshow(f2)

# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(expand_dims(omega, 2) * f2 + expand_dims(1 - omega, 2) * b)
# plt.subplot(1, 2, 2)
# plt.imshow(u)
# plt.show()
# %% [markdown]
# ## (b) Linearization + Gauss-Seidel

# %%
def gauss_seidel(A, b):
    N, _ = A.shape
    x = zeros((N, 1))

    L = sparse.tril(A, 0)
    U = sparse.triu(A, 1)
    
    for _ in range(15):
        x = sparse.linalg.factorized(b - U*x)
        
    x = x.reshape((256, 256,))
    return x


# %%
def LGS(f, b, omega, lmbda):
    """          
    b: backgound color image of size (M, N, 3)
    f: foreground color image of size (M, N, 3)
    omega: foreground mask of size (M, N)
    lmbda: parameter

    :returns u: blended image of size (M, N, 3)
    """
    
    u = b.copy()
    h, w, color = u.shape
    i = ones((h, w))
    I = sparse.eye(h*w)
    
    for t in range(10):
        for c in range(color):
            u_lap = hessian_matrix(u[:, :, c], omega, lmbda)
            f_lap = hessian_matrix(f[:, :, c], omega, lmbda)
            i_lap = hessian_matrix(i, omega, lmbda)

            b_c = -1*(((1-omega)*(u[:, :, c] - b[:, :, c])).reshape(-1, 1) + (u_lap-f_lap).sum(axis=1))
            A = i_lap + sparse.diags((1-omega).reshape(-1))

            x = gauss_seidel(A, b_c)

            u[:, :, c] += x
    return u #return b


# # %%
# # blend
# lmbda = 5  # change 
# u = LGS(f2, b, omega, lmbda)

# # display
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(b)
# plt.subplot(1, 2, 2)
# plt.imshow(f2)

# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(expand_dims(omega, 2) * f2 + expand_dims(1 - omega, 2) * b)
# plt.subplot(1, 2, 2)
# plt.imshow(u)
# plt.show()
# # %% [markdown]
# # ## (c) Linearization + SOR

# %%

def LSOR(f, b, omega, lmbda):
    """         
    b: backgound color image of size (M, N, 3)
    f: foreground color image of size (M, N, 3)
    omega: foreground mask of size (M, N)
    lmbda: parameter

    :returns u: blended image of size (M, N, 3)
    """
    
    # u = b.copy()
    # h, w, color = u.shape
    # i = ones((h, w))
    # I = sparse.eye(h*w)
    
    # for t in range(10):
    #     for c in range(color):
    #         u_lap = hessian_matrix(u[:, :, c], omega, lmbda)
    #         f_lap = hessian_matrix(f[:, :, c], omega, lmbda)
    #         i_lap = hessian_matrix(i, omega, lmbda)

    #         b_c = -1*(((1-omega)*(u[:, :, c] - b[:, :, c])).reshape(-1, 1) + (u_lap-f_lap).sum(axis=1))
    #         A = i_lap + sparse.diags((1-omega).reshape(-1))

    #         L = tril(A[:,:,i], -1)
    #         U = triu(A[:,:,i], 1)
    #         d = diag(A[:,:,i])
    #         D = diag(d)

    #         u[:, :, c] = linalg.solve((D+1*L),(1*c[:,:,i]-(1*U + (1-1)*D)*u[:,:,i]))
    # return u #return b

    u = b.copy()
    m,n,l = shape(b)
    A = zeros((m,n,l))
    coeff1 = array([2,0,0,0,0])
    coeff2 = array([8,-2,-2,-2,-2])
    c = zeros((m,n,l))
    n_iterations = 90
    E = array([])
    set_printoptions(threshold=256*256)
    for s in range(n_iterations):
        grad_E = GD_E(u, b, f, omega, lmbda)

        for i in range(l):
            t = u[:,:,i]
            t_ = sparse.csr_matrix(t.flatten('F'))

            ht1 = reshape( t_ * hessian_matrix(t,coeff1,lmbda), (m,n), order="F").toarray()
            ht2 = reshape( t_ * hessian_matrix(t,coeff2,lmbda), (m,n), order="F").toarray()

            background = ht1 @ (1-omega)
            foreground = ht2 @ omega

            A[:,:,i] = background + lmbda * foreground
            c[:,:,i] = A[:,:,i] @ u[:,:,i] - grad_E[:,:,i]

            L = tril(A[:,:,i], -1)
            U = triu(A[:,:,i], 1)
            d = diag(A[:,:,i])
            D = diag(d)

            A_ = D+L
            B_ = c[:,:,i] - U @ u[:,:,i] 
            u[:,:,i] = linalg.solve( A_, B_ )
    return u



# %%
# blend
lmbda = 2  # change 
u = LSOR(f2, b, omega, lmbda)

# display
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(b)
plt.subplot(1, 2, 2)
plt.imshow(f2)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(expand_dims(omega, 2) * f2 + expand_dims(1 - omega, 2) * b)
plt.subplot(1, 2, 2)
plt.imshow(u)
plt.show()
