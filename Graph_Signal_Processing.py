import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg
#import Networkx_Graphs
import networkx as nx
from code_utils.Mesh import *
import meshio

def conv_op_matrix(h,w,f_num=-0.2):
    """
    create 3 matrices A,B,C and use Kronecker product
    to generate convolution op matrix
    :param h: hight of original image
    :param w: width of orig image
    :param f_num: the number in the filter conv
    :return: op matrix of convolve a matrix size h,w with 3x3 filter with num value
    """
    # generate 3 matrices hxh
    ones = np.ones((h, h))
    basic_form = np.tril(ones, 1) - np.tril(ones, -2)
    basic_form[0, h - 1] = 1
    basic_form[h - 1, 0] = 1
    A = f_num * basic_form
    B = sparse.csr_matrix(A + np.eye(h) * (1 + -1*f_num))
    C = sparse.csr_matrix(A)
    A = sparse.csr_matrix(A)

    A_kron = np.tril(np.ones((w,w)),1) - np.tril(np.ones((w,w)),0)
    A_kron[w-1,0]=1
    A_kron = sparse.csr_matrix(A_kron)
    B_kron = sparse.csr_matrix(np.eye(w))
    C_kron = np.tril(np.ones((w,w)),-1) - np.tril(np.ones((w,w)),-2)
    C_kron[0,w-1]=1
    C_kron=sparse.csr_matrix(C_kron)
    return sparse.kron(A_kron,A)+sparse.kron(B_kron,B) + sparse.kron(C_kron,C)

def create_Walk_matrix_tile(image):
    """
    Currently not in use!
    take image and create a walk matrix without weights
    :param image: image
    :return: W_g
    """
    r,c = image.shape
    d1 = np.tile(np.append(np.ones(c-1), [0]), r)[:-1]
    d2 = np.append([0], d1[:c*(r-1)])
    d3 = np.ones(c*(r-1))
    d4 = d2[1:-1]
    upper_diags = sparse.diags([d1, d2, d3, d4], [1, c-1, c, c+1])
    W_g = upper_diags + upper_diags.T
    return W_g

def create_Walk_matrix(image_cs,h,sigma):
    """
    create a walk matrix with the weights as described
    :param image_cs: image in column stack
    :param h: hight of orig image
    :param sigma: for the denometer of exponent weight
    :return: W_g matrix image_cs*image_cs size
    """
    n = len(image_cs)
    W_g = sparse.lil_matrix((n, n))
    weight_fun = lambda a,b: np.exp((-1*np.abs(a-b)**2)/(2*sigma**2))
    for i in range(n):
        neigbors = [i+1,i+h-1,i+h,i+h+1]
        for j in neigbors:
            if j < n:
                W_g[i,j] =W_g[j,i] = weight_fun(image_cs[i],image_cs[i+1])
    return W_g

def create_Degree_matrix(h,w):
    """
    simply create a D_g with 8 connection but the edges
    :param h: hight of image
    :param w: width of image
    :return: D_g h*w X h*w
    """
    D_g = sparse.eye((h * w)) * 8
    D_g = sparse.lil_matrix(D_g)
    edges1 = [i for i in range(h)]
    edges2 = [i for i in range((w - 1) * h, w * h)]
    edges3 = [i * h for i in range(w)]
    edges4 = [(i * h - 1) for i in range(1, w)]
    edges = edges1 + edges2 + edges3+ edges4
    for i in edges:
        D_g[i, i] = 5
    D_g[0, 0] = D_g[0, w - 1] = D_g[h - 1, 0] = D_g[h - 1, w - 1] = 3
    return  D_g

def conjugate_gradient(im_path,L_method ='conv_op',gamma = 0.1,sigma=0.1):
    """

    :param im_path:
    :param L_method:
    :param gamma: regularization param
    :param sigma: weights param for W_g
    :return:
    """
    #plotting
    fig = plt.figure()
    fig.suptitle('marrige with kids')

    #load iimage in grayscale
    gray_scale = Image.open(im_path).convert('LA')
    z = np.array(gray_scale)[:,:,0]
    #plot orig image
    ax = fig.add_subplot('131')
    ax.imshow(z, cmap='gray')

    h,w = z.shape
    z_cs=z.T.flatten() #column stack
    noise = np.random.normal(scale=2.55,size=z_cs.shape[0])
    y_cs=z_cs+noise
    y= np.reshape(y_cs, (w, h)).T
    #plot noisy image
    ax = fig.add_subplot('132')
    ax.imshow(y,cmap='gray')
    if L_method =='conv op mat':
        L=conv_op_matrix(w,h)
    elif L_method =='graph laplacian':
        D_g= create_Degree_matrix(h,w)
        W_g = create_Walk_matrix(y_cs,h,sigma)
        L = D_g - W_g
    else:
        print("method not implemented, L_method can be 'conv op mat' or 'graph laplacian ")
    x_rec = linalg.cg((sparse.eye(L.shape[0])+gamma*L),y_cs)[0]

    #plot reconstract image
    ax = fig.add_subplot('133')
    ax.imshow(np.reshape(x_rec, (w, h)).T, cmap='gray')

    plt.show()

def denosie_3d(off_path):
    mesh = Mesh(off_path,is_meshio=True)
    #mesh=Mesh(off_path)

    mesh.render_pointcloud(snap_name = 'Toilet pointcloud')

    



def Denoising():
    im_path = os.path.join('data','al.jpg')
    #conjugate_gradient(im_path,L_method ='conv op mat')
    #conjugate_gradient(im_path,L_method ='graph laplacian')
    off_path = os.path.join('data','toilet_0010.off')
    denosie_3d(off_path)