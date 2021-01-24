
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg
from code_utils.Mesh import *
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize

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

def conjugate_gradient(im_path,L_method ='both',gamma = 0.1,sigma=0.001):
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
    ax = fig.add_subplot('221')
    ax.imshow(z, cmap='gray')

    h,w = z.shape
    z_cs=z.T.flatten() #column stack
    noise = np.random.normal(scale=4.55,size=z_cs.shape[0])
    y_cs=z_cs+noise
    y= np.reshape(y_cs, (w, h)).T
    #plot noisy image
    ax = fig.add_subplot('222')
    ax.imshow(y,cmap='gray')
    if L_method =='conv op mat':
        L=conv_op_matrix(w,h)
    elif L_method =='graph laplacian':
        D_g= create_Degree_matrix(h,w)
        W_g = create_Walk_matrix(y_cs,h,sigma)
        L = D_g - W_g
    else:
        L1 = conv_op_matrix(w, h)
        D_g= create_Degree_matrix(h,w)
        W_g = create_Walk_matrix(y_cs,h,sigma)
        L2 = D_g - W_g

        x_rec = linalg.cg((sparse.eye(L1.shape[0]) + gamma * L1), y_cs)[0]
        ax = fig.add_subplot('223')
        ax.imshow(np.reshape(x_rec, (w, h)).T, cmap='gray')
        x_rec = linalg.cg((sparse.eye(L2.shape[0]) + gamma * L2), y_cs)[0]
        ax = fig.add_subplot('224')
        ax.imshow(np.reshape(x_rec, (w, h)).T, cmap='gray')

        plt.show()
        return
    x_rec = linalg.cg((sparse.eye(L.shape[0])+gamma*L),y_cs)[0]
    #plot reconstract image
    ax = fig.add_subplot('223')
    ax.imshow(np.reshape(x_rec, (w, h)).T, cmap='gray')
    plt.show()

def constract_W_Pointcloud(mesh,sigma=0.1,r=1):
    distanses = cdist(mesh.v,mesh.v)
    valid_dist = (distanses <= r).astype('float')
    W = sparse.lil_matrix(valid_dist)
    r,c = W.nonzero()
    weights = np.exp(-1*(distanses[r,c]**2) / (2*sigma**2))
    #weights = np.exp(-(1/(2*sigma**2)*distanses[r,c]**2))
    W[r,c] =weights
    return W

def constract_D_Pointcloud(mesh):
    D = sparse.lil_matrix((mesh.v.shape[0],mesh.v.shape[0]))
    for i,degree in enumerate(np.array(mesh.vertex_degree())):
        D[i,i] = degree
    return D

def EVD(A,normed=False):
    """
    calculate the g Laplacian matrix eigen values and right eigen vectors
    :param A: matrix
     :param normed: True if normelized
    :return:sorted lists of eigenvalues and eigenvectors
    """
    if normed:
        w,v = np.linalg.eigh(normalize(A.A))
    else:
        w,v = np.linalg.eigh(A.A)
    #ids sorted from lowest
    idx = w.argsort()
    eigenValues = w[idx]
    eigenVectors = v[:, idx]
    return eigenValues,eigenVectors

def LPF(evals,taos= [3, 5, 10],plot = True):
    biggest_eval = evals[-1]
    low_pas = lambda t, x: np.exp(-t * x / biggest_eval)
    if plot:
        fig, ax = plt.subplots()
    lp=[]
    for tao in taos:
        lp.append(low_pas(tao, evals))
        if plot:
            ax.plot(low_pas(tao, evals), evals)
    if plot:
        plt.legend(taos)
        ax.set(xlabel='eigen values', ylabel='H^',
               title='Low pass filters')
        fig.savefig("LPF.png")
        plt.show()
    return lp

def denosie_3d(off_path,s=1,sigma=4,r=25):
    mesh = Mesh(off_path)
    mesh.render_pointcloud(snap_name = 'chair pointcloud')
    noise = np.random.normal(scale=0.01*s, size=mesh.v.shape)
    noisy_mesh = Mesh(v=mesh.v+noise,f=mesh.f)
    size = noisy_mesh.v.shape[0]
    W = constract_W_Pointcloud(noisy_mesh,sigma=sigma,r=r)
    #D = constract_D_Pointcloud(noisy_mesh)
    D= np.sum(W.A, axis=1) ** (-.5)
    D = np.diag(D)
    N = sparse.eye(size) - D @ W @ D
    evals,evecs = EVD(N,normed=True)
    ev_num= [0,1,3,9]
    print(f"eigen values: {evals}")
    np.savetxt('eigen_valus',evals)

    #color the pointcloud with coresponding eigen vectors:
    for i in ev_num:
        noisy_mesh.render_pointcloud(scalar_function= evecs[:,i], point_size=10 ,snap_name='Noisy Chair pointcloud colored by '+str(i)+' eigen vector')

    #now let's plot low pass filters
    taos = [3, 5, 10]
    lpf = LPF(evals,taos= taos, plot = False)
    psi = evecs
    for i in range(len(lpf)):
        A_h = np.diag(lpf[i])
        rec = psi @ A_h @ psi.T @ noisy_mesh.v
        rec_mesh = Mesh(v=rec, f=mesh.f)
        rec_mesh.render_pointcloud( point_size=10 ,snap_name='reconstruct Chair pointcloud by lpf with tao='+str(taos[i]))





def main():
    im_path = os.path.join('data', 'al.jpg')
    conjugate_gradient(im_path,L_method ='both')
    off_path = os.path.join('data', 'chair_0015.off')
    denosie_3d(off_path)


if __name__ == "__main__":
    main()