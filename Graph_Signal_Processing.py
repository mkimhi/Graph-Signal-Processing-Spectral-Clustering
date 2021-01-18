import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy import sparse


def conv_op_matrix(h,w,f_num=-0.2):
    """
    create 3 matrices A,B,C and use Kronecker product
    to generate convolution op matrix
    :param h: hight of original image
    :param w: width of orig image
    :param f_num: the number in the filter conv
    :return:
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

def Denoising():
    """

    :return:
    """
    #plotting
    fig = plt.figure()
    fig.suptitle('marrige with kids')

    #load iimage in grayscale
    im_path = os.path.join('data','al.jpg')
    gray_scale = Image.open(im_path).convert('LA')
    z = np.array(gray_scale)[:,:,0]

    ax = fig.add_subplot('121')
    ax.imshow(z,cmap='gray')

    h,w = z.shape
    z_0=z.T.flatten() #column stack
    noise = np.random.normal(scale=0.01,size=z_0.shape[0])
    y_0=z_0+noise
    y= np.reshape(y_0, (h, w))
    ax = fig.add_subplot('122')
    ax.imshow(y,cmap='gray')
    #plt.show()

    L=conv_op_matrix(w,h)

    print("X")