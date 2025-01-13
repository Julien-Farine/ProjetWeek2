import numpy as np
import math
import matplotlib.pyplot as plt

row = 10 #max 75
col = 10 #max 150
image = plt.imread("pns_original.png") #1200 * 600 * 3
bloc = image[(row-1)*8:row*8, (col-1)*8:col*8, 0] * 256
bloc = bloc.astype('int32')
bloc = bloc - 128

def dct(bloc): #taille du bloc 8*8
    sum = 0
    I = []
    for i in range(8):
        I.append(i-1)
    D = np.zeros((8,8))
    for k in I:
        for l in I:

            for i in I:
                for j in I:
                    sum += bloc[i][j] * math.cos(((2*i + 1)*k*math.pi)/16) * math.cos(((2*j + 1)*l*math.pi)/16)
            D[k][l] = 0.25 * sum
            if k == 0:
                D[k][l] *= 2**-0.5
            if l == 0:
                D[k][l] *= 2**-0.5
    return D

dctBloc = dct(bloc)
plt.imshow(dctBloc)
plt.show()
plt.imshow(bloc)
plt.show()