import numpy as np
import math
import matplotlib.pyplot as plt

row = 1 #max 75
col = 1 #max 150
image = plt.imread("pns_original.png") #1200 * 600 * 3
M = image[(row-1)*8:row*8, (col-1)*8:col*8, 2] * 256
M = M.astype('int32')
M += - 128

def P(n):
    P = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            P[i][j] = 0.5 * math.cos(((2*j+1)*i*math.pi)/16)
            if i == 0:
                P[i][j] = P[i][j] * 2**-0.5
    return P


P = P(8)
PT = np.transpose(P)
D = np.dot(np.dot(P, M), PT)
print(D)
