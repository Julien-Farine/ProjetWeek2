import numpy as np
import math
import matplotlib.pyplot as plt
#Choix du bloc 8*8
row = 30 #max 75
col = 50 #max 150
image = plt.imread("pns_original.png") #Lecture image
M = image[(row-1)*8:row*8, (col-1)*8:col*8, 2] * 255 # Prendre un bloc 8*8 + passage valeur de 0-1 --> 0-255
M = M.astype('int32') # Passage de float32 à int32
M += - 128 # Passage de 0-255 à -128-127

#Création de la matrice de Quantification Q
Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
              [12, 12, 14, 19, 26, 58, 60, 55],
              [14, 13, 16, 24, 40, 57, 69, 56],
              [14, 17, 22, 29, 51, 87, 80, 62],
              [18, 22, 37, 56, 68, 109, 103, 77],
              [24, 35, 55, 64, 81, 104, 113, 92],
              [49, 64, 78, 87, 103, 121, 120, 101],
              [72, 92, 95, 98, 112, 100, 103, 99]])

#Création de la matrice P
P = np.zeros((8,8))
for i in range(8):
    for j in range(8):
        P[i][j] = 0.5 * math.cos(((2*j+1)*i*math.pi)/16)
        if i == 0:
            P[i][j] = P[i][j] * 2**-0.5


PT = np.transpose(P) #P transposé (égal à P**-1)
D = np.dot(np.dot(P, M), PT)

for i in range(8):
    for j in range(8):
        D[i][j] = int(D[i][j] / Q[i][j])
        if j == 7: #Dernière colonne = 0
            D[i][j] = 0
        if i == 7: #Dernière ligne = 0
            D[i][j] = 0

# Décompression :
for i in range(8):
    for j in range(8):
        D[i][j] = int(D[i][j] * Q[i][j])

M_new = np.dot(np.dot(PT, D), P)

# Test

plt.subplot(1, 2, 1) # row 1, col 2 index 1
plt.imshow(M)
plt.subplot(1, 2, 2) # index 2
plt.imshow(M_new)
plt.show()
print(M - M_new)
