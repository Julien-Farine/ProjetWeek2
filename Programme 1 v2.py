import numpy as np
import math
import matplotlib.pyplot as plt

# Lecture de l'image
img_1 = plt.imread(r'C:\Users\Gab\Code\Projet DTC\img_1.png')

#Matrice de Quantification Q
Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
              [12, 12, 14, 19, 26, 58, 60, 55],
              [14, 13, 16, 24, 40, 57, 69, 56],
              [14, 17, 22, 29, 51, 87, 80, 62],
              [18, 22, 37, 56, 68, 109, 103, 77],
              [24, 35, 55, 64, 81, 104, 113, 92],
              [49, 64, 78, 87, 103, 121, 120, 101],
              [72, 92, 95, 98, 112, 100, 103, 99]])

def transformation_img(img):
    h,w=img.shape[:2]
    h=h-int(np.mod(h,8))
    w=w-int(np.mod(w,8))
    img=img[:h,:w,:]
    img_2=img.copy()
    couleurs=[]
    #Calcul de la matrice P
    for p in range(0,3):
        P = np.zeros((8, 8))
        for k in range(8):
            for i in range(8):
                if k==0:
                    P[k,i]=(np.cos(((2*i+1)*k*np.pi)/16)/2)/np.sqrt(2)
                else:
                    P[k,i]=np.cos(((2*i+1)*k*np.pi)/16)/2
        P_T = P.T
        #Compression
        D = np.empty((h//8,w//8), dtype=object)
        for i in range(0,h,8):
            for j in range(0,w,8):
                M = img_2[i:i+8,j:j+8, p]
                M = (M * 255).astype('int32')-128
                D[i//8, j//8] = np.dot(np.dot(P, M), P_T)
                D[i//8, j//8] = np.round(D[i//8, j//8]/Q).astype('int32')
                D[i//8, j//8][7, :] = 0  #Dernière ligne = 0
                D[i//8, j//8][:, 7] = 0  #Dernière colonne = 0
        #taux_decomp=np.count_nonzero(M) (taux de décompression à calculer)
        #Décompression
        M = np.empty((h//8,w//8), dtype=object)
        for i in range(0,h,8):
            for j in range(0,w,8):
                M[i//8, j//8] = np.dot(np.dot(P_T, D[i//8,j//8]*Q), P)
        M = (np.stack(M) + 128) / 255
        img_129 = np.zeros((h, w), dtype=float)
        for i in range(0,h,8):
            for j in range(0,w,8):
                img_129[i:i+8,j:j+8] = M[i//8, j//8]
        couleurs.append(img_129)
    img_finale=np.stack((couleurs[0],couleurs[1],couleurs[2]),axis=-1)
    return img_finale

img_finale=transformation_img(img_1)
#Comparaison des matrices en norme L2 relative
difference = np.linalg.norm(img_1 - img_finale) / np.linalg.norm(img_1)
print(f"Différence en norme L2 relative : {(difference)*100:.2f}%")

plt.figure(2)
plt.subplot(1, 2, 1)
plt.imshow(img_1)
plt.subplot(1, 2, 2)
plt.imshow(img_finale)
plt.show()