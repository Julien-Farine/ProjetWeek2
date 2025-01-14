import numpy as np
import math
from matplotlib import pyplot as plt
import time

#Matrice de quantification pour jpeg
Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
              [12, 12, 14, 19, 26, 58, 60, 55],
              [14, 13, 16, 24, 40, 57, 69, 56],
              [14, 17, 22, 29, 51, 87, 80, 62],
              [18, 22, 37, 56, 68, 109, 103, 77],
              [24, 35, 55, 64, 81, 104, 113, 92],
              [49, 64, 78, 87, 103, 121, 120, 101],
              [72, 92, 95, 98, 112, 100, 103, 99]])



img_1=plt.imread(r'C:\Users\Gab\Code\Projet DTC\img_1.png')


def show(img):
    plt.figure()
    plt.imshow(img)
    #plt.axis('off')
  

show(img_1)

#dimension de l'image
h,w=img_1.shape[:2]
h=h-int(np.mod(h,8))
w=w-int(np.mod(w,8))

#rognage de l'image en multiple de 8
img_1=img_1[:h,:w,:]
img_2=img_1.copy()


couleurs=[]
for p in range(0,3):
    img_2=img_1[:,:,p]
    h=img_2.shape[0]
    w=img_2.shape[1]

    #centrage des valeurs rgb
    img_128=img_2-np.ones([h,w])*128

    start=time.time()
    for m in range(0,h,8):
        for n in range(0,w,8):
            M=img_128[m:m+8,n:n+8]
            P=np.zeros_like(M)
            for k in range(0,8):
                for i in range(0,8):
                    if k==0:
                        P[k,i]=P[k,i]/np.sqrt(2)
                    else:
                        P[k,i]=np.cos(((2*i+1)*k*np.pi)/16)/2
            P_T=P.T
            D=np.dot(np.dot(P,M),P_T)
            if m==0 or n==0 or m==7 or n==7 :  #à voir
                img_128[m:m+8,n:n+8]=0
            img_128[m:m+8,n:n+8]=abs(np.divide(D,Q))
    ecart=time.time()-start
    img_129=img_128.copy()

    taux_decomp=np.count_nonzero(img_128)

    for m in range(0,h,8):
        for n in range(0,w,8):
            D=img_129[m:m+8,n:n+8]*Q
            M=np.dot(np.dot(P_T,D),P)
            M=(M+128)/255
            img_129[m:m+8,n:n+8]=M
    img_129 = np.zeros((h, w), dtype=float)
    couleurs.append(img_129)
    print(len(couleurs))
img_finale=np.stack((couleurs[0],couleurs[1],couleurs[2]),axis=-1)

# Comparaison des matrices en norme L2 relative
difference = np.linalg.norm(img_1 - img_finale) / np.linalg.norm(img_1)
print(f"Différence en norme L2 relative : {difference}")

print(D)
show(M)
#show(img_128)
#show(img_129)
print(img_finale.shape)
print(img_1.shape)
print(np.count_nonzero(D))
print(ecart)
print(couleurs)

plt.figure(6)
plt.subplot(1, 2, 1)
plt.imshow(img_1)
plt.subplot(1, 2, 2)
plt.imshow(img_finale)
plt.show()
