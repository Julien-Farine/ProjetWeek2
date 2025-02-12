import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage import gaussian_filter

# Lecture de l'image
img_1 = plt.imread('C:/Users/lucas/OneDrive/Images/image_bruitee.png')

# Vérifier si l'image est en RGBA (4 canaux) et la convertir en RGB si nécessaire
if img_1.shape[2] == 4:  # Si l'image a un canal alpha (RGBA)
    img_1 = img_1[:, :, :3]  # Convertir en RGB en supprimant le canal alpha

# Matrice de Quantification Q
Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
              [12, 12, 14, 19, 26, 58, 60, 55],
              [14, 13, 16, 24, 40, 57, 69, 56],
              [14, 17, 22, 29, 51, 87, 80, 62],
              [18, 22, 37, 56, 68, 109, 103, 77],
              [24, 35, 55, 64, 81, 104, 113, 92],
              [49, 64, 78, 87, 103, 121, 120, 101],
              [72, 92, 95, 98, 112, 100, 103, 99]])

def transformation_img(img):
    h, w = img.shape[:2]
    
    # Adapter les dimensions de l'image pour être divisibles par 8
    h_new = h - int(np.mod(h, 8))
    w_new = w - int(np.mod(w, 8))
    img_cropped = img[:h_new, :w_new, :]  # Image découpée aux dimensions multiples de 8
    
    img_2 = img_cropped.copy()
    
    # Appliquer un filtre gaussien pour réduire le bruit
    img_2 = gaussian_filter(img_2, sigma=0.5)  # sigma ajuste l'intensité du flou
    
    couleurs = []
    
    # Calcul de la matrice P
    for p in range(0, 3):
        P = np.zeros((8, 8))
        for k in range(8):
            for i in range(8):
                if k == 0:
                    P[k, i] = (np.cos(((2 * i + 1) * k * np.pi) / 16) / 2) / np.sqrt(2)
                else:
                    P[k, i] = np.cos(((2 * i + 1) * k * np.pi) / 16) / 2
        P_T = P.T
        
        # Compression
        D = np.empty((h_new // 8, w_new // 8), dtype=object)
        for i in range(0, h_new, 8):
            for j in range(0, w_new, 8):
                M = img_2[i:i + 8, j:j + 8, p]
                M = (M * 255).astype('int32') - 128
                D[i // 8, j // 8] = np.dot(np.dot(P, M), P_T)
                D[i // 8, j // 8] = np.round(D[i // 8, j // 8] / Q).astype('int32')
                D[i // 8, j // 8][7, :] = 0  # Dernière ligne = 0
                D[i // 8, j // 8][:, 7] = 0  # Dernière colonne = 0
                D[i // 8, j // 8][3:, :] = 0  # Mettre à zéro les lignes après la troisième
                D[i // 8, j // 8][:, 3:]=0
        # Décompression
        M = np.empty((h_new // 8, w_new // 8), dtype=object)
        for i in range(0, h_new, 8):
            for j in range(0, w_new, 8):
                M[i // 8, j // 8] = np.dot(np.dot(P_T, D[i // 8, j // 8] * Q), P)
        M = (np.stack(M) + 128) / 255
        img_129 = np.zeros((h_new, w_new), dtype=float)
        for i in range(0, h_new, 8):
            for j in range(0, w_new, 8):
                img_129[i:i + 8, j:j + 8] = M[i // 8, j // 8]
        couleurs.append(img_129)
    
    img_finale = np.stack((couleurs[0], couleurs[1], couleurs[2]), axis=-1)
    
    # Redimensionner l'image finale à la taille d'origine (h, w) de l'image
    img_finale_resized = np.zeros_like(img)  # Créer un tableau de la taille d'origine
    img_finale_resized[:h_new, :w_new, :] = img_finale[:h_new, :w_new, :]
    
    return img_finale_resized

# Mesurer le temps de calcul
start_time = time.time()

# Transformation de l'image
img_finale = transformation_img(img_1)

# Calcul du temps écoulé
end_time = time.time()
execution_time = end_time - start_time

# Affichage du temps de calcul
print(f"Temps de calcul: {execution_time:.2f} secondes")

# Comparaison des matrices en norme L2 relative
difference = np.linalg.norm(img_1 - img_finale) / np.linalg.norm(img_1)
print(f"Différence en norme L2 relative : {(difference) * 100:.2f}%")

# Affichage des images
plt.figure(2)
plt.subplot(1, 2, 1)
plt.imshow(img_1)
plt.subplot(1, 2, 2)
plt.imshow(img_finale)
plt.show()




"""
Image avec flou gaussien sans filtre

Temps de calcul pour la m�thode par le calcul: 0.77 secondes
Diff�rence obtenue via la m�thode passant par le calcul : 4.17%
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-0.053183876..1.1265721].
Temps de calcul pour la m�thode cv2: 0.44 secondes
Diff�rence en pourcentage entre l'image compress�e et l'image d'origine par cv2 : 1.54%
La m�thode passant par la librairie cv2 est meilleurs de 2.63%


Image avec flou gaussien avec filtre

Temps de calcul pour la m�thode par le calcul: 1.17 secondes
Diff�rence obtenue via la m�thode passant par le calcul : 6.07%
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-0.14445612..1.1840973].
Temps de calcul pour la m�thode cv2: 0.43 secondes
Diff�rence en pourcentage entre l'image compress�e et l'image d'origine par cv2 : 1.54%
La m�thode passant par la librairie cv2 est meilleurs de 4.54%
"""



