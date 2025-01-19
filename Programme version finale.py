import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import tkinter as tk
from tkinter import filedialog
from scipy.ndimage import gaussian_filter
import time


#Fenêtre de selection de l'image que l'on souhaite compresser
def selectionner_fichier():
    root = tk.Tk()
    root.withdraw()  # Masquer la fenêtre tkinter
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    return file_path

path_image=selectionner_fichier()




## Méthode par le calcul en utilisant matplotlib

# Lecture de l'image
img_1 = plt.imread(path_image)

#Matrice de quantification Q
Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
              [12, 12, 14, 19, 26, 58, 60, 55],
              [14, 13, 16, 24, 40, 57, 69, 56],
              [14, 17, 22, 29, 51, 87, 80, 62],
              [18, 22, 37, 56, 68, 109, 103, 77],
              [24, 35, 55, 64, 81, 104, 113, 92],
              [49, 64, 78, 87, 103, 121, 120, 101],
              [72, 92, 95, 98, 112, 100, 103, 99]])

#Matrice de quantification alternative
Q_alt = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])

def transformation_img(img, filtre=False, bruitage=False, Q=Q, save=False, titre='image finale.png'):
    h, w = img.shape[:2]
    # Adapter les dimensions de l'image pour être divisibles par 8
    h_new = h - int(np.mod(h, 8))
    w_new = w - int(np.mod(w, 8))
    img_cropped = img[:h_new, :w_new, :]  # Image découpée aux dimensions multiples de 8
    img_2 = img_cropped.copy()
    if bruitage:
        # Appliquer un filtre gaussien pour réduire le bruit
        img_2 = gaussian_filter(img_2, sigma=0.5)  # sigma ajuste l'intensité du flou
    couleurs = [] #liste pour stocker les différents canaux couleurs de l'image
    #Boucle pour les 3 couleurs de l'image (RGB)
    for p in range(0, 3):
        # Calcul de la matrice P
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
                # Filtrage des hautes fréquences : On met à zéro les coefficients au-delà d'une certaine limite
                # Ici, nous gardons uniquement les 3 premières lignes et colonnes (basses fréquences)
                if filtre: #filtre pouvant être appliqué à l'image 
                    D[i // 8, j // 8][3:, :] = 0  # Mettre à zéro les lignes après la troisième
                    D[i // 8, j // 8][:, 3:] = 0  # Mettre à zéro les colonnes après la troisième
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
    img_finale_resized = img.copy()  # Créer un tableau de la taille d'origine
    img_finale_resized[:h_new, :w_new, 0:3] = img_finale[:h_new, :w_new, :]
    if save: #sauvegarde l'image finale si on le souhaite
        # Normalisation de img_finale entre [0, 1]
        image_finale_plt = np.clip(img_finale_resized, 0, 1)
        # Sauvegarde de l'image
        plt.imsave(titre, image_finale_plt)
    return img_finale_resized



# Mesurer le temps de calcul
start_time = time.time()

image_finale_plt=transformation_img(img_1)

# Calcul du temps écoulé
end_time = time.time()
execution_time = end_time - start_time

# Affichage du temps de calcul
print(f"Temps de calcul pour la méthode par le calcul: {execution_time:.2f} secondes")

#comparaison entre l'image d'origine et l'image compresée (méthode de calcul marchant aussi avec la méthode utilisant opencv)
diff_plt = cv2.absdiff((img_1* 255).astype(np.uint8), (image_finale_plt * 255).astype(np.uint8))
diff_percentage_plt = (np.sum(diff_plt) / np.prod(diff_plt.shape)) * 100 / 255
print(f"Différence obtenue via la méthode passant par le calcul : {diff_percentage_plt:.2f}%")

#Comparaison des matrices en norme L2 relative
#difference = np.linalg.norm(img_1 - image_finale_plt) / np.linalg.norm(img_1)
#print(f"Différence en norme L2 relative par le calcul: {(difference)*100:.2f}%")

plt.figure(2)
plt.subplot(1, 2, 1)
plt.title('Image d\'origine')
plt.imshow(img_1)
plt.subplot(1, 2, 2)
plt.title('Image Transformée')
plt.imshow(image_finale_plt)
plt.show()






## Méthode par la libraire opencv

# Lire l'image en couleur (RGB) avec cv2
image = cv2.imread(path_image)

# Transformer les intensités en entiers entre 0 et 255, puis centrer pour se ramener entre -128 et 127
img_1 = image.astype(np.float32) - 128

# Définir la matrice de passage en fréquentiel de la DCT2
def dct2(block):
    return cv2.dct(block)

def idct2(block):
    return cv2.idct(block)

#fonction pour compresser l'image
def compress_block(block, Q):
    D = dct2(block)
    D_q = np.round(D / Q)
    return D_q

#fonction pour décompresser l'image
def decompress_block(block, Q):
    D_q = block * Q
    return idct2(D_q)

def transformation_img_cv2(image):
    h,w=image.shape[:2]
    h=h-int(np.mod(h,8))
    w=w-int(np.mod(w,8))
    img_1=image[:h,:w,:]
    img_2=img_1.copy()
    couleurs=[]
    for p in range(0,3):
        img_3=img_2[:,:,p]
        #transformation de l'image en multiple de 8

        #création de matrices de mêmes tailles que img_2
        compressed_image = np.zeros_like(img_3)
        decompressed_image = np.zeros_like(img_3)
        #compression puis décompression de l'image
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                block = img_3[i:i+8, j:j+8]
                compressed_block = compress_block(block, Q)
                compressed_image[i:i+8, j:j+8] = compressed_block
                decompressed_block = decompress_block(compressed_block, Q)
                decompressed_image[i:i+8, j:j+8] = decompressed_block
        # Re-transformer les valeurs entre -128 et 127 en réels entre 0 et 255
        decompressed_image = np.clip(decompressed_image + 128, 0, 255).astype(np.uint8)
        couleurs.append(decompressed_image)
    return cv2.merge((couleurs[0],couleurs[1],couleurs[2])), img_1



# Mesurer le temps de calcul
start_time = time.time()


image_finale, img_1=transformation_img_cv2(img_1)
#cv2.imwrite('image_finale_cv2.jpg',image_finale)

#convertion de l'image au bon format couleur
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_finale_cv2=cv2.cvtColor(image_finale, cv2.COLOR_BGR2RGB)
#plt.imsave('image_resultat_.jpg', image_finale_cv2)

# Calcul du temps écoulé
end_time = time.time()
execution_time = end_time - start_time

# Affichage du temps de calcul
print(f"Temps de calcul pour la méthode cv2: {execution_time:.2f} secondes")

#différence entre l'image d'origine et l'image transformé par cv2
diff_cv2 = cv2.absdiff((image * 255).astype(np.uint8), (image_finale_cv2 * 255).astype(np.uint8))
diff_percentage_cv2 = (np.sum(diff_cv2) / np.prod(diff_cv2.shape)) * 100 / 255
print(f"Différence en pourcentage entre l'image compressée et l'image d'origine par cv2 : {diff_percentage_cv2:.2f}%")


# Affichage des images pour comparaison visuelle
plt.subplot(1, 2, 1)
plt.title('Image Originale')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Image Transformée')
plt.imshow(image_finale_cv2)
plt.axis('off')

plt.show()




#comparaison des images obtenues via les deux méthodes 

diff_methods=diff_percentage_plt-diff_percentage_cv2
if diff_methods<0:
    print(f"La méthode passant par le calcul est meilleurs de {-diff_methods:.2f}%")
else:
    print(f"La méthode passant par la librairie cv2 est meilleurs de {diff_methods:.2f}%")
