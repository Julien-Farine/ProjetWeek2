import numpy as np
import cv2

# Lire l'image
image = cv2.imread('votre_image.jpg', cv2.IMREAD_GRAYSCALE)

# Tronquer l’image à des multiples de 8 en x et y
height, width = image.shape
height = height - height % 8
width = width - width % 8
image = image[:height, :width]

# Transformer les intensités en entiers entre 0 et 255, puis centrer pour se ramener entre -128 et 127
image = image.astype(np.float32) - 128

# Définir la matrice de passage en fréquentiel de la DCT2
def dct2(block):
    return cv2.dct(block)

def idct2(block):
    return cv2.idct(block)



# Matrice de quantification (exemple)
Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
              [12, 12, 14, 19, 26, 58, 60, 55],
              [14, 13, 16, 24, 40, 57, 69, 56],
              [14, 17, 22, 29, 51, 87, 80, 62],
              [18, 22, 37, 56, 68, 109, 103, 77],
              [24, 35, 55, 64, 81, 104, 113, 92],
              [49, 64, 78, 87, 103, 121, 120, 101],
              [72, 92, 95, 98, 112, 100, 103, 99]])

def compress_block(block, Q):
    D = dct2(block)
    D_q = np.round(D / Q)
    return D_q

compressed_image = np.zeros_like(image)

for i in range(0, height, 8):
    for j in range(0, width, 8):
        block = image[i:i+8, j:j+8]
        compressed_image[i:i+8, j:j+8] = compress_block(block, Q)




def decompress_block(block, Q):
    D_q = block * Q
    return idct2(D_q)

decompressed_image = np.zeros_like(image)

for i in range(0, height, 8):
    for j in range(0, width, 8):
        block = compressed_image[i:i+8, j:j+8]
        decompressed_image[i:i+8, j:j+8] = decompress_block(block, Q)

# Re-transformer les valeurs entre -128 et 127 en réels entre 0 et 1
decompressed_image = np.clip(decompressed_image + 128, 0, 255).astype(np.uint8)

# Sauvegarder l'image
cv2.imwrite('image_decompressee.jpg', decompressed_image)





# Comparaison des matrices en norme L2 relative
difference = np.linalg.norm(image - decompressed_image, ord='fro') / np.linalg.norm(image, ord='fro')
print(f"Différence en norme L2 relative : {difference}")

# Affichage des images pour comparaison visuelle
cv2.imshow('Image Originale', image)
cv2.imshow('Image Décompressée', decompressed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
