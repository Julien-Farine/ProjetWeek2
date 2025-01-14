import numpy as np
import math
import matplotlib.pyplot as plt

# Choix du bloc 8x8
row = 1  # max 75
col = 1  # max 150
image = plt.imread('C:/Users/lucas/OneDrive/Images/image_off.png')  # Lecture image
M = image[(row - 1) * 8:row * 8, (col - 1) * 8:col * 8, 2] * 255  # Prendre un bloc 8x8 + passage valeur de 0-1 --> 0-255
M = M.astype('int32')  # Passage de float32 à int32
M += -128  # Passage de 0-255 à -128-127

# Création de la matrice de quantification Q
Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
              [12, 12, 14, 19, 26, 58, 60, 55],
              [14, 13, 16, 24, 40, 57, 69, 56],
              [14, 17, 22, 29, 51, 87, 80, 62],
              [18, 22, 37, 56, 68, 109, 103, 77],
              [24, 35, 55, 64, 81, 104, 113, 92],
              [49, 64, 78, 87, 103, 121, 120, 101],
              [72, 92, 95, 98, 112, 100, 103, 99]])

# Création de la matrice P (DCT)
P = np.zeros((8, 8))
for i in range(8):
    for j in range(8):
        P[i][j] = 0.5 * math.cos(((2 * j + 1) * i * math.pi) / 16)
        if i == 0:
            P[i][j] = P[i][j] * 2**-0.5

PT = np.transpose(P)  # P transposé (égal à P**-1)
D = np.dot(np.dot(P, M), PT)

# Quantification
for i in range(8):
    for j in range(8):
        D[i][j] = int(D[i][j] / Q[i][j])
        if j == 7:  # Dernière colonne = 0
            D[i][j] = 0
        if i == 7:  # Dernière ligne = 0
            D[i][j] = 0

# Décompression (enlève cette partie si vous ne voulez pas la décompression)
# Suppression de la décompression

# Calcul de la différence en norme L2 relative (comparaison avant/après compression)
sizeRow = len(image)
sizeCol = len(image[0])

def image_comp(couleur):  # 0,1,2
    M = np.empty((sizeRow // 8, sizeCol // 8), dtype=object)
    for i in range(sizeRow // 8):
        for j in range(sizeCol // 8):
            block = image[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8, couleur]
            block = (block * 255).astype('int32')
            block -= 128

            # Stocker le bloc dans M
            M[i, j] = block

    D = np.empty((sizeRow // 8, sizeCol // 8), dtype=object)
    for i in range(sizeRow // 8):
        for j in range(sizeCol // 8):
            D[i][j] = np.dot(np.dot(P, M[i][j]), PT)
            for k in range(8):
                for l in range(8):
                    D[i][j][k][l] = int(D[i][j][k][l] / Q[k][l])
                    if l == 7:  # Dernière colonne = 0
                        D[i][j][k][l] = 0
                    if k == 7:  # Dernière ligne = 0
                        D[i][j][k][l] = 0

    # M_new = np.empty((sizeRow // 8, sizeCol // 8), dtype=object)
    # for i in range(sizeRow // 8):
    #     for j in range(sizeCol // 8):
    #         for k in range(8):
    #             for l in range(8):
    #                 D[i][j][k][l] = D[i][j][k][l] * Q[k][l]

            # M_new[i][j] = np.dot(np.dot(PT, D[i][j]), P)

    # M_new = (M_new + 128) / 255

    # Passer d'une matrice contenant des matrices 8x8 en matrice contenant des simples int
    output_matrix = np.zeros((sizeRow, sizeCol), dtype=float)

    # Remplir la matrice résultante
    for i in range(sizeRow // 8):
        for j in range(sizeCol // 8):
            # Placer le bloc 8x8 dans la position appropriée de la matrice de sortie
            output_matrix[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = D[i][j]
    return output_matrix

# Appliquer la méthode de compression/décompression sur chaque couleur
Rouge = image_comp(0)
Vert = image_comp(1)
Bleu = image_comp(2)

# Combiner les canaux R, G, B
combined_matrix = np.stack((Rouge, Vert, Bleu), axis=-1)

# Affichage des résultats
plt.subplot(1, 2, 1)  # row 1, col 2 index 1
plt.imshow(image)
plt.subplot(1, 2, 2)  # index 2
plt.imshow(combined_matrix)
plt.show()

# Comparaison des matrices en norme L2 relative
difference = np.linalg.norm(image - combined_matrix) / np.linalg.norm(image)
print(f"Différence en norme L2 relative : {difference}")
