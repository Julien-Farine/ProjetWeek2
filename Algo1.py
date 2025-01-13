import numpy as np
import math
import matplotlib.pyplot as plt

row = 1 #max 75
col = 1 #max 150
image = plt.imread("pns_original.png") #1200 * 600 * 3
bloc = image[(row-1)*8:row*8, (col-1)*8:col*8, 0] * 256
bloc = bloc.astype('int32')
bloc = bloc - 128
print(bloc)
plt.imshow(bloc)
plt.show()