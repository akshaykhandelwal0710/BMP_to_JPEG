import math
import numpy as np

def dct(ar):
    n = ar.shape[0]
    d = np.zeros(ar.shape, dtype = 'float32')
    for i in range(8):
        for j in range(8):
            for x in range(8):
                for y in range(8):
                    d[i, j] += math.cos((2 * x + 1) * i * math.pi / 16) * math.cos((2 * y + 1) * j * math.pi / 16) * ar[x, y]
            d[i, j] /= 4
            if (i == 0):
                d[i, j] /= math.sqrt(2)
            if (j == 0):
                d[i, j] /= math.sqrt(2)
    return d