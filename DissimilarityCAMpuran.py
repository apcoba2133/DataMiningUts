import numpy as np
from scipy.spatial.distance import pdist, squareform


# Fungsi untuk menghitung jarak Euclidean antara dua vektor numerik
def numeric_distance(x, y):
    return np.linalg.norm(x - y)


# Fungsi untuk menghitung jarak Hamming antara dua vektor nominal
def nominal_distance(x, y):
    return np.sum(x != y)


# Fungsi untuk menghitung jarak Ordinal antara dua vektor ordinal
def ordinal_distance(x, y):
    return np.sum(np.abs(x - y))


# Fungsi untuk menghitung matriks dissimilarity campuran
def mixed_dissimilarity(data_numeric, data_nominal, data_ordinal):
    # Menghitung jarak Euclidean antara data numeric
    numeric_distances = pdist(data_numeric, metric=numeric_distance)

    # Menghitung jarak Hamming antara data nominal
    nominal_distances = pdist(data_nominal, metric=nominal_distance)

    # Menghitung jarak Ordinal antara data ordinal
    ordinal_distances = pdist(data_ordinal, metric=ordinal_distance)

    # Menggabungkan jarak dari ketiga jenis data menjadi matriks dissimilarity
    mixed_dissimilarity = squareform(numeric_distances) + squareform(nominal_distances) + squareform(ordinal_distances)

    return mixed_dissimilarity


# Contoh penggunaan
data_numeric = np.array([0, 0,75, 0,25, 0,125, 0,05, 0,375],[0,75, 0, 1, 0,625, 0,8, 0,375],[0,25, 1, 0, 0,375, 0,2, 0,625],
[0,125, 0,625, 0,375, 0, 0,175, 0,25],[0,05, 0,8, 0,2, 0,175, 0, 0,425],[0,375, 0,375, 0,625, 0,25, 0,425, 0])
data_nominal = np.array([0, 1, 1, 0, 1,	1],[1, 0, 1, 1, 1,	1],[1, 1, 0, 1, 1, 1],[0, 1, 1, 0, 1, 1],[0, 1, 1, 0, 0, 1],
                        [1, 1, 1, 1, 1, 0])
data_ordinal = np.array([0, 0,5 , 1, 0, 0,5, 1],[0,5, 0, 0,5, 0,5, 0, 0,5],[1, 0,5, 0, 1, 0,5, 0],[0, 0,5, 1, 0, 0,5, 1],
                         [0,5, 0, 0,5, 0,5, 0, 0,5],[1, 0,5, 0, 1, 0,5, 0])

dissimilarity_matrix = mixed_dissimilarity(data_numeric, data_nominal, data_ordinal)
print(dissimilarity_matrix)
