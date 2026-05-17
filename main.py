import numpy as np

data_muatan = np.load("data_wajah_matriks.npz")
matriks_gamma = data_muatan['matriks']
label_wajah = data_muatan['label']

print(f"Data siap digunakan!")
print(f"Ukuran matriks Gamma: {matriks_gamma.shape}")

#lanjutkan codingan lau
