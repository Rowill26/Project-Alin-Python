import numpy as np
import math

print("memuat data wajah...")
data_muatan = np.load(r"C:\Users\rolla\Downloads\data_wajah_matriks.npz")
matriks_gamma = data_muatan['matriks']
print("data siap digunakan!")


matriks_foto = matriks_gamma[:1000] 
jumlah_gambar = matriks_foto.shape[0]

# Menghitung Rata-rata (Psi)
mean_face = np.mean(matriks_foto, axis=0)

# Menghitung Matriks Normalisasi (Phi)
matriks_normalisasi = matriks_foto - mean_face

# Matriks Kovarian (Trik Turk-Pentland: L = Phi * Phi^T)
kovarian = np.dot(matriks_normalisasi, matriks_normalisasi.T) / jumlah_gambar

def hitung_eigen_manual_dengan_QR(matriks, iterasi=50):
    #Menggunakan algoritma QR Decomposition sederhana
    n = matriks.shape[0]
    V = np.eye(n) 
    A_k = np.copy(matriks)
    
    for _ in range(iterasi):
        # (Gram-Schmidt)
        Q = np.zeros((n, n))
        R = np.zeros((n, n))
        for j in range(n):
            v = A_k[:, j]
            for i in range(j):
                R[i, j] = np.dot(Q[:, i], A_k[:, j])
                v = v - R[i, j] * Q[:, i]
            R[j, j] = np.linalg.norm(v)
            if R[j, j] > 0:
                Q[:, j] = v / R[j, j]
                
        A_k = np.dot(R, Q)
        V = np.dot(V, Q)
        
    nilai_eigen = np.diag(A_k)
    vektor_eigen = V
    return nilai_eigen, vektor_eigen

print("Menghitung nilai eigen dan vektor eigen secara manual...")
# Panggil fungsi manualnya
nilai_eigen, vektor_eigen_kovarian = hitung_eigen_manual_dengan_QR(kovarian)

# ngurutin nilai eigen secara descending (Sesuai Diagram di Dokumen)
urutan = np.argsort(nilai_eigen)[::-1]
vektor_eigen_kovarian = vektor_eigen_kovarian[:, urutan]

# U = Phi^T * Vektor Eigen
print("Membentuk Eigenface...")
eigenface = np.dot(matriks_normalisasi.T, vektor_eigen_kovarian)

print(f"SELESAI! Eigenface berhasil diekstrak dengan ukuran: {eigenface.shape}")

print("Menghitung vektor data training....")
vektor_training = np.dot(eigenface.T, matriks_normalisasi.T)
file_model  = r"C:\Users\rolla\Downloads\model_eigenface_final.npz"
np.savez_compressed(
    file_model,
    eigenface = eigenface,
    mean_face = mean_face,
    vektor_sudah_terlatih = vektor_training
)

print(f"File berhasil tersimpan ke : {file_model}")
print("File hasil training sudah siap digunakan.")