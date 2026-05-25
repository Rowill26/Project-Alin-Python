import numpy as np
import cupy as cp
import time

print("Memuat data wajah...")
data_muatan = np.load(r"C:\Users\rolla\OneDrive\Documents\Rolland's folder\Kuliah Semester 2\Aljabar Linear\Projek\src\data_wajah_matriks.npz")
matriks_gamma = data_muatan['matriks']
label_wajah = data_muatan['label']

print("Mentransfer seluruh data ke GPU RTX...")
matriks_foto_gpu = cp.asarray(matriks_gamma, dtype=cp.float32) 
jumlah_gambar = matriks_foto_gpu.shape[0]

print("Menghitung rata-rata dan normalisasi...")
mean_face = cp.mean(matriks_foto_gpu, axis=0)
matriks_normalisasi = matriks_foto_gpu - mean_face

print("Menghitung Matriks Kovarian...")
kovarian = cp.dot(matriks_normalisasi, matriks_normalisasi.T) / jumlah_gambar

#  FUNGSI GRAM-SCHMIDT MANUAL (Modifikasi N x K)
def manual_qr_gpu_nxk(A):
    n, k = A.shape
    Q = cp.zeros((n, k), dtype=cp.float32)
    R = cp.zeros((k, k), dtype=cp.float32)
    
    for j in range(k): 
        v = A[:, j]
        if j > 0:
            R[:j, j] = cp.dot(Q[:, :j].T, v)
            v = v - cp.dot(Q[:, :j], R[:j, j])
        R[j, j] = cp.linalg.norm(v)
        if R[j, j] > 0:
            Q[:, j] = v / R[j, j]
    return Q, R

# 2. ALGORITMA ITERASI SUBRUANG MANUAL
def hitung_eigen_top_K_manual(C, K=150, iterasi=10):
    n = C.shape[0]
    V = cp.random.randn(n, K, dtype=cp.float32)
    
    for it in range(iterasi):
        mulai = time.time()
        Y = cp.dot(C, V)
        V, R = manual_qr_gpu_nxk(Y)
        waktu = time.time() - mulai
        print(f"Iterasi pencarian ke-{it+1}/{iterasi} selesai dalam {waktu:.2f} detik")
        
    nilai_eigen = cp.diag(cp.dot(V.T, cp.dot(C, V)))
    return nilai_eigen, V

print("Mengekstrak 150 Ciri Wajah Utama (Top-150) secara manual...")
nilai_eigen, vektor_eigen_kovarian = hitung_eigen_top_K_manual(kovarian, K=150, iterasi=10)

print("Membentuk Eigenface dan menghitung bobot akhir...")
urutan = cp.argsort(nilai_eigen)[::-1]
vektor_eigen_kovarian = vektor_eigen_kovarian[:, urutan]

eigenface = cp.dot(matriks_normalisasi.T, vektor_eigen_kovarian)
vektor_training = cp.dot(eigenface.T, matriks_normalisasi.T)

print("Menarik hasil dari GPU ke CPU...")
eigenface_cpu = cp.asnumpy(eigenface)
mean_face_cpu = cp.asnumpy(mean_face)
vektor_training_cpu = cp.asnumpy(vektor_training)

file_model  = r"C:\Users\rolla\Downloads\model_eigenface_final.npz"
np.savez_compressed(
    file_model,
    eigenface=eigenface_cpu,
    mean_face=mean_face_cpu,
    vektor_sudah_terlatih=vektor_training_cpu,
    label=label_wajah 
)
print(f"SELESAI! {jumlah_gambar} wajah berhasil dipelajari dengan sempurna.")