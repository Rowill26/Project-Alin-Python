import cv2
import os
import numpy as np


def siapkan_data_training(folder_dataset, ukuran_target=(100, 100)):
    kumpulan_vektor_wajah = []
    label_wajah = [] 

    for nama_orang in os.listdir(folder_dataset):
        path_orang = os.path.join(folder_dataset, nama_orang)
        if os.path.isdir(path_orang):
            for nama_file in os.listdir(path_orang):
                path_gambar = os.path.join(path_orang, nama_file)
                
                #  OpenCV
                gambar = cv2.imread(path_gambar)
                
                if gambar is not None:
                    #  RGB ke Grayscale
                    gambar_gray = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
                    
                    # Resize gambar 
                    gambar_resized = cv2.resize(gambar_gray, ukuran_target)
                    
                    # Ubah matriks 2D menjadi vektor 1D 
                    vektor_1d = gambar_resized.flatten()
                    kumpulan_vektor_wajah.append(vektor_1d)
                    label_wajah.append(nama_orang)

    matriks_training = np.array(kumpulan_vektor_wajah)
    return matriks_training, label_wajah, ukuran_target

path_dataset = r"C:\Users\rolla\OneDrive\Documents\Rolland's folder\Kuliah Semester 2\Aljabar Linear\Projek\105_classes_pins_dataset"
matriks_gamma, label, dimensi = siapkan_data_training(path_dataset)
print(f"Ukuran Matriks Training yang dihasilkan: {matriks_gamma.shape}")
print(f"Jumlah label wajah yang diproses: {len(label)}")


nama_file_simpan = r"C:\Users\rolla\Downloads\data_wajah_matriks.npz"
np.savez_compressed(nama_file_simpan, matriks=matriks_gamma, label=label)
print(f"Data berhasil dikompres dan disimpan di:\n{nama_file_simpan}")