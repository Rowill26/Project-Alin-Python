import numpy as np
import cv2
import os

def siapkan_data_training(folder_dataset, ukuran_target=(100, 100)):
    kumpulan_vektor_wajah = []
    label_wajah = [] 

    for nama_orang in os.listdir(folder_dataset):
        path_orang = os.path.join(folder_dataset, nama_orang)
        if os.path.isdir(path_orang):
            for nama_file in os.listdir(path_orang):
                path_gambar = os.path.join(path_orang, nama_file)
                gambar = cv2.imread(path_gambar)
                if gambar is not None:
                    gambar_gray = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
                    gambar_resized = cv2.resize(gambar_gray, ukuran_target)
                    vektor_1d = gambar_resized.flatten()
                    kumpulan_vektor_wajah.append(vektor_1d)
                    label_wajah.append(nama_orang)

    matriks_training = np.array(kumpulan_vektor_wajah)
    return matriks_training, label_wajah, ukuran_target

def uji_satu_gambar(path_gambar_uji, ukuran_target=(100, 100)):
    gambar = cv2.imread(path_gambar_uji)
    if gambar is not None:
        gambar_gray = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
        gambar_resized = cv2.resize(gambar_gray, ukuran_target)
        vektor_uji = gambar_resized.flatten()
        
        return vektor_uji
    else:
        print(f"Error: Gambar pada '{path_gambar_uji}' tidak ditemukan atau format tidak didukung.")
        return None

path_foto_test = r"C:\Users\rolla\Downloads\foto_coba_coba.jpg" 

vektor_input = uji_satu_gambar(path_foto_test)

if vektor_input is not None:
    print(f"Berhasil! Ukuran vektor uji: {vektor_input.shape}")