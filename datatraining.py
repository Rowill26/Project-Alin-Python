import cv2
import os
import numpy as np

path_dataset = r"C:\Users\rolla\OneDrive\Documents\Rolland's folder\Kuliah Semester 2\Aljabar Linear\Projek\test\105_classes_pins_dataset"
def siapkan_data_training(folder_dataset, ukuran_target=(100, 100)):
    kumpulan_vektor_wajah = []
    label_wajah = [] 
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for nama_orang in os.listdir(folder_dataset):
        path_orang = os.path.join(folder_dataset, nama_orang)
        if os.path.isdir(path_orang):
            for nama_file in os.listdir(path_orang):
                path_gambar = os.path.join(path_orang, nama_file)
                gambar = cv2.imread(path_gambar)
                
                if gambar is not None:
                    gambar_gray = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
                    
                    wajah_terdeteksi = face_cascade.detectMultiScale(
                        gambar_gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
                    )
                    
                    if len(wajah_terdeteksi) > 0:
                        (x, y, w, h) = wajah_terdeteksi[0]
                        gambar_wajah_saja = gambar_gray[y:y+h, x:x+w]
                    else:
                        gambar_wajah_saja = gambar_gray 
                        
                    gambar_resized = cv2.resize(gambar_wajah_saja, ukuran_target)
                    vektor_1d = gambar_resized.flatten()
                    
                    kumpulan_vektor_wajah.append(vektor_1d)
                    label_wajah.append(nama_orang)

    matriks_training = np.array(kumpulan_vektor_wajah)
    return matriks_training, label_wajah, ukuran_target

matriks_gamma, label, dimensi = siapkan_data_training(path_dataset)

nama_file_simpan = r"C:\Users\rolla\Downloads\data_wajah_matriks.npz"
np.savez_compressed(nama_file_simpan, matriks=matriks_gamma, label=label)
print(f"Data TRAINING berhasil di-crop dan disimpan!")