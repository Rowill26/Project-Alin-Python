import tkinter as tk
from tkinter import filedialog # Untuk buka folder/file
from PIL import Image, ImageTk # Untuk memproses gambar
import os # Untuk membaca nama file
import numpy as np
import cv2
import time

# FUNGSI LOGIKA TOMBOL
path_dataset_terpilih = ""
path_gambar_terpilih = ""

def pilih_folder():
    # Membuka file explorer untuk memilih folder
    folder_path = filedialog.askdirectory()
    
    if folder_path:
        tampil_path = (folder_path[:20] + '...') if len(folder_path) > 20 else folder_path
        lbl_status_dataset.config(text=tampil_path, fg="black")
        
        global path_dataset_terpilih
        path_dataset_terpilih = folder_path

def pilih_gambar():
    # Membuka file explorer untuk memilih gambar
    global path_gambar_terpilih # 
    file_path = filedialog.askopenfilename(
        title="Pilih Gambar Wajah",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.pgm *.bmp *.webp")]
    )
    
    if file_path:
        # Ambil nama filenya saja
        path_gambar_terpilih = file_path
        nama_file = os.path.basename(file_path)
        lbl_status_image.config(text=nama_file, fg="black")
        
        # PROSES MENAMPILKAN GAMBAR KE GUI
        img = Image.open(file_path)
        img = img.resize((200, 200)) 
        img_tk = ImageTk.PhotoImage(img)
        
        canvas_test.config(image=img_tk, width=200, height=200, bg="white")
        canvas_test.image = img_tk 

def proses_pencocokan():
    print("Mulai menghitung Eigenface...")
    global path_dataset_terpilih, path_gambar_terpilih
    
    if not path_gambar_terpilih or not path_dataset_terpilih:
        lbl_result_value.config(text="ERROR: Pilih Dataset & Gambar dulu!", fg="red")
        return
        
    lbl_result_value.config(text="Sedang memproses...", fg="orange")
    root.update() 
    mulai_waktu = time.time()
    
    try:
        folder_aktif = os.path.dirname(os.path.abspath(__file__))
        path_model_npz = os.path.join(folder_aktif, "model_eigenface_final.npz")
        model = np.load(path_model_npz)
        eigenface = model['eigenface']
        mean_face = model['mean_face']
        vektor_fitur_training = model['vektor_sudah_terlatih']
        label_training = model['label'] 
        if 'vektor_fitur' in model:
                vektor_fitur_training = model['vektor_fitur']
        elif 'vektor_sudah_terlatih' in model:
                vektor_fitur_training = model['vektor_sudah_terlatih']
        else:
                raise KeyError("Variabel vektor fitur tidak ditemukan di model.npz")
                
            # 2. PROSES GAMBAR UJI
        gambar = cv2.imread(path_gambar_terpilih)
        if gambar is None:
            lbl_result_value.config(text="ERROR: Gambar tidak terbaca OpenCV!", fg="red")
            return
        gambar_gray = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
        gambar_resized = cv2.resize(gambar_gray, (100, 100))
        vektor_uji = gambar_resized.flatten()
        
        vektor_uji_normalisasi = vektor_uji - mean_face
        omega_uji = np.dot(eigenface.T, vektor_uji_normalisasi.T)

        jarak = np.linalg.norm(vektor_fitur_training - omega_uji.reshape(-1, 1), axis=0)
        index_terdekat = np.argmin(jarak)
        jarak_terdekat = jarak[index_terdekat]
        
        BATAS_KEMIRIPAN = 4000.0 
        print(f"Jarak kemiripan wajah ini: {jarak_terdekat:.2f}")
        
        if jarak_terdekat < BATAS_KEMIRIPAN:
            nama_orang = str(label_training[index_terdekat])
            lbl_result_value.config(text=f"MATCH: {nama_orang}", fg="green")

            folder_orang_tersebut = os.path.join(path_dataset_terpilih, nama_orang)
            if os.path.exists(folder_orang_tersebut):
                file_pertama = os.listdir(folder_orang_tersebut)[0]
                path_hasil = os.path.join(folder_orang_tersebut, file_pertama)
                try:
                    img_res = Image.open(path_hasil).resize((200, 200))
                    img_res_tk = ImageTk.PhotoImage(img_res)
                    canvas_result.config(image=img_res_tk, width=200, height=200)
                    canvas_result.config(image=img_res_tk)
                    canvas_result.image = img_res_tk
                except Exception as e:
                    print("Gagal memuat gambar hasil:", e)
        else:
                # Jika nilai minimum di atas batas kemiripan 
                lbl_result_value.config(text="TIDAK DIKENALI", fg="red")
                canvas_result.config(image='')

        waktu_eksekusi = time.time() - mulai_waktu
        lbl_time.config(text=f"Execution time: {waktu_eksekusi:.2f} seconds")

    except Exception as e:
        # Menangkap semua error diam-diam agar GUI tidak nyangkut
        lbl_result_value.config(text="ERROR! Cek terminal VS Code.", fg="red")
        print(f"TERJADI ERROR PADA PROGRAM: {e}")

# MEMBUAT JENDELA & FRAME UTAMA
root = tk.Tk()
root.title("Face Recognition - Tugas Aljabar Linear")
root.geometry("800x500")
root.resizable(False, False)
root.configure(bg="#FFF1DB")

BG_LEFT = "#E8D3AD"
BG_RIGHT = "#FDF4E3"
BTN_RUN = "#8B5A2B" 

#Bikin kotak dasarnya
left_frame = tk.Frame(root, bg=BG_LEFT, width=250)
left_frame.pack(side="left", fill="both", padx=10, pady=10)
left_frame.pack_propagate(False)

right_frame = tk.Frame(root, bg=BG_RIGHT)
right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

# PANEL KIRI dengan SCROLL (Menu Input)
# Buat canvas dan scrollbar untuk panel kiri
canvas_left = tk.Canvas(left_frame, bg=BG_LEFT, highlightthickness=0)
scrollbar_left = tk.Scrollbar(left_frame, orient="vertical", command=canvas_left.yview)
left_content = tk.Frame(canvas_left, bg=BG_LEFT)

left_content.bind(
    "<Configure>",
    lambda e: canvas_left.configure(scrollregion=canvas_left.bbox("all"))
)

canvas_left.create_window((0, 0), window=left_content, anchor="nw")
canvas_left.configure(yscrollcommand=scrollbar_left.set)

canvas_left.pack(side="left", fill="both", expand=True)
scrollbar_left.pack(side="right", fill="y")

# # Mouse wheel scroll
# def _on_mousewheel(event):
#     canvas_left.yview_scroll(int(-1*(event.delta/120)), "units")
# canvas_left.bind_all("<MouseWheel>", _on_mousewheel)

lbl_dataset = tk.Label(left_content, text="Insert Your Dataset", font=("Arial", 10, "bold"), bg=BG_LEFT)
lbl_dataset.pack(anchor="w") 

btn_dataset = tk.Button(left_content, text="Choose Folder", command=pilih_folder)
btn_dataset.pack(anchor="w", pady=(5, 5))

lbl_status_dataset = tk.Label(left_content, text="No File Chosen", fg="gray", bg=BG_LEFT)
lbl_status_dataset.pack(anchor="w", pady=(0, 15))

lbl_image = tk.Label(left_content, text="Insert Your Image", font=("Arial", 10, "bold"), bg=BG_LEFT)
lbl_image.pack(anchor="w")

btn_image = tk.Button(left_content, text="Choose File", command=pilih_gambar)
btn_image.pack(anchor="w", pady=(5, 5))

lbl_status_image = tk.Label(left_content, text="No File Chosen", fg="gray", bg="#EBCDA0")
lbl_status_image.pack(anchor="w", pady=(0, 20))

btn_run = tk.Button(left_content, text="RUN RECOGNITION", bg=BTN_RUN, fg="white", font=("Arial", 10, "bold"), command=proses_pencocokan)
btn_run.pack(fill="x", pady=10)

lbl_result_title = tk.Label(left_content, text="Result", font=("Arial", 10, "bold"), bg=BG_LEFT)
lbl_result_title.pack(anchor="w")

lbl_result_value = tk.Label(left_content, text="NONE", fg="green", font=("Arial", 10), bg=BG_LEFT)
lbl_result_value.pack(anchor="w")

lbl_dataset = tk.Label(left_content, text="Group 6 - Informatics D", font=("Arial", 8), bg=BG_LEFT)
lbl_dataset.pack(side="bottom")

# 4. PANEL KANAN (Display Gambar)
lbl_title_main = tk.Label(right_frame, text="Face Recognition", font=("Arial", 20, "bold"), bg=BG_RIGHT)
lbl_title_main.pack(pady=(0, 10))

image_container = tk.Frame(right_frame, bg=BG_RIGHT)
image_container.pack()

# Kotak 1: Test Image
lbl_test_img = tk.Label(image_container, text="Test Image", font=("Arial", 10), bg=BG_RIGHT)
lbl_test_img.grid(row=0, column=0, padx=20, pady=(0, 5), sticky="w")

canvas_test = tk.Label(image_container, bg="white", width=30, height=15, relief="solid", bd=1)
canvas_test.grid(row=1, column=0, padx=20)

# Kotak 2: Closest Result
lbl_res_img = tk.Label(image_container, text="Closest Result", font=("Arial", 10), bg=BG_RIGHT)
lbl_res_img.grid(row=0, column=1, padx=20, pady=(0, 5), sticky="w")

canvas_result = tk.Label(image_container, bg="white", width=30, height=15, relief="solid", bd=1)
canvas_result.grid(row=1, column=1, padx=20)

# Waktu Eksekusi
lbl_time = tk.Label(right_frame, text="Execution time: 00.00", font=("Arial", 9), fg="green", bg=BG_RIGHT)
lbl_time.pack(pady=(20, 0))

root.mainloop()