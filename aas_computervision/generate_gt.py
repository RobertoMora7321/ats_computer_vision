import os
import cv2
import pytesseract
import csv
import re

# ✅ Ganti dengan path folder gambar kamu
folder_path = r"C:\Users\HP\Documents\aas_computervision\test"
output_csv = os.path.join(folder_path, "..", "ground_truth.csv")

# ✅ Path ke file .exe langsung, BUKAN hanya folder!
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\HP\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Gagal membaca gambar: {image_path}")
        return None
    if img.shape[0] < 100:
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def clean_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

data = []
for filename in sorted(os.listdir(folder_path)):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(folder_path, filename)
        processed = preprocess_image(image_path)
        if processed is not None:
            raw_text = pytesseract.image_to_string(
                processed,
                config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )
            cleaned = clean_text(raw_text)
        else:
            cleaned = ""
        print(f"{filename} ➜ {cleaned}")
        data.append([filename, cleaned])

with open(output_csv, mode='w', encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image', 'ground_truth'])
    writer.writerows(data)

print(f"\n✅ ground_truth.csv berhasil dibuat di: {output_csv}")
print(f"Total data: {len(data)}")
