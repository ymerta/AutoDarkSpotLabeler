import streamlit as st
import cv2
import tempfile
import os
import json
from ultralytics import YOLO
from PIL import Image
from zipfile import ZipFile
import zipfile
import shutil

MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)


def is_selfie_opencv(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) != 1:
        return False

    (x, y, w, h) = faces[0]
    img_height, img_width = img.shape[:2]
    center_x = x + w / 2
    center_y = y + h / 2

    is_centered = (
        img_width * 0.3 < center_x < img_width * 0.7 and
        img_height * 0.3 < center_y < img_height * 0.7
    )
    is_large = (w * h) > (img_width * img_height * 0.2)

    return is_centered and is_large


st.set_page_config(page_title="Spot Etiketleyici (ZIP)", layout="wide")
st.title("🟤 ZIP ile Toplu Selfie + Spot Etiketleme Aracı")

uploaded_zip = st.file_uploader("📦 ZIP dosyası yükle (içinde .jpg/.png görseller olsun)", type=["zip"])

if uploaded_zip:
    temp_dir = tempfile.mkdtemp()
    image_output_dir = os.path.join(temp_dir, "images")
    annotation_output_dir = os.path.join(temp_dir, "annotations")
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(annotation_output_dir, exist_ok=True)


    with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
        zip_ref.extractall(temp_dir)


    all_image_paths = []
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                all_image_paths.append(os.path.join(root, file))

    json_files = []
    selfie_count = 0
    skipped_count = 0
    no_spot_count = 0

    with st.spinner("🔍 Etiketleme yapılıyor..."):
        for file_path in all_image_paths:
            file_name = os.path.basename(file_path)

            if not is_selfie_opencv(file_path):
                skipped_count += 1
                continue

            selfie_count += 1
            results = model.predict(source=file_path, conf=0.25, save=False)
            boxes = results[0].boxes.xyxy.cpu().numpy()

            if len(boxes) == 0:
                no_spot_count += 1
                continue


            label_data = {
                "filename": file_name,
                "selfie": "selfie",
                "spots": [{"x1": int(b[0]), "y1": int(b[1]), "x2": int(b[2]), "y2": int(b[3])} for b in boxes]
            }


            shutil.copy(file_path, os.path.join(image_output_dir, file_name))


            json_path = os.path.join(annotation_output_dir, file_name + ".json")
            with open(json_path, "w") as jf:
                json.dump(label_data, jf, indent=2)
            json_files.append(json_path)


    zip_output_path = os.path.join(temp_dir, "etiketler.zip")
    with ZipFile(zip_output_path, "w") as zipf:
        for root_dir, _, files in os.walk(temp_dir):
            for file in files:
                if file.endswith(".json") or file.lower().endswith((".jpg", ".png", ".jpeg")):
                    full_path = os.path.join(root_dir, file)
                    rel_path = os.path.relpath(full_path, temp_dir)
                    zipf.write(full_path, arcname=rel_path)


    st.success(f"✅ {len(json_files)} selfie ve spot içeren görsel etiketlendi.")
    if skipped_count > 0:
        st.warning(f"⚠️ {skipped_count} selfie olmayan görsel atlandı.")
    if no_spot_count > 0:
        st.info(f"ℹ️ {no_spot_count} selfie'de spot bulunmadığı için dahil edilmedi.")

    with open(zip_output_path, "rb") as f:
        st.download_button("⬇️ Etiketlenmiş Görselleri ve JSON'ları ZIP olarak indir", f, file_name="etiketler.zip", mime="application/zip")