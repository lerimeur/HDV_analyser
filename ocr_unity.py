import cv2
import pytesseract
import numpy as np
import pandas as pd
from tqdm import tqdm

# Variables globales pour ajuster la binarisation et la détection des contours
BINARY_THRESHOLD = 11
CLOSE_KERNEL_SIZE = 2
CONTOUR_AREA_THRESHOLD = 60  # Minimum contour area to be considered a valid text box

def pre_process_frame(frame):
    """Améliore l'image pour la rendre plus lisible pour l'OCR."""
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Amélioration du contraste en appliquant une égalisation d'histogramme
    equalized = cv2.equalizeHist(gray)

    # Application d'un filtre bilatéral pour réduire le bruit tout en gardant les bords nets
    filtered = cv2.bilateralFilter(equalized, 11, 17, 17)
    
    # Binarisation de l'image (seuil adaptatif pour mieux gérer les variations d'éclairage)
    binary = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, BINARY_THRESHOLD, 2)

    # Application d'une opération de fermeture pour connecter les zones de texte proches
    kernel = np.ones((CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return closed

def find_text_contours(processed_frame):
    """Trouve et isole les zones de texte à partir des contours détectés."""
    # Détection des contours
    contours, _ = cv2.findContours(processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrer les petits contours non pertinents
    text_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > CONTOUR_AREA_THRESHOLD]  # Ajuster la valeur de contourArea selon la taille des textes

    return text_contours

def extract_text_from_contours(frame, processed_frame, contours, frame_num):
    """Extrait le texte des contours sélectionnés dans une frame."""
    extracted_text = []
    
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        # Isoler la région d'intérêt (ROI)
        roi = frame[y:y+h, x:x+w]  
        
        # Découpe 35% des zones gauche et droite
        roi_left = roi[:, :int(w * 0.36)]  # Gauche
        roi_right = roi[:, int(w * 0.71):]  # Droite

        cv2.imwrite(f'natif/zone_left_{frame_num}_{i}.png', roi_left)
        cv2.imwrite(f'natif/zone_right_{frame_num}_{i}.png', roi_right)


        # Sauvegarde des images binarisées des zones gauche et droite
        binarized_left = processed_frame[y:y+h, x:x+int(w * 0.36)]
        binarized_right = processed_frame[y:y+h, x+int(w * 0.71):x+w]

        cv2.imwrite(f'output/zone_left_{frame_num}_{i}.png', binarized_left)
        cv2.imwrite(f'output/zone_right_{frame_num}_{i}.png', binarized_right)

        # Appliquer l'OCR sur les deux zones
        text_left = pytesseract.image_to_string(roi_left).strip()
        text_right = pytesseract.image_to_string(roi_right).strip()

        # Ajout au texte extrait
        extracted_text.append((text_left, text_right))

    return extracted_text

def extract_data_from_video(video_path, frame_interval=30):
    """Extrait les objets et prix d'une vidéo avec prétraitement et détection de contours."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Erreur lors de l'ouverture de la vidéo : {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Nombre total de frames dans la vidéo : {total_frames}")
    
    extracted_data = []
    
    for frame_num in tqdm(range(0, total_frames, frame_interval), desc="Traitement des frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            print(f"Impossible de lire la frame {frame_num}")
            continue

        # Prétraitement et détection des contours
        processed_frame = pre_process_frame(frame)
        contours = find_text_contours(processed_frame)

        # OCR sur les zones détectées
        raw_text_list = extract_text_from_contours(frame, processed_frame, contours, frame_num)
        
        # Filtrage et extraction des items et prix
        for text_left, text_right in raw_text_list:
            if text_left and text_right:  # Si les deux côtés sont lisibles
                extracted_data.append({"Item": text_left, "Price": text_right})

    cap.release()
    cv2.destroyAllWindows()

    if not extracted_data:
        print("Aucune donnée extraite.")
        return None

    # Créer un DataFrame pour stocker les résultats
    df = pd.DataFrame(extracted_data)

    # Suppression des doublons et correction des anomalies
    df = df.groupby("Item", as_index=False).agg({"Price": "max"})

    return df

# Exemple d'utilisation
video_path = 'unity_propre.mp4'  # Mets le bon chemin ici
df_extracted = extract_data_from_video(video_path)

# Sauvegarder le résultat final dans un CSV
if df_extracted is not None and not df_extracted.empty:
    df_extracted.to_csv("extracted_dofus_items_clean.csv", index=False)
    print("CSV sauvegardé avec succès.")
else:
    print("Aucune donnée à sauvegarder.")
