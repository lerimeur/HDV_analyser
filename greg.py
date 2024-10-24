import cv2
import pytesseract
import numpy as np
import re
import csv
import pandas as pd
import logging
from PIL import Image, ImageOps
from tqdm import tqdm
import os
from multiprocessing import Pool, cpu_count
from fuzzywuzzy import fuzz

# Set up logging
logging.basicConfig(filename='ocr.log', level=logging.INFO)

# Function to save CSV
def save_csv(names, prices, date, filename="data.csv"):
    assert len(names) == len(prices)
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        field = ["Name", "Price", "Date"]
        writer.writerow(field)
        for n, p in zip(names, prices):
            writer.writerow([n, p, date])
    print("CSV saved successfully.")

# Optimized binarization using numpy
def binarize_image_black_white(img, threshold=170):
    img2 = ImageOps.invert(img.convert("L"))
    img_array = np.array(img2)
    img_array = np.where(img_array < threshold, 0, 255)
    return Image.fromarray(img_array.astype(np.uint8))

# Frame processing function with OCR and binarization
def process_frame(frame):
    img = Image.fromarray(frame)
    img.save("images/cropped/cropped.png")
    nb_lines = 14
    names = []
    prices = []

    for line in range(nb_lines):
        box = (0, 0 + line * 62, 1200, 0 + (line + 1) * 62)
        img_line = img.crop(box)

        name_box = (50, 0, 370, 62)
        name_img = img_line.crop(name_box)
        s = name_img.size
        ratio = 5
        increase_img = name_img.resize((s[0] * ratio, s[1] * ratio), Image.Resampling.LANCZOS)

        increase_img = binarize_image_black_white(increase_img)
        increase_img.save(
            "images/cropped/decomposed/" + "{:02d}_name".format(line) + ".png"
        )
        text = pytesseract.image_to_string(increase_img, lang="fra", config="--psm 7")
        names.append(re.sub(r"^\s+|\s+$", "", text))

        price_box = (600, 0, 750, 62)
        price_img = img_line.crop(price_box)
        s = price_img.size
        ratio = 5
        increase_img = price_img.resize((s[0] * ratio, s[1] * ratio), Image.Resampling.LANCZOS)

        increase_img = binarize_image_black_white(increase_img, threshold=150)
        increase_img.save(
            "images/cropped/decomposed/" + "{:02d}_price".format(line) + ".png"
        )
        text = pytesseract.image_to_string(increase_img, config="--psm 6 digits")
        prices.append(re.sub(r"^\s+|\s+$", "", text))

    return names, prices

# Process frames in parallel using multiprocessing
def process_frames_in_parallel(frames):
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_frame, frames), total=len(frames), desc="Processing frames in parallel"))
    names, prices = zip(*results)
    return [name for sublist in names for name in sublist], [price for sublist in prices for price in sublist]

# Read frames from video
def read_frames(path, frame_interval=3):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        logging.error("Error opening video file")
        return []

    frames = []
    frame_count = 0
    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count * frame_interval)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            frame_count += 1
        else:
            break

    cap.release()
    print(f"Read {len(frames)} frames from the video.")
    return frames

# Main function to process video
def crop_video(path):
    filename = os.path.splitext(os.path.basename(path))[0]
    date = filename.split('-')
    date = date[1]+ '-' + date[2] + '-' + date[3]
    frames = read_frames(path)
    if not frames:
        return

    # Process frames in parallel
    names, prices = process_frames_in_parallel(frames)
    
    # Save results
    save_csv(names=names, prices=prices, date=date)

    df = pd.read_csv('data.csv')
    df['Name'] = df['Name'].fillna('_')
    df = df.drop_duplicates(subset='Name')
    # df = df[df['Name'].str.isalpha()]
    df.to_csv(f'{filename}.csv', index=False)
    print("Final CSV saved successfully.")

# Process all videos in a directory
def process_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".mp4"):
            crop_video(os.path.join(directory, filename))

if __name__ == "__main__":
    process_directory("./videos")
