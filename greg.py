import cv2
import pytesseract
import numpy as np
import re
import csv
import pandas as pd
import logging
from PIL import Image, ImageOps
from tqdm import tqdm

# Set up logging
logging.basicConfig(filename='ocr.log', level=logging.INFO)

# Set up constants
DATE = "2024-10-21"
FILENAME = "test_2"

def save_csv(names, prices):
    assert len(names) == len(prices)

    with open("data.csv", "w", newline="") as file:
        writer = csv.writer(file)
        field = ["Name", "Price", "Date"]
        writer.writerow(field)
        for n, p in zip(names, prices):
            writer.writerow([n, p, DATE])
    print("CSV saved successfully.")

def binarize_image_text(img):
    img2 = ImageOps.invert(img)
    table = []
    pixel_array = img2.load()
    for y in range(img2.size[1]):
        row = []
        for x in range(img2.size[0]):
            if pixel_array[x, y][1] < 150:
                row.append(0)
            else:
                row.append(255)
        table.append(row)
    final_img = Image.fromarray(np.array(table).astype(np.uint8))
    return final_img

def binarize_image_black_white(img, threshold=170):
    img2 = ImageOps.invert(img.convert("L"))
    table = []
    pixel_array = img2.load()
    for y in range(img2.size[1]):
        row = []
        for x in range(img2.size[0]):
            if pixel_array[x, y] < threshold:
                row.append(0)
            else:
                row.append(255)
        table.append(row)
    final_img = Image.fromarray(np.array(table).astype(np.uint8))
    return final_img

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

        increase_img = binarize_image_text(increase_img)
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

def read_frames(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        logging.error("Error opening video file")
        return

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break

    cap.release()
    print(f"Read {len(frames)} frames from the video.")
    return frames

def crop_video(path):
    frames = read_frames(path)
    if not frames:
        return

    names = []
    prices = []

    # Process each frame with a progress bar
    for i, frame in enumerate(tqdm(frames, desc="Processing frames")):
        frame_names, frame_prices = process_frame(frame)
        names.extend(frame_names)
        prices.extend(frame_prices)

    save_csv(names=names, prices=prices)

    df = pd.read_csv('data.csv')
    df = df.drop_duplicates(subset='Name')
    df = df[df['Name'].str.isalnum()]
    df.to_csv(FILENAME + '.csv', index=False)
    print("Final CSV saved successfully.")

if __name__ == "__main__":
    crop_video(path="./" + FILENAME + ".mp4")
