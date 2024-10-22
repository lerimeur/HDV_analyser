from PIL import Image, ImageOps
import pytesseract
import numpy as np
import re
import csv
import cv2
import pandas as pd

date = "2024-10-21"
filename = "ressource-" + date


def saveCSV(names, prices):
    assert len(names) == len(prices)

    with open("data.csv", "w", newline="") as file:
        writer = csv.writer(file)
        field = ["Name", "Price", "date"]
        writer.writerow(field)
        for n, p in zip(names, prices):
            writer.writerow([n, p, date])

def bynariseImageText(img):
    img2 = ImageOps.invert(img)
    table = []
    pixelArray = img2.load()
    for y in range(img2.size[1]):
        List = []
        for x in range(img2.size[0]):
            if pixelArray[x, y][1] < 150:
                List.append(0)
            else:
                List.append(255)
        table.append(List)
    final_img = Image.fromarray(np.array(table).astype(np.uint8))
    return final_img

def bynariseImageBlackWhite(img, threshold=170):
    img2 = ImageOps.invert(img.convert("L"))
    table = []
    pixelArray = img2.load()
    for y in range(img2.size[1]):
        List = []
        for x in range(img2.size[0]):
            if pixelArray[x, y] < threshold:
                List.append(0)
            else:
                List.append(255)
        table.append(List)
    final_img = Image.fromarray(np.array(table).astype(np.uint8))
    return final_img

def processFrame(frame):
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

        increase_img = bynariseImageText(increase_img)
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
           
        increase_img = bynariseImageBlackWhite(increase_img, threshold=150)
        increase_img.save(
            "images/cropped/decomposed/" + "{:02d}_price".format(line) + ".png"
        )
        text = pytesseract.image_to_string(increase_img, config="--psm 6 digits")
        prices.append(re.sub(r"^\s+|\s+$", "", text))

    return names, prices

def cropVideo(path):
    cap = cv2.VideoCapture(path)
    if (cap.isOpened()== False):
        print("Error opening video file")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    names = []
    prices = []

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            if frame_count % 5 == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_names, frame_prices = processFrame(frame)
                names = names + frame_names
                prices = prices + frame_prices
            frame_count += 1
        else:
            break

    cap.release()

    saveCSV(names=names, prices=prices)

    df = pd.read_csv('data.csv')
    df = df.drop_duplicates()
    df.to_csv(filename+'.csv', index=False)

if __name__ == "__main__":
    cropVideo(path="./"+ filename +".mp4")
