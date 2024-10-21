from PIL import Image, ImageOps
import pytesseract
import numpy as np
import os
import glob
import re


def bynariseImageBlackWhite(img, threshold=170):
    img2 = ImageOps.invert(img.convert("L"))
    # img.show()
    table = []
    pixelArray = img2.load()
    for y in range(img2.size[1]):  # binaryzate it
        List = []
        for x in range(img2.size[0]):
            if pixelArray[x, y] < threshold:
                List.append(0)
            else:
                List.append(255)
        table.append(List)
    final_img = Image.fromarray(
        np.array(table).astype(np.uint8)
    )  # load the image from array.
    return final_img


def cropImage(path="images/base/base_img.png"):
    """Split the image in line/price cropped images"""
    files = glob.glob("images/cropped/decomposed/*")
    for f in files:
        os.remove(f)
    img = Image.open(path)
    box = (625, 207, 1200, 837)
    img2 = img.crop(box)
    img2.save("images/cropped/cropped.png")
    nb_lines = int((837 - 207) / 45)
    for line in range(nb_lines):
        # Line generation
        box = (625, 207 + line * 45, 1200, 207 + (line + 1) * 45)
        img_line = img.crop(box)

        # Get name image
        name_box = (44, 0, 274, 44)
        name_img = img_line.crop(name_box)

        s = name_img.size
        ratio = 5
        increase_img = name_img.resize(
            (s[0] * ratio, s[1] * ratio), Image.Resampling.LANCZOS
        )

        # Binarise image for better IA
        increase_img = bynariseImage(increase_img, threshold=170)

        # Perform OCR using PyTesseract
        text = pytesseract.image_to_string(increase_img, lang="fra")

        # Print the extracted text without spaces at the end
        print(re.sub(r"^\s+|\s+$", "", text))

        increase_img.save(
            "images/cropped/decomposed/" + "{:02d}_name".format(line) + ".png"
        )

        # Get price image
        price_box = (430, 0, 545, 44)
        price_img = img_line.crop(price_box)

        #
        s = price_img.size
        ratio = 5
        increase_img = price_img.resize(
            (s[0] * ratio, s[1] * ratio), Image.Resampling.LANCZOS
        )

        # Binarise image for better IA
        increase_img = bynariseImage(increase_img, threshold=190)

        # Perform OCR using PyTesseract
        text = pytesseract.image_to_string(increase_img, config="--psm 6 digits")
        print(re.sub(r"^\s+|\s+$", "", text))

        increase_img.save(
            "images/cropped/decomposed/" + "{:02d}_price".format(line) + ".png"
        )
        print("----------------")


if __name__ == "__main__":
    cropImage(path="images/base/base_img_2.png")
