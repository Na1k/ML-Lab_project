from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
from reader import Reader
import cv2
import numpy as np


def create_collage(collage_path, df):
    collage = Image.new("RGBA", (1300, 600))
    x = 0
    y = 0
    for index, row in df.iterrows():
        if index > 0 and index % 500 == 0:
            image = load_img(row["Folder"] + row["Filename"])
            image = image.crop((row["Roi.X1"], row["Roi.Y1"], row["Roi.X2"], row["Roi.Y2"]))
            image = image.resize((100, 100))
            collage.paste(image, (x, y))
            x += 100
            if x > 1200:
                x = 0
                y += 100
    collage.save(collage_path)


def create_preprocessed_collage(collage_path, df):
    line = []
    columns = []
    for index, row in df.iterrows():
        if index > 0 and index % 500 == 0:
            x = row["Roi.X1"]
            y = row["Roi.Y1"]
            h = row["Roi.Y2"] - row["Roi.Y1"]
            w = row["Roi.X2"] - row["Roi.X1"]
            image = Reader.read_preprocess_img(Reader, row["Folder"] + row["Filename"], (100, 100), x, y, h, w)
            line.append(image)
            if len(line) == 13:
                columns.append(np.hstack(line))
                line = []
    collage = np.vstack(columns)
    collage = cv2.convertScaleAbs(collage, alpha=255.0)
    cv2.imwrite(collage_path, collage)
