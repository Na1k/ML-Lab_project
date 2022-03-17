import os

import cv2
from sklearn.model_selection import train_test_split
import pandas as pd
import shutil
import albumentations as A

path_to_ppm = r"C:\Users\kuest\Desktop\YOLO\realistic_data\data"  # ppm original images
meta_csv = r"C:\Users\kuest\Desktop\YOLO\realistic_data\meta\data.csv"  # csv with infos about bb and classid
all_img_result = r"C:\Users\kuest\Desktop\YOLO\dataset\images"  # all augmented images
all_labels_result = r"C:\Users\kuest\Desktop\YOLO\dataset\labels"  # all according images
destination = r"C:\Users\kuest\Desktop\YOLO\dataset\splitted_dataset"  # destination after train test split
# structure destination folder:
# |images
# |- train
# |- test
# |- val
# |labels
# |- train
# |- test
# |- val

transform = A.Compose([
    A.ColorJitter(p=1, contrast=0.8, saturation=0.68),
])


# convert to yolo bb format
def get_meta_info(rowdf, label, img_height, img_width):
    x1 = rowdf["Roi.X1"]
    y1 = rowdf["Roi.Y1"]
    x2 = rowdf["Roi.X2"]
    y2 = rowdf["Roi.Y2"]

    dw = 1. / img_width
    dh = 1. / img_height
    x = (x1 + (x2 - x1) / 2)
    y = (y1 + (y2 - y1) / 2)
    w = x2 - x1
    h = y2 - y1
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh

    c = str(label), str(x), str(y), str(w), str(h)
    c = " ".join(c)

    return c


def transform_img(_img):
    transformed = transform(image=_img)
    return transformed["image"]


def main():
    df = pd.read_csv(meta_csv, sep=";")

    df_copy = df.copy()

    for line in df_copy.iterrows():
        item = line[1]
        name = item["Image"].split(".")[0]
        image_src = os.path.join(path_to_ppm, name + ".ppm")
        label = item["ClassId"]

        # orig image
        img = cv2.imread(image_src)
        full_name = os.path.join(all_img_result, name + "_" + str(label))
        cv2.imwrite(full_name + ".png", img)

        # write label
        meta_info = get_meta_info(item, label, img.shape[0], img.shape[1])
        with open(os.path.join(all_labels_result, name + "_" + str(label) + ".txt"), 'w') as file:
            file.write(meta_info)

        # augment all images and copy labels with matching filename
        for i in range(0, 9):
            current_img_name = full_name + "_" + str(i) + ".png"
            df = df.append({'Image': current_img_name, 'ClassId': label}, ignore_index=True)
            transformed_img = transform_img(img)
            cv2.imwrite(current_img_name, transformed_img)
            with open(os.path.join(all_labels_result, name + "_" + str(label) + "_" + str(i) + ".txt"), 'w') as file:
                file.write(meta_info)

    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1:]

    # train, test, val (0.80, 0.05, 0.15)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=1234)
    X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.75, random_state=1234)

    # copy files into destination after train test split
    cases = [
        {
            "data": X_test,
            "labels": Y_test,
            "img_folder": os.path.join(destination, "images", "test"),
            "label_folder": os.path.join(destination, "labels", "test"),
            "prename": "test"
        }
        , {
            "data": X_train,
            "labels": Y_train,
            "img_folder": os.path.join(destination, "images", "train"),
            "label_folder": os.path.join(destination, "labels", "train"),
            "prename": "train"
        }
        , {
            "data": X_val,
            "labels": Y_val,
            "img_folder": os.path.join(destination, "images", "val"),
            "label_folder": os.path.join(destination, "labels", "val"),
            "prename": "val"
        }
    ]

    for case in cases:
        for i in range(0, len(case["data"])):
            name = case["data"].iloc[i]["Image"].split(".")[0]
            label = case["labels"].iloc[i]["ClassId"]
            if "\\" in name:
                name = name.split("\\")[-1]
            else:
                name = name + "_" + str(label)

            img_path_origin = os.path.join(all_img_result, name + ".png")
            img_path_dest = os.path.join(case["img_folder"], name + ".png")
            shutil.copyfile(img_path_origin, img_path_dest)

            label_path_origin = os.path.join(all_labels_result, name + ".txt")
            label_path_dest = os.path.join(case["label_folder"], name + ".txt")
            shutil.copyfile(label_path_origin, label_path_dest)


if __name__ == "__main__":
    main()
