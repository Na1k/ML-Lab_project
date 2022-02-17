import pandas as pd
import cv2
from skimage import exposure, transform


class Reader:
    df = pd.DataFrame()

    def read_data_to_df(self, folders=43, pre_path=r"./data/"):

        for i in range(0, folders):
            if i < 10:
                path = pre_path + str(i) + r"/GT-0000" + str(i) + ".csv"
            else:
                path = pre_path + str(i) + r"/GT-000" + str(i) + ".csv"

            df_tmp = pd.read_csv(path, sep=";")
            df_tmp.insert(1, "Folder", pre_path + str(i) + r"/")
            self.df = self.df.append(df_tmp)

        self.df.reset_index(inplace=True)
        self.df.drop("index", axis=1, inplace=True)
        return self.df

    def read_preprocess_img(self, image_path, size, x, y, h, w):
        image = cv2.imread(image_path)
        image = image[y:y + h, x:x + w]
        image = cv2.resize(image, size)  # transform.resize??
        image = exposure.equalize_adapthist(image, clip_limit=0.05)
        return image

    def read_img(self, image_path, clahe=True):
        image = cv2.imread(image_path)
        if clahe:
            image = exposure.equalize_adapthist(image, clip_limit=0.05)
        return image
        
    def read_preprocess_img_without_bounding(self, image_path, size, clahe=True, bounding=None):
        image = cv2.imread(image_path)
        if bounding is not None:
            image = image[bounding[1]:bounding[1] + bounding[2], bounding[0]:bounding[0] + bounding[3]]
        image = cv2.resize(image, size)  # transform.resize??
        if clahe: image = exposure.equalize_adapthist(image, clip_limit=0.05)
        return image
