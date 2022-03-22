from random import sample
from src.reader import Reader
from os import walk
import cv2
from itertools import cycle


def _get_random_signs(sign_df, sample_number, sign_size, with_clahe):
    signs = []
    for _, row in sign_df.sample(n=sample_number).iterrows():
        img = Reader().read_preprocess_img_without_bounding(row["Folder"]+"/"+row["Filename"],(sign_size,sign_size), clahe=with_clahe, resize=False)
        if with_clahe:
            img = img*255
        h, w = img.shape[:2]
        img = cv2.resize(img, (sign_size,sign_size))
        x1 = row["Roi.X1"]/w
        y1 = row["Roi.Y1"]/h
        x2 = row["Roi.X2"]/w
        y2 = row["Roi.Y2"]/h
        signs.append((img, int(row["ClassId"]), (x1,y1,x2,y2)))
    return signs
    
    
def _get_bg_images(path_to_bg_img_folder, size):
    images = []

    filenames = next(walk(path_to_bg_img_folder), (None, None, []))[2] 
    for img_name in filenames:
        images.append(Reader().read_preprocess_img_without_bounding(path_to_bg_img_folder+"/"+img_name, size, clahe=False))
    return images
    
    
def generate_raster_sign_image(number_of_images, number_of_signs, path_to_bg_img_folder, pre_path_sign, 
                                sign_size, raster_width, raster_height, sign_with_clahe, show_images=False):

    all_samples = [(a, b) for a in range(raster_width) for b in range(raster_height)]
    sign_df = Reader().read_data_to_df(folders=43, pre_path=pre_path_sign)
    
    labels = []
    data = []
    bg_images = _get_bg_images(path_to_bg_img_folder, (raster_width*sign_size, raster_height*sign_size))
    
    for raw_img, i in zip(cycle(bg_images), range(number_of_images)):
        img = raw_img.copy()
        signs = _get_random_signs(sign_df, number_of_signs, sign_size, sign_with_clahe)
        samples = sample(all_samples, number_of_signs)
        label = {}
        for sign, s in zip(signs, samples):
            sign_img, sign_label, sign_bounding = sign
            img[sign_size*s[1]:sign_size*(s[1]+1), sign_size*s[0]:sign_size*(s[0]+1)] = sign_img
            key = (sign_size*(s[0]+sign_bounding[0]), sign_size*(s[1]+sign_bounding[1]), 
                   sign_size*(s[0]+sign_bounding[2]), sign_size*(s[1]+sign_bounding[3]))
            label[key] = sign_label
        labels.append(label)
        data.append(img)
        
        if i % 250 == 0:
        	print("Iterated through", i, "signs")
        if show_images:
            for bounding in label:
                bounding = tuple(map(int, bounding))
                cv2.rectangle(img, bounding[:2], bounding[2:], (0, 0, 255), 1)
            cv2.namedWindow("image"+str(i), cv2.WINDOW_NORMAL)
            cv2.imshow("image"+str(i), img)
            cv2.waitKey()
            cv2.destroyAllWindows()
    return data, labels