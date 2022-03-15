
import numpy as np
import cv2

def predict_and_display_img(model, image, bounding):
    preds = model.predict(np.expand_dims(image, 0))[0]
    for i in range(int(len(bounding)/4)):
        (p_x1, p_y1, p_x2, p_y2) = preds[i * 4:(i + 1) * 4]
        (r_x1, r_y1, r_x2, r_y2) = bounding[i * 4:(i + 1) * 4]
        h, w = image.shape[:2]

        p_x1 = int(p_x1 * w)
        p_y1 = int(p_y1 * h)
        p_x2 = int(p_x2 * w)
        p_y2 = int(p_y2 * h)

        r_x1 = int(r_x1 * w)
        r_y1 = int(r_y1 * h)
        r_x2 = int(r_x2 * w)
        r_y2 = int(r_y2 * h)

        cv2.rectangle(image, (p_x1, p_y1), (p_x2, p_y2), (0, 255, 0), 1)
        cv2.rectangle(image, (r_x1, r_y1), (r_x2, r_y2), (0, 0, 255), 1)

    cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
    cv2.imshow("Output", image)
    cv2.waitKey(0)