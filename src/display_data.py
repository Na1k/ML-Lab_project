import cv2


class ImageDisplayer:

    def __init__(self, df, range_start, range_end):
        for i in range(range_start, range_end + 1):
            img = cv2.imread(df["Folder"].iloc[i] + df["Filename"].iloc[i])

            start_point = (df["Roi.X1"].iloc[i], df["Roi.Y1"].iloc[i])
            end_point = (df["Roi.X2"].iloc[i], df["Roi.Y2"].iloc[i])
            img = cv2.rectangle(img, start_point, end_point, (255, 0, 0), 1)

            img = cv2.resize(img, (200, 200))
            cv2.imshow("Image", img)
            k = cv2.waitKey(0)
            cv2.destroyAllWindows()
