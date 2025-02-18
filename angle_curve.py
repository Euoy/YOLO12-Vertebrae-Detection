import cv2
import numpy as np
import pandas
import matplotlib.pyplot as plt

def center_point(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

if __name__ == "__main__":
    img_name = "2.png"
    img = cv2.imread(f"D:\\bone_dataset\\{img_name}")
    c2_coords = pandas.read_csv("D:\\yolo11\\results\\c2.csv")
    c3_coords = pandas.read_csv("D:\\yolo11\\results\\c3.csv")
    c4_coords = pandas.read_csv("D:\\yolo11\\results\\c4.csv")
    c5_coords = pandas.read_csv("D:\\yolo11\\results\\c5.csv")
    c6_coords = pandas.read_csv("D:\\yolo11\\results\\c6.csv")
    c7_coords = pandas.read_csv("D:\\yolo11\\results\\c7.csv")

    coords = np.array([
            center_point(
                c2_coords.loc[c2_coords["picture_name"] == img_name].values[0][1],
                c2_coords.loc[c2_coords["picture_name"] == img_name].values[0][2],
                c2_coords.loc[c2_coords["picture_name"] == img_name].values[0][3],
                c2_coords.loc[c2_coords["picture_name"] == img_name].values[0][4]
                ),
            center_point(
                c3_coords.loc[c3_coords["picture_name"] == img_name].values[0][1],
                c3_coords.loc[c3_coords["picture_name"] == img_name].values[0][2],
                c3_coords.loc[c3_coords["picture_name"] == img_name].values[0][3],
                c3_coords.loc[c3_coords["picture_name"] == img_name].values[0][4]
                ),
            center_point(
                c4_coords.loc[c4_coords["picture_name"] == img_name].values[0][1],
                c4_coords.loc[c4_coords["picture_name"] == img_name].values[0][2],
                c4_coords.loc[c4_coords["picture_name"] == img_name].values[0][3],
                c4_coords.loc[c4_coords["picture_name"] == img_name].values[0][4]
                ),
            center_point(
                c5_coords.loc[c5_coords["picture_name"] == img_name].values[0][1],
                c5_coords.loc[c5_coords["picture_name"] == img_name].values[0][2],
                c5_coords.loc[c5_coords["picture_name"] == img_name].values[0][3],
                c5_coords.loc[c5_coords["picture_name"] == img_name].values[0][4]
                ),
            center_point(
                c6_coords.loc[c6_coords["picture_name"] == img_name].values[0][1],
                c6_coords.loc[c6_coords["picture_name"] == img_name].values[0][2],
                c6_coords.loc[c6_coords["picture_name"] == img_name].values[0][3],
                c6_coords.loc[c6_coords["picture_name"] == img_name].values[0][4]
                ),
            center_point(
                c7_coords.loc[c7_coords["picture_name"] == img_name].values[0][1],
                c7_coords.loc[c7_coords["picture_name"] == img_name].values[0][2],
                c7_coords.loc[c7_coords["picture_name"] == img_name].values[0][3],
                c7_coords.loc[c7_coords["picture_name"] == img_name].values[0][4]
                )
            ], np.int32)

    x = coords[:,0]
    y = coords[:,1]
    
    z = np.polyfit(x, y, 3)  # 3 is the degree of the polynomial
    p = np.poly1d(z)
    
    x_new = np.linspace(min(x), max(x), 200)
    y_new = p(x_new)
    new_coords = np.array([x_new, y_new], np.int32).T

    cv2.circle(img, (coords[0][0], coords[0][1]), 3, (0, 0, 255), -1)
    cv2.circle(img, (coords[1][0], coords[1][1]), 3, (0, 0, 255), -1)
    cv2.circle(img, (coords[2][0], coords[2][1]), 3, (0, 0, 255), -1)
    cv2.circle(img, (coords[3][0], coords[3][1]), 3, (0, 0, 255), -1)
    cv2.circle(img, (coords[4][0], coords[4][1]), 3, (0, 0, 255), -1)
    cv2.circle(img, (coords[5][0], coords[5][1]), 3, (0, 0, 255), -1)
    cv2.polylines(img, [coords], False, (0,255,255))

    cv2.imshow("img", img)
    cv2.waitKey(0)