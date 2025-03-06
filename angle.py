import cv2
import numpy as np
import pandas
import math


"""
Data Constants
"""
c2_img_path = "E:\\YOLO11-Vertebrae-Detection"
c7_img_path = "E:\\YOLO11-Vertebrae-Detection"
original_data_path = "E:\\bone_dataset"
c2_coords_csv_path = "E:\\YOLO11-Vertebrae-Detection\\results\\c2.csv"
c7_coords_csv_path = "E:\\YOLO11-Vertebrae-Detection\\results\\c7.csv"
NUMBER_OF_DATASETS = 5
"""
Data Constants
"""
    
"""
Function defs
"""

def line_eq(X, x1, y1, x2, y2):

    m = (y2 - y1) / (x2 - x1)

    return m * (X - x1) + y1

def edge_detection(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharped_img = contrast(gray, 100)
    sharped_img = hist_strech(sharped_img)
    sharped_img = cv2.equalizeHist(sharped_img)
    edges = img_binarized(sharped_img, np.min(sharped_img) + 150)
    cv2.imshow("bin", sharped_img)

    return edges

def coords_translate(points_to_transfer: list, original_img_number, c):

    if c == "c2":
        c2_coords = [
                [
                    original_c2_coords.loc[original_c2_coords["picture_name"] == (f"{original_img_number}.png")].values[0][1],
                    original_c2_coords.loc[original_c2_coords["picture_name"] == (f"{original_img_number}.png")].values[0][2]
                ],
                [
                    original_c2_coords.loc[original_c2_coords["picture_name"] == (f"{original_img_number}.png")].values[0][3],
                    original_c2_coords.loc[original_c2_coords["picture_name"] == (f"{original_img_number}.png")].values[0][4]
                ]
            ]
        final_c2_coords = [
            [
                c2_coords[0][0] + points_to_transfer[0][0],
                c2_coords[0][1] + points_to_transfer[0][1]
            ],
            [
                c2_coords[1][0] - (c2_img_width - points_to_transfer[1][0]),
                c2_coords[1][1] - (c2_img_height - points_to_transfer[1][1])
            ]
        ]
        final_c2_coords = np.int32(final_c2_coords)
        
        return final_c2_coords

    elif c == "c7":
        c7_coords = [
                [
                    original_c7_coords.loc[original_c7_coords["picture_name"] == (f"{original_img_number}.png")].values[0][1],
                    original_c7_coords.loc[original_c7_coords["picture_name"] == (f"{original_img_number}.png")].values[0][2]
                ],
                [
                    original_c7_coords.loc[original_c7_coords["picture_name"] == (f"{original_img_number}.png")].values[0][3],
                    original_c7_coords.loc[original_c7_coords["picture_name"] == (f"{original_img_number}.png")].values[0][4]
                ]
            ]
        final_c7_coords = [
            [
                c7_coords[0][0] + points_to_transfer[0][0],
                c7_coords[0][1] + points_to_transfer[0][1]
            ],
            [
                c7_coords[1][0] - (c7_img_width - points_to_transfer[1][0]),
                c7_coords[1][1] - (c7_img_height - points_to_transfer[1][1])
            ]
        ]
        final_c7_coords = np.int32(final_c7_coords)
        
        return final_c7_coords

def find_c2_bottom_points(img) -> list:

    left_half_x = []
    left_half_y = []
    right_half_x = []
    right_half_y = []

    # split the points of edges to left half and right half
    for i in range(len(c2_coords)):
        if (isbetween(c2_coords[i][0], (c2_img_width / 2) * c2_left_x_factor[0], (c2_img_width / 2) * c2_left_x_factor[1]) and isbetween(c2_coords[i][1], c2_img_height * c2_left_y_factor[0], c2_img_height * c2_left_y_factor[1])):
            left_half_x.append(c2_coords[i][0])
            left_half_y.append(c2_coords[i][1])
        elif (isbetween(c2_coords[i][0], (c2_img_width / 2) * (1 + c2_right_x_factor[0]), (c2_img_width / 2) * (1 + c2_right_x_factor[1])) and isbetween(c2_coords[i][1], c2_img_height * c2_right_y_factor[0], c2_img_height * c2_right_y_factor[1])):
            right_half_x.append(c2_coords[i][0])
            right_half_y.append(c2_coords[i][1])

    # for i in range(len(right_half_x)):
    #     cv2.circle(img, (right_half_x[i], right_half_y[i]), 1, (0, 0, 255), -1)
    # for i in range(len(left_half_x)):
    #     cv2.circle(img, (left_half_x[i], left_half_y[i]), 1, (0, 255, 0), -1)

    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    # find the bottom left and right points
    index = np.where(left_half_y == np.max(left_half_y))[0][0]
    bottom_left_y = left_half_y[index]
    bottom_left_x = left_half_x[index]
    index = np.where(right_half_y == np.max(right_half_y))[0][0]
    bottom_right_y = right_half_y[index]
    bottom_right_x = right_half_x[index]

    return [[bottom_left_x, bottom_left_y], [bottom_right_x, bottom_right_y]]

def find_c7_top_points(img) -> list:

    left_half_x = []
    left_half_y = []
    right_half_x = []
    right_half_y = []

    # split the points of edges to left half and right half
    for i in range(len(c7_coords)):
        if (isbetween(c7_coords[i][0], (c7_img_width / 2) * c7_left_x_factor[0], (c7_img_width / 2) * c7_left_x_factor[1]) and isbetween(c7_coords[i][1], c7_img_height * c7_left_y_factor[0], c7_img_height * c7_left_y_factor[1])):
            left_half_x.append(c7_coords[i][0])
            left_half_y.append(c7_coords[i][1])
        elif (isbetween(c7_coords[i][0], (c7_img_width / 2) * (1 + c7_right_x_factor[0]), (c7_img_width / 2) * (1 + c7_right_x_factor[1])) and isbetween(c7_coords[i][1], c7_img_height * c7_right_y_factor[0], c7_img_height * c7_right_y_factor[1])):
            right_half_x.append(c7_coords[i][0])
            right_half_y.append(c7_coords[i][1])

    # for i in range(len(right_half_x)):
    #     cv2.circle(img, (right_half_x[i], right_half_y[i]), 1, (0, 0, 255), -1)
    # for i in range(len(left_half_x)):
    #     cv2.circle(img, (left_half_x[i], left_half_y[i]), 1, (0, 255, 0), -1)

    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    # find the top left and right points
    index = np.where(left_half_y == np.min(left_half_y))[0][0]
    top_left_y = left_half_y[index]
    top_left_x = left_half_x[index]
    index = np.where(right_half_y == np.min(right_half_y))[0][0]
    top_right_y = right_half_y[index]
    top_right_x = right_half_x[index]

    return [[top_left_x, top_left_y], [top_right_x, top_right_y]]

def draw_lines(original_img, final_c2_coords):

    #Find and draw the line equation
    x = np.arange(0, original_img.shape[1])
    y = line_eq(x, final_c2_coords[0][0], final_c2_coords[0][1], final_c2_coords[1][0], final_c2_coords[1][1]).astype(np.int32)
    
    line = cv2.line(original_img, (x[0], y[0]), (x[-1], y[-1]), (0, 0, 255), 2)

    return line

def get_vetical_vector(vector):
    return [vector[1], -vector[0]]

def get_angle(vector1, vector2):
    return np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))

def img_binarized(img, threshold1):

    _, binary = cv2.threshold(img, threshold1, 255, cv2.THRESH_BINARY)

    return binary

def isbetween(x, a, b) -> bool:
    return a <= x <= b

def hist_strech(img):

    img_max = np.max(img)
    img_min = np.min(img)
    cdf = (img - img_min) * (255 / (img_max - img_min))
    cdf = cdf.astype(np.uint8)

    return cdf

def contrast(img, contrast):

    B = 0
    c = contrast / 255
    k = math.tan((45 + 44 * c) / 180 * math.pi)

    img = (img - 127.5 * (1 - B)) * k + 127.5 * (1 + B)
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img

"""
Function defs
"""

if __name__ == "__main__":
    
    """
    Initialization
    """

    original_c2_coords = pandas.read_csv(c2_coords_csv_path)
    original_c7_coords = pandas.read_csv(c7_coords_csv_path)
    c2_right_y_factor = [0.7, 0.9]
    c2_left_y_factor = [0.8, 0.92]
    c2_right_x_factor = [0.2, 0.6]
    c2_left_x_factor = [0.1, 0.8]
    c7_right_y_factor = [0.1, 0.3]
    c7_left_y_factor = [0.2, 0.4]
    c7_right_x_factor = [0, 1]
    c7_left_x_factor = [0, 1]

    """
    Initialization
    """

    # Find edges and load images
    for i in range(NUMBER_OF_DATASETS):

        img_number = i + 1
        original_img = cv2.imread(f"{original_data_path}\\{img_number}.png")
        c2_img = cv2.imread(f"{c2_img_path}\\{img_number}_c2.png")
        c7_img = cv2.imread(f"{c7_img_path}\\{img_number}_c7.png")
        c2_img_width = c2_img.shape[1]
        c2_img_height = c2_img.shape[0]
        c7_img_width = c7_img.shape[1]
        c7_img_height = c7_img.shape[0]

        c2_edges = edge_detection(c2_img)
        c2_indices = np.where(c2_edges != [0])
        c2_coords = np.column_stack((c2_indices[1], c2_indices[0])).tolist()

        c7_edges = edge_detection(c7_img)
        c7_indices = np.where(c7_edges != [0])
        c7_coords = np.column_stack((c7_indices[1], c7_indices[0])).tolist()

        c2_edges = cv2.cvtColor(c2_edges, cv2.COLOR_GRAY2BGR)
        c7_edges = cv2.cvtColor(c7_edges, cv2.COLOR_GRAY2BGR)

        c2_bottom_points = find_c2_bottom_points(c2_edges)
        c7_top_points = find_c7_top_points(c7_edges)

        cv2.circle(c2_edges, (c2_bottom_points[0][0], c2_bottom_points[0][1]), 3, (0, 0, 255), -1)
        cv2.circle(c2_edges, (c2_bottom_points[1][0], c2_bottom_points[1][1]), 3, (0, 0, 255), -1)
        cv2.circle(c7_edges, (c7_top_points[0][0], c7_top_points[0][1]), 3, (0, 0, 255), -1)
        cv2.circle(c7_edges, (c7_top_points[1][0], c7_top_points[1][1]), 3, (0, 0, 255), -1)

        final_c2_coords = coords_translate(c2_bottom_points, img_number, "c2")
        final_c7_coords = coords_translate(c7_top_points, img_number, "c7")
        line_img = draw_lines(original_img, final_c2_coords)
        line_img = draw_lines(line_img, final_c7_coords)
        
        vector1 = np.array(final_c2_coords[0]) - np.array(final_c2_coords[1])
        vector1 = get_vetical_vector(vector1)
        vector2 = np.array(final_c7_coords[0]) - np.array(final_c7_coords[1])
        vector2 = get_vetical_vector(vector2)

        angle = get_angle(vector1, vector2)
        angle_degree = np.degrees(angle)
        print(f"{angle} {angle_degree}")
        
        
        cv2.imshow("original", c7_img)
        cv2.imshow("c2", c2_edges)
        cv2.imshow("c7", c7_edges)
        cv2.imshow("line", line_img)
        cv2.waitKey(0)