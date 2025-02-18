import cv2
import numpy as np
import pandas


"""
Data Constants
"""
c2_img_path = "E:\\YOLO11-Vertebrae-Detection"
c7_img_path = "E:\\YOLO11-Vertebrae-Detection"
original_data_path = "E:\\bone_dataset"
c2_coords_csv_path = "E:\\YOLO11-Vertebrae-Detection\\results\\c2.csv"
c7_coords_csv_path = "E:\\YOLO11-Vertebrae-Detection\\results\\c7.csv"
c2_canny_threshold = [190, 230]
c7_canny_threshold = [150, 200]
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

def edge_detection(img, thresholds):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharped_img = cv2.equalizeHist(gray)
    gussian_img = cv2.GaussianBlur(sharped_img, (5, 5), 0)
    edges = cv2.Canny(gussian_img, thresholds[0], thresholds[1])

    return edges

def coords_translate(points_to_transfer: list, original_img_number):

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

def find_c2_bottom_points() -> list:

    left_half_x = []
    left_half_y = []
    right_half_x = []
    right_half_y = []

    # split the points of edges to left half and right half
    for i in range(len(c2_coords)):
        if(c2_coords[i][0] < c2_img_width / 2):
            left_half_x.append(c2_coords[i][0])
            left_half_y.append(c2_coords[i][1])
        elif (c2_coords[i][0] > c2_img_width / 2 and (c2_coords[i][1] <= c2_img_height * right_half_ty_factor and c2_coords[i][1] >= c2_img_height * right_half_by_factor)):
            right_half_x.append(c2_coords[i][0])
            right_half_y.append(c2_coords[i][1])

    # find the bottom left and right points
    index = np.where(left_half_y == np.max(left_half_y))[0][0]
    bottom_left_y = left_half_y[index]
    bottom_left_x = left_half_x[index]
    index = np.where(right_half_x == np.max(right_half_x))[0][0]
    bottom_right_y = right_half_y[index]
    bottom_right_x = right_half_x[index]

    return [[bottom_left_x, bottom_left_y], [bottom_right_x, bottom_right_y]]

def draw_lines(original_img, final_c2_coords):

    #Find and draw the line equation
    x = np.arange(0, original_img.shape[1])
    y = line_eq(x, final_c2_coords[0][0], final_c2_coords[0][1], final_c2_coords[1][0], final_c2_coords[1][1]).astype(np.int32)
    
    line = cv2.line(original_img, (x[0], y[0]), (x[-1], y[-1]), (0, 0, 255), 2)

    return line

"""
Function defs
"""

if __name__ == "__main__":
    
    """
    Initialization
    """

    original_c2_coords = pandas.read_csv(c2_coords_csv_path)
    right_half_ty_factor = 0.9
    right_half_by_factor = 0.75
    c2_x_offset = 20
    c7_x_offset = 20
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

        c2_edges = edge_detection(c2_img, c2_canny_threshold)
        c2_indices = np.where(c2_edges != [0])
        c2_coords = np.column_stack((c2_indices[1], c2_indices[0])).tolist()

        c7_edges = edge_detection(c7_img, c7_canny_threshold)
        c7_indices = np.where(c7_edges != [0])
        c7_coords = np.column_stack((c7_indices[1], c7_indices[0])).tolist()

        # find the points to remove from detection
        c2_index_to_pop = []
        for i in range(len(c2_coords)):
            if c2_coords[i][0] > c2_img_width - c2_x_offset:
                c2_index_to_pop.append(i)
        for i in range(len(c2_index_to_pop)):
            c2_coords.pop(c2_index_to_pop[i] - i)

        c7_index_to_pop = []
        for i in range(len(c7_coords)):
            if c7_coords[i][0] > c7_img_width - c7_x_offset:
                c7_index_to_pop.append(i)
        for i in range(len(c7_index_to_pop)):
            c7_coords.pop(c7_index_to_pop[i] - i)

        edges = cv2.cvtColor(c2_edges, cv2.COLOR_GRAY2BGR)

        c2_bottom_points = find_c2_bottom_points()

        # for i in range(len(right_half_x)):
        #     cv2.circle(edges, (right_half_x[i], right_half_y[i]), 1, (0, 0, 255), -1)

        cv2.circle(edges, (c2_bottom_points[0][0], c2_bottom_points[0][1]), 3, (0, 0, 255), -1)
        cv2.circle(edges, (c2_bottom_points[1][0], c2_bottom_points[1][1]), 3, (0, 0, 255), -1)

        final_c2_coords = coords_translate(c2_bottom_points, img_number)
        line_img = draw_lines(original_img, final_c2_coords)
        
        vector1 = np.array(final_c2_coords[0]) - np.array(final_c2_coords[1])
        print(vector1)
        
        
        cv2.imshow("img", edges)
        cv2.imshow("line", line_img)
        cv2.waitKey(0)