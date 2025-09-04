import cv2
import numpy as np
import pandas as pd
import math
from natsort import natsorted
from pathlib import Path
import matplotlib.pyplot as plt
import os
import alive_progress

class AngleSVACalculator():

    def __init__(self, result_save_path, original_vertebra_path, show_fig=False):
        """
        Initialization
        """

        self.result_save_path = result_save_path

        if not os.path.exists(f"{result_save_path}\\figs"):
            os.makedirs(f"{result_save_path}\\figs")
        if not os.path.exists(f"{result_save_path}\\processed_imgs"):
            os.makedirs(f"{result_save_path}\\processed_imgs")

        # Reading Image
        self.original_vertebra_path = original_vertebra_path
        self.c2_img_paths = natsorted(Path(f"{result_save_path}\\crops\\c2").rglob("*"))
        self.c7_img_paths = natsorted(Path(f"{result_save_path}\\crops\\c7").rglob("*"))

        self.c2_original_img_names = []
        self.c7_original_img_names = []
        for img_path in self.c2_img_paths:
            split = img_path.name.split("_c2")
            self.c2_original_img_names.append(f"{split[0]}{split[-1]}")
        for img_path in self.c7_img_paths:
            split = img_path.name.split("_c7")
            self.c7_original_img_names.append(f"{split[0]}{split[-1]}")
        self.common_img_names = natsorted(self.filter_different_list_value(self.c2_original_img_names, self.c7_original_img_names))
        self.c2_img_paths = (list)(map(lambda x: f"{result_save_path}\\crops\\c2\\" + x.split(".")[0] + "_c2." + x.split(".")[-1], self.common_img_names))
        self.c7_img_paths = (list)(map(lambda x: f"{result_save_path}\\crops\\c7\\" + x.split(".")[0] + "_c7." + x.split(".")[-1], self.common_img_names))

        # Reading CSV
        self.original_c2_coords = pd.read_csv(f"{result_save_path}\\c2.csv")
        self.original_c7_coords = pd.read_csv(f"{result_save_path}\\c7.csv")

        self.c2_right_y_factor = [0.6, 0.95]
        self.c2_left_y_factor = [0.8, 1]
        self.c2_right_x_factor = [0.1, 0.55]
        self.c2_left_x_factor = [0.2, 0.5]
        self.c7_right_y_factor = [0.4, 0.65]
        self.c7_left_y_factor = [0.82, 0.97]
        self.c7_right_x_factor = [0.75, 0.85]
        self.c7_left_x_factor = [0.5, 0.7]

        self.show_fig = show_fig

        """
        Initialization
        """

    """
    Function defs
    """

    def filter_different_list_value(self, list1, list2):
        return list(set(list1) & set(list2))

    def get_vetical_vector(self, vector):
        return [vector[1], -vector[0]]

    def get_angle(self, vector1, vector2):
        return np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))

    def img_binarized(self, img, threshold1):

        _, binary = cv2.threshold(img, threshold1, 255, cv2.THRESH_BINARY)

        return binary

    def isbetween(self, x, a, b) -> bool:
        return a <= x <= b

    def hist_strech(self, img):

        img_max = np.max(img)
        img_min = np.min(img)
        cdf = (img - img_min) * (255 / (img_max - img_min))
        cdf = cdf.astype(np.uint8)

        return cdf

    def contrast(self, img, contrast):

        B = 0
        c = contrast / 255
        k = math.tan((45 + 44 * c) / 180 * math.pi)

        img = (img - 127.5 * (1 - B)) * k + 127.5 * (1 + B)
        img = np.clip(img, 0, 255).astype(np.uint8)

        return img

    def line_eq(self, X, x1, y1, x2, y2):

        m = (y2 - y1) / (x2 - x1)

        return m * (X - x1) + y1
    
    
    def draw_long_line(self, img, points):

        #Find and draw the line equation
        x = np.arange(0, img.shape[1])
        y = self.line_eq(x, points[0][0], points[0][1], points[1][0], points[1][1]).astype(np.int32)
        
        cv2.line(img, (x[0], y[0]), (x[-1], y[-1]), (255, 0, 0), 2)
    
    def img_enhance(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = self.contrast(img, 30)

        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4,4))
        img = clahe.apply(img)

        img = self.hist_strech(img)

        return img

    def edge_detection(self, img):

        blur = cv2.GaussianBlur(img, (5, 5), 0)

        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                 cv2.THRESH_BINARY,11, 2)
        
        # Get max connected component
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        pixel_num = stats[:, cv2.CC_STAT_AREA]
        idx = np.where(pixel_num == np.max(pixel_num[1:]))[0]

        lcc = np.zeros(binary.shape, dtype=np.uint8)
        lcc[labels == idx] = 255

        return lcc

    def coords_translate(self, points_to_transfer: list, original_img_name, c):

        if c == "c2":
            c2_coords = [
                    self.original_c2_coords.loc[self.original_c2_coords["picture_name"] == original_img_name].values[0][1],
                    self.original_c2_coords.loc[self.original_c2_coords["picture_name"] == original_img_name].values[0][2]
                    ]
            final_c2_coords = [
                    c2_coords[0] + points_to_transfer[0],
                    c2_coords[1] + points_to_transfer[1]
                ]
            final_c2_coords = np.int32(final_c2_coords)
            
            return final_c2_coords

        elif c == "c7":
            c7_coords = [
                    self.original_c7_coords.loc[self.original_c7_coords["picture_name"] == original_img_name].values[0][1],
                    self.original_c7_coords.loc[self.original_c7_coords["picture_name"] == original_img_name].values[0][2]
                ]
            final_c7_coords = [
                c7_coords[0] + points_to_transfer[0],
                c7_coords[1] + points_to_transfer[1]
                ]
            final_c7_coords = np.int32(final_c7_coords)
            
            return final_c7_coords

    def find_c2_bottom_points(self, reversed=False) -> list:

        left_half_x = []
        left_half_y = []
        right_half_x = []
        right_half_y = []

        
        if reversed:
            c2_left_y_factor = self.c2_right_y_factor
            c2_right_y_factor = self.c2_left_y_factor
            c2_left_x_factor = [1 - self.c2_right_x_factor[1], 1 - self.c2_right_x_factor[0]]
            c2_right_x_factor = [1 - self.c2_left_x_factor[1], 1 - self.c2_left_x_factor[0]]
        else:
            c2_left_y_factor = self.c2_left_y_factor
            c2_right_y_factor = self.c2_right_y_factor
            c2_left_x_factor = self.c2_left_x_factor
            c2_right_x_factor = self.c2_right_x_factor


        # split the points of edges to left half and right half
        for i in range(len(self.c2_coords)):
            if (self.isbetween(self.c2_coords[i][0], (self.c2_img_width / 2) * c2_left_x_factor[0], (self.c2_img_width / 2) * c2_left_x_factor[1]) and self.isbetween(self.c2_coords[i][1], self.c2_img_height * c2_left_y_factor[0], self.c2_img_height * c2_left_y_factor[1])):
                left_half_x.append(self.c2_coords[i][0])
                left_half_y.append(self.c2_coords[i][1])
            elif (self.isbetween(self.c2_coords[i][0], (self.c2_img_width / 2) * (1 + c2_right_x_factor[0]), (self.c2_img_width / 2) * (1 + c2_right_x_factor[1])) and self.isbetween(self.c2_coords[i][1], self.c2_img_height * c2_right_y_factor[0], self.c2_img_height * c2_right_y_factor[1])):
                right_half_x.append(self.c2_coords[i][0])
                right_half_y.append(self.c2_coords[i][1])

        # find the bottom left and right points
        index = np.where(left_half_y == np.max(left_half_y))[0][-1]
        bottom_left_y = left_half_y[index]
        bottom_left_x = left_half_x[index]
        index = np.where(right_half_y == np.max(right_half_y))[0][0]
        bottom_right_y = right_half_y[index]
        bottom_right_x = right_half_x[index]

        return [[bottom_left_x, bottom_left_y], [bottom_right_x, bottom_right_y]]

    def find_c7_bottom_points(self, reversed=False) -> list:

        left_half_x = []
        left_half_y = []
        right_half_x = []
        right_half_y = []
        sva_x = []
        sva_y = []

        sva_y_factor = [0.01, 0.15]
        if reversed:
            c7_left_y_factor = self.c7_right_y_factor
            c7_right_y_factor = self.c7_left_y_factor
            c7_left_x_factor = [1 - self.c7_right_x_factor[1], 1 - self.c7_right_x_factor[0]]
            c7_right_x_factor = [1 - self.c7_left_x_factor[1], 1 - self.c7_left_x_factor[0]]
            sva_x_factor = [0.1, 0.8]
        else:
            c7_left_y_factor = self.c7_left_y_factor
            c7_right_y_factor = self.c7_right_y_factor
            c7_left_x_factor = self.c7_left_x_factor
            c7_right_x_factor = self.c7_right_x_factor
            sva_x_factor = [0.2, 0.9]

        # split the points of edges to left half and right half
        for i in range(len(self.c7_coords)):
            if (self.isbetween(self.c7_coords[i][0], (self.c7_img_width / 2) * c7_left_x_factor[0], (self.c7_img_width / 2) * c7_left_x_factor[1]) and self.isbetween(self.c7_coords[i][1], self.c7_img_height * c7_left_y_factor[0], self.c7_img_height * c7_left_y_factor[1])):
                left_half_x.append(self.c7_coords[i][0])
                left_half_y.append(self.c7_coords[i][1])
            elif (self.isbetween(self.c7_coords[i][0], (self.c7_img_width / 2) * (1 + c7_right_x_factor[0]), (self.c7_img_width / 2) * (1 + c7_right_x_factor[1])) and self.isbetween(self.c7_coords[i][1], self.c7_img_height * c7_right_y_factor[0], self.c7_img_height * c7_right_y_factor[1])):
                right_half_x.append(self.c7_coords[i][0])
                right_half_y.append(self.c7_coords[i][1])
            
            if reversed:
                if (self.isbetween(self.c7_coords[i][0], (self.c7_img_width / 2) * sva_x_factor[0], (self.c7_img_width / 2) * sva_x_factor[1]) and self.isbetween(self.c7_coords[i][1], self.c7_img_height * sva_y_factor[0], self.c7_img_height * sva_y_factor[1])):
                    sva_x.append(self.c7_coords[i][0])
                    sva_y.append(self.c7_coords[i][1])
            else:
                if (self.isbetween(self.c7_coords[i][0], (self.c7_img_width / 2) * (1 + sva_x_factor[0]), (self.c7_img_width / 2) * (1 + sva_x_factor[1])) and self.isbetween(self.c7_coords[i][1], self.c7_img_height * sva_y_factor[0], self.c7_img_height * sva_y_factor[1])):
                    sva_x.append(self.c7_coords[i][0])
                    sva_y.append(self.c7_coords[i][1])

        # find the top left and right points
        index = np.where(left_half_y == np.max(left_half_y))[0][-1]
        top_left_y = left_half_y[index]
        top_left_x = left_half_x[index]
        index = np.where(right_half_y == np.max(right_half_y))[0][0]
        top_right_y = right_half_y[index]
        top_right_x = right_half_x[index]
        if reversed:
            index = np.where(sva_y == np.min(sva_y))[0][-1]
        else:
            index = np.where(sva_y == np.min(sva_y))[0][0]
        right_top_y = sva_y[index]
        right_top_x = sva_x[index]

        return [[top_left_x, top_left_y], [top_right_x, top_right_y]], [right_top_x, right_top_y]
    
    def get_c2_center_point(self, original_img_name):
        return [
            (self.original_c2_coords.loc[self.original_c2_coords["picture_name"] == original_img_name].values[0][3]) - (self.c2_img_width / 2),
            (self.original_c2_coords.loc[self.original_c2_coords["picture_name"] == original_img_name].values[0][4]) - (self.c2_img_height / 2)
        ]
    
    def SVA(self, c2_center_point, sva_point, reversed=False):
        
        if reversed:
            return c2_center_point[0] - sva_point[0]
        else:
            return sva_point[0] - c2_center_point[0]
        
    def reversed_check(self, img_name):
        c2_coord_x = self.original_c2_coords.loc[self.original_c2_coords["picture_name"] == img_name].values[0][1]
        c7_coord_x = self.original_c7_coords.loc[self.original_c7_coords["picture_name"] == img_name].values[0][1]
        return c2_coord_x > c7_coord_x
        

    """
    Function defs
    """

    def run(self):

        index = 0
        # Find edges and load images
        with alive_progress.alive_bar(len(self.common_img_names), title="Calculating...") as bar:
            for _ in self.common_img_names:
                try:
                    original_img_name = self.common_img_names[index]
                    original_img = cv2.imread(f"{self.original_vertebra_path}\\{original_img_name}")
                    c2_img = cv2.imread(self.c2_img_paths[index])
                    c7_img = cv2.imread(self.c7_img_paths[index])
                    self.c2_img_width = c2_img.shape[1]
                    self.c2_img_height = c2_img.shape[0]
                    self.c7_img_width = c7_img.shape[1]
                    self.c7_img_height = c7_img.shape[0]

                    reversed = self.reversed_check(original_img_name)
                    # original_img_name_without_extension = original_img_name.split(".")[0]
                    # original_img_name_extension = original_img_name.split(".")[1]
                    
                    # image enhancement
                    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                    enhanced_c2_img = self.img_enhance(c2_img)
                    enhanced_c7_img = self.img_enhance(c7_img)
                    # cv2.imwrite(f"{self.result_save_path}\\crops\\c2\\{original_img_name_without_extension}_c2_enh.{original_img_name_extension}", enhanced_c2_img)
                    # cv2.imwrite(f"{self.result_save_path}\\crops\\c7\\{original_img_name_without_extension}_c7_enh.{original_img_name_extension}", enhanced_c7_img)
                    
                    # find edges
                    c2_edges = self.edge_detection(enhanced_c2_img)
                    c2_indices = np.where(c2_edges != [0])
                    self.c2_coords = np.column_stack((c2_indices[1], c2_indices[0])).tolist()

                    c7_edges = self.edge_detection(enhanced_c7_img)
                    c7_indices = np.where(c7_edges != [0])
                    self.c7_coords = np.column_stack((c7_indices[1], c7_indices[0])).tolist()

                    # cv2.imwrite(f"{self.result_save_path}\\crops\\c2\\{original_img_name_without_extension}_c2_edges.{original_img_name_extension}", c2_edges)
                    # cv2.imwrite(f"{self.result_save_path}\\crops\\c7\\{original_img_name_without_extension}_c7_edges.{original_img_name_extension}", c7_edges)

                    # find c2 and c7 points(sva points)
                    c2_bottom_points = self.find_c2_bottom_points(reversed)
                    c7_bottom_points, c7_right_top_point = self.find_c7_bottom_points(reversed)
                    final_c2_coords = [self.coords_translate(c2_bottom_points[0], original_img_name, "c2"), self.coords_translate(c2_bottom_points[1], original_img_name, "c2")]
                    final_c7_coords = [self.coords_translate(c7_bottom_points[0], original_img_name, "c7"), self.coords_translate(c7_bottom_points[1], original_img_name, "c7")]
                    final_sva_point = self.coords_translate(c7_right_top_point, original_img_name, "c7")
                    
                    # calculate angle
                    vector1 = np.array(final_c2_coords[0]) - np.array(final_c2_coords[1])
                    vector2 = np.array(final_c7_coords[0]) - np.array(final_c7_coords[1])
                    angle = self.get_angle(vector1, vector2)
                    angle_degree = np.degrees(angle).round(2)

                    c2_center_point = np.array(self.get_c2_center_point(original_img_name)).astype(int)
                    sva = self.SVA(c2_center_point, final_sva_point, reversed)

                    # print(f"{original_img_name}的cobb angle為 {angle_degree} 度")
                    # print(f"{original_img_name}的SVA為 {sva} pixel")

                    # Plot
                    c2_edges = cv2.cvtColor(c2_edges, cv2.COLOR_GRAY2RGB)
                    c7_edges = cv2.cvtColor(c7_edges, cv2.COLOR_GRAY2RGB)
                    cv2.circle(c2_edges, (c2_bottom_points[0][0], c2_bottom_points[0][1]), 3, (255, 0, 0), -1)
                    cv2.circle(c2_edges, (c2_bottom_points[1][0], c2_bottom_points[1][1]), 3, (255, 0, 0), -1)
                    cv2.circle(c7_edges, (c7_bottom_points[0][0], c7_bottom_points[0][1]), 3, (255, 0, 0), -1)
                    cv2.circle(c7_edges, (c7_bottom_points[1][0], c7_bottom_points[1][1]), 3, (255, 0, 0), -1)
                    cv2.circle(c7_edges, (c7_right_top_point[0], c7_right_top_point[1]), 3, (0, 0, 255), -1)

                    # cobb angle
                    self.draw_long_line(original_img, final_c2_coords)
                    self.draw_long_line(original_img, final_c7_coords)
                    # SVA
                    if reversed:
                        cv2.line(original_img, (c2_center_point[0], c2_center_point[1]), (c2_center_point[0] - sva, c2_center_point[1]), (0, 0, 255), 2)
                        cv2.line(original_img, (c2_center_point[0] - sva, c2_center_point[1]), (final_sva_point[0], final_sva_point[1]), (0, 255, 0), 2)
                        
                    else:
                        cv2.line(original_img, (c2_center_point[0], c2_center_point[1]), (c2_center_point[0] + sva, c2_center_point[1]), (0, 0, 255), 2)
                        cv2.line(original_img, (c2_center_point[0] + sva, c2_center_point[1]), (final_sva_point[0], final_sva_point[1]), (0, 255, 0), 2)
                    cv2.circle(original_img, (c2_center_point[0], c2_center_point[1]), 3, (255, 0, 0), -1)
                    cv2.circle(original_img, (final_sva_point[0], final_sva_point[1]), 3, (255, 0, 0), -1)
                    

                    plt.figure(dpi=300)
                    plt.subplot(3, 3, 2)
                    plt.title("c2 original")
                    plt.axis("off")
                    plt.imshow(c2_img)
                    plt.subplot(3, 3, 3)
                    plt.title("c7 original")
                    plt.axis("off")
                    plt.imshow(c7_img,)
                    plt.subplot(3, 3, 5)
                    plt.title("c2 image enhanced")
                    plt.axis("off")
                    plt.imshow(enhanced_c2_img, cmap="gray")
                    plt.subplot(3, 3, 6)
                    plt.title("c7 image enhanced")
                    plt.axis("off")
                    plt.imshow(enhanced_c7_img, cmap="gray")
                    plt.subplot(3, 3, 8)
                    plt.title("c2 edges")
                    plt.axis("off")
                    plt.imshow(c2_edges)
                    plt.subplot(3, 3, 9)
                    plt.title("c7 edges")
                    plt.axis("off")
                    plt.imshow(c7_edges)
                    plt.subplot(1, 3, 1)
                    plt.title(f"cobb angle = {angle_degree} degree\nC2 C7 SVA = {sva} pixel\n{original_img_name}")
                    plt.axis("off")
                    plt.imshow(original_img)
                    if self.show_fig:
                        plt.show()
                    plt.savefig(f"{self.result_save_path}\\figs\\{original_img_name}")
                    plt.close()

                    cv2.imwrite(f"{self.result_save_path}\\processed_imgs\\{original_img_name}", original_img,)

                except Exception as e:
                    print("Error on image: ", original_img_name)
                    print(e)
                
                index += 1
                bar()