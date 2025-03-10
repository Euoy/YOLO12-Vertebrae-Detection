from ultralytics import YOLO
import cv2
import pandas
import os
from pathlib import Path
from natsort import natsorted

class Cutter():
    def __init__(self, model_path, dataset_path, predict_save_path, crop_save_path, csv_save_path):

        self.model = YOLO(model_path)
        self.dataset_path = Path(dataset_path)
        self.predict_save_path = predict_save_path
        self.crop_save_path = crop_save_path
        self.csv_save_path = csv_save_path

        self.imgs = []
        self.imgs_path = natsorted(self.dataset_path.rglob("*.png"))
        self.img_names = []
        for img_path in self.imgs_path:
            self.imgs.append(cv2.imread(img_path))
            self.img_names.append(img_path.name)

        """
        確認路徑是否存在
        """
        if not os.path.exists(predict_save_path):
            os.makedirs(predict_save_path)
        if not os.path.exists(crop_save_path):
            os.makedirs(crop_save_path)
        if not os.path.exists(csv_save_path):
            os.makedirs(csv_save_path)
        """
        確認路徑是否存在
        """

        """
        初始化變數
        """
        self.x1s, self.y1s, self.x2s, self.y2s, self.confidences, self.names = [], [], [], [], [], []
        self.columns = ["picture_name", "x1", "y1", "x2", "y2", "confidence"]

        #初始化二維陣列(因為轉csv時的陣列大小要相同)(6x500)
        for i in range(6):
            self.x1s.append([])
            self.y1s.append([])
            self.x2s.append([])
            self.y2s.append([])
            self.confidences.append([])
            for j in range(self.img_names.__len__()):
                self.x1s[i].append(None)
                self.y1s[i].append(None)
                self.x2s[i].append(None)
                self.y2s[i].append(None)
                self.confidences[i].append(None)
        """
        初始化變數
        """

    def save_crop(self, x1, y1, x2, y2, img, bone_name, crop_name):
        """
        儲存切割後的圖片。
        """
        crop_img = img[int(y1):int(y2), int(x1):int(x2)]
        save_path = f"{self.crop_save_path}\\{bone_name}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = f"{self.crop_save_path}\\{bone_name}\\{crop_name}"
        cv2.imwrite(save_path, crop_img)

    def save_croods(self,x1, y1, x2, y2, confidence, id, img_number):
        """
        儲存坐標(tensor)至二維陣列中。
        每個骨節只會儲存一個座標，保留可信度最高的。
        """
        self.x1s[id][img_number] = x1
        self.y1s[id][img_number] = y1
        self.x2s[id][img_number] = x2
        self.y2s[id][img_number] = y2
        self.confidences[id][img_number] = confidence

    def save_csv(self,id):
        """
        將骨節資料儲存為csv。
        """
        #x1s, y1d, x2s, y2s, confidences陣列大小要相同
        dataframe = pandas.DataFrame(columns=self.columns, data={"picture_name": self.img_names, "x1": self.x1s[id], "y1": self.y1s[id], "x2": self.x2s[id], "y2": self.y2s[id], "confidence": self.confidences[id]})
        name = "c" + str(id + 2) + ".csv"
        if not os.path.exists(self.csv_save_path):
            os.makedirs(self.csv_save_path)
        dataframe.to_csv(f"{self.csv_save_path}\\{name}", index=False)

    def run(self):
        #因為要取得所預測圖片的檔名進行分類，所以使用迴圈一張一張預測
        img_number = 0
        for img_name in self.img_names:
            img = self.imgs[img_number]
            result = self.model(img)[0]

            #儲存預測結果
            result.save(f"{self.predict_save_path}\\{img_name}")

            #偵測每個預測框的座標，並儲存切割後的圖片及座標
            for box in result.boxes:
                bone_id = int(box.cls.item())
                bone_name = self.model.names[bone_id]
                confidence = box.conf.item()
                #如果可信度小於0.5就跳過
                if(confidence < 0.5):
                    continue
                
                x1 = box.xyxy[0][0].item()
                y1 = box.xyxy[0][1].item()
                x2 = box.xyxy[0][2].item()
                y2 = box.xyxy[0][3].item()

                if not(self.x1s[bone_id][img_number] != None and confidence < self.confidences[bone_id][img_number]):
                    self.save_croods(x1, y1, x2, y2, confidence, bone_id, img_number)

                crop_name = img_name.split(".")[0] + "_" + bone_name + ".png"
                
                if not(os.path.exists(f"{self.crop_save_path}\\{bone_name}\\{crop_name}") and self.confidences[bone_id][img_number] > confidence):
                    #儲存切割後的圖片(因為只要保存單一個可信度最高的骨節，所以不使用內建的result.save_crop())
                    self.save_crop(x1, y1, x2, y2, img, bone_name, crop_name)
            
            img_number += 1

        for i in range(6):
            self.save_csv(i)