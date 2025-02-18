from ultralytics import YOLO
import cv2
import pandas
import os


if __name__ == "__main__":

    model = YOLO("E:\\YOLO11-Vertebrae-Detection\\best.pt")

    """
    設定變數
    """
    NUMBER_OF_DATASETS = 500
    NUMBER_OF_BONES = 6
    dataset_path = "E:\骨刺dataset"
    predict_save_path = "E:\\YOLO11-Vertebrae-Detection\\results\\predicts"
    crop_save_path = "E:\\YOLO11-Vertebrae-Detection\\results\\crops"
    csv_save_path = "E:\\YOLO11-Vertebrae-Detection\\results"
    """
    設定變數
    """

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
    x1s, y1s, x2s, y2s, confidences, names = [], [], [], [], [], []
    columns = ["picture_name", "x1", "y1", "x2", "y2", "confidence"]

    #初始化二維陣列(因為轉csv時的陣列大小要相同)(6x500)
    for i in range(NUMBER_OF_BONES):
        x1s.append([])
        y1s.append([])
        x2s.append([])
        y2s.append([])
        confidences.append([])
        for j in range(NUMBER_OF_DATASETS):
            x1s[i].append(None)
            y1s[i].append(None)
            x2s[i].append(None)
            y2s[i].append(None)
            confidences[i].append(None)
    """
    初始化變數
    """

    def save_crop(x1, y1, x2, y2, img, bone_name, crop_name):
        """
        儲存切割後的圖片。
        """
        crop_img = img[int(y1):int(y2), int(x1):int(x2)]
        save_path = f"{crop_save_path}\\{bone_name}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = f"{crop_save_path}\\{bone_name}\\{crop_name}"
        cv2.imwrite(save_path, crop_img)
    
    def save_croods(x1, y1, x2, y2, confidence, id, img_number):
        """
        儲存坐標(tensor)至二維陣列中。
        每個骨節只會儲存一個座標，保留可信度最高的。
        """
        x1s[id][img_number] = x1
        y1s[id][img_number] = y1
        x2s[id][img_number] = x2
        y2s[id][img_number] = y2
        confidences[id][img_number] = confidence

    def save_csv(id):
        """
        將骨節資料儲存為csv。
        """
        #x1s, y1d, x2s, y2s, confidences陣列大小要相同
        dataframe = pandas.DataFrame(columns=columns, data={"picture_name": names, "x1": x1s[id], "y1": y1s[id], "x2": x2s[id], "y2": y2s[id], "confidence": confidences[id]})
        name = "c" + str(i + 2) + ".csv"
        if not os.path.exists(csv_save_path):
            os.makedirs(csv_save_path)
        dataframe.to_csv(f"{csv_save_path}\\{name}", index=False)
        
    #因為要取得所預測圖片的檔名進行分類，所以使用迴圈一張一張預測
    for img_number in range(NUMBER_OF_DATASETS):
        img_name = str(img_number + 1) + ".png"
        img = cv2.imread(f"{dataset_path}\\{img_name}")
        result = model(img)[0]
        names.append(img_name)

        #儲存預測結果
        result.save(f"{predict_save_path}\\{img_name}")

        #偵測每個預測框的座標，並儲存切割後的圖片及座標
        for box in result.boxes:
            bone_id = int(box.cls.item())
            bone_name = model.names[bone_id]
            confidence = box.conf.item()
            #如果可信度小於0.5就跳過
            if(confidence < 0.5):
                continue
            
            x1 = box.xyxy[0][0].item()
            y1 = box.xyxy[0][1].item()
            x2 = box.xyxy[0][2].item()
            y2 = box.xyxy[0][3].item()

            if not(x1s[bone_id][img_number] != None and confidence < confidences[bone_id][img_number]):
                save_croods(x1, y1, x2, y2, confidence, bone_id, img_number)

            crop_name = str(img_number + 1) + "_" + bone_name + ".png"
            
            if not(os.path.exists(f"{crop_save_path}\\{bone_name}\\{crop_name}") and confidences[bone_id][img_number] > confidence):
                #儲存切割後的圖片(因為只要保存單一個可信度最高的骨節，所以不使用內建的result.save_crop())
                save_crop(x1, y1, x2, y2, img, bone_name, crop_name)

    for i in range(NUMBER_OF_BONES):
        save_csv(i)

    print("done!")

    