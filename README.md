# 頸椎骨節標記定位以及Cobb Angle SVA 計算
> Based on YOLOv12  (輸出和訓練圖集並沒有在此repo內)

測試時使用之環境：  
| 平台 | 版本 |
| --- | --- |
| Python| 3.12.3 |
| GPU | RTX3070 |
| CPU | AMD R5 7600 |
| RAM | DDR5 32G |

# 如何使用

1. 安裝Python（建議使用3.12.3）

2. 首先先將原始碼下載，可以直接從github上下載或是使用指令：  
```
git clone https://github.com/Euoy/YOLO12-Vertebrae-Detection
```

3. 在原始碼資料夾底下運行以下指令安裝依賴：
```
pip install -r requirements.txt
```

4. 在此repo中的Release下載訓練好的YOLO模型

5. 將path.yaml內路徑更改為自己設定的路徑

6. 執行以下指令並根據需要選擇模式即可：
```
python main.py
```