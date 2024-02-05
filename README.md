# Realtime Sign Language Recognizer Based on Deep Learning

## 一、簡介
### 動機
世界衛生組織預計 2025 年有近 25 億人有一定程度的聽力損失，且至少有 7 億人需要聽力康復。由於文字主要依靠聽覺來理解並記憶，文字溝通對聽障人士而言十分費勁。因此我們想讓聽障人士可以直接使用手語與一般聽人溝通，拉近彼此之間的距離。

### 目標
使用深度學習實作即時手語辨識系統，以臺灣使用的手語作為資料集。並結合文字轉語音與語音轉文字功能，使聽障人士與一般聽人皆能使用自己最熟悉的母語與彼此溝通。

## 二、技術方法
### 手語辨識 
以深度學習實作，自行錄製手語動作作為資料集。使用MediaPipe偵測骨架座標，一個手語動作做一秒，並以30幀的速度紀錄動作變化，然後以30個frame的骨架座標作為一筆輸入資料，每個手語動作錄製180筆資料。最後以LSTM和GRU作為基礎模型架構來訓練，並比較不同模型的效能。

## 語音辨識 / 合成 
語音辨識部分將即時錄製到的語音降噪後分段，再將分段的語音傳送到 Google API取得語音轉文字的結果。語音合成同樣使用Google API取得文字轉 語音的結果。 

![image](https://github.com/ty0601/signLanguage/assets/71759327/fb497d67-3850-40b7-8917-f9a3bbfc60aa)

## 三、系統架構
![image](https://github.com/ty0601/signLanguage/assets/71759327/c04ecee4-c111-46b8-9f7e-28e0e8f96494)

## 四、成果展示
### 手語辨識 
![image](https://github.com/ty0601/signLanguage/assets/71759327/42bf28aa-84b5-4b73-8e6f-60ece5305471)

### 語音辨識
![image](https://github.com/ty0601/signLanguage/assets/71759327/2d1719dc-d4e2-4ca3-87ed-c8df459766ab)
