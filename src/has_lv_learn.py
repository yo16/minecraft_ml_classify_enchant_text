import os
import numpy as np
import cv2
import json
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



# 画像ファイルの前処理
# 画像ファイルは、BGRで、背景は黒(0,0,0)、文字色は白(255,255,255)の２色
# 高さは15pxで固定、幅はいろいろあるものを、高さ15px、幅200pxとする
# 背景を0、文字部分を1とした、flattenな１次元配列とする
# 200x15=3,000要素の0/1の配列になる
def pre_process_image(img_path: str) -> np.ndarray:
    # グレイスケールで読む
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 横幅を200まで拡張（背景は黒）
    height, width = img.shape
    img_new = np.zeros((height, 200), dtype=np.uint8)
    img_new[:height, :width] = img
    cv2.imwrite("./tmp/img_new.png", img_new)
    
    return img_new.flatten()


DATA_MAP_FILE = "./data/data_map.json"
# ファイル名をjsonファイルから探してレベルを返す
# 毎回jsonファイルを開くけど、、、まぁ多くないからいいや
def get_level(filename: str) -> int:
    with open(DATA_MAP_FILE, mode="r", encoding="utf-8") as f:
        data_map = json.load(f)
    for file_info in data_map["data"]:
        if file_info["file_name"] == filename:
            # みつけた
            return file_info["level"]
    return -1


# 学習ファイルを読む
def load_data(target_dir):
    images = []
    labels = []
    for filename in os.listdir(target_dir):
        #print(filename)
        if filename.lower().endswith(".png"):
            img_path = os.path.join(target_dir, filename)
            # 読みながら前処理
            img = pre_process_image(img_path)
            
            # このファイルのlabelを取得
            lv = get_level(filename)
            assert lv >= 0, f"{filename}が見つからないみたいです"

            images.append(img)
            labels.append(0 if lv==0 else 1)

    return np.array(images), np.array(labels)



# メイン
def main():
    # データとラベルを読む
    train_images, train_labels = load_data("./data/has_lv/train")
    test_images, test_labels = load_data("./data/has_lv/test")

    # SVM(SVC)で学習
    model = SVC(kernel='linear')
    model.fit(train_images, train_labels)

    # 保存
    with open('./models/has_lv.pkl', 'wb') as file:
        pickle.dump(model, file)

    # 予測と評価
    #predictions = model.predict(test_images)
    #accuracy = accuracy_score(test_labels, predictions)
    #print(f'テスト精度: {accuracy}')
    # １つずつ確認
    for i, label in enumerate(test_labels):
        ret = model.predict([test_images[i]])
        print(f"{label} - {ret[0]}")



if __name__=="__main__":
    #file_path = "./data/has_lv/test/knock_back1.png"
    #ret = pre_process_image(file_path)
    #print(ret)
    #print(sum(ret))
    #print(ret.shape)

    #ret1, ret2 = load_data("./data/has_lv/test")
    #print(ret1)
    #print(ret1.shape)
    #print(ret2)
    #print(ret2.shape)

    main()

