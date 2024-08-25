import cv2
import numpy as np
import json


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
