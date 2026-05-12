import pickle
import json
import numpy as np
import os

INPUT_FILE = "whk_old.pkl"
OUTPUT_FILE = "whk_old.json"

def convert(obj):
    if obj is None or isinstance(obj, (int, float, str, bool)):
        return obj

    if isinstance(obj, (list, tuple, set)):
        return [convert(i) for i in obj]

    if isinstance(obj, dict):
        return {str(k): convert(v) for k, v in obj.items()}

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    return str(obj)

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到文件: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "rb") as f:
        data = pickle.load(f)

    data_json = convert(data)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data_json, f, ensure_ascii=False, indent=4)

    print(f"✅ 转换完成：{OUTPUT_FILE}")

if __name__ == "__main__":
    main()