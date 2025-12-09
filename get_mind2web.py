from datasets import load_dataset
import json

ds = load_dataset("osunlp/Online-Mind2Web")
print(ds)
print(ds.keys())  # 看有哪些 split

# 任选一个 split，比如 "test" 或 "validation"（以实际keys为准）
split_name = list(ds.keys())[0]
print(ds[split_name].column_names)

print(ds[split_name][0])

split_name = list(ds.keys())[0]
data = ds[split_name].to_list()

with open("./data/online_mind2web.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)