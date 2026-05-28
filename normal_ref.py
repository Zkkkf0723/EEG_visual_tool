import json

# 加载JSON文件
with open(r"d:\work\json\5_stat_info_dict_ALL_AVG_0514.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 查看所有年龄段
print("年龄段:", list(data.keys()))

# 查看某个年龄段的导联列表
age_key = "0-6"
sample_keys = list(data[age_key].keys())[:10]
print("示例key:", sample_keys)

# 获取某个频段某个导联的均值和标准差
mean_val = data[age_key].get("delta_Fp1-A1_mean")
std_val = data[age_key].get("delta_Fp1-A1_std")
print(f"delta Fp1-A1: 均值={mean_val:.2f}, 标准差={std_val:.2f}")