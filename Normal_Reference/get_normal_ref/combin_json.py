#合并json的工具

import os
import json






def verify_merging(folder_path, merged_file_path, merge_mode="key"):
    """
    merge_mode:
      "key" -> 按文件名作为键合并
      "list" -> 列表拼接合并
    """
    # 1. 加载合并后的文件
    with open(merged_file_path, 'r', encoding='utf-8') as f:
        merged_data = json.load(f)

    # 2. 统计原始文件的数据
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    original_file_count = len(json_files)

    print("===== 开始校验 =====")
    print(f"原文件夹内 JSON 文件总数: {original_file_count}")

    if merge_mode == "key":
        merged_key_count = len(merged_data.keys())
        print(f"合并后 JSON 的 Key 总数: {merged_key_count}")

        if original_file_count == merged_key_count:
            print("✅ 校验通过：文件数量与 Key 数量完全一致，无丢失！")
        else:
            print("❌ 校验失败：数量不匹配，可能存在同名文件覆盖或读取遗漏。")

    elif merge_mode == "list":
        total_original_items = 0
        for filename in json_files:
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                sub_data = json.load(f)
                total_original_items += len(sub_data)

        merged_item_count = len(merged_data)
        print(f"原始所有列表元素总和: {total_original_items}")
        print(f"合并后列表元素总总数: {merged_item_count}")

        if total_original_items == merged_item_count:
            print("✅ 校验通过：元素总数完全一致！")
        else:
            print("❌ 校验失败：元素数量不一致。")






def merge_json_files(folder_path, output_filename):
    merged_data = {}  # 如果你的JSON最外层是列表，这里改用 merged_data = []

    # 遍历文件夹下的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)

                    # 核心逻辑：根据你的文件名和数据结构进行合并
                    # 示例 A：如果最外层是字典，且你想把“文件名”作为 Key
                    file_key = os.path.splitext(filename)[0]  # 去掉 .json 后缀
                    merged_data[file_key] = data

                    # 示例 B：如果最外层是字典，想直接把属性合并（相同key会被覆盖）
                    # merged_data.update(data)

                    # 示例 C：如果最外层是列表，想把数据拼接进去
                    # merged_data.extend(data)

                except json.JSONDecodeError:
                    print(file_path)

    # 保存合并后的文件
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

    print(f"合并完成！结果已保存至: {output_filename}")


# --- 使用示例 ---
# 替换为你的 JSON 文件夹路径
my_folder = "./data_json"
output_file = "./combined_result_0611.json"

# merge_json_files(my_folder, output_file)


#
# verify_merging(my_folder,output_file,merge_mode="key")

with open(output_file, 'r', encoding='utf-8') as f:
    merged_data = json.load(f)

    print(merged_data["age_0_6_10_normal_ref_0526"]['alpha_info'])

    output_filename = "f_combined_result.json"
    out_dict = {}
    for k,v in merged_data.items():
        if "_group" in k:
            k = k.replace("_group","")

        out_dict[k] = v


    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(out_dict, f, ensure_ascii=False, indent=4)

    print(out_dict["age_0_6_15_normal_ref_0526"].keys())



