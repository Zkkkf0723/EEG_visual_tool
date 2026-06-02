import os


def generate_tree(path, indent=""):
    if not os.path.isdir(path):
        return

    files = os.listdir(path)
    for i, file in enumerate(files):
        # 排除隐藏文件
        if file.startswith('.'):
            continue

        is_last = (i == len(files) - 1)
        current_indent = "└── " if is_last else "├── "

        print(indent + current_indent + file)

        # 如果是文件夹，递归打印
        sub_path = os.path.join(path, file)
        if os.path.isdir(sub_path):
            next_indent = indent + ("    " if is_last else "│   ")
            generate_tree(sub_path, next_indent)


# 填入你想查看的路径
generate_tree(r"G:\project_\Normal_Reference")