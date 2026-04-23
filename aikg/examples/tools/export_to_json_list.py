import os
import json

def extract_impl_info(root_dir, output_file):
    results = []

    # 检查根目录是否存在
    if not os.path.exists(root_dir):
        print(f"错误: 找不到目录 {root_dir}")
        return

    # 遍历 root 目录下的所有子项
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        
        # 确保处理的是目录
        if os.path.isdir(subdir_path):
            json_path = os.path.join(subdir_path, "impl_info.json")
            
            # 检查该子目录下是否存在 impl_info.json
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # 提取关键词，使用 get 保证稳定性
                        profile = data.get("profile", {})
                        extracted_data = {
                            "id": data.get("id"),
                            "parent_id": data.get("parent_id"),
                            "speedup": profile.get("speedup", 0),
                            "profile_dict": profile
                        }
                        results.append(extracted_data)
                except Exception as e:
                    print(f"处理文件 {json_path} 时出错: {e}")

    # 将结果写入新的 JSON 文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"提取成功！已保存至: {output_file}")
    except Exception as e:
        print(f"写入文件时出错: {e}")

# 使用示例
if __name__ == "__main__":
    target_root = "/mnt/lustre-client/zhangzizheng/AIKG/akg/aikg/evolve_database/test_bk"  # 替换为你的实际目录路径
    output_name = "extracted_impl_data.json"
    extract_impl_info(target_root, output_name)