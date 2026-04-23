import os
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def load_impls(root_dir: str, level_dirs: List[str]):
    """
    扫描给定 root_dir/levelX 下所有 impl_info.json，返回：
    - impl_by_id: {impl_id: impl_info_dict}
    - children_map: {impl_id: [child_impl_id, ...]}
    - op_by_impl: {impl_id: op_name}
    """
    impl_by_id: Dict[str, dict] = {}
    children_map: Dict[str, List[str]] = defaultdict(list)
    op_by_impl: Dict[str, str] = {}

    for level in level_dirs:
        level_path = os.path.join(root_dir, level)
        if not os.path.isdir(level_path):
            print(f"[WARN] level dir not found: {level_path}")
            continue

        # level 下每个算子目录
        for op_dir_name in os.listdir(level_path):
            op_dir = os.path.join(level_path, op_dir_name)
            if not os.path.isdir(op_dir):
                continue

            # op_dir 下每个实现目录
            for impl_dir_name in os.listdir(op_dir):
                impl_dir = os.path.join(op_dir, impl_dir_name)
                if not os.path.isdir(impl_dir):
                    continue

                impl_info_path = os.path.join(impl_dir, "impl_info.json")
                if not os.path.isfile(impl_info_path):
                    continue

                try:
                    with open(impl_info_path, "r", encoding="utf-8") as f:
                        info = json.load(f)
                except Exception as e:
                    print(f"[WARN] fail to load json: {impl_info_path}, err={e}")
                    continue

                impl_id = info.get("id")
                parent_id = info.get("parent_id")
                op_name = info.get("op_name") or info.get("task_info", {}).get("op_name", "UnknownOp")

                if impl_id is None:
                    print(f"[WARN] impl_info.json missing id: {impl_info_path}")
                    continue

                impl_by_id[impl_id] = info
                op_by_impl[impl_id] = op_name

                if parent_id:
                    children_map[parent_id].append(impl_id)

    return impl_by_id, children_map, op_by_impl


def find_leaf_impls(impl_by_id: Dict[str, dict], children_map: Dict[str, List[str]]) -> List[str]:
    """
    找到所有“没有孩子的实现 id”（叶子）。
    """
    leaf_ids = []
    for impl_id in impl_by_id.keys():
        if impl_id not in children_map or len(children_map[impl_id]) == 0:
            leaf_ids.append(impl_id)
    return leaf_ids


def backtrack_lineage(
    leaf_id: str,
    impl_by_id: Dict[str, dict]
) -> List[str]:
    """
    从 leaf_id 一直沿 parent_id 往上回溯到根，返回从根 -> ... -> leaf 的 id 列表。
    """
    chain = []
    current_id: Optional[str] = leaf_id

    while current_id is not None:
        info = impl_by_id.get(current_id)
        if info is None:
            # 说明 parent 在别处（不在当前 level 或未扫描到），链条到此为止
            break
        chain.append(current_id)
        parent_id = info.get("parent_id")
        if not parent_id or parent_id == current_id:
            break
        current_id = parent_id

    # 当前顺序是 leaf -> root，需要翻转
    chain.reverse()
    return chain


def extract_speedup(info: dict) -> Optional[float]:
    """
    从 impl_info.json 里抽取 profile 信息中的加速比。
    根据你真实的 json 结构修改这里的字段路径。
    假设：info["profile"]["speedup"]
    """
    profile = info.get("profile")
    if profile is None:
        return None

    # 根据你的实际字段名做修改：
    # 例如可能是 "speedup", "accel_ratio", "throughput_speedup" 等
    for key in ["speedup"]:
        if key in profile:
            try:
                return float(profile[key])
            except Exception:
                pass

    return None


def plot_lineages(
    impl_by_id: Dict[str, dict],
    op_by_impl: Dict[str, str],
    leaf_ids: List[str],
    out_dir: str
):
    """
    对每个叶子构建一条 lineage（从 root 到 leaf），提取每一代的加速比，画折线图。
    每条 lineage 生成一张 png，文件名格式：{op_name}_{leaf_id[:8]}.png
    """
    os.makedirs(out_dir, exist_ok=True)

    for leaf_id in leaf_ids:
        chain = backtrack_lineage(leaf_id, impl_by_id)
        if len(chain) <= 1:
            # 只有自己，没有父代，不画图也行，看需求
            continue

        speedups = []
        generations = list(range(len(chain)))
        for impl_id in chain:
            info = impl_by_id[impl_id]
            s = extract_speedup(info)
            speedups.append(s)

        # 如果一条链完全没有 speedup 信息，跳过
        if all(v is None for v in speedups):
            continue

        # 简单处理：没有 speedup 的点用 None，Matplotlib 会断线；
        # 也可以选择插值 / 跳过等
        op_name = op_by_impl.get(leaf_id, "UnknownOp")

        plt.figure(figsize=(6, 4), dpi=150)
        plt.plot(
            generations,
            speedups,
            marker="o",
            linestyle="-",
            label=f"leaf: {leaf_id[:8]}"
        )
        plt.xlabel("Generation index (0 = earliest ancestor)")
        plt.ylabel("Speedup")
        plt.title(f"Evolution of speedup for op: {op_name}")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()

        # 文件名中去掉不合法字符
        safe_op_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in op_name)
        out_path = os.path.join(out_dir, f"{safe_op_name}_{leaf_id[:8]}.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

        print(f"[INFO] saved plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Draw evolution speedup line charts.")
    parser.add_argument(
        "--root",
        type=str,
        default="/mnt/lustre-client/zhangzizheng/AIKG/akg/aikg/evolve_database",
        help="根目录（包含 level1/level2 等）"
    )
    parser.add_argument(
        "--levels",
        type=str,
        nargs="+",
        default=["level1", "level2"],
        help="需要扫描的 level 目录名列表"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="evolve_plots",
        help="输出图片目录"
    )
    args = parser.parse_args()

    impl_by_id, children_map, op_by_impl = load_impls(args.root, args.levels)
    print(f"[INFO] loaded {len(impl_by_id)} impls.")

    leaf_ids = find_leaf_impls(impl_by_id, children_map)
    print(f"[INFO] found {len(leaf_ids)} leaf impls.")

    plot_lineages(impl_by_id, op_by_impl, leaf_ids, args.out)


if __name__ == "__main__":
    main()