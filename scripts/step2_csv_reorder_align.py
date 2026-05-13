"""
Step2: CSV汇总、重命名、样本平衡（升级版：兼容Step1新目录结构 + 方案B平衡）
从多个 Step1 输出目录汇总到统一仓库 aligned_data
运行: python scripts/step2_csv_reorder_align.py
参数修改: 修改本文件开头的 INPUT_ROOTS, OUTPUT_ROOT 等配置
"""

import sys
import os
import shutil
from collections import defaultdict

# 动态获取项目根目录并添加到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# ======================================
# 一、配置入口：你主要改这里
# ======================================

# 1) 多个输入根目录（兼容多种写法）
# 允许填：
#   - 包含 processed_main / processed_sub 的根目录
#   - 或直接填 processed_main / processed_sub
#   - 或直接填 processed_sub/class_15（也能识别）
INPUT_ROOTS = [
    r" ",  
]

# 2) 统一汇总后的输出根目录（新的总仓库）
OUTPUT_ROOT = r" "

# 3) 是否只复制而不删除原文件
COPY_INSTEAD_OF_MOVE = True  # True=复制；False=移动

# 4) 是否对主片段做全局样本平衡（跨 liquid + 跨 Class）
BALANCE_MAIN_SAMPLES = True

# 5) 是否对子片段做样本平衡（方案B：组内Hi/Lo对齐 -> 跨组全局平衡）
BALANCE_SUB_SAMPLES = True

# 6) 统一输出目录结构名
PROCESSED_MAN_NAME = "processed_main"
PROCESSED_SUB_NAME = "processed_sub"
SUB_HI_NAME = "Hi"   # 输出时统一为 Hi
SUB_LO_NAME = "Lo"   # 输出时统一为 Lo

# 7) 为避免误删，多余文件移动到 extra 子目录
EXTRA_DIR_NAME = "extra"

# 8) Step1 目录关键字（输入侧）
STEP1_CLASS_PREFIX = "class_"   # 输入侧是 class_15
STEP2_CLASS_PREFIX = "Class"    # 输出侧统一为 Class15

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ======================================
# 二、工具函数
# ======================================

def safe_copy_or_move(src, dst, copy_mode=True):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if copy_mode:
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)

def is_csv_file(p):
    return os.path.isfile(p) and p.lower().endswith(".csv")

def scan_csv_under(dir_path):
    csv_files = []
    if not dir_path or not os.path.isdir(dir_path):
        return csv_files
    for cur, _, files in os.walk(dir_path):
        for f in files:
            if f.lower().endswith(".csv"):
                csv_files.append(os.path.join(cur, f))
    csv_files.sort()
    return csv_files

def normalize_class_label(x: str):
    """
    输入可能是：
      - class_15 / class_5
      - Class15 / CLASS_10
      - class15 / class10
    输出统一为：Class15 / Class10 ...
    """
    if not x:
        return "ClassUnknown"
    s = str(x).strip()
    sl = s.lower()

    if sl.startswith("class_"):
        num = s.split("_", 1)[1]
        return f"{STEP2_CLASS_PREFIX}{num}" if num.isdigit() else "ClassUnknown"

    if sl.startswith("class") and len(sl) > len("class"):
        num = s[len("class"):]
        return f"{STEP2_CLASS_PREFIX}{num}" if num.isdigit() else "ClassUnknown"

    return "ClassUnknown"

def parse_liquid_and_class_from_filename(filename):
    """
    从文件名解析 liquid 与 class（作为兜底）
    兼容:
      yogurt_01_class_15_001.csv
      yogurt_Class15_0001.csv
      yogurt_class_15_H0_001.csv
    """
    name_no_ext = os.path.splitext(os.path.basename(filename))[0]
    parts = name_no_ext.split("_")
    liquid = parts[0] if parts else "Unknown"

    class_name = None
    for i, p in enumerate(parts):
        pl = p.lower()
        if pl == "class" and i + 1 < len(parts) and parts[i + 1].isdigit():
            class_name = f"{STEP2_CLASS_PREFIX}{parts[i + 1]}"
            break
        if pl.startswith("class_"):
            class_name = normalize_class_label(p)
            break
        if pl.startswith("class") and len(pl) > len("class"):
            class_name = normalize_class_label(p)
            break

    if class_name is None:
        class_name = "ClassUnknown"
    return liquid, class_name

def parse_from_step1_path(csv_path: str):
    """
    从 Step1 新目录结构推断：
      processed_main/class_15/liquid/xxx.csv
      processed_sub/class_15/liquid/high/xxx.csv
      processed_sub/class_15/liquid/low/xxx.csv
    返回：
      liquid, class_label(Class15), sub_state(Hi/Lo/None)
    """
    parts = os.path.normpath(csv_path).split(os.sep)

    liquid, class_from_name = parse_liquid_and_class_from_filename(os.path.basename(csv_path))
    class_label = class_from_name
    sub_state = None

    # 从路径里找 processed_main / processed_sub
    if PROCESSED_MAN_NAME in parts:
        idx = parts.index(PROCESSED_MAN_NAME)
        # 期望：.../processed_main/class_15/liquid/file.csv
        if idx + 2 < len(parts):
            class_label = normalize_class_label(parts[idx + 1])
            liquid = parts[idx + 2]
        return liquid, class_label, None

    if PROCESSED_SUB_NAME in parts:
        idx = parts.index(PROCESSED_SUB_NAME)
        # 期望：.../processed_sub/class_15/liquid/high/file.csv
        if idx + 3 < len(parts):
            class_label = normalize_class_label(parts[idx + 1])
            liquid = parts[idx + 2]
            hl = parts[idx + 3].lower()
            if hl == "high":
                sub_state = SUB_HI_NAME
            elif hl == "low":
                sub_state = SUB_LO_NAME
        return liquid, class_label, sub_state

    return liquid, class_label, sub_state

def locate_step1_dirs(any_path: str):
    """
    输入 any_path 可能是：
      - 根目录（包含 processed_main / processed_sub）
      - processed_main
      - processed_sub
      - processed_sub/class_15
    返回：
      main_dir, sub_dir
    """
    p = os.path.normpath(any_path)

    if not os.path.exists(p):
        return None, None

    # 情况1：直接是 processed_main
    if os.path.basename(p) == PROCESSED_MAN_NAME:
        parent = os.path.dirname(p)
        sub_dir = os.path.join(parent, PROCESSED_SUB_NAME)
        return p, sub_dir if os.path.isdir(sub_dir) else None

    # 情况2：直接是 processed_sub
    if os.path.basename(p) == PROCESSED_SUB_NAME:
        parent = os.path.dirname(p)
        main_dir = os.path.join(parent, PROCESSED_MAN_NAME)
        return main_dir if os.path.isdir(main_dir) else None, p

    # 情况3：是 processed_sub/class_15 这类
    if os.path.basename(os.path.dirname(p)) == PROCESSED_SUB_NAME and os.path.basename(p).lower().startswith("class"):
        # 回溯到 parent
        parent = os.path.dirname(os.path.dirname(p))
        main_dir = os.path.join(parent, PROCESSED_MAN_NAME)
        sub_dir = os.path.join(parent, PROCESSED_SUB_NAME)
        return main_dir if os.path.isdir(main_dir) else None, sub_dir if os.path.isdir(sub_dir) else None

    # 情况4：一般根目录
    main_dir = os.path.join(p, PROCESSED_MAN_NAME)
    sub_dir = os.path.join(p, PROCESSED_SUB_NAME)
    return (main_dir if os.path.isdir(main_dir) else None,
            sub_dir if os.path.isdir(sub_dir) else None)

def list_step1_groups_main(main_root):
    """
    Step1 main结构：
      processed_main/class_15/liquid/*.csv
    返回 group 列表：[(class_label, liquid, dir_path), ...]
    """
    groups = []
    if not main_root or not os.path.isdir(main_root):
        return groups
    for cls in sorted(os.listdir(main_root)):
        cls_path = os.path.join(main_root, cls)
        if not os.path.isdir(cls_path) or not cls.lower().startswith("class"):
            continue
        class_label = normalize_class_label(cls)
        for liquid in sorted(os.listdir(cls_path)):
            liq_path = os.path.join(cls_path, liquid)
            if os.path.isdir(liq_path):
                groups.append((class_label, liquid, liq_path))
    return groups

def list_step1_groups_sub(sub_root):
    """
    Step1 sub结构：
      processed_sub/class_15/liquid/high/*.csv
      processed_sub/class_15/liquid/low/*.csv
    返回 group 列表：[(class_label, liquid, hi_dir, lo_dir), ...]
    """
    groups = []
    if not sub_root or not os.path.isdir(sub_root):
        return groups
    for cls in sorted(os.listdir(sub_root)):
        cls_path = os.path.join(sub_root, cls)
        if not os.path.isdir(cls_path) or not cls.lower().startswith("class"):
            continue
        class_label = normalize_class_label(cls)
        for liquid in sorted(os.listdir(cls_path)):
            liq_path = os.path.join(cls_path, liquid)
            if not os.path.isdir(liq_path):
                continue
            hi_dir = os.path.join(liq_path, "high")
            lo_dir = os.path.join(liq_path, "low")
            # 允许缺其中之一
            groups.append((class_label, liquid, hi_dir if os.path.isdir(hi_dir) else None, lo_dir if os.path.isdir(lo_dir) else None))
    return groups

def balance_dir_to_target(dir_path, target, extra_dir_name="extra"):
    """
    把 dir_path 下的 csv 保留前 target 个，多余移动到 extra/
    """
    if not os.path.isdir(dir_path):
        return 0, 0
    files = sorted([f for f in os.listdir(dir_path) if f.lower().endswith(".csv") and os.path.isfile(os.path.join(dir_path, f))])
    n = len(files)
    if n <= target:
        return n, 0
    extra_dir = os.path.join(dir_path, extra_dir_name)
    os.makedirs(extra_dir, exist_ok=True)
    to_move = files[target:]
    for f in to_move:
        shutil.move(os.path.join(dir_path, f), os.path.join(extra_dir, f))
    return target, len(to_move)

def global_balance_groups(group_dir_list, extra_dir_name="extra"):
    """
    group_dir_list: [dir1, dir2, ...]
    跨组全局最小平衡
    """
    counts = {}
    for d in group_dir_list:
        if not os.path.isdir(d):
            continue
        c = len([f for f in os.listdir(d) if f.lower().endswith(".csv") and os.path.isfile(os.path.join(d, f))])
        counts[d] = c
    if not counts:
        print("  ⚠ 未找到任何可平衡的组目录，跳过。")
        return

    global_min = min(counts.values())
    print(f"  组样本数统计: {[counts[d] for d in counts]}")
    print(f"  全局目标 = {global_min}")

    for d, c in counts.items():
        kept, moved = balance_dir_to_target(d, global_min, extra_dir_name)
        if moved > 0:
            print(f"  裁剪: {d} | {c} -> {kept} | extra {moved}")

# ======================================
# 三、第一步：定位输入目录并汇总
# ======================================

found_info = []
for root in INPUT_ROOTS:
    main_dir, sub_dir = locate_step1_dirs(root)
    found_info.append((root, main_dir, sub_dir))

print("=== 扫描到的输入目录结构（按Step1新结构定位） ===")
for root, m, s in found_info:
    print(f"输入: {root}")
    print(f"  main: {m}")
    print(f"  sub : {s}")
print("=================================================\n")

# 输出根目录结构（对齐你的新阶段命名）
output_man_root = os.path.join(OUTPUT_ROOT, PROCESSED_MAN_NAME)
output_sub_root = os.path.join(OUTPUT_ROOT, PROCESSED_SUB_NAME)
os.makedirs(output_man_root, exist_ok=True)
os.makedirs(output_sub_root, exist_ok=True)

# 计数器：按 (Class, liquid, kind) 分组编号
counter_main = defaultdict(int)
counter_hi = defaultdict(int)
counter_lo = defaultdict(int)

# ---- 汇总主片段（Step1 main）----
for _, main_dir, _ in found_info:
    if not main_dir or not os.path.isdir(main_dir):
        continue

    groups = list_step1_groups_main(main_dir)
    if not groups:
        print(f"⚠ main 下没有找到 Step1 结构组: {main_dir}")
        continue

    for class_label, liquid, liq_dir in groups:
        files = scan_csv_under(liq_dir)
        if not files:
            continue

        print(f"▶ 汇总主片段: {liq_dir} | -> {class_label}/{liquid} | {len(files)} files")

        out_group_dir = os.path.join(output_man_root, class_label, liquid)
        os.makedirs(out_group_dir, exist_ok=True)

        for src in files:
            key = (class_label, liquid)
            counter_main[key] += 1
            idx = counter_main[key]
            new_name = f"{liquid}_{class_label}_{idx:04d}.csv"
            dst = os.path.join(out_group_dir, new_name)
            safe_copy_or_move(src, dst, COPY_INSTEAD_OF_MOVE)

# ---- 汇总子片段（Step1 sub）----
for _, _, sub_dir in found_info:
    if not sub_dir or not os.path.isdir(sub_dir):
        continue

    groups = list_step1_groups_sub(sub_dir)
    if not groups:
        print(f"⚠ sub 下没有找到 Step1 结构组: {sub_dir}")
        continue

    for class_label, liquid, hi_dir, lo_dir in groups:
        # Hi
        if hi_dir and os.path.isdir(hi_dir):
            files = scan_csv_under(hi_dir)
            if files:
                print(f"▶ 汇总子片段Hi: {hi_dir} | -> {class_label}/{liquid}/Hi | {len(files)} files")
                out_dir = os.path.join(output_sub_root, class_label, liquid, SUB_HI_NAME)
                os.makedirs(out_dir, exist_ok=True)
                for src in files:
                    key = (class_label, liquid)
                    counter_hi[key] += 1
                    idx = counter_hi[key]
                    new_name = f"{liquid}_{class_label}_{SUB_HI_NAME}_{idx:04d}.csv"
                    dst = os.path.join(out_dir, new_name)
                    safe_copy_or_move(src, dst, COPY_INSTEAD_OF_MOVE)

        # Lo
        if lo_dir and os.path.isdir(lo_dir):
            files = scan_csv_under(lo_dir)
            if files:
                print(f"▶ 汇总子片段Lo: {lo_dir} | -> {class_label}/{liquid}/Lo | {len(files)} files")
                out_dir = os.path.join(output_sub_root, class_label, liquid, SUB_LO_NAME)
                os.makedirs(out_dir, exist_ok=True)
                for src in files:
                    key = (class_label, liquid)
                    counter_lo[key] += 1
                    idx = counter_lo[key]
                    new_name = f"{liquid}_{class_label}_{SUB_LO_NAME}_{idx:04d}.csv"
                    dst = os.path.join(out_dir, new_name)
                    safe_copy_or_move(src, dst, COPY_INSTEAD_OF_MOVE)

print("\n🎯 第一步：汇总 + 重命名 完成！")
print(f"主片段输出根目录: {output_man_root}")
print(f"子片段输出根目录: {output_sub_root}")

# ======================================
# 四、第二步：主片段全局样本数平衡（跨液体 + 跨 Class）
# ======================================

if BALANCE_MAIN_SAMPLES:
    print("\n=== 主片段全局样本数平衡（跨液体 + 跨 Class）===")

    group_dirs = []
    for cls in sorted(os.listdir(output_man_root)):
        cls_dir = os.path.join(output_man_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for liquid in sorted(os.listdir(cls_dir)):
            g = os.path.join(cls_dir, liquid)
            if os.path.isdir(g):
                group_dirs.append(g)

    if not group_dirs:
        print("  ⚠ 未找到任何主片段组目录，跳过。")
    else:
        global_balance_groups(group_dirs, EXTRA_DIR_NAME)
        print("✅ 主片段全局平衡完成。")

# ======================================
# 五、第三步：子片段样本平衡（方案B）
# 方案B：组内先对齐Hi/Lo -> 再跨组做全局最小平衡
# ======================================

if BALANCE_SUB_SAMPLES:
    print("\n=== 子片段样本平衡（方案B：组内Hi/Lo对齐 -> 跨组全局平衡）===")

    # 1) 收集所有 (Class, liquid) 的 Hi/Lo 目录
    pairs = []
    for cls in sorted(os.listdir(output_sub_root)):
        cls_dir = os.path.join(output_sub_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for liquid in sorted(os.listdir(cls_dir)):
            liq_dir = os.path.join(cls_dir, liquid)
            if not os.path.isdir(liq_dir):
                continue
            hi_dir = os.path.join(liq_dir, SUB_HI_NAME)
            lo_dir = os.path.join(liq_dir, SUB_LO_NAME)
            if os.path.isdir(hi_dir) or os.path.isdir(lo_dir):
                pairs.append((cls, liquid, hi_dir, lo_dir))

    if not pairs:
        print("  ⚠ 未找到任何子片段组目录，跳过。")
    else:
        # 2) 组内对齐：每组 Hi/Lo -> min(Hi,Lo)
        print("  [1/2] 组内对齐 Hi/Lo ...")
        for cls, liquid, hi_dir, lo_dir in pairs:
            if not os.path.isdir(hi_dir) or not os.path.isdir(lo_dir):
                # 缺一侧时不做对齐（保持现状）
                continue
            hi_files = sorted([f for f in os.listdir(hi_dir) if f.lower().endswith(".csv") and os.path.isfile(os.path.join(hi_dir, f))])
            lo_files = sorted([f for f in os.listdir(lo_dir) if f.lower().endswith(".csv") and os.path.isfile(os.path.join(lo_dir, f))])
            target = min(len(hi_files), len(lo_files))
            if target == 0:
                continue

            if len(hi_files) > target:
                extra = os.path.join(hi_dir, EXTRA_DIR_NAME)
                os.makedirs(extra, exist_ok=True)
                for f in hi_files[target:]:
                    shutil.move(os.path.join(hi_dir, f), os.path.join(extra, f))

            if len(lo_files) > target:
                extra = os.path.join(lo_dir, EXTRA_DIR_NAME)
                os.makedirs(extra, exist_ok=True)
                for f in lo_files[target:]:
                    shutil.move(os.path.join(lo_dir, f), os.path.join(extra, f))

        # 3) 跨组全局平衡：先统计每组 Hi（或 Lo）数量，再裁到 global_min
        print("  [2/2] 跨组全局最小平衡 ...")
        hi_group_dirs = []
        lo_group_dirs = []
        for cls, liquid, hi_dir, lo_dir in pairs:
            if os.path.isdir(hi_dir):
                hi_group_dirs.append(hi_dir)
            if os.path.isdir(lo_dir):
                lo_group_dirs.append(lo_dir)

        print("  - 平衡 Hi 组 ...")
        global_balance_groups(hi_group_dirs, EXTRA_DIR_NAME)

        print("  - 平衡 Lo 组 ...")
        global_balance_groups(lo_group_dirs, EXTRA_DIR_NAME)

        print("✅ 子片段方案B平衡完成。")

print("\n🎉 全部处理完成！")
print(f"输出总目录: {OUTPUT_ROOT}")