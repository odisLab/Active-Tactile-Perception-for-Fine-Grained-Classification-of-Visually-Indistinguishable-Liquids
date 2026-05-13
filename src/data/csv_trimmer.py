# src/data/csv_trimmer.py
import os
import pandas as pd
import numpy as np
from skimage.filters import threshold_otsu
from scipy.ndimage import median_filter


# =========================
# 工具函数
# =========================
def safe_dirname(s: str) -> str:
    """Windows 兼容的文件夹名清洗"""
    s = str(s).strip()
    for ch in ['<', '>', ':', '"', '/', '\\', '|', '?', '*']:
        s = s.replace(ch, '_')
    return s if s else "unnamed"


def compute_threshold(data: pd.Series, mode: str, fixed_value: float) -> float:
    arr = np.asarray(data, dtype=float)
    if mode == "fixed":
        return float(fixed_value)
    if mode == "auto_mean":
        return float(np.nanmean(arr))
    if mode == "auto_otsu":
        try:
            return float(threshold_otsu(arr))
        except Exception:
            return float(np.nanmedian(arr))
    return float(fixed_value)


def extract_segments(mask: np.ndarray):
    """从 bool 序列中抽取连续 True 段 (start_idx, end_idx)"""
    segments = []
    is_on = False
    start = None
    for i, v in enumerate(mask):
        if v and not is_on:
            is_on = True
            start = i
        elif (not v) and is_on:
            is_on = False
            segments.append((start, i - 1))
    if is_on:
        segments.append((start, len(mask) - 1))
    return segments


def extract_segments_with_minlen(mask: np.ndarray, min_samples: int):
    """只返回长度 >= min_samples 的连续 True 段"""
    raw = extract_segments(mask)
    return [(s, e) for (s, e) in raw if (e - s + 1) >= min_samples]


def save_csv(df: pd.DataFrame, filename: str, folder: str):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    df.to_csv(path, index=False)
    print("✔ Saved:", path)


def list_csv_files(input_folder: str, recursive: bool = False):
    """
    列出 CSV 文件路径列表。
    - recursive=False：只扫描 input_folder 顶层（与你原逻辑一致）
    - recursive=True：递归扫描子目录
    """
    csv_paths = []
    if not os.path.isdir(input_folder):
        return csv_paths

    if not recursive:
        for f in sorted(os.listdir(input_folder)):
            if f.lower().endswith(".csv"):
                csv_paths.append(os.path.join(input_folder, f))
        return csv_paths

    for root, _, files in os.walk(input_folder):
        for f in files:
            if f.lower().endswith(".csv"):
                csv_paths.append(os.path.join(root, f))
    csv_paths.sort()
    return csv_paths


# =========================
# ✅ 核心入口函数：供 Step1 调用
# =========================
def trim_csv_files(input_folder: str, name_list: list, config):
    """
    Step1 核心：从原始 CSV 中提取主片段和子片段，并按你要求的目录结构保存。

    输出结构：
      processed_main/class_x/<liquid_name>/*.csv
      processed_sub/class_x/<liquid_name>/high/*.csv
      processed_sub/class_x/<liquid_name>/low/*.csv

    返回：
      {
        "main_root": <processed_main>,
        "sub_root": <processed_sub>,
        "log_file": <skipped_segments.log>,
        "num_files": 扫描到的 csv 文件数
      }
    """

    # -------------------------
    # 参数（优先取 config）
    # -------------------------
    sampling_rate = getattr(config, "SAMPLING_RATE", 5000)

    signal_channels = getattr(
        config,
        "SIGNAL_CHANNELS",
        ['ch0', 'ch1', 'ch2', 'ch3', 'ch6', 'ch7', 'ch8', 'ch10', 'ch14', 'ch15']
    )
    time_col = getattr(config, "TIME_COL", "time_ms")
    trigger_ch = getattr(config, "TRIGGER_CH", "ch10")
    subcut_ch = getattr(config, "SUBCUT_CH", "ch8")

    trigger_threshold_value = getattr(config, "TRIGGER_THRESHOLD", 15000)
    subcut_threshold_value = getattr(config, "SUBCUT_THRESHOLD", 15000)

    trigger_threshold_mode = getattr(config, "TRIGGER_THRESHOLD_MODE", "fixed")  # fixed/auto_mean/auto_otsu
    subcut_threshold_mode = getattr(config, "SUBCUT_THRESHOLD_MODE", "fixed")    # fixed/auto_mean/auto_otsu

    min_segment_duration_ms = getattr(config, "MIN_SEGMENT_DURATION_MS", 2000)
    min_subsegment_duration_ms = getattr(config, "MIN_SUBSEGMENT_DURATION_MS", 500)
    median_filter_kernel = getattr(config, "MEDIAN_FILTER_KERNEL", 5)

    # 递归扫描开关（默认 False：不改变你原行为）
    recursive_scan = getattr(config, "STEP1_RECURSIVE_SCAN", False)

    # Step1 输出目录（优先用 config 的全局输出目录；没有则默认写到 input_folder 下面）
    output_main_root = getattr(config, "STEP1_OUTPUT_MAIN", os.path.join(input_folder, "processed_main"))
    output_sub_root = getattr(config, "STEP1_OUTPUT_SUB", os.path.join(input_folder, "processed_sub"))

    # 日志目录：放在 input_folder/logs（与原脚本一致）
    log_path = os.path.join(input_folder, "logs")
    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(log_path, "skipped_segments.log")

    def log(msg: str):
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        print("[LOG]", msg)

    # 创建根输出目录
    os.makedirs(output_main_root, exist_ok=True)
    os.makedirs(output_sub_root, exist_ok=True)

    # -------------------------
    # 扫描 CSV
    # -------------------------
    csv_paths = list_csv_files(input_folder, recursive=recursive_scan)
    if not csv_paths:
        print("⚠ 目录下没有找到 csv 文件:", input_folder)
        return {
            "main_root": output_main_root,
            "sub_root": output_sub_root,
            "log_file": log_file,
            "num_files": 0,
        }

    if not name_list:
        raise ValueError("name_list 为空：请在 config.NAME_LIST 或调用参数中提供命名数组")

    num_names = len(name_list)
    global_file_index = 0

    # -------------------------
    # 主处理循环
    # -------------------------
    for file_path in csv_paths:
        filename = os.path.basename(file_path)
        print(f"\n🔎 Processing CSV #{global_file_index+1}: {filename}")

        # 根据全局顺序号循环命名
        liquid_name = name_list[global_file_index % num_names]
        liquid_run_idx = global_file_index // num_names + 1
        liquid_run_str = f"{liquid_run_idx:02d}"
        global_file_index += 1

        liquid_dirname = safe_dirname(liquid_name)

        # 读取数据
        try:
            df = pd.read_csv(file_path, encoding="utf-8", low_memory=False)
        except Exception as e:
            log(f"读取失败 {filename}: {e}")
            continue

        # 校验列
        if time_col not in df.columns:
            log(f"{filename} 缺少时间列 {time_col} → 跳过")
            continue
        if trigger_ch not in df.columns or subcut_ch not in df.columns:
            log(f"{filename} 缺少触发通道 {trigger_ch}/{subcut_ch} → 跳过")
            continue

        # 只保留需要的列
        keep_cols = [c for c in [time_col] + list(signal_channels) if c in df.columns]
        df = df[keep_cols].copy()

        # 清理异常值
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        if df.empty:
            log(f"{filename} 清洗后为空 → 跳过")
            continue

        # 中值滤波消抖
        if median_filter_kernel and median_filter_kernel >= 3:
            df[trigger_ch] = median_filter(df[trigger_ch], median_filter_kernel)
            df[subcut_ch] = median_filter(df[subcut_ch], median_filter_kernel)

        # ---- 主触发阈值 ----
        thr = compute_threshold(df[trigger_ch], trigger_threshold_mode, trigger_threshold_value)
        print(f"  Trigger threshold used: {thr:.2f}")
        main_mask = df[trigger_ch].values > thr

        min_main_samples = int(min_segment_duration_ms / 1000.0 * sampling_rate)
        main_segments = extract_segments_with_minlen(main_mask, min_main_samples)
        print(f"  有效主片段（长度≥{min_segment_duration_ms} ms）: {len(main_segments)} 段")

        # ---- 次触发阈值 ----
        sub_thr = compute_threshold(df[subcut_ch], subcut_threshold_mode, subcut_threshold_value)
        print(f"  Subcut threshold used: {sub_thr:.2f}")

        # 每个文件最多保留 3 段主片段（与你原逻辑一致）
        main_count = 0

        for seg_idx, (s, e) in enumerate(main_segments, start=1):
            if main_count >= 3:
                log(f"{filename} 有效主片段超过 3 段，第 {seg_idx} 段及之后全部跳过")
                break

            seg_df = df.iloc[s:e + 1].reset_index(drop=True)
            duration_ms = len(seg_df) / sampling_rate * 1000.0

            # 在主片段内部，用 ch8 划分高/低电平
            sub_values = seg_df[subcut_ch].values
            sub_mask_high = sub_values > sub_thr
            sub_mask_low = ~sub_mask_high
            min_sub_samples = int(min_subsegment_duration_ms / 1000.0 * sampling_rate)

            high_segments = extract_segments_with_minlen(sub_mask_high, min_sub_samples)
            if len(high_segments) == 0:
                log(f"{filename} 主片段 #{seg_idx} 内无满足条件的 ch8 高电平子段 → 整段跳过")
                continue

            low_segments = extract_segments_with_minlen(sub_mask_low, min_sub_samples)

            # 合格主片段计数 + class
            main_count += 1
            class_label = f"class_{5 * main_count}"  # 1→class_5, 2→class_10, 3→class_15
            base_name = f"{liquid_dirname}_{liquid_run_str}_{class_label}"

            # ==============================
            # ✅ 输出目录：按你要求的“两级/三级”
            # main: class -> liquid
            # sub : class -> liquid -> high/low
            # ==============================
            main_out_dir = os.path.join(output_main_root, class_label, liquid_dirname)
            sub_liquid_dir = os.path.join(output_sub_root, class_label, liquid_dirname)
            sub_high_dir = os.path.join(sub_liquid_dir, "high")
            sub_low_dir = os.path.join(sub_liquid_dir, "low")
            os.makedirs(main_out_dir, exist_ok=True)
            os.makedirs(sub_high_dir, exist_ok=True)
            os.makedirs(sub_low_dir, exist_ok=True)

            # ===== 保存主片段 =====
            main_filename = f"{base_name}_{main_count:03d}.csv"
            save_csv(seg_df, main_filename, main_out_dir)

            print(
                f"  >> 主片段 #{main_count} | {liquid_dirname}_{liquid_run_str} | 时长 {duration_ms:.1f} ms "
                f"| 高子段 {len(high_segments)} 个 | 低子段 {len(low_segments)} 个"
            )

            # ===== 保存高电平子段 =====
            for k, (ss, ee) in enumerate(high_segments):
                part_df = seg_df.iloc[ss:ee + 1].reset_index(drop=True)
                sub_dur_ms = len(part_df) / sampling_rate * 1000.0
                part_filename = f"{base_name}_H{k}_{main_count:03d}.csv"
                save_csv(part_df, part_filename, sub_high_dir)
                print(f"     -> 高电平子段 H{k} 时长 {sub_dur_ms:.1f} ms")

            # ===== 保存低电平子段 =====
            for k, (ss, ee) in enumerate(low_segments):
                part_df = seg_df.iloc[ss:ee + 1].reset_index(drop=True)
                sub_dur_ms = len(part_df) / sampling_rate * 1000.0
                part_filename = f"{base_name}_L{k}_{main_count:03d}.csv"
                save_csv(part_df, part_filename, sub_low_dir)
                print(f"     -> 低电平子段 L{k} 时长 {sub_dur_ms:.1f} ms")

    # 返回根目录（稳定、不依赖最后一次循环）
    return {
        "main_root": output_main_root,
        "sub_root": output_sub_root,
        "log_file": log_file,
        "num_files": len(csv_paths),
    }