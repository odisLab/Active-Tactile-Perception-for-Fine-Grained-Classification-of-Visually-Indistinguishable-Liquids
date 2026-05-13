import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import stft

# ===================== 全局配置参数（全部汇总在代码头部） =====================
# 0) 输入/输出目录配置（你只需要改这里）
RAW_DATA_ROOT = r" "  # ✅ 总目录：可包含多个自组子文件夹
OUTPUT_ROOT = None  # None=自动：与RAW_DATA_ROOT同级生成 “<RAW_DATA_ROOT名>_STFT”

# 0.1) 输入目录筛选配置（支持 OR / AND）
# INCLUDE_KEYWORDS_MODE:
#   - "OR"：命中任意关键词就保留（你当前需求）
#   - "AND"：必须全部命中才保留
INCLUDE_DIR_KEYWORDS = [" "]     # 例：["Class5"] 或 ["A1", "Class5"]
INCLUDE_KEYWORDS_MODE = "OR"          # ✅ 你要的 OR
EXCLUDE_DIR_KEYWORDS = ["_STFT"]      # 例：排除历史输出目录，避免重复处理
KEYWORD_MATCH_MODE = "path"           # "path"：匹配完整路径；"name"：只匹配文件夹名
CASE_INSENSITIVE = True               # 英文匹配是否忽略大小写

# 1) STFT核心参数
sampling_rate = 5000.0  # 采样率（Hz）
nperseg = 512          # STFT窗长
noverlap = 384         # 重叠长度（75%重叠）
nfft = nperseg          # FFT点数
fmax = 1000.0           # 保留的最高频率（Hz），设为None则保留全频段

# 2) 通道配置（核心：适配10个有效通道）
MISSING_CHANNELS = [ ]              # 无信号的缺失通道编号（仅用于打印说明）
VALID_CHANNELS_NUM = 12                # 有效通道数
REQUIRED_COLS = 1 + VALID_CHANNELS_NUM # 数据列数要求：1列时间 + 10列通道
# 有效通道的物理编号，用于NPY命名标注
VALID_PHYSICAL_CHANNELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


# ===================== 目录筛选工具函数（新增：OR关系支持） =====================
def _norm_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    return s.lower() if CASE_INSENSITIVE else s


def _match_keywords(target: str,
                    include_keywords=None,
                    include_mode="OR",
                    exclude_keywords=None) -> bool:
    """
    include_mode:
      - "OR": include_keywords 中任意一个命中即可通过（include为空则不过滤）
      - "AND": include_keywords 全部命中才通过（include为空则不过滤）
    exclude：命中任意一个直接淘汰
    """
    target_n = _norm_text(target)
    include_keywords = include_keywords or []
    exclude_keywords = exclude_keywords or []
    include_mode = (include_mode or "OR").upper()

    # exclude：任意命中 -> 淘汰
    for kw in exclude_keywords:
        if _norm_text(kw) in target_n:
            return False

    # include：空列表 -> 不过滤，直接通过
    if len(include_keywords) == 0:
        return True

    # include：OR / AND
    if include_mode == "AND":
        return all(_norm_text(kw) in target_n for kw in include_keywords)
    else:  # 默认 OR
        return any(_norm_text(kw) in target_n for kw in include_keywords)


def discover_input_roots(root_dir: str,
                         include_keywords=None,
                         include_mode="OR",
                         exclude_keywords=None,
                         match_mode="path"):
    """
    在root_dir下发现“包含数据文件的子目录”，并按关键词筛选。
    返回：筛选后的子目录列表（绝对路径）。
    规则：只要该目录（含其子目录）中存在 csv/xlsx/xls 文件，就认为是候选输入目录。
    另外做一次“父子目录去重”：保留更浅层目录，避免重复处理。
    """
    root_abs = os.path.abspath(root_dir)
    candidates = []

    for dirpath, dirnames, filenames in os.walk(root_abs):
        has_data_file = any(fn.lower().endswith((".csv", ".xlsx", ".xls")) for fn in filenames)
        if not has_data_file:
            continue

        key_target = dirpath if match_mode == "path" else os.path.basename(dirpath)
        if _match_keywords(
            key_target,
            include_keywords=include_keywords,
            include_mode=include_mode,
            exclude_keywords=exclude_keywords
        ):
            candidates.append(dirpath)

    # 去重 + 父子目录去重：优先保留更浅的
    candidates = sorted(set(candidates), key=lambda p: (len(p.split(os.sep)), p))
    filtered = []
    for p in candidates:
        if not any(p.startswith(prev + os.sep) for prev in filtered):
            filtered.append(p)

    return filtered


# ===================== STFT核心处理函数（保持不变） =====================
def compute_stft(signal, fs):
    """
    对单通道时域信号做STFT，转换为分贝（dB）的频域特征
    :param signal: 单通道时域信号数组 (N,)
    :param fs: 采样率（Hz）
    :return: (f, t, mag_db) → 频率轴、时间轴、幅值分贝矩阵
    """
    f, t, Zxx = stft(
        signal,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary=None,
        padded=False,
    )
    # 计算幅值并转换为dB（加极小值避免log(0)）
    mag = np.abs(Zxx)
    mag_db = 20 * np.log10(mag + 1e-12)

    # 截断频率范围到fmax
    if fmax is not None:
        idx = f <= fmax
        f = f[idx]
        mag_db = mag_db[idx, :]

    return f, t, mag_db


def save_single_channel_npy(f, t, mag_db, save_dir, physical_channel_id):
    """
    保存单通道STFT数据到NPY文件（包含元信息）
    :param f: 频率轴数组 (freq,)
    :param t: 时间轴数组 (time,)
    :param mag_db: 幅值分贝矩阵 (freq, time)
    :param save_dir: 保存目录
    :param physical_channel_id: 物理通道编号（如1、2、5等）
    """
    channel_stft_data = {
        "frequency": f,
        "time": t,
        "magnitude_dB": mag_db,
        "model_input": mag_db[np.newaxis, :, :],  # (1, freq, time)
        "sampling_rate": sampling_rate,
        "physical_channel_id": physical_channel_id
    }
    npy_save_path = os.path.join(save_dir, f"Channel_{physical_channel_id}_STFT.npy")
    np.save(npy_save_path, channel_stft_data)
    print(f"    ✅ 保存通道{physical_channel_id}：{npy_save_path}")


# ===================== 批量处理CSV/Excel生成STFT NPY（核心功能不变，仅改“输入目录发现/筛选”） =====================
def process_stft_single_file(input_file_path, raw_data_root_abs, output_root_abs):
    """
    处理单个CSV/Excel文件，生成10个有效通道的STFT NPY
    """
    # 1. 强制路径校验（杜绝空值/非字符串）
    if not isinstance(input_file_path, str) or len(input_file_path.strip()) == 0:
        print("    ❌ 输入文件路径为空，跳过")
        return None
    if not isinstance(raw_data_root_abs, str) or len(raw_data_root_abs.strip()) == 0:
        print("    ❌ 原始数据根目录为空，跳过")
        return None
    if not isinstance(output_root_abs, str) or len(output_root_abs.strip()) == 0:
        print("    ❌ 输出根目录为空，跳过")
        return None

    # 2. 标准化路径（统一绝对路径+分隔符）
    input_file_abs = os.path.abspath(input_file_path)
    raw_data_root_abs = os.path.abspath(raw_data_root_abs)
    output_root_abs = os.path.abspath(output_root_abs)

    # 3. 安全计算相对路径（保持目录结构）
    try:
        rel_path = os.path.relpath(input_file_abs, start=raw_data_root_abs)
    except ValueError:
        parent_dir = os.path.basename(os.path.dirname(input_file_abs))
        file_name = os.path.basename(input_file_abs)
        rel_path = os.path.join(parent_dir, file_name)

    # 4. 构建输出目录
    rel_dir = os.path.dirname(rel_path)
    file_name = os.path.basename(input_file_abs)
    file_base = os.path.splitext(file_name)[0]
    fmax_suffix = "fullband" if fmax is None else f"fmax{int(fmax)}Hz"
    file_stft_dir = f"{file_base}_STFT_npy_{fmax_suffix}"

    output_dir = os.path.join(
        output_root_abs,
        rel_dir if rel_dir else "unknown_dir",
        file_stft_dir
    )
    os.makedirs(output_dir, exist_ok=True)

    # 5. 读取数据（兼容CSV/Excel）
    try:
        if input_file_abs.endswith('.csv'):
            df = pd.read_csv(input_file_abs, encoding='utf-8')
        elif input_file_abs.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(input_file_abs)
        else:
            print(f"    ❌ 不支持的文件格式：{input_file_abs}")
            return None
    except Exception as e:
        print(f"    ❌ 读取文件失败：{str(e)}")
        return None

    # 6. 校验列数（1时间+10通道=11列）
    if df.shape[1] < REQUIRED_COLS:
        print(f"    ❌ 列数不足（需{REQUIRED_COLS}列，当前{df.shape[1]}列），跳过")
        return None

    # 7. 逐通道处理STFT
    channel_cols = df.columns[1:REQUIRED_COLS].tolist()
    print(f"    识别到{VALID_CHANNELS_NUM}个有效通道：{channel_cols}（缺失通道：{MISSING_CHANNELS}）")

    for channel_col, phys_id in zip(channel_cols, VALID_PHYSICAL_CHANNELS):
        signal = df[channel_col].values
        f, t, mag_db = compute_stft(signal, sampling_rate)
        save_single_channel_npy(f, t, mag_db, output_dir, phys_id)

    return output_dir


def batch_process_stft(raw_data_root,
                       output_root=None,
                       include_dir_keywords=None,
                       include_keywords_mode="OR",
                       exclude_dir_keywords=None,
                       match_mode="path"):
    """
    批量处理原始数据目录下的所有CSV/Excel文件，生成STFT NPY
    - raw_data_root 允许是“总目录”
    - 按关键词筛选输入目录（支持 OR/AND）
    """
    raw_data_root_abs = os.path.abspath(raw_data_root)

    # 1. 初始化输出根目录（兜底空值）
    if output_root is None or not isinstance(output_root, str) or len(output_root.strip()) == 0:
        output_root = os.path.join(
            os.path.dirname(raw_data_root_abs),
            f"{os.path.basename(raw_data_root_abs)}_STFT"
        )
    output_root_abs = os.path.abspath(output_root)
    os.makedirs(output_root_abs, exist_ok=True)

    # 2. 发现并筛选输入目录
    input_roots = discover_input_roots(
        raw_data_root_abs,
        include_keywords=include_dir_keywords,
        include_mode=include_keywords_mode,
        exclude_keywords=exclude_dir_keywords,
        match_mode=match_mode
    )

    print(f"📌 原始数据总目录：{raw_data_root_abs}")
    print(f"📌 STFT输出根目录：{output_root_abs}")
    print(f"📌 目录筛选 include={include_dir_keywords} (mode={include_keywords_mode}) exclude={exclude_dir_keywords} match={match_mode}")
    print(f"✅ 命中 {len(input_roots)} 个输入目录\n")

    if not input_roots:
        print("❌ 未找到任何符合筛选条件且包含CSV/Excel的输入目录！请检查路径/关键词")
        return output_root_abs

    # 3. 在所有命中的输入目录中匹配文件
    all_files = []
    for r in input_roots:
        for pattern in [
            os.path.join(r, "**", "*.csv"),
            os.path.join(r, "**", "*.xlsx"),
            os.path.join(r, "**", "*.xls")
        ]:
            all_files.extend(glob.glob(pattern, recursive=True))
    all_files = sorted(set(all_files))

    if not all_files:
        print("❌ 在命中的输入目录内未找到任何CSV/Excel文件！")
        return output_root_abs

    # 4. 逐文件处理：relpath 仍以“总目录”作为start，保持相对目录结构
    total_files = len(all_files)
    print(f"✅ 共找到 {total_files} 个文件，开始批量处理...\n")

    for idx, file_path in enumerate(all_files, 1):
        print(f"[{idx}/{total_files}] 处理文件：{file_path}")
        process_stft_single_file(file_path, raw_data_root_abs, output_root_abs)

    print(f"\n🎉 批量处理完成！所有NPY文件已保存到：{output_root_abs}")
    return output_root_abs


# ===================== 主调用入口（底部不再放配置，只调用） =====================
if __name__ == "__main__":
    batch_process_stft(
        RAW_DATA_ROOT,
        output_root=OUTPUT_ROOT,
        include_dir_keywords=INCLUDE_DIR_KEYWORDS,
        include_keywords_mode=INCLUDE_KEYWORDS_MODE,
        exclude_dir_keywords=EXCLUDE_DIR_KEYWORDS,
        match_mode=KEYWORD_MATCH_MODE
    )