"""
所有超参数、路径、常量集中管理
运行前只需修改此文件顶部的 "用户配置区"
"""

import os
import numpy as np

# ==================== 项目根目录自动检测 ====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ==============================================================================
#                        👤 用户配置区（集中修改）
# ==============================================================================
"""
⚠️ 所有需要修改的参数都在这里！
"""

# ==================== 1. 单/双分支模式切换 ====================
# ✅ 修改此处即可切换单分支/双分支训练
USE_DUAL_BRANCH = True  # False=单分支, True=双分支

# ==================== 2. 数据路径配置 ====================
# 原始CSV输入目录（Step1 使用）
RAW_CSV_DIR = r' '

# Step1 输出目录（主路和子路）
STEP1_OUTPUT_MAIN = os.path.join(PROJECT_ROOT, "data", "trimed_data","dataset_root", "processed_main")
STEP1_OUTPUT_SUB = os.path.join(PROJECT_ROOT, "data", "trimed_data","dataset_root", "processed_sub")

# Step3 输入目录（STFT 转换前）
STEP3_INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "aligned_data", "dataset_root")

# 单分支数据根目录（Step4/5/6 使用）
DATA_ROOT = os.path.join(PROJECT_ROOT, "input_data", "processed_main", "dataset_root_STFT")

# 双分支数据根目录（Step4/5/6 使用 只在 USE_DUAL_BRANCH=True 时生效）
DUAL_BRANCH_ROOT = os.path.join(PROJECT_ROOT, "data", "aligned_data", "dataset_root_STFT", "Class_root")

# ==================== 3. Step1 命名配置 ====================
# 标准化命名数组（循环使用）
NAME_LIST = ['coke','peanut','nosugar','corn']

# ==================== 4. Step3 关键词过滤配置 ====================
# 包含/排除关键词（用于筛选处理哪些文件夹）
INCLUDE_DIR_KEYWORDS = []           # 空列表 = 包含所有
EXCLUDE_DIR_KEYWORDS = ['_STFT']    # 排除已处理的 STFT 文件夹
INCLUDE_KEYWORDS_MODE = 'OR'        # 'OR' 或 'AND'
KEYWORD_MATCH_MODE = 'name'         # 'name' 或 'path'
MATCH_MODE = 'prefix'               # 'prefix' 或 'exact'

# ==================== 5. 信号处理参数 ====================
# 采样率（Hz）
SAMPLING_RATE = 5000

# Step1 裁剪参数
SIGNAL_CHANNELS = [ ]
TIME_COL = "time_ms"
TRIGGER_CH = " "                 # 主触发通道
SUBCUT_CH = " "                   # 次触发通道
TRIGGER_THRESHOLD = 15000
SUBCUT_THRESHOLD = 15000
MIN_SEGMENT_DURATION_MS = 2000
MIN_SUBSEGMENT_DURATION_MS = 500
MEDIAN_FILTER_KERNEL = 5

# Step3 STFT参数
STFT_NPERSEG = 512
STFT_NOVERLAP = 384
STFT_NFFT = 512
STFT_FMAX = 1000.0
STFT_SHAPE = (101, 193)             # (频率bins, 时间bins)

# ==================== 6. 通道与图结构配置 ====================
NUM_CHANNELS = 12                                              # ✅ 保持12（完整拓扑结构）
VALID_CHANNELS_NUM = 12                                        # ✅ 实际有效通道数
MISSING_CHANNELS = []                                      # ✅ 缺失通道
VALID_PHYSICAL_CHANNELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # ✅ 实际存在的物理通道

# 通道物理坐标（用于构建邻接矩阵）- 保留12个位置
CHANNEL_COORDS = np.array([
    [0, 1], [1, 0], [1, 2], [2, 1],   # Channel 1, 2, 3, 4
    [0, 5], [1, 4], [1, 6], [2, 5],   # Channel 5, 6, 7, 8
    [0, 9], [1, 8], [1,10], [2, 9]    # Channel 9, 10, 11, 12
])

# ==================== 7. 模型参数 ====================
NUM_CLASSES = 6
GNN_HIDDEN_DIM = 64
TRANSFORMER_NHEAD = 1
TRANSFORMER_LAYERS = 2
TRANSFORMER_DIM_FEEDFORWARD = 256

# ==================== 8. 训练参数 ====================
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
VALID_RATIO = 0.2
LR_STEP_SIZE = 10
LR_GAMMA = 0.8

# Dataset 增强开关
ENABLE_NORMALIZE = True
ENABLE_AUGMENT = True
AUG_PROB = 0.8
AUG_NOISE_STD = 0.02
AUG_GAIN_RANGE = (0.8, 1.2)
AUG_TIME_SHIFT = 8
AUG_FREQ_MASK_MAX = 12
AUG_TIME_MASK_MAX = 18

# ==================== 9. 数据集划分策略 ====================
# 划分策略选择: "random" / "stratified_ratio" / "stratified_fixed"
SPLIT_STRATEGY = "stratified_ratio"
TEST_SAMPLES_PER_CLASS = 20         # 仅在 stratified_fixed 模式下生效

# ==================== 10. 双分支专属配置（仅当 USE_DUAL_BRANCH=True 时生效）====================
# Phase E 和 Phase H 子目录名
PHASE_E_SUBDIR = "Hi"               # 激励段子目录
PHASE_H_SUBDIR = "Lo"               # 保持段子目录

# Phase E 激励策略选择: "fixed_quota" 或 "multi_window"
E_STRATEGY = "multi_window"

# -------- 策略一：fixed_quota 参数 --------
E_FIXED_QUOTA_SEC = 1.5             # 固定截取时长（秒）
E_FIXED_ALIGN = "end"               # 对齐方式："end" 或 "start"
E_FIXED_PADDING_MODE = "edge"       # 填充模式："edge" 或 "zero"

# -------- 策略二：multi_window 参数 --------
E_MULTI_NUM_WINDOWS = 3             # 子窗口数量
E_MULTI_WINDOW_SEC = 1.0            # 每个子窗口时长（秒）
E_MULTI_SAMPLING = "uniform"        # 采样方式："uniform" 或 "random"
E_MULTI_RANDOM_SEED = 42            # 随机采样种子（保证可复现）
E_MULTI_AGGREGATION = "mean"        # 聚合方式："mean" / "max" / "attention"

# Phase H 配置
PHASE_H_FIXED_SEC = 2.0             # Phase H 固定时长（秒）
PHASE_H_ALIGN = "start"             # 对齐方式："start" 或 "end"
PHASE_H_PADDING_MODE = "edge"       # 填充模式

# 双分支模型配置
BRANCH_WEIGHT_SHARING = False       # 分支权重共享：True（E/H共享编码器）/ False（独立编码器）
BRANCH_FUSION_MODE = "concat"       # 融合方式："concat" / "gated" / "attention"
FUSION_ATTENTION_HEADS = 4          # Attention 融合时的注意力头数（仅当 BRANCH_FUSION_MODE="attention" 时生效）

# 双分支数据集行为配置
DUAL_BRANCH_SKIP_INCOMPLETE = True  # True: 跳过 E/H 不配对的样本
DUAL_ENABLE_AUTO_MATCH = True       # True: 启用文件名自动匹配（容错模式）

# ==================== 11. 评估模块配置 ====================
# Module A: 标准分类指标
EVAL_ENABLE_METRICS = True
EVAL_METRICS_AVERAGE = "macro"      # "macro" / "micro" / "weighted"
#  - macro: 各类别指标的简单平均（不加权），每类同等重要，对少数类更敏感
#  - weighted: 按类别样本数加权平均，更能反映类别不均衡情形，易被多数类主导
EVAL_ENABLE_MSE_RMSE = True
EVAL_PRINT_TO_CONSOLE = True
EVAL_SAVE_TO_CSV = True

# ==================== 11.A 训练过程指标记录（Step5 可选增强） ====================
# 是否在 Step5 训练过程中按 epoch 记录 Train/Val 指标并保存到实验目录
TRAIN_LOG_ENABLE = True
TRAIN_LOG_CSV_NAME = "training_history.csv"
TRAIN_LOG_JSON_NAME = "training_summary.json"
# 是否在训练日志中包含 MSE/RMSE（同时受 EVAL_ENABLE_MSE_RMSE 控制）
TRAIN_LOG_INCLUDE_MSE = True
# 是否每个 epoch 在控制台打印一行 Train/Val 指标摘要
TRAIN_LOG_PRINT_EVERY_EPOCH = True


# Module B: 混淆矩阵
EVAL_ENABLE_CONFUSION_MATRIX = True
EVAL_CONFUSION_NORMALIZE = True
EVAL_CONFUSION_TO_CONSOLE = True
EVAL_CONFUSION_TO_CSV = True

# Module C: 学习曲线
EVAL_ENABLE_LEARNING_CURVE = False  # 默认关闭（需重训练多次，耗时）
EVAL_LEARNING_CURVE_FRACS = [0.1, 0.25, 0.5, 0.75, 1.0]
EVAL_LEARNING_CURVE_REPEATS = 1
EVAL_LEARNING_CURVE_METRIC = "accuracy"
EVAL_LEARNING_CURVE_CSV_SUFFIX = "learning_curve"

# 通用评估输出配置
EVAL_CSV_SUFFIX_MAIN = "eval"
EVAL_RUN_NAME = "experiment"

# ==================== 12. 诊断模块配置 ====================
DIAG_ENABLE = True
DIAG_OUTPUT_DIR = None              # None = 使用 EVAL_OUTPUT_DIR/DATASET_NAME

# 1. 表征导出配置
DIAG_ENABLE_EMBEDDING_EXPORT = True
DIAG_EXPORT_LAYERS = ['raw_stft', 'node_feat', 'transformer']
DIAG_CSV_PREFIX = 'diag'
DIAG_EXPORT_FORMAT = 'csv'          # 'csv' / 'npy' / 'both'

# 2. 元信息配置
DIAG_METADATA_COLUMNS = ['sample_id', 'source_file', 'class_label', 'split']

# 3. 降维配置
DIAG_ENABLE_DIMENSIONALITY_REDUCTION = False
DIAG_DIM_REDUCTION_METHODS = ['pca_2d', 'pca_3d', 'umap_2d']
DIAG_UMAP_N_NEIGHBORS = 15
DIAG_UMAP_MIN_DIST = 0.1

# 4. 可分性指标配置
DIAG_ENABLE_SEPARABILITY_METRICS = False
DIAG_LINEAR_CLASSIFIER = 'logistic'  # 'logistic' / 'svm'
DIAG_LINEAR_TEST_RATIO = 0.3

# 5. 错分样本分析配置
DIAG_ENABLE_MISCLASSIFICATION_ANALYSIS = False
DIAG_CONFIDENCE_BINS = [0.0, 0.5, 0.7, 0.9, 0.95, 1.0]

# 6. 频段敏感性测试配置
DIAG_ENABLE_FREQUENCY_SENSITIVITY = True
DIAG_FREQ_MASK_BANDS = [
    (0, 50), (50, 150),(150,300), (300, 500), (500, 1000), (1000, 2000)
]

# 7. 时间窗敏感性测试配置
DIAG_ENABLE_TIME_SENSITIVITY = False
DIAG_TIME_MASK_WINDOWS = 10

# 8. 自动扫描配置
DIAG_ENABLE_AUTO_SCAN = True
DIAG_AUTO_SCAN_FREQ_SEGMENTS = 10
DIAG_AUTO_SCAN_TIME_SEGMENTS = 10
DIAG_AUTO_SCAN_TOPK = 5

# 9. 通道消融配置
DIAG_ENABLE_CHANNEL_ABLATION = False
DIAG_ABLATION_MODE = 'zero'         # 'zero' / 'drop'

# 10. 输出格式配置
DIAG_SAVE_PLOTS = False
DIAG_VERBOSE = True

# ==================== 12.A 双分支随机多通道消融（Step6 可选增强） ====================
# 在 Step6 诊断中，对双分支输入同时进行随机多通道消融（k=2/4/6/8/10，多次采样）
DIAG_MULTI_CHANNEL_ABLATION_ENABLE = False
DIAG_MULTI_ABLATION_K_LIST = [2, 4, 6, 8, 10]
DIAG_MULTI_ABLATION_MIN_TRIALS_PER_K = 5
DIAG_MULTI_ABLATION_RANDOM_SEED = 42
# 是否额外保存 precision/recall/f1（使用 EVAL_METRICS_AVERAGE 作为平均策略）
DIAG_MULTI_ABLATION_SAVE_EXTRA_METRICS = True


# ==================== 13. 其他配置 ====================
RANDOM_SEED = 42
DEVICE = "cuda"                     # "cuda" / "cpu"

# ==============================================================================
#                     🔧 系统自动配置区（无需修改）
# ==============================================================================
"""
⚠️ 以下代码由系统自动管理，除非理解其作用，否则不要修改
"""

# ==================== Step3 STFT 输出目录自动生成 ====================
def make_stft_output_dir(input_dir: str, suffix: str = "_STFT") -> str:
    """在输入目录同级创建 STFT 输出目录"""
    parent = os.path.dirname(input_dir.rstrip(r"\/"))
    leaf = os.path.basename(input_dir.rstrip(r"\/"))
    return os.path.join(parent, leaf + suffix)

STEP3_OUTPUT_DIR = make_stft_output_dir(STEP3_INPUT_DIR)

# ==================== 双分支路径自动派生 ====================
if USE_DUAL_BRANCH:
    # 自动构建 Phase E 和 Phase H 完整路径
    PHASE_E_DIR = DUAL_BRANCH_ROOT  # ✅ 指向根目录
    PHASE_H_DIR = DUAL_BRANCH_ROOT  # ✅ 指向根目录

    
    # 双分支 STFT 参数（继承全局设置）
    DUAL_STFT_NPERSEG = STFT_NPERSEG
    DUAL_STFT_NOVERLAP = STFT_NOVERLAP
    DUAL_STFT_NFFT = STFT_NFFT
    DUAL_STFT_FMAX = STFT_FMAX
    DUAL_FREQ_BINS = 103 
    
    # 自动计算目标时间帧数
    if E_STRATEGY == "fixed_quota":
        E_TARGET_TIME_BINS = int(E_FIXED_QUOTA_SEC * SAMPLING_RATE / (DUAL_STFT_NPERSEG - DUAL_STFT_NOVERLAP))
    elif E_STRATEGY == "multi_window":
        E_TARGET_TIME_BINS = int(E_MULTI_WINDOW_SEC * SAMPLING_RATE / (DUAL_STFT_NPERSEG - DUAL_STFT_NOVERLAP))
    else:
        E_TARGET_TIME_BINS = 50
    
    H_TARGET_TIME_BINS = int(PHASE_H_FIXED_SEC * SAMPLING_RATE / (DUAL_STFT_NPERSEG - DUAL_STFT_NOVERLAP))
    
    # 数据增强继承
    DUAL_ENABLE_NORMALIZE = ENABLE_NORMALIZE
    DUAL_ENABLE_AUGMENT = ENABLE_AUGMENT
    
    # 双分支诊断配置
    DUAL_DIAG_ENABLE_EMBEDDING_EXPORT = True
    DUAL_DIAG_EXPORT_LAYERS = [
        'node_feat_e',      # Phase E GNN 输出
        'node_feat_h',      # Phase H GNN 输出
        'node_feat_fused',  # 融合后特征
        'transformer'       # Transformer 输出
    ]
    DUAL_DIAG_CSV_PREFIX = 'dual_diag'

# ==================== 输出目录配置（✅ 关键修改）====================
# ✅ 根据模式选择正确的数据集基础名称
if USE_DUAL_BRANCH:
    # 双分支模式：从 DUAL_BRANCH_ROOT 提取名称
    _dual_branch_root_normalized = DUAL_BRANCH_ROOT.rstrip('/\\')
    DATASET_BASE_NAME = os.path.basename(_dual_branch_root_normalized)
else:
    # 单分支模式：从 DATA_ROOT 提取名称
    DATASET_BASE_NAME = os.path.basename(DATA_ROOT)

# 🔍 调试输出（可选，正式使用时可删除）
print(f"[Config] 数据集模式: {'双分支' if USE_DUAL_BRANCH else '单分支'}")
if USE_DUAL_BRANCH:
    print(f"[Config] DUAL_BRANCH_ROOT: {DUAL_BRANCH_ROOT}")
print(f"[Config] DATASET_BASE_NAME: {DATASET_BASE_NAME}")

# ==================== 实验会话管理（✅ 关键修改）====================
def get_experiment_hash(config_obj=None):
    """
    根据关键超参数生成哈希值（用于判断是否需要新建实验目录）
    当数据路径或关键参数变化时，会生成不同的 hash，从而创建新目录
    """
    import hashlib
    import json
    
    # ✅ 根据模式选择正确的数据路径
    if USE_DUAL_BRANCH:
        data_path = DUAL_BRANCH_ROOT  # 双分支：使用双分支根目录
    else:
        data_path = DATA_ROOT  # 单分支：使用单分支根目录
    
    # 提取关键参数（只有这些变化时才创建新目录）
    key_params = {
        'data_root': data_path,  # ✅ 核心：数据路径变化会改变 hash
        'num_classes': NUM_CLASSES,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'epochs': EPOCHS,
        'gnn_hidden_dim': GNN_HIDDEN_DIM,
        'transformer_layers': TRANSFORMER_LAYERS,
    }
    
    # 🔍 调试输出（可选）
    # print(f"[Config] Hash 计算参数: data_root={data_path}")
    
    param_str = json.dumps(key_params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()[:8]


def ensure_eval_dirs():
    """确保评估目录存在"""
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)


def get_or_create_experiment_name():
    """
    获取实验名称（带会话管理：参数相同则复用，参数变化则递增）
    
    工作原理：
    1. 计算当前参数的 hash
    2. 如果 hash 与上次相同 → 复用现有实验名
    3. 如果 hash 不同 → 扫描已有编号，递增创建新实验名
    
    Returns:
        str: 实验名称（如 "sauce-like_STFT_001" 或 "milk-like_STFT_001"）
    """
    base_path = os.path.join(PROJECT_ROOT, "evaluation_results")
    os.makedirs(base_path, exist_ok=True)
    
    # 会话文件
    session_file = os.path.join(base_path, ".current_session.json")
    current_hash = get_experiment_hash()
    
    # 检查现有会话
    if os.path.exists(session_file):
        try:
            import json
            with open(session_file, 'r') as f:
                session = json.load(f)
            
            # 参数未变化 → 复用现有目录名
            if session.get('param_hash') == current_hash:
                existing_name = session['experiment_name']
                print(f"[Config] 复用现有实验: {existing_name}")
                return existing_name
        except:
            pass
    
    # 需要新目录：扫描现有编号
    existing_numbers = []
    for dirname in os.listdir(base_path):
        if dirname.startswith(DATASET_BASE_NAME + "_"):
            suffix = dirname[len(DATASET_BASE_NAME)+1:]
            if suffix.isdigit():
                existing_numbers.append(int(suffix))
    
    next_number = max(existing_numbers) + 1 if existing_numbers else 1
    experiment_name = f"{DATASET_BASE_NAME}_{next_number:03d}"
    
    print(f"[Config] 创建新实验: {experiment_name}")
    
    # 保存新会话
    import json
    from datetime import datetime
    session = {
        'experiment_name': experiment_name,
        'param_hash': current_hash,
        'created_at': datetime.now().isoformat(),
    }
    with open(session_file, 'w') as f:
        json.dump(session, f, indent=2)
    
    return experiment_name


def get_dataset_name():
    """供 Step5/6 调用，延迟生成实验目录"""
    return get_or_create_experiment_name()


# ==================== 🔥 智能路径选择（Step4/5/6 自动适配）====================
def get_active_data_root():
    """
    根据 USE_DUAL_BRANCH 自动返回正确的数据路径
    
    - 单分支: 返回 DATA_ROOT
    - 双分支: 返回 DUAL_BRANCH_ROOT
    
    供 Step4/5/6 统一调用
    """
    if USE_DUAL_BRANCH:
        return DUAL_BRANCH_ROOT
    else:
        return DATA_ROOT


def get_active_dataset_description():
    """
    返回当前激活的数据集描述（用于打印）
    """
    if USE_DUAL_BRANCH:
        return {
            'mode': '双分支',
            'root': DUAL_BRANCH_ROOT,
            'phase_e': PHASE_E_DIR if USE_DUAL_BRANCH else None,
            'phase_h': PHASE_H_DIR if USE_DUAL_BRANCH else None,
            'strategy': E_STRATEGY if USE_DUAL_BRANCH else None
        }
    else:
        return {
            'mode': '单分支',
            'root': DATA_ROOT,
            'phase_e': None,
            'phase_h': None,
            'strategy': None
        }


# ==================== 双分支配置验证函数（延迟调用）====================
def validate_dual_branch_config():
    """
    双分支配置验证（只在 Step4/5/6 中手动调用）
    
    ⚠️ 不在 config.py 导入时自动执行，避免影响 Step1/2/3
    """
    if not USE_DUAL_BRANCH:
        return True  # 单分支模式，跳过验证
    
    errors = []
    
    # 验证路径存在
    if not os.path.exists(DUAL_BRANCH_ROOT):
        errors.append(f"DUAL_BRANCH_ROOT 路径不存在: {DUAL_BRANCH_ROOT}")
    
    # 验证策略
    if E_STRATEGY not in ["fixed_quota", "multi_window"]:
        errors.append(f"E_STRATEGY 必须是 'fixed_quota' 或 'multi_window'，当前值: {E_STRATEGY}")
    
    if BRANCH_FUSION_MODE not in ["concat", "gated", "attention"]:
        errors.append(f"BRANCH_FUSION_MODE 必须是 'concat', 'gated' 或 'attention'，当前值: {BRANCH_FUSION_MODE}")
    
    if errors:
        raise ValueError("双分支配置错误:\n" + "\n".join(f"  ❌ {e}" for e in errors))
    
    # 验证通过，打印配置信息
    print(f"\n{'='*70}")
    print(f"✅ 双分支配置验证通过")
    print(f"   Phase E 路径: {PHASE_E_DIR}")
    print(f"   Phase H 路径: {PHASE_H_DIR}")
    print(f"   Phase E 策略: {E_STRATEGY}")
    print(f"   融合模式: {BRANCH_FUSION_MODE}")
    print(f"   权重共享: {BRANCH_WEIGHT_SHARING}")
    print(f"{'='*70}\n")
    
    return True


# ==================== 默认使用基础名称（不递增），Step5/6 会主动调用 get_dataset_name() ====================
DATASET_NAME = DATASET_BASE_NAME

# 模型保存目录（使用基础名称，避免每次导入都创建）
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", DATASET_NAME)

# 评估结果输出目录（使用基础名称）
EVAL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "evaluation_results", DATASET_NAME)

# 日志目录
LOG_DIR = os.path.join(PROJECT_ROOT, "logs", DATASET_NAME)


# ==================== 辅助函数：打印路径横幅 ====================
def print_eval_paths_banner():
    """打印评估路径信息"""
    active_root = get_active_data_root()
    print(f"\n{'='*70}")
    print(f"📊 当前模式: {'双分支' if USE_DUAL_BRANCH else '单分支'}")
    print(f"📂 数据根目录: {active_root}")
    print(f"💾 评估输出目录: {EVAL_OUTPUT_DIR}")
    print(f"{'='*70}\n")


def print_project_banner():
    """打印项目横幅（兼容 step6）"""
    print("\n" + "=" * 70)
    print("🔬 GNN-Transformer 深度诊断工具")
    print("=" * 70)
    print(f"📁 项目根目录: {PROJECT_ROOT}")
    print(f"📊 数据模式: {'双分支' if USE_DUAL_BRANCH else '单分支'}")
    if USE_DUAL_BRANCH:
        print(f"📂 数据根目录: {DUAL_BRANCH_ROOT}")
    else:
        print(f"📂 数据根目录: {DATA_ROOT}")
    print(f"💾 评估输出: {EVAL_OUTPUT_DIR}")
    print("=" * 70 + "\n")
