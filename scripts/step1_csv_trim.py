"""
Step1: CSV信号裁剪
从原始CSV中提取主片段和子片段
运行: python scripts/step1_csv_trim.py
参数修改: configs/config.py中修改RAW_CSV_DIR、STEP1相关参数、NAME_LIST
"""
import sys
import os
# 动态获取项目根目录并添加到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from configs import config
from src.data.csv_trimmer import trim_csv_files

if __name__ == "__main__":
    print("=" * 60)
    print("Step1: CSV信号裁剪")
    print("=" * 60)
    print(f"项目根目录: {project_root}")
    print(f"输入目录: {config.RAW_CSV_DIR}")
    print(f"样本命名列表: {config.NAME_LIST}")
    print(f"采样率: {config.SAMPLING_RATE}Hz")
    print(f"主触发通道: {config.TRIGGER_CH}, 阈值: {config.TRIGGER_THRESHOLD}")
    print(f"次触发通道: {config.SUBCUT_CH}, 阈值: {config.SUBCUT_THRESHOLD}")
    print("=" * 60)

    # 调用核心函数
    result = trim_csv_files(config.RAW_CSV_DIR, config.NAME_LIST, config)
    print("\n✅ Step1完成")
    print(f"主数据根目录: {result.get('main_root')}")
    print(f"子数据根目录: {result.get('sub_root')}")
    print(f"日志文件: {result.get('log_file')}")
    print(f"扫描CSV数量: {result.get('num_files')}")
