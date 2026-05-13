"""
Step3: STFT时频变换
从CSV生成NPY格式频域特征
运行: python scripts/step3_stft.py
参数修改: configs/config.py中修改STEP3_INPUT_DIR、STFT参数
"""
import sys
import os
# 动态获取项目根目录并添加到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from configs import config
from src.data.stft_processor import batch_process_stft

if __name__ == "__main__":
    print("=" * 60)
    print("Step3: STFT时频变换")
    print("=" * 60)
    print(f"项目根目录: {project_root}")
    print(f"输入目录: {config.STEP3_INPUT_DIR}")
    print(f"STFT参数: nperseg={config.STFT_NPERSEG}, fmax={config.STFT_FMAX}Hz")
    print(f"输出shape: {config.STFT_SHAPE}")
    print("=" * 60)

    output_dir = batch_process_stft(
        raw_data_root=config.STEP3_INPUT_DIR,
        output_root=config.STEP3_OUTPUT_DIR,
        include_dir_keywords=getattr(config, 'INCLUDE_DIR_KEYWORDS', []),
        include_keywords_mode=getattr(config, 'INCLUDE_KEYWORDS_MODE', 'OR'),
        exclude_dir_keywords=getattr(config, 'EXCLUDE_DIR_KEYWORDS', []),
        match_mode=getattr(config, 'KEYWORD_MATCH_MODE', 'path')
    )

    print(f"\n✅ Step3完成 -> {output_dir}")
