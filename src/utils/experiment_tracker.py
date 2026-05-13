"""
实验追踪模块 - 增强版
为每次训练提供完整的可追溯性支持
- 实验唯一ID
- 配置快照保存
- 超参数记录到CSV
- Checkpoint关联
- 实验元数据
"""
import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Any
import torch

class ExperimentTracker:
    """实验追踪器"""

    def __init__(self, config, custom_id=None):
        """
        初始化实验追踪器

        Args:
            config: 配置对象
            custom_id: 自定义实验ID（可选，不提供则自动生成）
        """
        self.config = config

        # 生成唯一实验ID
        if custom_id:
            self.experiment_id = custom_id
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            hash_str = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:4]
            self.experiment_id = f"exp_{timestamp}_{hash_str}"

        # 创建实验专属目录
        self.exp_dir = os.path.join(config.EVAL_OUTPUT_DIR, self.experiment_id)
        os.makedirs(self.exp_dir, exist_ok=True)

        # 记录实验开始时间
        self.start_time = datetime.now()
        self.end_time = None

        # 元数据字典
        self.metadata = {
            'experiment_id': self.experiment_id,
            'start_time': self.start_time.isoformat(),
            'config_snapshot': {},
            'hyperparameters': {},
            'model_architecture': {},
            'data_info': {},
            'training_info': {},
            'results': {}
        }

    def save_config_snapshot(self):
        """保存完整配置快照"""
        config_dict = {}

        # 提取所有大写的配置项（常量）
        for attr in dir(self.config):
            if attr.isupper() and not attr.startswith('_'):
                value = getattr(self.config, attr)
                # 转换为可JSON序列化的格式
                if hasattr(value, 'tolist'):  # numpy数组
                    value = value.tolist()
                config_dict[attr] = value

        # 保存为JSON
        config_path = os.path.join(self.exp_dir, 'config_snapshot.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        self.metadata['config_snapshot'] = config_dict
        print(f"✅ 配置快照已保存: {config_path}")

        return config_dict

    def extract_hyperparameters(self):
        """提取关键超参数用于CSV记录"""
        hyperparams = {
            'batch_size': self.config.BATCH_SIZE,
            'learning_rate': self.config.LEARNING_RATE,
            'weight_decay': self.config.WEIGHT_DECAY,
            'epochs': self.config.EPOCHS,
            'lr_step_size': self.config.LR_STEP_SIZE,
            'lr_gamma': self.config.LR_GAMMA,
            'valid_ratio': self.config.VALID_RATIO,
            'random_seed': self.config.RANDOM_SEED,
        }

        self.metadata['hyperparameters'] = hyperparams
        return hyperparams

    def extract_model_architecture(self):
        """提取模型架构参数"""
        model_arch = {
            'num_classes': self.config.NUM_CLASSES,
            'num_channels': self.config.NUM_CHANNELS,
            'gnn_hidden_dim': self.config.GNN_HIDDEN_DIM,
            'transformer_nhead': self.config.TRANSFORMER_NHEAD,
            'transformer_layers': self.config.TRANSFORMER_LAYERS,
            'transformer_dim_feedforward': self.config.TRANSFORMER_DIM_FEEDFORWARD,
        }

        self.metadata['model_architecture'] = model_arch
        return model_arch

    def record_data_info(self, train_size, test_size):
        """记录数据集信息"""
        data_info = {
            'data_root': self.config.DATA_ROOT,
            'train_size': train_size,
            'test_size': test_size,
            'total_size': train_size + test_size,
            'enable_normalize': self.config.ENABLE_NORMALIZE,
            'enable_augment': self.config.ENABLE_AUGMENT,
        }

        if self.config.ENABLE_AUGMENT:
            data_info['aug_prob'] = self.config.AUG_PROB
            data_info['aug_noise_std'] = self.config.AUG_NOISE_STD

        self.metadata['data_info'] = data_info
        return data_info

    def record_training_info(self, total_epochs, best_epoch, best_val_acc):
        """记录训练过程信息"""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()

        training_info = {
            'total_epochs': total_epochs,
            'best_epoch': best_epoch,
            'best_val_acc': best_val_acc,
            'duration_seconds': duration,
            'duration_minutes': duration / 60,
            'end_time': self.end_time.isoformat(),
        }

        self.metadata['training_info'] = training_info
        return training_info

    def record_results(self, metrics):
        """记录评估结果"""
        self.metadata['results'] = metrics
        return metrics

    def save_metadata(self):
        """保存实验元数据"""
        metadata_path = os.path.join(self.exp_dir, 'experiment_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        print(f"✅ 实验元数据已保存: {metadata_path}")
        return metadata_path

    def get_checkpoint_path(self, val_acc):
        """生成带实验ID的checkpoint路径"""
        checkpoint_dir = self.config.CHECKPOINT_DIR
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 格式：exp_20251219_120530_a7f3_best_acc0.8750.pth
        checkpoint_name = f"{self.experiment_id}_best_acc{val_acc:.4f}.pth"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

        return checkpoint_path

    def get_csv_path(self, suffix='eval'):
        """生成CSV文件路径（实验专属目录）"""
        csv_name = f"results_{suffix}.csv"
        csv_path = os.path.join(self.exp_dir, csv_name)
        return csv_path

    def generate_summary_report(self):
        """生成实验摘要报告（文本格式）"""
        report_path = os.path.join(self.exp_dir, 'experiment_summary.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write(f"实验报告 - {self.experiment_id}\n")
            f.write("="*70 + "\n\n")

            f.write("## 基本信息\n")
            f.write(f"实验ID: {self.experiment_id}\n")
            f.write(f"开始时间: {self.metadata['start_time']}\n")
            if self.end_time:
                f.write(f"结束时间: {self.end_time.isoformat()}\n")
                f.write(f"耗时: {self.metadata['training_info'].get('duration_minutes', 0):.2f} 分钟\n")
            f.write("\n")

            f.write("## 超参数配置\n")
            for key, value in self.metadata['hyperparameters'].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

            f.write("## 模型架构\n")
            for key, value in self.metadata['model_architecture'].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

            f.write("## 数据集信息\n")
            for key, value in self.metadata['data_info'].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

            if 'results' in self.metadata and self.metadata['results']:
                f.write("## 评估结果\n")
                for key, value in self.metadata['results'].items():
                    f.write(f"{key}: {value:.4f}\n")
                f.write("\n")

            f.write("="*70 + "\n")

        print(f"✅ 实验摘要已保存: {report_path}")
        return report_path

    def get_full_info_for_csv(self):
        """获取用于CSV保存的完整信息行"""
        info = {
            'experiment_id': self.experiment_id,
            'timestamp': self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # 添加超参数
        info.update(self.metadata['hyperparameters'])

        # 添加模型架构关键参数
        info['gnn_hidden'] = self.metadata['model_architecture']['gnn_hidden_dim']
        info['transformer_layers'] = self.metadata['model_architecture']['transformer_layers']

        # 添加数据集大小
        info['train_size'] = self.metadata['data_info'].get('train_size', 0)
        info['test_size'] = self.metadata['data_info'].get('test_size', 0)

        # 添加训练信息
        if 'training_info' in self.metadata and self.metadata['training_info']:
            info['best_epoch'] = self.metadata['training_info'].get('best_epoch', 0)
            info['duration_min'] = self.metadata['training_info'].get('duration_minutes', 0)

        return info


def load_experiment_metadata(experiment_id, eval_output_dir="evaluation_results"):
    """
    根据实验ID加载完整的实验元数据

    Args:
        experiment_id: 实验ID
        eval_output_dir: 评估结果根目录

    Returns:
        metadata: 实验元数据字典
    """
    exp_dir = os.path.join(eval_output_dir, experiment_id)
    metadata_path = os.path.join(exp_dir, 'experiment_metadata.json')

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"未找到实验 {experiment_id} 的元数据文件")

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    return metadata


def compare_experiments(experiment_ids, eval_output_dir="evaluation_results"):
    """
    对比多个实验的配置和结果

    Args:
        experiment_ids: 实验ID列表
        eval_output_dir: 评估结果根目录

    Returns:
        comparison: 对比表格（字典列表）
    """
    comparison = []

    for exp_id in experiment_ids:
        try:
            metadata = load_experiment_metadata(exp_id, eval_output_dir)

            row = {
                'experiment_id': exp_id,
                'batch_size': metadata['hyperparameters']['batch_size'],
                'learning_rate': metadata['hyperparameters']['learning_rate'],
                'epochs': metadata['hyperparameters']['epochs'],
                'gnn_hidden': metadata['model_architecture']['gnn_hidden_dim'],
                'transformer_layers': metadata['model_architecture']['transformer_layers'],
            }

            # 添加结果指标
            if 'results' in metadata and metadata['results']:
                row.update(metadata['results'])

            comparison.append(row)

        except Exception as e:
            print(f"⚠️  加载实验 {exp_id} 失败: {e}")

    return comparison
