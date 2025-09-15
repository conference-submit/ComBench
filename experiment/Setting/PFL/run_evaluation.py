#!/usr/bin/env python3
"""
PFL评估运行脚本
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any

from pfl_evaluator import PFLEvaluator
from model.register import get_available_models


def setup_environment(config: Dict[str, Any]):
    """设置环境变量"""
    env_config = config.get('environment', {})
    
    # 设置必需的环境变量
    required_vars = env_config.get('required_env_vars', {})
    for var_name, var_value in required_vars.items():
        os.environ[var_name] = str(var_value)
    
    # 设置可选的环境变量（如果未设置）
    optional_vars = env_config.get('optional_env_vars', {})
    for var_name, var_value in optional_vars.items():
        if var_name not in os.environ:
            os.environ[var_name] = str(var_value)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PFL评估系统')
    
    # 基本参数
    parser.add_argument('--data', type=str,
                        help='输入数据文件路径（JSONL格式）')
    parser.add_argument('--output', type=str, default='results',
                        help='输出目录（默认：results）')
    
    # 模型参数
    parser.add_argument('--model', type=str, 
                        help='要评估的模型名称（如果不指定，会列出可用模型）')
    parser.add_argument('--models', type=str, nargs='+',
                        help='要评估的多个模型名称')
    parser.add_argument('--list-models', action='store_true',
                        help='列出所有可用模型')
    
    # 评估参数
    parser.add_argument('--max-instances', type=int, default=None,
                        help='最大评估实例数（用于测试）')
    parser.add_argument('--similarity-threshold', type=float, default=0.8,
                        help='补丁相似度阈值（默认：0.8）')
    parser.add_argument('--enable-resume', action='store_true',
                        help='启用断点续跑模式（不添加日期后缀，跳过已处理的结果）')

    
    # 配置文件
    parser.add_argument('--config', type=str,
                        help='配置文件路径（JSON格式）')
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return {}


def _infer_dataset_name_from_path(data_path: str) -> str:
    """从数据路径推导数据集名称"""
    import re
    from pathlib import Path
    
    # 从文件名中提取项目名称
    file_name = Path(data_path).name
    
    # 匹配模式：{project}_compiler_errors_extracted.json
    match = re.search(r'([^_]+)_compiler_errors_extracted\.json$', file_name)
    if match:
        project_name = match.group(1)
        return f"{project_name}_extracted"
    
    # 匹配模式：{project}_repair_analysis.jsonl
    match = re.search(r'([^_]+)_repair_analysis\.jsonl$', file_name)
    if match:
        project_name = match.group(1)
        return f"{project_name}_repair"
    
    # 匹配模式：{project}_{type}.json(l)
    match = re.search(r'([^_]+)_([^_]+)\.(json|jsonl)$', file_name)
    if match:
        project_name = match.group(1)
        data_type = match.group(2)
        return f"{project_name}_{data_type}"
    
    # 如果无法匹配，使用文件名（去掉扩展名）
    return Path(data_path).stem



def run_single_evaluation(data_path: str, model_name: str, 
                         max_instances: int = None, similarity_threshold: float = 0.8, 
                         repo_path: str = None, dataset_name: str = None, output_dir: str = 'results',
                         enable_resume: bool = False):
    """运行单个模型评估"""
    # 创建评估器
    evaluator = PFLEvaluator(similarity_threshold=similarity_threshold, output_dir=output_dir, enable_resume=enable_resume)
    
    # 检测数据格式并推导dataset_name（如果未提供）
    if not dataset_name:
        dataset_name = _infer_dataset_name_from_path(data_path)
    
    print(f"数据路径: {data_path}")
    print(f"数据集名称: {dataset_name}")
    print(f"仓库路径: {repo_path}")
    
    # 加载数据
    instances = evaluator.load_data(data_path)
    if not instances:
        print("没有加载到任何数据实例")
        return None
    
    # 限制实例数量
    if max_instances and max_instances < len(instances):
        instances = instances[:max_instances]
        print(f"限制实例数量为: {max_instances}")
    
    # 评估模型
    result = evaluator.evaluate_model(
        model_name, 
        instances, 
        data_path=data_path, 
        repo_path=repo_path,
        dataset_name=dataset_name
    )
    
    # 打印结果
    evaluator.print_evaluation_summary({model_name: result})
    
    return result


def run_batch_evaluation(data_path: str, model_names: List[str], 
                        max_instances: int = None, similarity_threshold: float = 0.8, 
                        repo_path: str = None, dataset_name: str = None, output_dir: str = 'results',
                        max_workers: int = None, enable_resume: bool = False):
    """运行批量模型评估"""
    # 创建评估器
    evaluator = PFLEvaluator(similarity_threshold=similarity_threshold, output_dir=output_dir, enable_resume=enable_resume)
    
    # 检测数据格式并推导dataset_name（如果未提供）
    if not dataset_name:
        dataset_name = _infer_dataset_name_from_path(data_path)
    
    print(f"数据路径: {data_path}")
    print(f"数据集名称: {dataset_name}")
    print(f"仓库路径: {repo_path}")
    
    # 加载数据
    instances = evaluator.load_data(data_path)
    if not instances:
        print("没有加载到任何数据实例")
        return None
    
    # 限制实例数量
    if max_instances and max_instances < len(instances):
        instances = instances[:max_instances]
        print(f"限制实例数量为: {max_instances}")
    
    # 批量评估
    results = evaluator.batch_evaluate(
        model_names, 
        instances, 
        repo_path=repo_path,
        dataset_name=dataset_name,
        max_workers=max_workers
    )
    
    # 打印结果
    evaluator.print_evaluation_summary(results)
    
    return results


def run_config_evaluation(config_path: str):
    """基于配置文件运行评估"""
    config = load_config(config_path)
    if not config:
        return None
    
    # 设置环境变量
    setup_environment(config)
    
    # 获取配置参数
    datasets = config.get('datasets', [])
    models = config.get('models', [])
    settings = config.get('settings', {})
    
    # 从配置文件读取参数
    config_output_dir = settings.get('output_dir', 'results')
    max_instances = settings.get('max_instances')
    similarity_threshold = settings.get('similarity_threshold', 0.8)
    # 仅使用 max_workers（并行度），不存在则为 None，评估器内部按 CPU 推断
    configured_workers = settings.get('max_workers')
    # 断点续跑相关参数
    enable_resume = settings.get('enable_resume', False)

    
    all_results = {}
    
    # 遍历每个数据集
    for dataset in datasets:
        print(f"开始评估数据集: {dataset.get('name', 'unknown')}")
        dataset_name = dataset.get('name', 'unknown')
        data_path = dataset.get('path')
        repo_path = dataset.get('repo_path')
        
        if not data_path or not os.path.exists(data_path):
            print(f"数据集路径不存在: {data_path}")
            continue
        
        # 检查repo_path是否存在
        if repo_path and not os.path.exists(repo_path):
            print(f"仓库路径不存在: {repo_path}，将使用数据路径推导")
            repo_path = None
        
        # 获取要评估的模型
        dataset_models = dataset.get('models', models)
        model_names = [model.get('name') for model in dataset_models if model.get('enabled', True)]
        
        if not model_names:
            continue
        
        # 运行批量评估
        results = run_batch_evaluation(
            data_path, 
            model_names, 
            max_instances, 
            similarity_threshold, 
            repo_path=repo_path,
            dataset_name=dataset_name,
            output_dir=config_output_dir,
            max_workers=configured_workers,
            enable_resume=enable_resume
        )
        
        if results:
            all_results[dataset_name] = results
    
    return all_results


def main():
    """主函数"""
    args = parse_args()
    
    # 基于配置文件运行
    if args.config:
        run_config_evaluation(args.config)
        return
    
    # 如果没有配置文件，使用默认配置
    default_config = {
        "environment": {
            "required_env_vars": {
                "USE_CUSTOM_OPENAI_API": "true"
            },
            "optional_env_vars": {}
        }
    }
    setup_environment(default_config)
    
    # 列出可用模型
    if args.list_models:
        available_models = get_available_models()
        print("可用模型:")
        for model in available_models:
            print(f"  - {model}")
        return
    
    # 检查是否需要数据文件
    if not args.data and not args.list_models and not args.config:
        print("需要指定数据文件或使用--list-models查看可用模型")
        return
    
    # 如果指定了数据文件，检查其存在性
    if args.data and not os.path.exists(args.data):
        print(f"数据文件不存在: {args.data}")
        return
    
    # 确定要评估的模型
    if args.models:
        # 批量评估
        if not args.data:
            print("批量评估需要指定数据文件")
            return
        run_batch_evaluation(args.data, args.models, args.max_instances, args.similarity_threshold, output_dir=args.output, enable_resume=args.enable_resume)
    elif args.model:
        # 单个评估
        if not args.data:
            print("单个模型评估需要指定数据文件")
            return
        run_single_evaluation(args.data, args.model, args.max_instances, args.similarity_threshold, output_dir=args.output, enable_resume=args.enable_resume)
    else:
        # 没有指定模型，列出可用模型
        available_models = get_available_models()
        print("请指定要评估的模型。可用模型:")
        for model in available_models:
            print(f"  - {model}")
        print("\n使用 --model 指定单个模型，或 --models 指定多个模型")
        print("使用 --list-models 查看所有可用模型")


if __name__ == "__main__":
    main() 