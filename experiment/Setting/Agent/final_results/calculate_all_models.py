import json
import os
import glob
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))
from path_config import get_path

# 添加编译错误模块路径
compilation_error_path = str(get_path('compilation_error_data'))
if compilation_error_path not in sys.path:
    sys.path.append(compilation_error_path)

from error_classifier import ErrorClassifier

# 定义标签和类型映射
labels = ['declaration', 'type', 'member', 'syntax', 'function', 'template', 'semantic']
type_map = {
    'declaration': 0,
    'type': 1,
    'member': 2,
    'syntax': 3,
    'function': 4,
    'template': 5,
    'semantic': 6,
}

def calculate_model_stats(file_path):
    """计算单个模型的统计数据"""
    cs_values = [0] * 7
    sc_values = [0] * 7
    em_values = [0] * 7
    unknown_types = 0  # 统计未知类型的个数
    
    # 创建错误分类器
    classifier = ErrorClassifier()
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                obj = json.loads(line)
                # 使用ErrorClassifier获取main_type
                if 'pfl_patches' in obj and obj['pfl_patches']:
                    # 从pfl_patches中获取错误信息
                    error_message = obj['pfl_patches'][0] if isinstance(obj['pfl_patches'], list) else str(obj['pfl_patches'])
                    main_type_enum, _ = classifier.identify_error_type(str(error_message))
                    main_type = main_type_enum.value
                else:
                    # 如果没有pfl_patches，使用原有的error_classification
                    main_type = obj['error_classification']['main_type']
                
                idx = type_map.get(main_type.replace('_error', ''), None)
                if idx is None:
                    unknown_types += 1
                    continue
                
                cs_values[idx] += 1
                intent = obj['llm_analysis']['intent_consistency']
                contained = obj['string_match_analysis']['patches_contained_in_ground_truth']
                
                if intent in ('consistent', 'partially_consistent'):
                    sc_values[idx] += 1
                if contained:
                    em_values[idx] += 1
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
    
    return {
        'cs_values': cs_values,
        'sc_values': sc_values,
        'em_values': em_values,
        'unknown_types': unknown_types
    }

def main():
    # 获取所有项目的目录
    base_dir = "."
    project_dirs = [d for d in os.listdir(base_dir) if d.startswith('results-') and os.path.isdir(os.path.join(base_dir, d))]
    
    # 存储所有模型的结果
    all_models_data = {}
    
    for project_dir in project_dirs:
        project_path = os.path.join(base_dir, project_dir, 'sem_eq')
        if not os.path.exists(project_path):
            continue
            
        print(f"\n处理项目: {project_dir}")
        
        # 查找所有comparison.jsonl文件
        jsonl_files = glob.glob(os.path.join(project_path, '*_comparison.jsonl'))
        
        for jsonl_file in jsonl_files:
            # 提取模型名称
            filename = os.path.basename(jsonl_file)
            model_name = filename.replace('_comparison.jsonl', '')
            
            print(f"  处理模型: {model_name}")
            
            # 计算统计数据
            stats = calculate_model_stats(jsonl_file)
            if stats is not None:
                # 如果模型已存在，累加数据；否则直接赋值
                if model_name in all_models_data:
                    # 累加各个metric的值
                    for i in range(7):
                        all_models_data[model_name]['cs_values'][i] += stats['cs_values'][i]
                        all_models_data[model_name]['sc_values'][i] += stats['sc_values'][i]
                        all_models_data[model_name]['em_values'][i] += stats['em_values'][i]
                    all_models_data[model_name]['unknown_types'] += stats['unknown_types']
                else:
                    all_models_data[model_name] = stats
    
    # 输出结果
    print("\n" + "="*80)
    print("所有模型的统计结果:")
    print("="*80)
    
    for model_name, stats in all_models_data.items():
        print(f"\n# {model_name}")
        print(f"overall_values = {[cs + sc + em for cs, sc, em in zip(stats['cs_values'], stats['sc_values'], stats['em_values'])]}")
        print(f"cs_values = {stats['cs_values']}")
        print(f"sc_values = {stats['sc_values']}")
        print(f"em_values = {stats['em_values']}")
        print(f"unknown_types = {stats['unknown_types']}")
    
    # 计算所有模型的汇总数据
    print("\n" + "="*80)
    print("所有模型的汇总统计:")
    print("="*80)
    
    if all_models_data:
        # 初始化汇总数组
        total_cs = [0] * 7
        total_sc = [0] * 7
        total_em = [0] * 7
        total_unknown_types = 0
        
        # 累加所有模型的数据
        for stats in all_models_data.values():
            for i in range(7):
                total_cs[i] += stats['cs_values'][i]
                total_sc[i] += stats['sc_values'][i]
                total_em[i] += stats['em_values'][i]
            total_unknown_types += stats['unknown_types']
        
        # 计算总体值
        total_overall = [cs + sc + em for cs, sc, em in zip(total_cs, total_sc, total_em)]
        
        print(f"\n# 所有模型汇总")
        print(f"overall_values = {total_overall}")
        print(f"cs_values = {total_cs}")
        print(f"sc_values = {total_sc}")
        print(f"em_values = {total_em}")
        print(f"total_unknown_types = {total_unknown_types}")
        
        # 输出CSV格式便于复制
        print("\n" + "="*80)
        print("CSV格式 (便于复制到Excel):")
        print("="*80)
        print("Model,Declaration,Type,Member,Syntax,Function,Template,Semantic")
        for model_name, stats in all_models_data.items():
            overall = [cs + sc + em for cs, sc, em in zip(stats['cs_values'], stats['sc_values'], stats['em_values'])]
            print(f"{model_name}," + ",".join(map(str, overall)))
        
        print(f"Total," + ",".join(map(str, total_overall)))

if __name__ == "__main__":
    main()