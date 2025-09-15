#!/usr/bin/env python3
"""
脚本用于根据instance_index和error_index匹配两个JSONL文件的数据
从claude-3-7-sonnet_detailed_results.jsonl和openssl_repair_analysis.jsonl中提取数据
"""

import json
import sys
from typing import Dict, List, Any, Optional

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def find_matching_analysis_record(analysis_data: List[Dict[str, Any]], 
                                instance_index: int, 
                                error_index: int) -> Optional[Dict[str, Any]]:
    """
    在analysis数据中查找匹配的记录
    根据instance_index定位到行，根据error_index定位到error_lines数组中的项
    """
    # instance_index是从1开始的，需要转换为0基索引
    if instance_index <= 0 or instance_index > len(analysis_data):
        return None
    
    analysis_record = analysis_data[instance_index - 1]
    
    # 检查error_index是否在error_lines数组范围内
    error_lines = analysis_record.get('error_lines', [])
    if error_index <= 0 or error_index > len(error_lines):
        return None
    
    return analysis_record

def extract_compilation_related_paths(analysis_record: Dict[str, Any], 
                                    error_index: int) -> Dict[str, Any]:
    """
    从compilation_related_paths_details数组中提取对应error_index的项
    """
    compilation_details = analysis_record.get('compilation_related_paths_details', [])
    if error_index <= 0 or error_index > len(compilation_details):
        return analysis_record.get('compilation_related_paths', {})
    
    return compilation_details[error_index - 1]

def process_data(claude_file: str, analysis_file: str, output_file: str):
    """处理数据并生成输出文件"""
    
    # 加载数据
    print("正在加载claude-3-7-sonnet_detailed_results.jsonl...")
    claude_data = load_jsonl(claude_file)
    
    print("正在加载openssl_repair_analysis.jsonl...")
    analysis_data = load_jsonl(analysis_file)
    
    print(f"Claude数据记录数: {len(claude_data)}")
    print(f"Analysis数据记录数: {len(analysis_data)}")
    
    # 处理数据
    output_records = []
    matched_count = 0
    unmatched_count = 0
    
    for claude_record in claude_data:
        instance_index = claude_record.get('instance_index')
        error_index = claude_record.get('error_index')
        
        if not instance_index or not error_index:
            print(f"警告: 跳过缺少instance_index或error_index的记录: {claude_record}")
            unmatched_count += 1
            continue
        
        # 查找匹配的analysis记录
        analysis_record = find_matching_analysis_record(analysis_data, instance_index, error_index)
        
        if not analysis_record:
            print(f"警告: 未找到匹配的analysis记录 - instance_index: {instance_index}, error_index: {error_index}")
            unmatched_count += 1
            continue
        
        # 提取所需字段
        error_lines = analysis_record.get('error_lines', [])
        error_details = analysis_record.get('error_details', [])
        
        # 根据error_index获取对应的error_line和error_detail
        error_line = error_lines[error_index - 1] if error_index <= len(error_lines) else ""
        error_detail = error_details[error_index - 1] if error_index <= len(error_details) else ""
        
        # 提取compilation_related_paths
        compilation_related_path = extract_compilation_related_paths(analysis_record, error_index)
        
        # 构建输出记录
        output_record = {
            "instance_index": instance_index,
            "error_index": error_index,
            "failure_commit": analysis_record.get('failure_commit', ''),
            "repair_commit": analysis_record.get('repair_commit', ''),
            "error_line": error_line,
            "error_detail": error_detail,
            "compilation_related_path": compilation_related_path,
            "original_error_lines": error_lines,  # 保留原始error_lines数组
            "workflow_name": analysis_record.get('workflow_name', ''),
            "job_name": analysis_record.get('job_name', [])
        }
        
        output_records.append(output_record)
        matched_count += 1
    
    # 保存结果
    print(f"正在保存结果到 {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in output_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"处理完成!")
    print(f"匹配成功: {matched_count} 条记录")
    print(f"匹配失败: {unmatched_count} 条记录")
    print(f"输出文件: {output_file}")

def main():
    """主函数"""
    if len(sys.argv) != 4:
        print("用法: python match_data.py <claude_file> <analysis_file> <output_file>")
        print("示例: python match_data.py claude-3-7-sonnet_detailed_results.jsonl openssl_repair_analysis.jsonl matched_results.jsonl")
        sys.exit(1)
    
    claude_file = sys.argv[1]
    analysis_file = sys.argv[2]
    output_file = sys.argv[3]
    
    try:
        process_data(claude_file, analysis_file, output_file)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
