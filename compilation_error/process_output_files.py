#!/usr/bin/env python3
"""
Process all compilation error files in the output folder

Requirements:
1. Filter all compilation workflows not specified in project_configs.json
2. Filter all warning as error types and do not save
3. Output an array representing the type of each line in error_lines
4. Merge identical errors from different jobs of the same commit_sha and same workflow
5. Output results to compilation_error directory
"""

import json
import os
import sys
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import time

# Import error classifier
from error_classifier import ErrorClassifier, ErrorType

def load_project_configs(config_file: str = "../project_configs.json") -> Dict:
    """Load project configuration file"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error: Unable to load project configuration file {config_file}: {e}")
        return {}

def load_compiler_errors_data(file_path: str) -> List[Dict]:
    """Load compilation error data, prioritize reading from output folder, read from backup if not exists, and fix encoding issues"""
    file_name = os.path.basename(file_path)
    output_file_path = os.path.join("output", file_name)
    backup_file_path = file_path
    
    # If output file exists, only read output file
    if os.path.exists(output_file_path):
        print(f"  Reading from output directory: {output_file_path}")
        try:
            with open(output_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            errors = data.get('compiler_errors', [])
            
            # Fix encoding issues
            print(f"  Starting to fix encoding issues...")
            for error_record in errors:
                error_lines = error_record.get('error_lines', [])
                fixed_error_lines = [fix_encoding_issues(line) for line in error_lines]
                error_record['error_lines'] = fixed_error_lines
            print(f"  Encoding issues fixed")
            
            # Sort by created_at time from recent to old
            errors.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            print(f"  Sorted by time: from newest to oldest")
            print(f"  Read from output: {len(errors)} records")
            
            return errors
        except Exception as e:
            print(f"  读取output文件失败: {e}")
            return []
    
    # 如果output文件不存在，从backup目录读取
    if os.path.exists(backup_file_path):
        print(f"  从backup目录读取: {backup_file_path}")
        try:
            with open(backup_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            errors = data.get('compiler_errors', [])
            
            # Fix encoding issues
            print(f"  Starting to fix encoding issues...")
            for error_record in errors:
                error_lines = error_record.get('error_lines', [])
                fixed_error_lines = [fix_encoding_issues(line) for line in error_lines]
                error_record['error_lines'] = fixed_error_lines
            print(f"  Encoding issues fixed")
            
            # Sort by created_at time from recent to old
            errors.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            print(f"  Sorted by time: from newest to oldest")
            print(f"  从backup读取: {len(errors)} 条记录")
            
            return errors
        except Exception as e:
            print(f"  读取backup文件失败: {e}")
            return []
    else:
        print(f"  文件不存在: {backup_file_path}")
        return []

def filter_by_workflow(compiler_errors: List[Dict], allowed_workflows: List[str]) -> List[Dict]:
    """根据允许的workflow过滤编译错误"""
    filtered_errors = []
    
    for error_record in compiler_errors:
        workflow_name = error_record.get('workflow_name', '')
        
        # 检查是否在允许的workflow列表中
        if workflow_name in allowed_workflows:
            filtered_errors.append(error_record)
    
    return filtered_errors

def filter_by_error_type(compiler_errors: List[Dict], classifier: ErrorClassifier) -> List[Dict]:
    """过滤掉warning as error类型的错误"""
    filtered_errors = []
    
    for error_record in compiler_errors:
        error_lines = error_record.get('error_lines', [])
        
        # 检查是否包含warning as error
        has_werror = False
        for error_line in error_lines:
            main_type, _ = classifier.identify_error_type(error_line)
            if main_type == ErrorType.WERROR:
                has_werror = True
                break
        
        # 如果不包含warning as error，则保留
        if not has_werror:
            filtered_errors.append(error_record)
    
    return filtered_errors

def fix_encoding_issues(text):
    """修复编码问题：将日志中用 ?...?, 表示的引号内容统一替换为单引号包裹内容。

    覆盖以下情况：
    - ?dynamic_cast? -> 'dynamic_cast'
    - ?x? -> 'x'
    - ?void*? -> 'void*'
    - ?struct rocksdb::KeyLockWaiter*? -> 'struct rocksdb::KeyLockWaiter*'
    - ?class std::unordered_map<...>*? -> 'class std::unordered_map<...>*'

    同时保留对可能残留的单侧问号的兜底替换。
    """
    import re

    # 1) 先处理最通用的成对问号：允许内容内含空格与符号，但不包含问号本身
    text = re.sub(r"\?([^?\n]+)\?", r"'\1'", text)

    # 2) 兜底：若仍有单侧的问号误当作引号出现在单词/符号边界，替换为单引号
    #    前置问号：后面紧跟非空白字符
    text = re.sub(r"\?(?=\S)", "'", text)
    #    后置问号：前面是非空白字符
    text = re.sub(r"(?<=\S)\?", "'", text)

    return text

def merge_similar_errors(compiler_errors: List[Dict]) -> List[Dict]:
    """
    合并同一个commit_sha和workflow的不同job中的相同错误
    将job_name和job_id改成数组
    只有相同或相似的错误才合并
    """
    from difflib import SequenceMatcher
    
    def similarity(a, b):
        """计算两个字符串的相似度"""
        return SequenceMatcher(None, a, b).ratio()
    
    def are_error_lines_similar(line1, line2):
        """比较两个错误行是否相似，忽略路径前缀差异，并忽略所有引号(')和问号(?)"""
        import re

        def normalize(text: str) -> str:
            # 忽略所有单引号和问号
            return re.sub(r"[\'?]", "", text)

        nline1 = normalize(line1)
        nline2 = normalize(line2)

        # 如果归一化后完全相等，直接返回True
        if nline1 == nline2:
            return True

        # 解析错误行格式：path:line:column: error: message
        # 使用正则表达式匹配
        pattern = r'^(.+?):(\d+):(\d+):\s*error:\s*(.+)$'

        match1 = re.match(pattern, nline1)
        match2 = re.match(pattern, nline2)

        if not match1 or not match2:
            # 如果格式不匹配，使用归一化后的原始比较
            return nline1 == nline2

        # 提取各部分
        path1, line_num1, col1, message1 = match1.groups()
        path2, line_num2, col2, message2 = match2.groups()

        # 比较行号、列号和错误消息
        if line_num1 != line_num2 or col1 != col2 or message1 != message2:
            return False

        # 比较路径后缀（忽略前缀差异）
        # 提取路径的最后部分进行比较
        path1_parts = path1.split('/')
        path2_parts = path2.split('/')

        # 从后往前比较，找到相同的后缀
        min_len = min(len(path1_parts), len(path2_parts))
        for i in range(1, min_len + 1):
            suffix1 = '/'.join(path1_parts[-i:])
            suffix2 = '/'.join(path2_parts[-i:])
            if suffix1 == suffix2:
                return True

        # 如果路径后缀也不匹配，返回False
        return False
    
    def are_errors_similar(error1, error2, similarity_threshold=0.8):
        """判断两个错误是否相似"""
        # 只比较error_lines，要求完全匹配
        error_lines1 = error1.get('error_lines', [])
        error_lines2 = error2.get('error_lines', [])
        
        # 如果error_lines数量不同，认为不相似
        if len(error_lines1) != len(error_lines2):
            return False
        
        # 如果error_lines为空，只有都为空才认为相似
        if len(error_lines1) == 0:
            return len(error_lines2) == 0
        
        # 比较每一行，忽略路径前缀差异
        for line1, line2 in zip(error_lines1, error_lines2):
            if not are_error_lines_similar(line1, line2):
                return False
        
        return True
    
    # 按commit_sha和workflow分组
    grouped_errors = defaultdict(list)
    
    for error in compiler_errors:
        commit_sha = error.get('commit_sha', '')
        workflow_name = error.get('workflow_name', '')
        key = (commit_sha, workflow_name)
        grouped_errors[key].append(error)
    
    merged_errors = []
    
    for (commit_sha, workflow_name), errors in grouped_errors.items():
        if len(errors) == 1:
            # 只有一个错误，直接添加
            error = errors[0].copy()
            # 将job_name和job_id转换为数组
            error['job_name'] = [error.get('job_name', '')]
            error['job_id'] = [error.get('job_id', '')]
            merged_errors.append(error)
        else:
            # 多个错误，需要判断相似度后再合并
            # 将相似的错误分组
            similar_groups = []
            processed_indices = set()
            
            for i, error1 in enumerate(errors):
                if i in processed_indices:
                    continue
                
                # 创建新的相似组
                similar_group = [error1]
                processed_indices.add(i)
                
                # 查找与error1相似的其他错误
                for j, error2 in enumerate(errors[i+1:], i+1):
                    if j in processed_indices:
                        continue
                    
                    if are_errors_similar(error1, error2):
                        similar_group.append(error2)
                        processed_indices.add(j)
                
                similar_groups.append(similar_group)
            
            # 处理每个相似组
            for similar_group in similar_groups:
                if len(similar_group) == 1:
                    # 组内只有一个错误，直接添加
                    error = similar_group[0].copy()
                    error['job_name'] = [error.get('job_name', '')]
                    error['job_id'] = [error.get('job_id', '')]
                    
                    
                    merged_errors.append(error)
                else:
                    # 组内有多个相似错误，需要合并
                    # 使用第一个错误作为基础
                    base_error = similar_group[0].copy()
                    
                    # 收集所有job信息
                    job_names = []
                    job_ids = []
                    
                    all_error_details = []
                    
                    for error in similar_group:
                        job_name = error.get('job_name', '')
                        job_id = error.get('job_id', '')
                        workflow_id = error.get('workflow_id', '')
                        created_at = error.get('created_at', '')
                        

                        if job_name not in job_names:
                            job_names.append(job_name)
                        
                        if job_id not in job_ids:
                            job_ids.append(job_id)
                        
                        
                        # 选择长度更长的error_lines和error_details
                        error_lines = error.get('error_lines', [])
                        error_details = error.get('error_details', [])
                        
                        current_details_length = sum(len(detail) for detail in error_details)
                        existing_details_length = sum(len(detail) for detail in all_error_details)
                        
                        # 如果当前错误的error_details更长，替换所有error_details
                        if current_details_length > existing_details_length:
                            all_error_details = error_details.copy()
                    
                    # 更新基础错误
                    base_error['job_name'] = job_names
                    base_error['job_id'] = job_ids
                    base_error['error_lines'] = similar_group[0].get('error_lines', [])  # 使用第一个错误的error_lines
                    base_error['error_details'] = similar_group[0].get('error_details', []) 
                    
                    # 移除重新计算错误类型的逻辑，避免重复分类
                    # 错误类型将在后续的classify_error_lines函数中统一计算
                    
                    merged_errors.append(base_error)
    
    return merged_errors

def deduplicate_by_error_lines(compiler_errors: List[Dict]) -> List[Dict]:
    """
    对于不同commit但error_lines一致的情况，只保留时间更新的commit的所有记录
    注意：可能有多条记录因为不同job，所以保留时间更新的commit的所有job记录
    """
    from difflib import SequenceMatcher
    
    def are_error_lines_identical(error1, error2):
        """比较两个错误的error_lines是否完全一致"""
        error_lines1 = error1.get('error_lines', [])
        error_lines2 = error2.get('error_lines', [])
        
        if len(error_lines1) != len(error_lines2):
            return False
        
        # 编码问题已在第一步修复，直接比较
        return error_lines1 == error_lines2
    
    # 按error_lines分组
    error_lines_groups = defaultdict(list)
    
    for error in compiler_errors:
        # 编码问题已在第一步修复，直接使用error_lines作为key
        error_lines = error.get('error_lines', [])
        key = tuple(error_lines)  # 转换为tuple以便作为dict的key
        
        error_lines_groups[key].append(error)
    
    deduplicated_errors = []
    
    for error_lines_key, errors in error_lines_groups.items():
        if len(errors) == 1:
            # 只有一个错误，直接添加
            deduplicated_errors.append(errors[0])
        else:
            # 多个错误，需要按时间排序，保留最新的commit的所有记录
            # 按created_at时间排序，从最新到最旧
            sorted_errors = sorted(errors, key=lambda x: x.get('created_at', ''), reverse=True)
            
            # 获取最新的commit_sha
            latest_commit_sha = sorted_errors[0].get('commit_sha', '')
            latest_created_at = sorted_errors[0].get('created_at', '')
            
            print(f"  发现相同error_lines的多个commit:")
            for error in sorted_errors:
                commit_sha = error.get('commit_sha', '')
                created_at = error.get('created_at', '')
                job_names = error.get('job_name', [])
                if isinstance(job_names, str):
                    job_names = [job_names]
                print(f"    Commit: {commit_sha}, Time: {created_at}, Jobs: {job_names}")
            
            # 只保留最新commit的所有记录
            latest_commit_errors = [error for error in sorted_errors 
                                  if error.get('commit_sha', '') == latest_commit_sha]
            
            print(f"  保留最新commit {latest_commit_sha} 的 {len(latest_commit_errors)} 条记录")
            
            deduplicated_errors.extend(latest_commit_errors)
    
    return deduplicated_errors

def deduplicate_records(compiler_errors: List[Dict]) -> List[Dict]:
    """
    去除commit_sha、workflow_name、job_name相同的重复记录
    保留最新的记录（基于created_at时间戳）
    假设输入的job_name和job_id都是字符串格式
    """
    # 使用(commit_sha, workflow_name, job_name)作为唯一键
    unique_records = {}
    
    for error in compiler_errors:
        commit_sha = error.get('commit_sha', '')
        workflow_name = error.get('workflow_name', '')
        job_name = error.get('job_name', '')  # 假设是字符串
        
        if job_name:  # 确保job_name不为空
            key = (commit_sha, workflow_name, job_name)
            created_at = error.get('created_at', '')
            
            # 如果键不存在，或者当前记录更新，则更新记录
            if key not in unique_records or created_at > unique_records[key].get('created_at', ''):
                # 创建记录的副本
                record_copy = error.copy()
                unique_records[key] = record_copy
    
    return list(unique_records.values())

def classify_error_lines(compiler_errors: List[Dict], classifier: ErrorClassifier) -> Dict:
    """为每个错误记录分类error_lines"""
    results = {}
    
    for i, error_record in enumerate(compiler_errors):
        error_lines = error_record.get('error_lines', [])
        error_details = error_record.get('error_details', [])
        commit_sha = error_record.get('commit_sha', '') or error_record.get('failure_commit', '')
        workflow_name = error_record.get('workflow_name', '')
        job_names = error_record.get('job_name', [])
        if isinstance(job_names, str):
            job_names = [job_names]
        
        # 修复error_lines中的编码问题
        fixed_error_lines = []
        for error_line in error_lines:
            fixed_line = fix_encoding_issues(error_line)
            fixed_error_lines.append(fixed_line)
        
        # 修复error_details中的编码问题
        fixed_error_details = []
        for error_detail in error_details:
            fixed_detail = fix_encoding_issues(error_detail)
            fixed_error_details.append(fixed_detail)
        
        # 为每一行error_line分类
        error_line_types = []
        for error_line in fixed_error_lines:
            main_type, detailed_type = classifier.identify_error_type(error_line)
            
            # 使用主要类型（main type）
            error_line_types.append(main_type.value)
        
        # 保存结果
        results[i] = {
            'commit_sha': commit_sha,
            'workflow_name': workflow_name,
            'job_name': job_names,
            'error_lines': fixed_error_lines,
            'error_details': fixed_error_details,
            'error_line_types': error_line_types,
            'error_count': len(fixed_error_lines)
        }
    
    return results

def calculate_statistics(compiler_errors: List[Dict], project_name: str, allowed_workflows: List[str]) -> Dict:
    """计算编译错误的统计信息"""
    stats = {
        'total_errors': len(compiler_errors),
        'commits_with_errors': len(set(err.get('commit_sha', '') for err in compiler_errors if err.get('commit_sha', ''))),
        'workflows_with_errors': defaultdict(int),
        'error_types_summary': defaultdict(int),
        'error_types_summary_detailed': defaultdict(int),
        'jobs_with_errors': defaultdict(int)
    }
    
    # Import error classifier用于获取主类型
    from error_classifier import ErrorClassifier
    classifier = ErrorClassifier()
    
    for error in compiler_errors:
        # 统计workflow
        workflow_name = error.get('workflow_name', '')
        if workflow_name:
            stats['workflows_with_errors'][workflow_name] += 1
        
        # 统计job（处理数组格式）
        job_names = error.get('job_name', [])
        if isinstance(job_names, list):
            for job_name in job_names:
                if job_name and isinstance(job_name, str):
                    stats['jobs_with_errors'][job_name] += 1
        else:
            # 兼容旧格式
            job_name = job_names
            if job_name and isinstance(job_name, str):
                stats['jobs_with_errors'][job_name] += 1
        
        # 统计错误类型
        error_lines = error.get('error_lines', [])
        for error_line in error_lines:
            # Fix encoding issues
            fixed_error_line = fix_encoding_issues(error_line)
            main_type, detailed_type = classifier.identify_error_type(fixed_error_line)
            # 统计主类型
            stats['error_types_summary'][main_type.value] += 1
            # 统计详细类型
            stats['error_types_summary_detailed'][detailed_type.value] += 1
    
    # 转换为普通字典
    stats['workflows_with_errors'] = dict(stats['workflows_with_errors'])
    stats['error_types_summary'] = dict(stats['error_types_summary'])
    stats['error_types_summary_detailed'] = dict(stats['error_types_summary_detailed'])
    stats['jobs_with_errors'] = dict(stats['jobs_with_errors'])
    
    return stats

def update_statistics_csv(statistics: List[Dict], output_file: str = "detailed_repository_statistics.csv"):
    """更新统计CSV文件"""
    fieldnames = [
        '仓库名称', '错误数量', 'error_lines数量', '唯一commit数', '主要语言',
        '平均每commit错误数', '平均每错误行数', '主要工作流', '主要作业', '主要错误类型'
    ]
    
    try:
        # 按错误数量从多到少排序
        sorted_statistics = sorted(statistics, key=lambda x: x['error_count'], reverse=True)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for stat in sorted_statistics:
                row = {
                    '仓库名称': stat['repository_name'],
                    '错误数量': stat['error_count'],
                    'error_lines数量': stat['error_lines_count'],
                    '唯一commit数': stat['unique_commits'],
                    '主要语言': stat['main_language'],
                    '平均每commit错误数': stat['avg_errors_per_commit'],
                    '平均每错误行数': stat['avg_lines_per_error'],
                    '主要工作流': stat['main_workflow'],
                    '主要作业': stat['main_job'],
                    '主要错误类型': stat['main_error_type']
                }
                writer.writerow(row)
        
        print(f"统计文件已更新: {output_file} (按错误数量排序)")
        
    except Exception as e:
        print(f"更新CSV文件失败: {e}")

def process_backup_files():
    """处理backup文件夹中的所有文件"""
    
    # 加载项目配置
    project_configs = load_project_configs()
    if not project_configs:
        print("错误: 无法加载项目配置文件")
        return
    
    # 创建错误分类器
    classifier = ErrorClassifier()
    
    # 获取所有可能的JSON文件路径
    backup_dir = Path("backup_20250727_182717")
    output_dir = Path("output")
    
    # 收集所有文件路径（包括output和backup）
    json_files = []
    
    # 检查output目录
    if output_dir.exists():
        output_files = list(output_dir.glob("*_compiler_errors_extracted.json"))
        json_files.extend(output_files)
        print(f"在output目录找到 {len(output_files)} 个文件")
    
    # 检查backup目录
    if backup_dir.exists():
        backup_files = list(backup_dir.glob("*_compiler_errors_extracted.json"))
        json_files.extend(backup_files)
        print(f"在backup目录找到 {len(backup_files)} 个文件")
    else:
        print(f"警告: backup目录不存在: {backup_dir}")
    
    # 去重，按文件名分组
    unique_files = {}
    for file_path in json_files:
        file_name = file_path.name
        if file_name not in unique_files:
            unique_files[file_name] = file_path
        else:
            # 如果output目录的文件存在，优先使用output目录的文件
            current_file = unique_files[file_name]
            if output_dir.exists() and (output_dir / file_name).exists():
                unique_files[file_name] = output_dir / file_name
    
    json_files = list(unique_files.values())
    
    print(f"找到 {len(json_files)} 个编译错误文件")
    print("=" * 60)
    
    total_processed = 0
    total_filtered_workflow = 0
    total_filtered_werror = 0
    total_filtered_path = 0
    total_classified = 0
    total_merged = 0
    total_deduplicated = 0
    total_error_lines_deduplicated = 0
    all_statistics = []
    
    for json_file in sorted(json_files):
        print(f"\n处理文件: {json_file.name}")
        
        # 从文件名提取项目名
        project_name = json_file.name.replace("_compiler_errors_extracted.json", "")
        
        # 检查项目是否在配置中
        if project_name not in project_configs:
            print(f"  跳过: 项目 {project_name} 不在配置中")
            continue
        
        project_config = project_configs[project_name]
        allowed_workflows = project_config.get('compilation_workflows', [])
        
        print(f"  项目: {project_name}")
        print(f"  允许的workflows: {allowed_workflows}")
        
        # 加载编译错误数据
        compiler_errors = load_compiler_errors_data(json_file)
        if not compiler_errors:
            print(f"  跳过: 没有编译错误数据")
            continue
        
        print(f"  原始错误记录数: {len(compiler_errors)}")
        
        # 第一步：去重
        deduplicated_errors = deduplicate_records(compiler_errors)
        deduplicated_count = len(compiler_errors) - len(deduplicated_errors)
        total_deduplicated += deduplicated_count
        
        print(f"  去重后: {len(deduplicated_errors)} 条记录 (去重了 {deduplicated_count} 条)")
        
        # 按workflow过滤
        workflow_filtered = filter_by_workflow(deduplicated_errors, allowed_workflows)
        workflow_filtered_count = len(deduplicated_errors) - len(workflow_filtered)
        total_filtered_workflow += workflow_filtered_count
        
        print(f"  按workflow过滤后: {len(workflow_filtered)} 条记录 (过滤掉 {workflow_filtered_count} 条)")
        
        # 按错误类型过滤（过滤warning as error）
        type_filtered = filter_by_error_type(workflow_filtered, classifier)
        werror_filtered_count = len(workflow_filtered) - len(type_filtered)
        total_filtered_werror += werror_filtered_count
        
        print(f"  按错误类型过滤后: {len(type_filtered)} 条记录 (过滤掉 {werror_filtered_count} 条)")
        
        if not type_filtered:
            print(f"  跳过: 过滤后没有剩余记录")
            continue
    
        # 合并相同错误
        merged_errors = merge_similar_errors(type_filtered)
        merged_count = len(type_filtered) - len(merged_errors)
        total_merged += merged_count
        
        print(f"  合并相同错误后: {len(merged_errors)} 条记录 (合并了 {merged_count} 条)")
        
        # 按error_lines去重（保留最新commit的所有记录）
        error_lines_deduplicated = deduplicate_by_error_lines(merged_errors)
        error_lines_deduplicated_count = len(merged_errors) - len(error_lines_deduplicated)
        total_error_lines_deduplicated += error_lines_deduplicated_count
        
        print(f"  按error_lines去重后: {len(error_lines_deduplicated)} 条记录 (去重了 {error_lines_deduplicated_count} 条)")
        
        # 分类error_lines
        classified_results = classify_error_lines(error_lines_deduplicated, classifier)
        total_classified += len(classified_results)
        
        print(f"  分类完成: {len(classified_results)} 条记录")
        
        # 收集统计数据
        all_error_types = []
        all_commits = set()
        all_workflows = Counter()
        all_jobs = Counter()
        
        for record in classified_results.values():
            error_line_types = record.get('error_line_types', [])
            all_error_types.extend(error_line_types)
            
            commit_sha = record.get('commit_sha', '')
            if commit_sha:
                all_commits.add(commit_sha)
            
            workflow_name = record.get('workflow_name', '')
            if workflow_name:
                all_workflows[workflow_name] += 1
            
            job_names = record.get('job_name', [])
            if isinstance(job_names, list):
                for job_name in job_names:
                    if job_name and isinstance(job_name, str):
                        all_jobs[job_name] += 1
            else:
                # 兼容旧格式
                job_name = job_names
                if job_name and isinstance(job_name, str):
                    all_jobs[job_name] += 1
        
        # 统计错误类型
        error_type_counter = Counter(all_error_types)
        most_common_error_type = error_type_counter.most_common(1)[0][0] if error_type_counter else "unknown"
        
        # 计算统计数据
        error_count = len(classified_results)
        error_lines_count = len(all_error_types)
        unique_commits = len(all_commits)
        avg_errors_per_commit = error_count / unique_commits if unique_commits > 0 else 0
        avg_lines_per_error = error_lines_count / error_count if error_count > 0 else 0
        
        # 主要工作流和作业
        main_workflow = all_workflows.most_common(1)[0][0] if all_workflows else "unknown"
        main_job = all_jobs.most_common(1)[0][0] if all_jobs else "unknown"
        
        # 从配置中获取主要语言
        main_language = project_config.get('main_language', 'Unknown')
        
        statistics = {
            'repository_name': project_name,
            'error_count': error_count,
            'error_lines_count': error_lines_count,
            'unique_commits': unique_commits,
            'main_language': main_language,
            'avg_errors_per_commit': round(avg_errors_per_commit, 2),
            'avg_lines_per_error': round(avg_lines_per_error, 1),
            'main_workflow': main_workflow,
            'main_job': main_job,
            'main_error_type': most_common_error_type
        }
        
        all_statistics.append(statistics)
        
        # 保存处理后的文件到compilation_error目录
        output_file = f"{project_name}_compiler_errors_extracted.json"
        
        # 将分类结果转换为原始格式，只添加error_line_types字段
        processed_compiler_errors = []
        for record in classified_results.values():
            # 找到对应的原始记录
            original_record = None
            for orig_record in error_lines_deduplicated:
                if (orig_record.get('commit_sha', '') == record.get('commit_sha', '') and
                    orig_record.get('workflow_name', '') == record.get('workflow_name', '') and
                    orig_record.get('job_name', []) == record.get('job_name', [])): # Compare job_name as list
                    original_record = orig_record.copy()
                    break
            
            if original_record:
                # 添加error_line_types字段，并移除旧的line_types字段
                original_record['error_line_types'] = record.get('error_line_types', [])
                # 移除旧的line_types字段（如果存在）
                if 'line_types' in original_record:
                    del original_record['line_types']
                
                # 更新编码修复后的error_lines和error_details
                original_record['error_lines'] = record.get('error_lines', [])
                original_record['error_details'] = record.get('error_details', [])
                
                processed_compiler_errors.append(original_record)
        
        # 重新计算统计信息（基于编码修复后的数据）
        statistics = calculate_statistics(processed_compiler_errors, project_name, allowed_workflows)
        
        # 构建输出数据结构
        output_data = {
            'metadata': {
                'project_name': project_name,
                'original_records': len(compiler_errors),
                'deduplicated_records': deduplicated_count,
                'workflow_filtered': workflow_filtered_count,
                'werror_filtered': werror_filtered_count,
                'merged_records': merged_count,
                'final_records': len(classified_results),
                'processing_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'allowed_workflows': allowed_workflows,
                'statistics': statistics
            },
            'compiler_errors': processed_compiler_errors
        }
        
        # 保存文件
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"  已保存到: {output_file}")
        except Exception as e:
            print(f"  保存文件失败: {e}")
        
        print(f"  分类完成: {len(classified_results)} 条记录")
        
        total_processed += 1
    
    # 更新CSV统计文件
    if all_statistics:
        print(f"\n正在更新统计CSV文件...")
        update_statistics_csv(all_statistics)
        print(f"完成！共分析了 {len(all_statistics)} 个项目")
    
    # 打印总结
    print(f"\n{'='*60}")
    print("处理总结:")
    print(f"  处理的项目数: {total_processed}")
    print(f"  去重的记录数: {total_deduplicated}")
    print(f"  按workflow过滤的记录数: {total_filtered_workflow}")
    print(f"  按错误类型过滤的记录数: {total_filtered_werror}")
    print(f"  按路径合法性过滤的记录数: {total_filtered_path}")
    print(f"  合并相同错误的记录数: {total_merged}")
    print(f"  按error_lines去重的记录数: {total_error_lines_deduplicated}")
    print(f"  最终分类的记录数: {total_classified}")
    print(f"{'='*60}")

def main():
    """主函数"""
    print("处理backup文件夹中的编译错误文件")
    print("=" * 60)
    
    # 检查当前目录
    current_dir = Path('.')
    backup_dir = current_dir / "backup_20250727_182717"
    output_dir = current_dir / "output"
    
    if not backup_dir.exists() and not output_dir.exists():
        print("错误: 当前目录下没有找到backup_20250727_182717文件夹或output文件夹")
        print("请确保在compilation_error目录下运行此脚本")
        return
    
    if backup_dir.exists():
        print(f"发现backup目录: {backup_dir}")
    if output_dir.exists():
        print(f"发现output目录: {output_dir}")
    
    
    # 处理文件
    process_backup_files()

if __name__ == "__main__":
    main() 