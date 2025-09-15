#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from typing import List, Dict
from collections import defaultdict
import time
import argparse
import re
import chardet

# General error detection configuration
# Use regular expressions to match error messages with line numbers
CPP_ERROR_PATTERNS = [
    r'\.(cpp|cc|cxx|c|h|hpp|hxx|inl):\d+:\d+: (error|fatal error):',  # GCC/Clang style
    r'\.(cpp|cc|cxx|c|h|hpp|hxx|inl)\(\d+\): (error|fatal error)',     # MSVC style
]

# Required error patterns, at least one must be included
REQUIRED_ERROR_PATTERNS = [
    ': error:',     
    ': fatal error:', 
    'compilation failed',
    'undefined reference',
    'multiple definition',
    'collect2: error',
    'internal compiler error'
]

EXCLUDE_KEYWORDS = [
    'test',
    'from /',
    'required from',
    'warning:',
    'instantiated from',
    'in instantiation of',
    '##[',
    'runner version',
    'operating system',
    'image:',
    'run:',
    'shell:',
    '.sh',
    '.py',
    'ninja log:',
    'ninja: build stopped:',
    'bootstrap:',
    'stage',
    'GIT_VERSION=',
    'GEN ',
    'CC ',
    '* new build flags',
    '* new link flags',
    'Unexpected error attempting to determine',  # Exclude file search errors
    'UNKNOWN: unknown error',  # Exclude unknown errors
    'Error: UNKNOWN:',  # Exclude unknown errors
    'Post job cleanup',  # Exclude cleanup information
    'ERROR DETAILS:',  # Exclude error detail links
    'RequestId:',  # Exclude request ID
    'Time:',  # Exclude timestamp
    'NoAuthenticationInformation',  # Exclude authentication errors
    'Server failed to authenticate',  # Exclude authentication errors
    'stat ',  # 排除文件状态检查错误
    'error attempting to determine',  # 排除文件检查错误
    'CHECK',
    'SPV',
    #'file not found',
    #'No such file or directory',
    'module',
    'version control',
    'static assertion failed',
    'clang-format'
]

def get_repo_config(repo_name: str) -> Dict[str, str]:
    """根据仓库名生成配置，包含compilation_workflows"""
    # 读取项目配置文件
    import json
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'project_configs.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        all_configs = json.load(f)
    if repo_name not in all_configs:
        raise ValueError(f"项目 {repo_name} 未在project_configs.json中配置")
    repo_cfg = all_configs[repo_name]
    return {
        'input_file': f'../ci_failures/{repo_name}_ci_failures.json',
        'output_file': f'output/{repo_name}_compiler_errors_extracted.json',
        'compilation_workflows': repo_cfg.get('compilation_workflows', [])
    }

def detect_encoding(file_path: str) -> str:
    """检测文件编码"""
    # 读取文件的前10000字节来检测编码
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)
        result = chardet.detect(raw_data)
        return result['encoding']

def convert_to_utf8(text: str, source_encoding: str) -> str:
    """将文本转换为UTF-8编码"""
    try:
        # 先将文本编码为字节
        bytes_data = text.encode(source_encoding, errors='replace')
        # 然后用UTF-8解码
        return bytes_data.decode('utf-8', errors='replace')
    except Exception as e:
        print(f"转换编码时出错: {e}")
        return text

def load_ci_failures(file_path: str, max_records: int = None) -> List[Dict]:
    """从JSON Lines文件中加载CI失败记录，支持限制数量"""
    failures = []
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在")
        return failures
    
    print(f"开始加载 {file_path}...")
    try:
        # 检测文件编码
        detected_encoding = detect_encoding(file_path)
        print(f"检测到文件编码: {detected_encoding}")
        
        # 统一使用UTF-8读取
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line_num, line in enumerate(f, 1):
                if max_records and len(failures) >= max_records:
                    print(f"已达到最大记录数限制 {max_records}，停止加载")
                    break
                    
                line = line.strip()
                if line:
                    try:
                        # 如果检测到的编码不是UTF-8，进行转换
                        if detected_encoding and detected_encoding.lower() != 'utf-8':
                            line = convert_to_utf8(line, detected_encoding)
                            
                        failure_info = json.loads(line)
                        failures.append(failure_info)
                        
                        # 每1000条显示进度
                        if len(failures) % 1000 == 0:
                            print(f"已加载 {len(failures)} 条记录...")
                            
                    except json.JSONDecodeError as e:
                        print(f"解析第 {line_num} 行时出错: {e}")
                        
        print(f"成功加载 {len(failures)} 条失败记录")
    except Exception as e:
        print(f"读取文件时出错: {e}")
    
    return failures

def clean_log_line(line: str) -> str:
    """清理日志行，移除时间戳、workflow前缀和后面的第一个空格"""
    # 移除时间戳和后面的第一个空格
    line = line.strip()
    timestamp_pattern = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\s'
    line = re.sub(timestamp_pattern, '', line)
    
    # 移除workflow前缀，如 [GitHub CI/basic_clang]   | 
    prefix_pattern = r'^\[[^\]]+\]\s*\|\s*'
    line = re.sub(prefix_pattern, '', line)
    
    return line

def is_include_stack_line(line: str) -> bool:
    """检查是否是包含栈信息行"""
    line_lower = line.lower()
    return line_lower.startswith('in file included from')

def is_note_line(line: str) -> bool:
    """检查是否是note行"""
    line_lower = line.lower()
    return 'note:' in line_lower

def is_warning_line(line: str) -> bool:
    """检查是否是warning行"""
    line_lower = line.lower()
    return 'warning:' in line_lower

def is_error_line(line: str) -> bool:
    """检查是否是错误行"""
    line_lower = line.lower()
    
    # 首先检查是否应该排除
    for exclude in EXCLUDE_KEYWORDS:
        if exclude.lower() in line_lower:
            return False
    
    # 检查是否包含必需的错误模式
    has_required = False
    for pattern in REQUIRED_ERROR_PATTERNS:
        if pattern.lower() in line_lower:
            has_required = True
            break
    
    if not has_required:
        return False
    
    # 对于其他错误，检查是否匹配C/C++文件错误模式
    for pattern in CPP_ERROR_PATTERNS:
        if re.search(pattern, line, re.IGNORECASE):
            return True
    
    
    return False

def is_error_line_no_exclude(line: str) -> bool:
    """检查是否是错误行"""
    line_lower = line.lower()
    
    # 检查是否包含必需的错误模式
    has_required = False
    for pattern in REQUIRED_ERROR_PATTERNS:
        if pattern.lower() in line_lower:
            has_required = True
            break
    
    if not has_required:
        return False
    
    # 对于其他错误，检查是否匹配C/C++文件错误模式
    for pattern in CPP_ERROR_PATTERNS:
        if re.search(pattern, line, re.IGNORECASE):
            return True
    
    
    return False

def is_build_progress_line(line: str) -> bool:
    """检查是否是构建进度信息行"""
    line_lower = line.lower()
    return (
        # 编译进度信息
        ('[' in line and '@' in line and 'processes' in line) or
        # 编译目标信息
        ('obj/' in line and ('.o' in line or '.a' in line)) or
        # 性能指标信息
        ('metric' in line_lower and 'count' in line_lower and 'total' in line_lower) or
        # ninja构建信息
        any(x in line for x in ['CXX', 'AR', 'LINK', 'CC'])
    )

def is_build_system_line(line: str) -> bool:
    """判断是否为构建系统相关的行，并返回其是否有用
    
    Args:
        line: 要检查的行
        
    Returns:
        tuple[bool, bool]: (是否为构建系统行, 是否有用)
        - 第一个布尔值表示是否为构建系统相关的行
        - 第二个布尔值表示该行是否有用（如果第一个值为True）
    """
    line_strip = line.strip()
    
    # 构建系统输出
    if line_strip.startswith(('* ', '+ ', '++ ', '##[', 'make', 'ninja:', 'gmake:','[', 'gcc', 'ERROR:')):
        return True
        
    # 构建状态信息
    if any(x in line_strip for x in ['Waiting for unfinished jobs', 'exit.status']):
        return True
        
    return False

def collect_error_context(lines: List[str], error_line_index: int) -> List[str]:
    """收集错误行及其上下文信息
    
    Args:
        lines: 所有日志行的列表
        error_line_index: 错误行的索引
        
    Returns:
        包含错误上下文的行列表
    """
    error_context = []
    
    # 向前查找包含栈信息和相关代码
    include_stack = []
    j = error_line_index - 1
    while j >= 0 and j >= error_line_index - 20:  # 最多向前看20行
        prev_line = clean_log_line(lines[j])
        if not prev_line or is_build_system_line(prev_line) or is_build_progress_line(prev_line):
            j -= 1
            continue
            
        # 检查是否是包含栈信息
        if is_error_line_no_exclude(prev_line) or is_warning_line(prev_line):
            break
        elif (prev_line.startswith('In file included from') or  # 标准包含栈格式
            prev_line.startswith('from ') or  # 简化的包含栈格式
            (prev_line.startswith('./') and ':' in prev_line) or  # 相对路径包含
            (prev_line.startswith('/') and ':' in prev_line)):  # 绝对路径包含
            include_stack.insert(0, prev_line)
        j -= 1
    
    # 添加所有包含栈信息（按顺序）
    if include_stack:
        error_context.extend(include_stack)
    
    # 添加错误行本身
    error_line = clean_log_line(lines[error_line_index])
    if not is_build_system_line(error_line) and not is_build_progress_line(error_line):
        error_context.append(error_line)
    
    # 向后查找相关信息
    j = error_line_index + 1
    while j < len(lines) and j < error_line_index + 15:  # 最多向后看15行
        next_line = clean_log_line(lines[j])
        if not next_line or is_build_system_line(next_line) or is_build_progress_line(next_line):
            break
        # 如果遇到新的错误行（除本行外）就停止
        if is_error_line_no_exclude(next_line) or is_warning_line(next_line):
            break
        # 如果遇到 'generated' 相关的总结行就停止（但包含该行）
        if 'generated' in next_line.lower():
            break
        if (next_line.startswith('In file included from') or  # 标准包含栈格式
        next_line.startswith('from ') or  # 简化的包含栈格式
        (next_line.startswith('./') and ':' in next_line) or  # 相对路径包含
        (next_line.startswith('/') and ':' in next_line)):  # 绝对路径包含
            break
        # 其他全部收集
        error_context.append(next_line)
        j += 1
    
    return error_context

def extract_error_lines_simple(logs: str) -> tuple[List[str], List[str]]:
    """使用简单字符串匹配提取真正的错误行及其上下文
    
    Returns:
        tuple: (error_lines, error_details) 
        - error_lines: 只包含错误行本身（已去重，最多10个）
        - error_details: 包含错误行及其上下文
    """
    if not logs:
        return [], []
    
    error_lines = []
    error_details = []
    seen_error_lines = set()  # 用于去重
    lines = logs.split('\n')
    
    i = 0
    while i < len(lines):
        line = clean_log_line(lines[i])
        if not line:
            i += 1
            continue
        
        # 如果是错误行
        if is_error_line(line.strip()):
            error_line = line.strip()
            # 去重检查
            if error_line not in seen_error_lines:
                seen_error_lines.add(error_line)
                error_lines.append(error_line)
                
                # 收集错误上下文
                error_context = collect_error_context(lines, i)
                
                # 将收集到的上下文作为一个完整的错误信息
                if error_context:
                    context_str = '\n'.join(error_context)
                    error_details.append(context_str)
            
            # 更新索引到最后处理的行
            i = i + 1
        else:
            i += 1
        
    
    return error_lines, error_details

def classify_error_type_simple(error_lines: List[str]) -> Dict[str, int]:
    """使用新的ErrorClassifier分类错误类型"""
    # 导入新的错误分类器
    from error_classifier import ErrorClassifier
    
    classifier = ErrorClassifier()
    return classifier.classify_error_lines(error_lines)

def extract_compiler_errors_simple(failures: List[Dict], batch_size: int = 50, compilation_workflows: List[str] = None) -> List[Dict]:
    """使用简单字符串匹配快速提取编译错误, 支持workflow过滤和去重"""
    compiler_errors = []
    total = len(failures)
    start_time = time.time()
    print(f"开始分析 {total} 条CI失败记录，查找编译错误...")
    
    # 用于去重的集合，记录已处理的commit+workflow+job组合
    seen_combinations = set()
    duplicate_count = 0
    
    for i, failure in enumerate(failures):
        # workflow过滤
        workflow_name = failure.get('workflow_name', '')
        if compilation_workflows and len(compilation_workflows) > 0:
            if workflow_name not in compilation_workflows:
                continue
        
        # 去重检查：基于commit_sha + workflow_id + job_id的组合
        commit_sha = failure.get('commit_sha', '')
        workflow_id = failure.get('workflow_id', '')
        job_id = failure.get('job_id', '')
        combination_key = (commit_sha, workflow_id, job_id)
        
        if combination_key in seen_combinations:
            duplicate_count += 1
            continue
        
        seen_combinations.add(combination_key)
        
        # 显示进度
        if i % batch_size == 0:
            elapsed = time.time() - start_time
            if i > 0:
                rate = i / elapsed
                eta = (total - i) / rate if rate > 0 else 0
                print(f"已处理 {i}/{total} 条记录 ({i/total*100:.1f}%) - "
                      f"速度: {rate:.1f}条/秒, 预计剩余: {eta:.1f}秒")
        
        logs = failure.get('failure_logs', '')
        # 分行处理日志
        error_lines, error_details = extract_error_lines_simple(logs)
        if error_lines:  # 只有找到错误行才保存
            # 分类错误类型
            error_types = classify_error_type_simple(error_lines)
            compiler_error_info = {
                'commit_sha': commit_sha,
                'branch': failure.get('branch', ''),
                'workflow_name': workflow_name,
                'job_name': failure.get('job_name', ''),
                'workflow_id': workflow_id,
                'job_id': job_id,
                'created_at': failure.get('created_at', ''),
                'error_lines': error_lines,
                'error_details': error_details,  # 添加新字段
                'error_count': len(error_lines),
                'error_types': dict(error_types)
            }
            compiler_errors.append(compiler_error_info)
    
    elapsed = time.time() - start_time
    rate = total / elapsed if elapsed > 0 else 0
    print(f"处理完成，找到 {len(compiler_errors)} 条包含编译错误的记录")
    print(f"去重统计：跳过了 {duplicate_count} 条重复记录")
    print(f"总耗时: {elapsed:.1f}秒, 平均速度: {rate:.1f}条/秒")
    return compiler_errors

def process_ci_failures_streaming(file_path: str, compilation_workflows: List[str] = None, batch_size: int = 50) -> List[Dict]:
    """逐行处理CI失败记录，避免一次性加载所有数据到内存"""
    compiler_errors = []
    total_processed = 0
    start_time = time.time()
    
    # 用于去重的集合，记录已处理的commit+workflow+job组合
    seen_combinations = set()
    duplicate_count = 0
    
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在")
        return compiler_errors
    
    print(f"开始逐行处理 {file_path}...")
    
    try:
        # 检测文件编码
        detected_encoding = detect_encoding(file_path)
        print(f"检测到文件编码: {detected_encoding}")
        
        # 逐行读取和处理
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # 如果检测到的编码不是UTF-8，进行转换
                    if detected_encoding and detected_encoding.lower() != 'utf-8':
                        line = convert_to_utf8(line, detected_encoding)
                    
                    failure = json.loads(line)
                    total_processed += 1
                    
                    # workflow过滤
                    workflow_name = failure.get('workflow_name', '')
                    if compilation_workflows and len(compilation_workflows) > 0:
                        if workflow_name not in compilation_workflows:
                            continue
                    
                    # 去重检查：基于commit_sha + workflow_id + job_id的组合
                    commit_sha = failure.get('commit_sha', '')
                    workflow_id = failure.get('workflow_id', '')
                    job_id = failure.get('job_id', '')
                    combination_key = (commit_sha, workflow_id, job_id)
                    
                    if combination_key in seen_combinations:
                        duplicate_count += 1
                        continue
                    
                    seen_combinations.add(combination_key)
                    
                    # 显示进度
                    if total_processed % batch_size == 0:
                        elapsed = time.time() - start_time
                        if total_processed > 0:
                            rate = total_processed / elapsed
                            print(f"已处理 {total_processed} 条记录 - "
                                  f"速度: {rate:.1f}条/秒, 耗时: {elapsed:.1f}秒")
                    
                    # 处理日志提取编译错误
                    logs = failure.get('failure_logs', '')
                    error_lines, error_details = extract_error_lines_simple(logs)
                    
                    if error_lines:  # 只有找到错误行才保存
                        # 分类错误类型
                        error_types = classify_error_type_simple(error_lines)
                        compiler_error_info = {
                            'commit_sha': commit_sha,
                            'branch': failure.get('branch', ''),
                            'workflow_name': workflow_name,
                            'job_name': failure.get('job_name', ''),
                            'workflow_id': workflow_id,
                            'job_id': job_id,
                            'created_at': failure.get('created_at', ''),
                            'error_lines': error_lines,
                            'error_details': error_details,
                            'error_count': len(error_lines),
                            'error_types': dict(error_types)
                        }
                        compiler_errors.append(compiler_error_info)
                        
                except json.JSONDecodeError as e:
                    print(f"解析第 {line_num} 行时出错: {e}")
                except Exception as e:
                    print(f"处理第 {line_num} 行时出错: {e}")
    
    except Exception as e:
        print(f"读取文件时出错: {e}")
    
    elapsed = time.time() - start_time
    rate = total_processed / elapsed if elapsed > 0 else 0
    print(f"逐行处理完成，总共处理了 {total_processed} 条记录")
    print(f"找到 {len(compiler_errors)} 条包含编译错误的记录")
    print(f"去重统计：跳过了 {duplicate_count} 条重复记录")
    print(f"总耗时: {elapsed:.1f}秒, 平均速度: {rate:.1f}条/秒")
    
    return compiler_errors

def analyze_error_patterns_simple(compiler_errors: List[Dict]) -> Dict:
    """简单分析错误模式统计"""
    stats = {
        'total_errors': len(compiler_errors),
        'commits_with_errors': len(set(err['commit_sha'] for err in compiler_errors if err['commit_sha'])),
        'workflows_with_errors': defaultdict(int),
        # 主类型统计（保持向后兼容）
        'error_types_summary': defaultdict(int),
        # 详细类型统计（新增字段）
        'error_types_summary_detailed': defaultdict(int),
        'jobs_with_errors': defaultdict(int)
    }
    
    # 导入错误分类器用于获取主类型
    from error_classifier import ErrorClassifier
    classifier = ErrorClassifier()
    
    for error in compiler_errors:
        # 统计workflow
        workflow_name = error['workflow_name']
        stats['workflows_with_errors'][workflow_name] += 1
        
        # 统计job
        job_name = error['job_name']
        stats['jobs_with_errors'][job_name] += 1
        
        # 只统计主要错误类型
        error_lines = error.get('error_lines', [])
        for error_line in error_lines:
            main_type, detailed_type = classifier.identify_error_type(error_line)
            # 统计主类型
            stats['error_types_summary'][main_type.value] += 1
            # 统计详细类型
            stats['error_types_summary_detailed'][detailed_type.value] += 1
    
    return stats

def generate_error_classification_report(compiler_errors: List[Dict]) -> Dict:
    """生成详细的错误分类报告 - 使用新的ErrorClassifier"""
    # 导入新的错误分类器
    from error_classifier import ErrorClassifier
    
    classifier = ErrorClassifier()
    return classifier.generate_classification_report(compiler_errors)

def save_results(compiler_errors: List[Dict], stats: Dict, output_file: str, report: Dict = None):
    """保存结果到文件"""
    result = {
        'metadata': {
            'total_compiler_errors': len(compiler_errors),
            'extraction_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'method': 'enhanced_string_matching_with_classification',
            'statistics': dict(stats)
        },
        'compiler_errors': compiler_errors
    }
    
    # 添加详细的错误分类报告
    if report:
        result['error_classification_report'] = report
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到 {output_file}")
    except Exception as e:
        print(f"保存文件时出错: {e}")

def print_summary(stats: Dict, report: Dict = None):
    """打印统计摘要"""
    print("\n" + "="*60)
    print("编译错误提取摘要 (增强版错误分类)")
    print("="*60)
    print(f"总编译错误记录数: {stats['total_errors']}")
    print(f"涉及的commit数: {stats['commits_with_errors']}")
    
    if report:
        print(f"\n=== 错误分类报告 ===")
        summary = report['summary']
        print(f"错误类型总数: {summary['total_error_types']}")
        print(f"错误实例总数: {summary['total_error_instances']}")
        if summary['most_common_error']:
            error_type, count = summary['most_common_error']
            print(f"最常见错误: {error_type} ({count} 次)")
        
        print(f"\n=== 按类别统计 ===")
        for category, count in sorted(report['by_category'].items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"  {category.replace('_', ' ').title()}: {count} 次")
        
        print(f"\n=== 前10种具体错误类型 ===")
        for error_type, count in list(report['top_errors'].items())[:10]:
            print(f"  {error_type}: {count} 次")
    
    print(f"\n涉及的workflows (前10个):")
    workflows = sorted(stats['workflows_with_errors'].items(), key=lambda x: x[1], reverse=True)[:10]
    for workflow, count in workflows:
        print(f"  {workflow}: {count} 次")
    
    if not report:
        print(f"\n主要错误类型统计:")
        for error_type, count in sorted(stats['error_types_summary'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {error_type}: {count} 次")

        # 新增详细错误类型统计
        print(f"\n详细错误类型统计 (前20):")
        for error_type, count in sorted(stats['error_types_summary_detailed'].items(), key=lambda x: x[1], reverse=True)[:20]:
            print(f"  {error_type}: {count} 次")
    
    print(f"\n涉及的jobs (前10个):")
    jobs = sorted(stats['jobs_with_errors'].items(), key=lambda x: x[1], reverse=True)[:10]
    for job, count in jobs:
        print(f"  {job}: {count} 次")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='编译错误提取工具 (增强版错误分类系统)')
    parser.add_argument('repo', help='要分析的仓库名')
    args = parser.parse_args()
    # 获取仓库配置
    config = get_repo_config(args.repo)
    input_file = config['input_file']
    output_file = config['output_file']
    compilation_workflows = config.get('compilation_workflows', [])
    print(f"编译错误提取工具 (增强版错误分类系统) - {args.repo.upper()}")
    print("="*50)
    # 检查输入文件
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
        return
    print(f"发现输入文件: {input_file}")
    print("开始逐行处理所有记录（避免内存溢出）...")
    # 使用逐行处理函数，避免一次性加载所有数据到内存
    compiler_errors = process_ci_failures_streaming(input_file, compilation_workflows=compilation_workflows)
    if not compiler_errors:
        print("没有找到编译错误，退出")
        return
    # 分析错误模式
    stats = analyze_error_patterns_simple(compiler_errors)
    # 生成错误分类报告
    report = generate_error_classification_report(compiler_errors)
    # 保存结果
    save_results(compiler_errors, stats, output_file, report)
    # 打印摘要
    print_summary(stats, report)
    print(f"\n完成！详细结果已保存到 {output_file}")

if __name__ == "__main__":
    main() 