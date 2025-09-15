#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
编译错误复现和验证脚本

该脚本被batch_reproduce_project.py调用，用于：
1. 调用reproduce_error.py运行CI workflow
2. 从输出日志中提取编译错误
3. 与find_repair_patch目录下的元数据对比
4. 判断错误信息是否一致，确定reproduce是否成功

使用方法:
    python reproduce_compiler_errors.py <project_name> <failure_commit> <job_name> <workflow_name> [--output OUTPUT]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re

def run_reproduce_script(project_name: str, failure_commit: str, job_name: str, workflow_name: str, 
                        workflow_id: str = None, line_number: Optional[int] = None, timeout: int = 43200) -> Tuple[bool, str, str]:
    """
    运行reproduce_error.py脚本
    
    Returns:
        Tuple[bool, str, str]: (是否成功, 标准输出, 标准错误)
    """
    print(f"\n{'='*60}")
    print("开始编译错误复现和验证")
    print(f"项目: {project_name}")
    print(f"Commit: {failure_commit}")
    print(f"Job: {job_name}")
    print(f"Workflow: {workflow_name}")
    if workflow_id:
        print(f"Workflow ID: {workflow_id}")
    print(f"{'='*60}\n")
    
    print("🚀 开始运行复现脚本...")
    
    script_dir = Path(__file__).parent.parent
    reproduce_script = script_dir / "reproduce_error.py"
    
    cmd = [
        "python3", str(reproduce_script),
        project_name, failure_commit, job_name, workflow_name
    ]
    
    # 添加workflow_id参数（如果提供）
    if workflow_id:
        cmd.append(workflow_id)
    else:
        # 如果没有提供workflow_id，使用默认值
        cmd.append("0")
    
    if line_number:
        cmd.extend(['--line-number', str(line_number)])
    
    print(f"🔧 执行命令: {' '.join(cmd)}")
    
    try:
        # 运行命令并捕获输出
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(script_dir)
        )
        
        success = result.returncode == 0
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        
        if success:
            print("✅ 复现脚本运行成功")
        else:
            print(f"❌ 复现脚本运行失败，退出码: {result.returncode}")
            
        return success, stdout, stderr
        
    except subprocess.TimeoutExpired:
        print(f"⏰ 复现脚本运行超时 ({timeout}秒)")
        return False, "", "超时"
    except Exception as e:
        print(f"❌ 运行复现脚本时出错: {e}")
        return False, "", str(e)

def extract_log_file_path(stdout: str) -> Optional[str]:
    """从输出中提取日志文件路径"""
    # 查找日志文件路径
    log_pattern = r"📝 日志文件: (.*\.log)"
    match = re.search(log_pattern, stdout)
    if match:
        log_file = match.group(1).strip()
        # 如果是相对路径，转换为绝对路径
        if not os.path.isabs(log_file):
            script_dir = Path(__file__).parent.parent
            log_file = str(script_dir / log_file)
        print(f"✅ 找到日志文件: {log_file}")
        return log_file
    
    print("❌ 未在输出中找到日志文件路径")
    print(f"输出内容: {repr(stdout[:1000])}")  # 调试信息
    return None

def extract_compiler_errors(log_content: str) -> List[str]:
    """从日志中提取编译错误"""
    print("🔍 从日志中提取编译错误...")
    from error_extractor import ErrorExtractor
    extractor = ErrorExtractor()
    errors = extractor.extract_errors(log_content)
    
    print(f"✅ 提取到 {len(errors)} 个编译错误")
    return errors

def load_expected_errors(project_name: str, failure_commit: str, job_name: str, workflow_name: str) -> List[Dict]:
    """加载预期的错误信息"""
    print("📚 加载预期的错误信息...")
    
    from metadata_loader import MetadataLoader
    
    loader = MetadataLoader()
    
    # 获取元数据文件路径
    metadata_file = loader.get_metadata_file_path(project_name)
    if not metadata_file:
        print(f"❌ 未找到项目 {project_name} 的元数据文件")
        return []
    
    # 加载所有记录
    records = loader.load_metadata_records(metadata_file)
    if not records:
        print("❌ 未加载到任何记录")
        return []
    
    # 查找匹配的记录
    matching_record = loader.find_matching_record(records, failure_commit, job_name, workflow_name)
    if not matching_record:
        print("❌ 未找到匹配的记录")
        return []
    
    error_lines = matching_record.get('error_lines', [])
    print(f"✅ 找到匹配的记录，包含 {len(error_lines)} 个预期错误")
    return [{'error_lines': [err]} for err in error_lines]

def compare_errors(actual_errors: List[str], expected_errors: List[Dict]) -> Dict:
    """比较实际错误和预期错误"""
    print("\n🔍 比较实际错误和预期错误...")
    
    from error_matcher import ErrorMatcher
    
    matcher = ErrorMatcher()
    
    # 使用ErrorMatcher进行匹配
    match_result = matcher.match_errors(actual_errors, expected_errors)
    
    # 判断是否成功匹配
    match_count = match_result.get('match_count', 0)
    similarity_score = match_result.get('similarity_score', 0.0)
    success = (
        match_count >= len(expected_errors) * 0.8 and  # 至少匹配80%的预期错误
        similarity_score >= 0.6                        # 总体相似度至少0.6
    )
    
    # 转换为原有格式以保持兼容性
    result = {
        'success': success,
        'reason': '所有错误都成功匹配' if success else match_result.get('reason', '未知错误'),
        'actual_count': match_result.get('actual_count', 0),
        'expected_count': match_result.get('expected_count', 0),
        'matched_errors': match_result.get('matched_errors', []),
        'similarity_score': similarity_score,
        'match_count': match_count  # 添加匹配数量字段
    }
    
    print(f"📊 匹配结果:")
    print(f"   实际错误数: {result['actual_count']}")
    print(f"   预期错误数: {result['expected_count']}")
    print(f"   匹配错误数: {match_count}")
    print(f"   总体相似度: {similarity_score:.3f}")
    print(f"   匹配结果: {'✅ 成功' if success else '❌ 失败'}")
    print(f"   匹配原因: {match_result.get('reason', '未知错误')}")
    
    # 显示详细的匹配信息
    matched_errors = match_result.get('matched_errors', [])
    if matched_errors:
        print("🔍 匹配详情:")
        for i, match in enumerate(matched_errors[:3], 1):  # 只显示前3个匹配
            similarity = match.get('similarity', 0.0)
            actual_error = match.get('actual_error', '未知错误')
            expected_error = match.get('expected_error', '未知错误')
            print(f"   匹配{i} (相似度: {similarity:.3f}):")
            print(f"     实际: {actual_error[:80]}...")
            expected_text = str(expected_error)
            print(f"     预期: {expected_text[:80]}...")
    
    return result

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('project_name',
                       help='项目名称 (如: llvm)')
    parser.add_argument('failure_commit',
                       help='失败的commit SHA')
    parser.add_argument('job_name',
                       help='作业名称')
    parser.add_argument('workflow_name',
                       help='workflow名称')
    parser.add_argument('workflow_id',
                       help='workflow ID (可选，默认为0)')
    parser.add_argument('--output',
                       help='输出结果到JSON文件')
    parser.add_argument('--timeout', type=int, default=43200,
                       help='运行超时时间(秒)')
    parser.add_argument('--line-number', type=int,
                       help='JSONL文件中的行号，用于日志文件命名')
    parser.add_argument('--reuse-log', action='store_true',
                       help='如果日志文件已存在，则直接复用，不重新运行复现脚本')
    
    args = parser.parse_args()
    
    start_time = time.time()

    # 构造日志文件名（和原有脚本一致）
    line_num = args.line_number if args.line_number is not None else 0
    log_file = f"logs/act_{args.project_name}_line{line_num}.log"
    print(f"   📝 日志文件: {log_file}")
    log_file_path = log_file
    if not os.path.isabs(log_file_path):
        script_dir = Path(__file__).parent.parent
        log_file_path = str(script_dir / log_file_path)

    # 如果日志文件已存在且启用了复用选项，直接加载
    if args.reuse_log and os.path.exists(log_file_path):
        print(f"✅ 日志文件已存在，启用复用模式，直接加载: {log_file_path}")
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                log_content = f.read()
            actual_errors = extract_compiler_errors(log_content)
            expected_errors = load_expected_errors(
                args.project_name,
                args.failure_commit,
                args.job_name,
                args.workflow_name
            )
            comparison = compare_errors(actual_errors, expected_errors)
            result = {
                'success': comparison.get('success', False),
                'stage': 'verification',
                'reason': comparison.get('reason', '未知错误'),
                'log_file': log_file_path,
                'actual_errors': actual_errors,
                'expected_errors': expected_errors,
                'comparison': comparison,
                'elapsed_time': time.time() - start_time
            }
        except Exception as e:
            result = {
                'success': False,
                'stage': 'error_processing',
                'reason': f'处理错误时发生异常: {e}',
                'log_file': log_file_path,
                'elapsed_time': time.time() - start_time
            }
    else:
        if os.path.exists(log_file_path):
            print(f"📝 日志文件已存在，但未启用复用模式，将重新运行复现脚本: {log_file_path}")
        else:
            print(f"📝 日志文件不存在，将运行复现脚本生成: {log_file_path}")
        # 步骤1: 运行复现脚本
        success, stdout, stderr = run_reproduce_script(
            args.project_name,
            args.failure_commit,
            args.job_name,
            args.workflow_name,
            args.workflow_id,
            args.line_number,
            args.timeout
        )
        if not success:
            result = {
                'success': False,
                'stage': 'reproduce_script',
                'reason': '复现脚本运行失败',
                'stdout': stdout,
                'stderr': stderr,
                'elapsed_time': time.time() - start_time
            }
        else:
            # 步骤2: 提取日志文件路径
            log_file = extract_log_file_path(stdout)
            if not log_file:
                result = {
                    'success': False,
                    'stage': 'log_extraction',
                    'reason': '未找到日志文件',
                    'stdout': stdout,
                    'stderr': stderr,
                    'elapsed_time': time.time() - start_time
                }
            else:
                try:
                    # 步骤3: 读取日志文件
                    with open(log_file, 'r', encoding='utf-8') as f:
                        log_content = f.read()
                    # 步骤4: 提取编译错误
                    actual_errors = extract_compiler_errors(log_content)
                    # 步骤5: 加载预期错误
                    expected_errors = load_expected_errors(
                        args.project_name,
                        args.failure_commit,
                        args.job_name,
                        args.workflow_name
                    )
                    # 步骤6: 比较错误
                    comparison = compare_errors(actual_errors, expected_errors)
                    result = {
                        'success': comparison.get('success', False),
                        'stage': 'verification',
                        'reason': comparison.get('reason', '未知错误'),
                        'log_file': log_file,
                        'actual_errors': actual_errors,
                        'expected_errors': expected_errors,
                        'comparison': comparison,
                        'elapsed_time': time.time() - start_time
                    }
                except Exception as e:
                    result = {
                        'success': False,
                        'stage': 'error_processing',
                        'reason': f'处理错误时发生异常: {e}',
                        'log_file': log_file,
                        'elapsed_time': time.time() - start_time
                    }
    # 输出结果
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n📄 结果已保存到: {args.output}")
        except Exception as e:
            print(f"\n❌ 保存结果文件失败: {e}")
    # 设置退出码
    sys.exit(0 if result['success'] else 1)

if __name__ == "__main__":
    main() 