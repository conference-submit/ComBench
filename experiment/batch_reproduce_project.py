#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Set standard output unbuffered
import sys
import io
sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(), 'wb', 0), write_through=True)
"""
Batch run all records of the project and save results to JSONL file

This script automatically processes all records of the specified project, runs compilation error reproduction verification,
and saves the results of each record to a JSONL file. Supports continuing from where it was interrupted.
The script executes two phases in sequence: the first phase uses the passed successful records for diversification selection,
the second phase re-performs diversification selection based on the first phase results.

Usage:
    python3 batch_reproduce_project.py <project_name> [options]
    
Examples:
    python3 batch_reproduce_project.py llvm
    python3 batch_reproduce_project.py systemd 
    python3 batch_reproduce_project.py llvm --start-from 100 --max-records 50

"""

import argparse
import json
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

def load_project_records(project_name):
    """Load all records for the specified project"""
    print(f"📚 Loading {project_name} project records...")
    
    try:
        # First try to read JSONL file from find_repair_patch directory
        jsonl_file = Path(f"../find_repair_patch/{project_name}_repair_analysis.jsonl")
        
        if jsonl_file.exists():
            print(f"📄 Reading from find_repair_patch directory: {jsonl_file}")
            records = []
            
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        record = json.loads(line)
                        record['_line_number'] = line_num
                        records.append(record)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Line {line_num} JSON format error: {e}")
                        continue
            
            print(f"✅ 从JSONL文件加载了 {len(records)} 条记录")
            return records
        else:
            print(f"📄 JSONL文件不存在: {jsonl_file}")
            print("📄 尝试从compilation_error目录读取JSON数据...")
            
            # 从compilation_error目录读取JSON数据
            json_file = Path(f"../compilation_error/{project_name}_compiler_errors_extracted.json")
            
            if not json_file.exists():
                print(f"❌ JSON文件也不存在: {json_file}")
                return []
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 从JSON数据中提取compiler_errors数组
            if 'compiler_errors' in data:
                compiler_errors = data['compiler_errors']
                records = []
                for line_num, record in enumerate(compiler_errors, 1):
                    # 添加行号信息
                    record['_line_number'] = line_num
                    # 将commit_sha映射为failure_commit以保持兼容性
                    if 'commit_sha' in record:
                        record['failure_commit'] = record['commit_sha']
                    records.append(record)
                
                print(f"✅ 从JSON文件加载了 {len(records)} 条记录")
                return records
            else:
                print(f"❌ JSON文件中没有找到compiler_errors字段")
                return []
                
    except Exception as e:
        print(f"❌ 加载记录失败: {e}")
        return []

def select_records_with_diverse_jobs(records, max_records=None, successful_records=None):
    """选择具有多样化job_name的记录，确保每次尝试不同的job_name，并为每个记录指定选择的job_name"""
    if not records:
        return []
    
    # 统计所有可用的job_name
    job_name_count = {}
    for record in records:
        job_name = record.get('job_name', '')
        if isinstance(job_name, list):
            for job in job_name:
                job_name_count[job] = job_name_count.get(job, 0) + 1
        else:
            job_name_count[job_name] = job_name_count.get(job_name, 0) + 1
    
    print(f"📊 发现 {len(job_name_count)} 种不同的job_name:")
    for job_name, count in sorted(job_name_count.items()):
        print(f"   - {job_name}: {count} 条记录")
    
    # 按job_name分组记录，同时记录每个记录来自哪个job_name
    records_by_job = {}
    record_job_mapping = {}  # 记录每个记录对应的job_name
    
    for record in records:
        job_name = record.get('job_name', '')
        if isinstance(job_name, list):
            for job in job_name:
                if job not in records_by_job:
                    records_by_job[job] = []
                records_by_job[job].append(record)
                # 为每个记录记录它对应的job_name
                record_id = id(record)
                if record_id not in record_job_mapping:
                    record_job_mapping[record_id] = []
                record_job_mapping[record_id].append(job)
        else:
            if job_name not in records_by_job:
                records_by_job[job_name] = []
            records_by_job[job_name].append(record)
            # 为每个记录记录它对应的job_name
            record_id = id(record)
            record_job_mapping[record_id] = [job_name]
    
    # 第一阶段：轮询选择记录，确保每个job_name都有机会被选择
    selected_records = []
    job_names = list(records_by_job.keys())
    job_indices = {job: 0 for job in job_names}
    
    # 每个job_name选择1条记录
    records_per_job = 1
    
    print(f"🔄 第一阶段：每个job_name选择 {records_per_job} 条记录...")
    
    # 轮询每个job_name，每个选择指定数量的记录
    for job_name in job_names:
        job_records = records_by_job[job_name]
        for i in range(min(records_per_job, len(job_records))):
            if job_indices[job_name] < len(job_records):
                record = job_records[job_indices[job_name]]
                
                # 为记录添加selected_job_name字段
                record_copy = record.copy()
                record_copy['selected_job_name'] = job_name
                
                selected_records.append(record_copy)
                job_indices[job_name] += 1
    
    print(f"✅ 第一阶段选择了 {len(selected_records)} 条记录")
    
    # 限制记录数量
    if max_records and len(selected_records) > max_records:
        selected_records = selected_records[:max_records]
        print(f"📌 限制为前{max_records}条记录")
    
    print(f"✅ 总共选择了 {len(selected_records)} 条记录，涵盖 {len(set(job_names))} 种不同的job_name")
    return selected_records

def select_records_with_successful_jobs(records, max_records=None, successful_records=None):
    """第二阶段：只选择包含成功过的job name的记录"""
    if not records:
        return []
    
    # 从成功记录中提取成功过的job name
    successful_job_names = set()
    if successful_records:
        for record_id, record in successful_records.items():
            job_name = record.get('selected_job_name', '')
            if not job_name:
                job_name = record.get('job_name', '')
            if isinstance(job_name, list):
                job_name = job_name[0] if job_name else ''
            if job_name:
                successful_job_names.add(job_name)
    
    print(f"📊 发现 {len(successful_job_names)} 个成功过的job name: {list(successful_job_names)}")
    
    if not successful_job_names:
        print("⚠️ 没有发现成功过的job name，返回空列表")
        return []
    
    # 只选择包含成功job name的记录
    selected_records = []
    
    for record in records:
        record_job_names = record.get('job_name', [])
        if isinstance(record_job_names, str):
            record_job_names = [record_job_names]
        
        # 检查这个记录是否包含成功过的job name
        has_successful_job = any(job in successful_job_names for job in record_job_names)
        
        if has_successful_job:
            # 为记录添加selected_job_name字段，优先选择成功过的job name
            record_copy = record.copy()
            
            # 找到第一个成功过的job name
            selected_job = None
            for job in record_job_names:
                if job in successful_job_names:
                    selected_job = job
                    break
            
            # 如果没有成功过的job name，使用第一个
            if not selected_job and record_job_names:
                selected_job = record_job_names[0]
            
            record_copy['selected_job_name'] = selected_job or ''
            selected_records.append(record_copy)
    
    print(f"📊 找到 {len(selected_records)} 条包含成功job name的记录")
    
    # 限制记录数量
    if max_records and len(selected_records) > max_records:
        selected_records = selected_records[:max_records]
        print(f"📌 限制为前{max_records}条记录")
    
    print(f"✅ 第二阶段选择了 {len(selected_records)} 条记录（仅包含成功过的job name）")
    return selected_records

def load_completed_records(output_file):
    """加载已完成的记录，返回已处理的记录集合和成功记录字典"""
    completed_records = set()
    successful_records = {}  # 记录成功的记录，用于update模式
    
    if not Path(output_file).exists():
        print(f"📄 输出文件不存在，将从头开始处理")
        return completed_records, successful_records
    
    print(f"📄 检查已完成的记录...")
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    # 使用failure_commit + job_name + workflow_name作为唯一标识
                    # 优先使用selected_job_name，如果没有则使用job_name
                    job_name = record.get('selected_job_name', '')
                    if not job_name:
                        job_name = record.get('job_name', '')
                    
                    # 确保job_name和workflow_name是字符串，如果是列表则取第一个元素
                    if isinstance(job_name, list):
                        job_name = job_name[0] if job_name else ''
                    
                    workflow_name = record.get('workflow_name', '')
                    if isinstance(workflow_name, list):
                        workflow_name = workflow_name[0] if workflow_name else ''
                    
                    record_id = (
                        record.get('failure_commit', ''),
                        job_name,
                        workflow_name
                    )
                    completed_records.add(record_id)
                    
                    # 如果是成功的记录，保存到成功记录字典中
                    if record.get('reproduce', False):
                        successful_records[record_id] = record
                        
                except json.JSONDecodeError:
                    continue
        
        print(f"✅ 发现 {len(completed_records)} 条已完成的记录")
        print(f"✅ 其中 {len(successful_records)} 条是成功的记录")
        return completed_records, successful_records
        
    except Exception as e:
        print(f"⚠️ 读取已完成记录时发生错误: {e}")
        return completed_records, successful_records





def clean_bitcoin_docker():
    """清理Bitcoin项目的Docker容器"""
    try:
        # 尝试删除ci_win64容器
        result = subprocess.run(
            ["docker", "rm", "ci_win64","-f"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("🐳 已清理Docker容器 ci_win64")
        else:
            # 容器可能不存在，这是正常的
            if "No such container" not in result.stderr:
                print(f"⚠️ 清理Docker容器时出现警告: {result.stderr.strip()}")
        
    except Exception as e:
        print(f"⚠️ 清理Docker容器时发生异常: {e}")





def run_single_reproduce(record, project_name, timeout=1800):
    """运行单条记录的复现测试，返回结果记录"""
    line_num = record.get('_line_number', '?')
    failure_commit = record.get('failure_commit', '')
    workflow_name = record.get('workflow_name', '')
    
    # 优先使用selected_job_name，如果没有则使用job_name
    job_name = record.get('selected_job_name', '')
    if not job_name:
        job_name = record.get('job_name', '')
    
    workflow_id = record.get('workflow_id', '0')  # 获取workflow_id，默认为'0'
    
    # 确保job_name和workflow_name是字符串，如果是列表则取第一个元素
    if isinstance(job_name, list):
        job_name = job_name[0] if job_name else ''
    
    if isinstance(workflow_name, list):
        workflow_name = workflow_name[0] if workflow_name else ''
    
    # 确保workflow_id是字符串
    if isinstance(workflow_id, int):
        workflow_id = str(workflow_id)
    
    # 针对bitcoin项目，每次运行前清理Docker容器
    if project_name.lower() == 'bitcoin':
        clean_bitcoin_docker()
    
    # 创建日志目录
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"🔄 处理记录 {line_num}")
    print(f"   Commit: {failure_commit}")
    print(f"   Job: {job_name}")
    print(f"   Workflow: {workflow_name}")
    print(f"   Workflow ID: {workflow_id}")
    print(f"   📝 日志文件: logs/act_{project_name}_line{line_num}.log")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # 构建命令
    cmd = [
        'python3', 'reproduce/reproduce_compiler_errors.py',
        project_name, failure_commit, job_name, workflow_name, workflow_id,
        '--line-number', str(line_num),
        '--output', f'temp_result_{line_num}.json'
    ]


    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        success = result.returncode == 0
        elapsed_time = time.time() - start_time
        
        # 尝试读取详细结果
        detailed_result = None
        temp_result_file = f'temp_result_{line_num}.json'
        if Path(temp_result_file).exists():
            try:
                with open(temp_result_file, 'r', encoding='utf-8') as f:
                    detailed_result = json.load(f)
                # 清理临时文件
                Path(temp_result_file).unlink()
            except Exception as e:
                print(f"⚠️ 读取详细结果失败: {e}")
        
        # 构建结果记录 - 复制原始记录的所有字段
        result_record = record.copy()  # 复制原始记录的所有字段
        
        # 移除临时添加的行号字段
        if '_line_number' in result_record:
            del result_record['_line_number']
        
        # 确保job_name是成功的那个job name
        selected_job_name = record.get('selected_job_name', '')
        if selected_job_name:
            result_record['job_name'] = selected_job_name
        
        # 添加复现结果字段
        result_record['reproduce'] = success
        
        # 添加复现相关的元数据（可选）
        result_record['reproduce_metadata'] = {
            'elapsed_time': elapsed_time,
            'timestamp': datetime.now().isoformat(),
            'detailed_result': detailed_result
        }
        
        # 如果复现失败，添加错误信息
        if not success:
            result_record['reproduce_metadata']['error_stdout'] = result.stdout[-1000:] if result.stdout else ""
            result_record['reproduce_metadata']['error_stderr'] = result.stderr[-500:] if result.stderr else ""
        
        if success:
            print(f"✅ 记录 {line_num} 处理成功 (耗时: {elapsed_time:.1f}秒)")
            # 打印详细的成功信息，方便人工查验
            if detailed_result:
                print(f"   成功阶段: {detailed_result.get('stage', 'unknown')}")
                print(f"   成功原因: {detailed_result.get('reason', 'unknown')}")
                actual_errors = detailed_result.get('actual_errors', [])
                if actual_errors:
                    print("   实际错误:")
                    for err in actual_errors[:3]:  # 只显示前3个
                        print(f"      - {err}")
                    if len(actual_errors) > 3:
                        print(f"      ... (还有 {len(actual_errors) - 3} 个错误)")
                expected_errors = detailed_result.get('expected_errors', [])
                if expected_errors:
                    print("   预期错误:")
                    for err in expected_errors[:3]:  # 只显示前3个
                        if isinstance(err, dict) and 'error_lines' in err:
                            # 新格式: {'error_lines': [err]}
                            error_text = err['error_lines'][0] if err['error_lines'] else str(err)
                        elif isinstance(err, dict) and 'error_text' in err:
                            # 旧格式: {'error_text': err}
                            error_text = err.get('error_text', '')
                        else:
                            # 其他格式
                            error_text = str(err)
                        print(f"      - {error_text}")
                    if len(expected_errors) > 3:
                        print(f"      ... (还有 {len(expected_errors) - 3} 个错误)")
                if 'comparison' in detailed_result and detailed_result['comparison']:
                    comparison = detailed_result['comparison']
                    print(f"   匹配统计: {comparison.get('match_count', 0)}/{comparison.get('expected_count', 0)} 匹配成功")
                    print(f"   相似度: {comparison.get('similarity_score', 0):.3f}")
        else:
            print(f"❌ 记录 {line_num} 处理失败 (耗时: {elapsed_time:.1f}秒)")
            # 打印详细的失败原因
            if detailed_result:
                print(f"   失败阶段: {detailed_result.get('stage', 'unknown')}")
                print(f"   失败原因: {detailed_result.get('reason', 'unknown')}")
                actual_errors = detailed_result.get('actual_errors', [])
                if actual_errors:
                    print("   实际错误:")
                    for err in actual_errors: 
                        print(f"      - {err}")
                expected_errors = detailed_result.get('expected_errors', [])
                if expected_errors:
                    print("   预期错误:")
                    for err in expected_errors: 
                        if isinstance(err, dict) and 'error_lines' in err:
                            # 新格式: {'error_lines': [err]}
                            error_text = err['error_lines'][0] if err['error_lines'] else str(err)
                        elif isinstance(err, dict) and 'error_text' in err:
                            # 旧格式: {'error_text': err}
                            error_text = err.get('error_text', '')
                        else:
                            # 其他格式
                            error_text = str(err)
                        print(f"      - {error_text}")
            if result.stderr:
                print(f"   错误输出: {result.stderr[-200:]}")
        
        return result_record
        
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        print(f"⏰ 记录 {line_num} 处理超时 (耗时: {elapsed_time:.1f}秒)")
        print(f"   超时限制: {timeout}秒")
        
        # 记录超时结果 - 复制原始记录的所有字段
        result_record = record.copy()  # 复制原始记录的所有字段
        
        # 移除临时添加的行号字段
        if '_line_number' in result_record:
            del result_record['_line_number']
        
        # 确保job_name是成功的那个job name
        selected_job_name = record.get('selected_job_name', '')
        if selected_job_name:
            result_record['job_name'] = selected_job_name
        
        # 添加复现结果字段
        result_record['reproduce'] = False
        
        # 添加复现相关的元数据
        result_record['reproduce_metadata'] = {
            'elapsed_time': elapsed_time,
            'timestamp': datetime.now().isoformat(),
            'error': 'timeout',
            'error_stderr': "处理超时"
        }
        
        return result_record
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ 记录 {line_num} 处理异常 (耗时: {elapsed_time:.1f}秒)")
        print(f"   异常类型: {type(e).__name__}")
        print(f"   异常信息: {str(e)}")
        
        # 记录异常结果 - 复制原始记录的所有字段
        result_record = record.copy()  # 复制原始记录的所有字段
        
        # 移除临时添加的行号字段
        if '_line_number' in result_record:
            del result_record['_line_number']
        
        # 确保job_name是成功的那个job name
        selected_job_name = record.get('selected_job_name', '')
        if selected_job_name:
            result_record['job_name'] = selected_job_name
        
        # 添加复现结果字段
        result_record['reproduce'] = False
        
        # 添加复现相关的元数据
        result_record['reproduce_metadata'] = {
            'elapsed_time': elapsed_time,
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'error_stderr': str(e)
        }
        
        return result_record

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="Batch run all records of the project and save results to JSONL file",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('project_name', help='项目名称 (如: llvm, systemd, opencv)')
    parser.add_argument('--start-from', type=int, default=1, help='从指定行号开始处理 (默认: 1)')
    parser.add_argument('--max-records', type=int, help='最大处理记录数')
    parser.add_argument('--timeout', type=int, default=18000, help='单个记录的超时时间(秒) (默认: 3600)')
    
    args = parser.parse_args()
    
    print(f"🚀 开始批量处理{args.project_name}项目的编译错误复现测试")
    print("=" * 80)
    
    # 加载记录
    records = load_project_records(args.project_name)
    if not records:
        print("❌ 未找到可处理的记录")
        sys.exit(1)
    
    if args.max_records is None:
        args.max_records = len(records)
    
    # 根据参数过滤记录
    if args.start_from > 1:
        records = [r for r in records if r.get('_line_number', 0) >= args.start_from]
        print(f"📌 从第{args.start_from}行开始处理，剩余 {len(records)} 条记录")
    
    # 生成输出文件名（固定名称，无时间戳）
    output_dir = Path("reproduce/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{args.project_name}_batch_reproduce_results.jsonl"
    backup_file = output_dir / f"{args.project_name}_batch_reproduce_results_backup.jsonl"
    
    # 删除旧文件，每次都从头开始
    if output_file.exists():
        print(f"🗑️ 删除旧的结果文件: {output_file}")
        output_file.unlink()
    
    if backup_file.exists():
        print(f"🗑️ 删除旧的备份文件: {backup_file}")
        backup_file.unlink()
    
    # ==================== 第一阶段 ====================
    print(f"\n🎯 开始第一阶段处理...")
    print(f"=" * 80)
    
    # 每次都从头开始，不使用已完成的记录
    completed_records = set()
    successful_records = {}
    
    # 第一阶段：使用多样化选择策略
    print(f"🔄 第一阶段：使用多样化选择策略...")
    stage1_records = select_records_with_diverse_jobs(records, args.max_records, successful_records)
    
    if args.max_records and len(stage1_records) > args.max_records:
        stage1_records = stage1_records[:args.max_records]
        print(f"📌 限制处理前{args.max_records}条记录")
    
    print(f"📄 结果将保存到: {output_file}")
    print(f"📊 第一阶段总计需要处理 {len(stage1_records)} 条记录")
    
    # 程序开始前删除备份文件
    if backup_file.exists():
        print(f"🗑️ 删除旧的备份文件: {backup_file}")
        backup_file.unlink()
    
    # 处理第一阶段所有记录
    all_records_to_process = stage1_records
    
    if not all_records_to_process:
        print(f"✅ 第一阶段没有需要处理的记录！")
        print(f"📊 统计信息:")
        print(f"   总记录数: {len(records)}")
        print(f"   待处理记录数: 0")
    else:
        print(f"📋 第一阶段需要处理 {len(all_records_to_process)} 条记录")
        
        # 批量处理第一阶段
        success_count = 0
        total_start_time = time.time()
        results = []  # 存储所有结果记录
        
        for i, record in enumerate(all_records_to_process, 1):
            print(f"\n📈 第一阶段进度: {i}/{len(all_records_to_process)} ({i/len(all_records_to_process)*100:.1f}%)")
            
            # 每次都运行复现测试，不检查是否已成功
            result_record = run_single_reproduce(record, args.project_name, args.timeout)
            results.append(result_record)
            
            if result_record.get('reproduce', False):
                success_count += 1
            
            # 每次处理完一条记录后，立即写入备份文件（热更新）
            print(f"📝 正在保存第 {i} 条记录到备份文件...")
            with open(backup_file, 'w', encoding='utf-8') as f:
                for result_record in results:
                    f.write(json.dumps(result_record, ensure_ascii=False) + '\n')
            
            # 每处理10条记录显示一次统计
            if i % 10 == 0 or i == len(all_records_to_process):
                elapsed = time.time() - total_start_time
                avg_time = elapsed / i
                remaining = (len(all_records_to_process) - i) * avg_time
                
                print(f"\n📊 第一阶段阶段性统计 ({i}/{len(all_records_to_process)}):")
                print(f"   成功: {success_count}/{i} ({success_count/i*100:.1f}%)")
                print(f"   平均耗时: {avg_time:.1f}秒/条")
                print(f"   已用时间: {elapsed/60:.1f}分钟")
                print(f"   预计剩余: {remaining/60:.1f}分钟")
        
        # 处理完成后，将备份文件替换为最终文件并删除备份文件
        print(f"📝 正在将备份文件替换为最终文件...")
        import os
        os.replace(backup_file, output_file)
        print(f"✅ 第一阶段所有结果已保存到: {output_file}")
        
        # 第一阶段最终统计
        total_elapsed = time.time() - total_start_time
        
        # 计算总体成功数量（包括之前成功的记录）
        total_success_count = success_count  # 现在success_count已经包含了所有成功的记录
        
        print(f"\n🎯 第一阶段批量处理完成！")
        print(f"=" * 80)
        print(f"📊 第一阶段最终统计:")
        print(f"   总记录数: {len(records)}")
        print(f"   本次处理记录数: {len(all_records_to_process)}")
        print(f"   本次成功数量: {success_count}")
        print(f"   本次失败数量: {len(all_records_to_process) - success_count}")
        print(f"   总成功数量: {total_success_count}")
        if len(all_records_to_process) > 0:
            print(f"   本次成功率: {success_count/len(all_records_to_process)*100:.1f}%")
        print(f"   总体成功率: {total_success_count/len(records)*100:.1f}%")
        print(f"   总耗时: {total_elapsed/60:.1f}分钟")
        if len(all_records_to_process) > 0:
            print(f"   平均耗时: {total_elapsed/len(all_records_to_process):.1f}秒/条")
        print(f"📄 详细结果已保存到: {output_file}")
        
        # 生成第一阶段简要统计文件
        summary_file = output_dir / f"{args.project_name}_batch_reproduce_summary.json"
        summary = {
            'total_records': len(records),
            'processed_records': len(all_records_to_process),
            'success_count': success_count,
            'failure_count': len(all_records_to_process) - success_count,
            'total_success_count': total_success_count,
            'success_rate': success_count / len(all_records_to_process) if len(all_records_to_process) > 0 else 0,
            'overall_success_rate': total_success_count / len(records) if len(records) > 0 else 0,
            'total_elapsed_seconds': total_elapsed,
            'average_time_per_record': total_elapsed / len(all_records_to_process) if len(all_records_to_process) > 0 else 0,
            'timestamp': datetime.now().isoformat(),
            'output_file': str(output_file),
            'stage': 1
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"📈 第一阶段统计摘要已保存到: {summary_file}")
    
    # ==================== 第二阶段 ====================
    print(f"\n🎯 开始第二阶段处理...")
    print(f"=" * 80)
    
    # 第二阶段：重新加载第一阶段的运行结果，只选择成功过的job name
    print(f"🔄 第二阶段：重新加载第一阶段的运行结果，只选择成功过的job name...")
    
    # 重新加载第一阶段的运行结果
    updated_completed_records, updated_successful_records = load_completed_records(output_file)
    print(f"📊 第一阶段运行后，成功记录数: {len(updated_successful_records)}")
    
    # 使用更新后的successful_records，只选择成功过的job name
    stage2_records = select_records_with_successful_jobs(records, args.max_records, updated_successful_records)
    
    if args.max_records and len(stage2_records) > args.max_records:
        stage2_records = stage2_records[:args.max_records]
        print(f"📌 限制处理前{args.max_records}条记录")
    
    print(f"📄 结果将保存到: {output_file}")
    print(f"📊 第二阶段总计需要处理 {len(stage2_records)} 条记录")
    
    # 程序开始前删除备份文件
    if backup_file.exists():
        print(f"🗑️ 删除旧的备份文件: {backup_file}")
        backup_file.unlink()
    
    # 处理第二阶段所有记录
    all_records_to_process = stage2_records
    
    if not all_records_to_process:
        print(f"✅ 第二阶段没有需要处理的记录！")
        print(f"📊 统计信息:")
        print(f"   总记录数: {len(records)}")
        print(f"   待处理记录数: 0")
    else:
        print(f"📋 第二阶段需要处理 {len(all_records_to_process)} 条记录")
        
        # 批量处理第二阶段
        success_count = 0
        total_start_time = time.time()
        results = []  # 存储所有结果记录
        
        for i, record in enumerate(all_records_to_process, 1):
            print(f"\n📈 第二阶段进度: {i}/{len(all_records_to_process)} ({i/len(all_records_to_process)*100:.1f}%)")
            
            # 每次都运行复现测试，不检查是否已成功
            result_record = run_single_reproduce(record, args.project_name, args.timeout)
            results.append(result_record)
            
            if result_record.get('reproduce', False):
                success_count += 1
            
            # 每次处理完一条记录后，立即写入备份文件（热更新）
            print(f"📝 正在保存第 {i} 条记录到备份文件...")
            with open(backup_file, 'w', encoding='utf-8') as f:
                for result_record in results:
                    f.write(json.dumps(result_record, ensure_ascii=False) + '\n')
            
            # 每处理10条记录显示一次统计
            if i % 10 == 0 or i == len(all_records_to_process):
                elapsed = time.time() - total_start_time
                avg_time = elapsed / i
                remaining = (len(all_records_to_process) - i) * avg_time
                
                print(f"\n📊 第二阶段阶段性统计 ({i}/{len(all_records_to_process)}):")
                print(f"   成功: {success_count}/{i} ({success_count/i*100:.1f}%)")
                print(f"   平均耗时: {avg_time:.1f}秒/条")
                print(f"   已用时间: {elapsed/60:.1f}分钟")
                print(f"   预计剩余: {remaining/60:.1f}分钟")
        
        # 处理完成后，将备份文件替换为最终文件并删除备份文件
        print(f"📝 正在将备份文件替换为最终文件...")
        import os
        os.replace(backup_file, output_file)
        print(f"✅ 第二阶段所有结果已保存到: {output_file}")
        
        # 第二阶段最终统计
        total_elapsed = time.time() - total_start_time
        
        # 计算总体成功数量（包括之前成功的记录）
        total_success_count = success_count  # 现在success_count已经包含了所有成功的记录
        
        print(f"\n🎯 第二阶段批量处理完成！")
        print(f"=" * 80)
        print(f"📊 第二阶段最终统计:")
        print(f"   总记录数: {len(records)}")
        print(f"   本次处理记录数: {len(all_records_to_process)}")
        print(f"   本次成功数量: {success_count}")
        print(f"   本次失败数量: {len(all_records_to_process) - success_count}")
        print(f"   总成功数量: {total_success_count}")
        if len(all_records_to_process) > 0:
            print(f"   本次成功率: {success_count/len(all_records_to_process)*100:.1f}%")
        print(f"   总体成功率: {total_success_count/len(records)*100:.1f}%")
        print(f"   总耗时: {total_elapsed/60:.1f}分钟")
        if len(all_records_to_process) > 0:
            print(f"   平均耗时: {total_elapsed/len(all_records_to_process):.1f}秒/条")
        print(f"📄 详细结果已保存到: {output_file}")
        
        # 生成第二阶段简要统计文件
        summary_file = output_dir / f"{args.project_name}_batch_reproduce_summary.json"
        summary = {
            'total_records': len(records),
            'processed_records': len(all_records_to_process),
            'success_count': success_count,
            'failure_count': len(all_records_to_process) - success_count,
            'total_success_count': total_success_count,
            'success_rate': success_count / len(all_records_to_process) if len(all_records_to_process) > 0 else 0,
            'overall_success_rate': total_success_count / len(records) if len(records) > 0 else 0,
            'total_elapsed_seconds': total_elapsed,
            'average_time_per_record': total_elapsed / len(all_records_to_process) if len(all_records_to_process) > 0 else 0,
            'timestamp': datetime.now().isoformat(),
            'output_file': str(output_file),
            'stage': 2
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"📈 第二阶段统计摘要已保存到: {summary_file}")
    
    print(f"\n🎉 所有阶段处理完成！")
    print(f"=" * 80)

if __name__ == "__main__":
    main() 