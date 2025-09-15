#!/usr/bin/env python3
"""
Extract parameters from JSON files in compilation_error directory and automatically reproduce compilation errors

This script reads error records from the specified project's *_compiler_errors_extracted.json file,
and automatically runs reproduction scripts to reproduce compilation errors.

Usage:
    python extract_and_reproduce.py --project <project_name> [options]

Parameters:
    --project STR         Project name to process (required, e.g.: llvm, opencv)

Options:
    --line-number N       Process record with specified line number (starting from 1)
    --first-n N           Process first N records
    --dry-run             Only show operations to be performed, do not actually run
    --rebuild             Force rebuild environment
    --no-switch           Do not switch to specified commit
    --list                List all available records
    --repo-path PATH      指定仓库路径 (默认使用项目配置中的路径)
    --reuse-log PATH      指定要复用的日志目录或日志文件路径，不重新运行复现脚本
    
示例:
    python extract_and_reproduce.py --project llvm --line-number 1
    python extract_and_reproduce.py --project opencv --first-n 3 --rebuild
    python extract_and_reproduce.py --project curl --no-switch
    python extract_and_reproduce.py --project curl --list
    python extract_and_reproduce.py --project llvm --repo-path /path/to/llvm
    python extract_and_reproduce.py --project llvm --line-number 1 --reuse-log
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from path_config import get_path, get_compilation_error_file, get_repair_patch_file, get_project_repo_path

# 设置标准输出无缓冲
import sys
import io
sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(), 'wb', 0), write_through=True)


def load_project_config(project_name):
    """加载项目配置"""
    config_file = get_path('project_configs')
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            configs = json.load(f)
        
        if project_name not in configs:
            print(f"警告: 项目 '{project_name}' 在配置文件中未找到")
            return None
        
        return configs[project_name]
        
    except FileNotFoundError:
        print(f"错误: 配置文件不存在: {config_file}")
        return None
    except Exception as e:
        print(f"错误: 读取配置文件失败: {e}")
        return None


def select_compilable_job(job_names, project_config):
    """根据项目配置选择可编译的job"""
    if not project_config:
        # 如果没有项目配置，返回第一个job
        if isinstance(job_names, list) and job_names:
            return job_names[0]
        return job_names
    
    reproducible_jobs = project_config.get('reproducible_jobs', [])
    
    if not reproducible_jobs:
        print("警告: 项目配置中没有可复现的job列表，使用第一个job")
        if isinstance(job_names, list) and job_names:
            return job_names[0]
        return job_names
    
    # 如果job_names是单个字符串，转换为列表
    if isinstance(job_names, str):
        job_names = [job_names]
    
    # 优先选择在reproducible_jobs中的job
    for job in job_names:
        if job in reproducible_jobs:
            print(f"选择可编译的job: {job}")
            return job
    
    # 如果没有找到可编译的job，返回None
    print("错误: 没有找到可编译的job，跳过此记录")
    return None


def load_compiler_errors_data(jsonl_file, project_name):
    """从find_repair_patch目录的JSONL文件中加载编译错误数据，如果不存在则从compilation_error目录读取JSON数据"""
    records = []
    
    # 首先尝试从JSONL文件读取
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    record = json.loads(line.strip())
                    # 添加行号信息
                    record['_line_number'] = line_num
                    # JSONL文件中已经有failure_commit字段，不需要映射
                    records.append(record)
        print(f"从JSONL文件加载了 {len(records)} 条记录")
        return records
            
    except FileNotFoundError:
        print(f"JSONL文件不存在: {jsonl_file}")
        print("尝试从compilation_error目录读取JSON数据...")
        
        # 从compilation_error目录读取JSON数据
        json_file = get_compilation_error_file(project_name)
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 从JSON数据中提取compiler_errors数组
            if 'compiler_errors' in data:
                compiler_errors = data['compiler_errors']
                for line_num, record in enumerate(compiler_errors, 1):
                    # 添加行号信息
                    record['_line_number'] = line_num
                    # 将commit_sha映射为failure_commit以保持兼容性
                    if 'commit_sha' in record:
                        record['failure_commit'] = record['commit_sha']
                    records.append(record)
                
                print(f"从JSON文件加载了 {len(records)} 条记录")
                return records
            else:
                print(f"错误: JSON文件中没有找到compiler_errors字段")
                sys.exit(1)
                
        except FileNotFoundError:
            print(f"错误: JSON文件也不存在: {json_file}")
            sys.exit(1)
        except Exception as e:
            print(f"错误: 读取JSON文件失败: {e}")
            sys.exit(1)
            
    except Exception as e:
        print(f"错误: 读取JSONL文件失败: {e}")
        sys.exit(1)


def list_records(records, project_name, project_config=None):
    """列出所有记录"""
    print(f"项目 '{project_name}' 的可用记录:")
    print("=" * 80)
    
    for record in records:
        line_num = record.get('_line_number', '?')
        failure_commit = record.get('failure_commit', 'N/A')[:12]
        workflow_name = record.get('workflow_name', 'N/A')
        job_name = record.get('job_name', 'N/A')
        
        # 根据项目配置选择可编译的job
        compilable_job_name = select_compilable_job(job_name, project_config)
        
        error_count = len(record.get('error_lines', []))
        
        print(f"行号: {line_num}")
        print(f"  失败commit: {failure_commit}")
        print(f"  Workflow: {workflow_name}")
        print(f"  作业: {compilable_job_name}")
        print(f"  错误数量: {error_count}")
        print("-" * 40)


def run_reproduction(project_name, failure_commit, job_name, workflow_name, line_number=None, dry_run=False, rebuild=False, no_switch=False, repo_path=None, reuse_log=None):
    """运行复现脚本"""
    # 构造日志文件名（和reproduce_compiler_errors.py一致）
    line_num = line_number if line_number is not None else 0
    log_file = f"logs/act_{project_name}_line{line_num}.log"
    script_dir = Path(__file__).parent
    log_file_path = script_dir / log_file
    
    # 如果显式指定了复用日志路径，则优先将其作为目录处理，找默认日志文件；若是文件就直接读取
    if reuse_log:
        reuse_path = Path(reuse_log)
        candidate_file = None
        if reuse_path.is_dir():
            candidate_file = reuse_path / f"act_{project_name}_line{line_num}.log"
        elif reuse_path.is_file():
            candidate_file = reuse_path
        else:
            # 既不是存在的目录也不是文件，直接失败
            print(f"❌ 指定的复用日志路径不存在: {reuse_path}")
            return False

        if candidate_file.exists():
            print(f"✅ 启用复用模式，使用日志: {candidate_file}")
            try:
                with open(candidate_file, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                print(f"\n📄 日志文件内容:")
                print("=" * 80)
                print(log_content)
                print("=" * 80)
            except Exception as e:
                print(f"❌ 读取日志文件失败: {e}")
                return False
            print("✅ 编译错误复现验证成功 (复用模式)")
            return True
        else:
            print(f"❌ 未找到可复用的日志文件: {candidate_file}")
            return False
    
    # 使用绝对路径调用reproduce_error.py
    reproduce_script = script_dir / 'reproduce_error.py'
    
    # 如果没有指定仓库路径，使用默认路径
    if not repo_path:
        try:
            repo_path = get_project_repo_path(project_name)
        except KeyError:
            print(f"警告: 项目 '{project_name}' 的仓库路径未配置，请使用 --repo-path 参数指定")
            return False
    
    # 使用新的编译错误复现脚本
    cmd = [
        'python3', str(reproduce_script),
        project_name, failure_commit, job_name, workflow_name
    ]
    
    if line_number:
        cmd.extend(['--line-number', str(line_number)])
    
    if dry_run:
        cmd.append('--dry-run')  # 注意：新脚本可能不支持dry_run，但先保留
    
    if rebuild:
        cmd.append('--force-rebuild')
    
    if no_switch:
        cmd.append('--no-switch')
    
    if repo_path:
        cmd.extend(['--repo-path', repo_path])
    
    print(f"运行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=False)
        success = result.returncode == 0
        
        if success:
            print("✅ 编译错误复现验证成功")
        else:
            print("❌ 编译错误复现验证失败")
        
        return success
    except Exception as e:
        print(f"错误: 运行复现脚本失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Extract parameters from JSON files in compilation_error directory and automatically reproduce compilation errors",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--project',
                       required=True,
                       help='要处理的项目名称 (例如: llvm, opencv)')
    parser.add_argument('--line-number', type=int,
                       help='处理指定行号的记录 (从1开始)')
    parser.add_argument('--first-n', type=int,
                       help='处理前N条记录')
    parser.add_argument('--dry-run', action='store_true',
                       help='仅显示将要执行的操作，不实际运行')
    parser.add_argument('--rebuild', action='store_true',
                       help='强制重新构建环境 (对应 reproduce_error.py 的 --force-rebuild)')
    parser.add_argument('--list', action='store_true',
                       help='列出所有可用的记录')
    parser.add_argument('--no-switch', action='store_true',
                       help='不切换到指定的commit (对应 reproduce_error.py 的 --no-switch)')
    parser.add_argument('--repo-path', type=str,
                       help='指定仓库路径 (对应 reproduce_error.py 的 --repo-path)')
    parser.add_argument('--reuse-log', type=str,
                        help='指定要复用的日志文件路径，不重新运行复现脚本')
    
    args = parser.parse_args()
    
    # 根据项目名称构建JSONL文件路径 - 从find_repair_patch目录读取
    jsonl_file = get_repair_patch_file(args.project)

    # 加载记录
    print(f"加载JSONL文件: {jsonl_file}")
    records = load_compiler_errors_data(jsonl_file, args.project)
    print(f"加载了 {len(records)} 条记录")
    
    # 加载项目配置
    project_config = load_project_config(args.project)
    
    # 如果是列表模式，显示记录并退出
    if args.list:
        list_records(records, args.project, project_config)
        return
    
    # 过滤记录
    filtered_records = records
    
    if args.line_number:
        # 处理指定行号
        target_record = None
        for record in filtered_records:
            if record.get('_line_number') == args.line_number:
                target_record = record
                break
        
        if not target_record:
            print(f"错误: 未找到行号 {args.line_number} 的记录")
            sys.exit(1)
        
        filtered_records = [target_record]
    
    elif args.first_n:
        # 处理前N条记录
        filtered_records = filtered_records[:args.first_n]
    
    elif not args.list:
        # 如果没有指定任何过滤条件，处理第一条记录
        if filtered_records:
            filtered_records = [filtered_records[0]]
            print("未指定记录，将处理第一条记录")
        else:
            print("错误: 没有可处理的记录")
            sys.exit(1)
    
    print(f"将处理 {len(filtered_records)} 条记录")
    
    # 处理记录
    success_count = 0
    for i, record in enumerate(filtered_records, 1):
        line_num = record.get('_line_number', '?')
        failure_commit = record.get('failure_commit', '')
        workflow_name = record.get('workflow_name', '')
        workflow_id = record.get('workflow_id', '')
        job_name = record.get('job_name', '')
        
        # 根据项目配置选择可编译的job
        compilable_job_name = select_compilable_job(job_name, project_config)
        
        # 如果没有找到可编译的job，跳过此记录
        if compilable_job_name is None:
            print(f"跳过记录 {line_num}: 没有可编译的job")
            continue
        
        project_name = args.project
        
        print(f"\n{'='*60}")
        print(f"处理记录 {i}/{len(filtered_records)} (行号: {line_num})")
        print(f"项目: {project_name}")
        print(f"失败commit: {failure_commit}")
        print(f"Workflow: {workflow_name}")
        print(f"作业: {compilable_job_name}")
        print(f"{'='*60}")
        
        # 验证必要参数
        if not failure_commit:
            print("错误: 缺少failure_commit")
            continue
        
        if not workflow_name:
            print("错误: 缺少workflow_name")
            continue
        
        # 运行复现
        success = run_reproduction(
            project_name, failure_commit, compilable_job_name, workflow_name, line_num, args.dry_run, args.rebuild, args.no_switch, args.repo_path, args.reuse_log
        )
        
        if success:
            success_count += 1
            print(f"✓ 记录 {line_num} 处理成功")
        else:
            print(f"✗ 记录 {line_num} 处理失败")
    
    print(f"\n总结: {success_count}/{len(filtered_records)} 条记录处理成功")


if __name__ == "__main__":
    main() 

if __name__ == "__main__":
    main() 


if __name__ == "__main__":
    main() 
