#!/usr/bin/env python3
"""
Batch extraction of compilation errors script

Automatically detect which repositories have input files but no output files, then call extract_compiler_errors.py in parallel for processing
"""

import os
import subprocess
import sys
from pathlib import Path

def get_existing_output_files():
    """Get existing output files in compilation_error folder"""
    compilation_error_dir = Path(__file__).parent
    existing_files = set()
    
    for file_path in compilation_error_dir.glob("*_compiler_errors_extracted.json"):
        # Extract repository name
        filename = file_path.name
        repo_name = filename.replace("_compiler_errors_extracted.json", "")
        existing_files.add(repo_name)
    
    return existing_files

def get_available_input_files():
    """Get available input files in ci_failures folder"""
    ci_failures_dir = Path(__file__).parent.parent / "ci_failures"
    available_repos = set()
    
    for file_path in ci_failures_dir.glob("*_ci_failures.json"):
        # Extract repository name
        filename = file_path.name
        repo_name = filename.replace("_ci_failures.json", "")
        available_repos.add(repo_name)
    
    return available_repos

def count_lines_with_wc(file_path):
    """Use wc -l to count file lines"""
    try:
        result = subprocess.run(['wc', '-l', str(file_path)], capture_output=True, text=True)
        if result.returncode == 0:
            return int(result.stdout.strip().split()[0])
        else:
            print(f"wc -l count for {file_path} failed: {result.stderr}")
            return 0
    except Exception as e:
        print(f"Error counting lines in {file_path}: {e}")
        return 0

def print_unprocessed_file_stats(repos_to_process):
    """打印未处理文件的统计信息（用wc -l）"""
    ci_failures_dir = Path(__file__).parent.parent / "ci_failures"
    print(f"\n{'='*60}")
    print("未处理文件统计信息 (按物理行数)")
    print(f"{'='*60}")
    total_lines = 0
    file_stats = []
    for repo in sorted(repos_to_process):
        input_file = ci_failures_dir / f"{repo}_ci_failures.json"
        if input_file.exists():
            line_count = count_lines_with_wc(input_file)
            total_lines += line_count
            file_stats.append((repo, line_count))
            print(f"  {repo}: {line_count:,} 行")
    print(f"\n总计: {len(file_stats)} 个文件，{total_lines:,} 行数据")
    print(f"{'='*60}")


def start_background_process(repo_name):
    """启动后台进程处理指定仓库"""
    try:
        cmd = [sys.executable, "extract_compiler_errors.py", repo_name]
        print(f"🚀 启动后台进程: {repo_name.upper()}")
        
        # 在后台运行，不等待完成
        process = subprocess.Popen(
            cmd, 
            cwd=Path(__file__).parent,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        print(f"✅ {repo_name.upper()} 后台进程已启动 (PID: {process.pid})")
        return True
        
    except Exception as e:
        print(f"❌ {repo_name.upper()} 启动失败: {e}")
        return False

def main():
    """主函数"""
    print("批量编译错误提取工具 (后台并行版本)")
    print("="*60)
    
    # 获取已存在的输出文件
    existing_outputs = get_existing_output_files()
    print(f"已存在的输出文件: {sorted(existing_outputs)}")
    
    # 获取可用的输入文件
    available_inputs = get_available_input_files()
    print(f"可用的输入文件: {sorted(available_inputs)}")
    
    # 找出需要处理的仓库
    repos_to_process = available_inputs - existing_outputs
    print(f"\n需要处理的仓库: {sorted(repos_to_process)}")
    
    if not repos_to_process:
        print("所有仓库都已处理完成！")
        return
    
    # 显示将要处理的仓库
    print(f"\n将处理 {len(repos_to_process)} 个仓库:")
    for repo in sorted(repos_to_process):
        print(f"  - {repo}")
    
    # 询问用户是否继续
    response = input(f"\n将启动 {len(repos_to_process)} 个后台进程。是否继续？(y/N): ")
    if response.lower() != 'y':
        print("操作已取消")
        return
    
    # 打印未处理文件的统计信息
    print_unprocessed_file_stats(repos_to_process)

    # 启动所有后台进程
    print("\n开始启动后台进程...")
    success_count = 0
    
    for repo in sorted(repos_to_process):
        if start_background_process(repo):
            success_count += 1
    
    # 输出结果
    print(f"\n{'='*60}")
    print("后台进程启动完成")
    print(f"{'='*60}")
    print(f"成功启动: {success_count}/{len(repos_to_process)} 个后台进程")
    print("所有进程将在后台运行，请稍后检查输出文件")
    print("可以使用 'ps aux | grep extract_compiler_errors' 查看运行中的进程")

if __name__ == "__main__":
    main() 