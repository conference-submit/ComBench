#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metadata loading module

Load expected compilation error information from JSONL files in find_repair_patch directory
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional

class MetadataLoader:
    """Metadata loader"""
    
    def __init__(self):
        # Set find_repair_patch directory path
        script_dir = Path(__file__).parent
        self.find_repair_patch_dir = script_dir.parent.parent / "find_repair_patch"
        self.compilation_error_dir = script_dir.parent.parent / "compilation_error"
        
        if not self.find_repair_patch_dir.exists():
            print(f"⚠️ find_repair_patch directory does not exist: {self.find_repair_patch_dir}")
        
        if not self.compilation_error_dir.exists():
            print(f"⚠️ compilation_error directory does not exist: {self.compilation_error_dir}")
    
    def get_metadata_file_path(self, project_name: str) -> Optional[Path]:
        """
        Get project metadata file path
        
        Args:
            project_name: Project name
            
        Returns:
            Metadata file path, returns None if not exists
        """
        # First search in find_repair_patch directory
        possible_filenames = [
            f"{project_name}_repair_analysis.jsonl",
            f"{project_name}_repair_analysis.json",
            f"{project_name.lower()}_repair_analysis.jsonl",
            f"{project_name.upper()}_repair_analysis.jsonl",
        ]
        
        for filename in possible_filenames:
            file_path = self.find_repair_patch_dir / filename
            if file_path.exists():
                return file_path
        
        # If not found in find_repair_patch directory, try compilation_error directory
        compilation_error_filenames = [
            f"{project_name}_compiler_errors_extracted.json",
            f"{project_name.lower()}_compiler_errors_extracted.json",
            f"{project_name.upper()}_compiler_errors_extracted.json",
        ]
        
        for filename in compilation_error_filenames:
            file_path = self.compilation_error_dir / filename
            if file_path.exists():
                return file_path
        
        return None
    
    def load_metadata_records(self, file_path: Path) -> List[Dict]:
        """
        Load all records from JSONL or JSON file
        
        Args:
            file_path: File path
            
        Returns:
            Record list
        """
        records = []
        
        try:
            # 检查文件扩展名来决定读取方式
            if file_path.suffix.lower() == '.json':
                # JSON格式文件
                with open(file_path, 'r', encoding='utf-8') as f:
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
                    
                    print(f"📚 从JSON文件{file_path}加载了 {len(records)} 条记录")
                else:
                    print(f"❌ JSON文件中没有找到compiler_errors字段")
                    
            else:
                # JSONL格式文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            record = json.loads(line)
                            record['_line_number'] = line_num
                            records.append(record)
                        except json.JSONDecodeError as e:
                            print(f"⚠️ 解析第{line_num}行JSON时出错: {e}")
                            continue
                            
                print(f"📚 从JSONL文件{file_path}加载了 {len(records)} 条记录")
            
        except FileNotFoundError:
            print(f"❌ 文件不存在: {file_path}")
        except Exception as e:
            print(f"❌ 读取文件时出错: {e}")
        
        return records
    
    def find_matching_record(self, records: List[Dict], failure_commit: str, 
                           job_name: str, workflow_name: str) -> Optional[Dict]:
        """
        根据条件查找匹配的记录
        
        Args:
            records: 记录列表
            failure_commit: 失败的commit SHA
            job_name: 作业名称
            workflow_name: workflow名称
            
        Returns:
            匹配的记录，如果未找到则返回None
        """
        for record in records:
            # 检查commit匹配
            record_commit = record.get('failure_commit', '')
            if record_commit and failure_commit:
                # 支持短commit SHA匹配
                if len(failure_commit) >= 7 and len(record_commit) >= 7:
                    if record_commit.startswith(failure_commit[:7]) or failure_commit.startswith(record_commit[:7]):
                        commit_match = True
                    else:
                        commit_match = record_commit == failure_commit
                else:
                    commit_match = record_commit == failure_commit
            else:
                commit_match = False
            
            # 检查job名称匹配（支持模糊匹配）
            record_job = record.get('job_name', '')
            
            # 处理job_name可能是列表的情况
            if isinstance(record_job, list):
                job_match = job_name in record_job
            else:
                job_match = (
                    record_job == job_name or
                    job_name in record_job or
                    record_job in job_name
                )
            
            # 检查workflow名称匹配（支持模糊匹配）
            record_workflow = record.get('workflow_name', '')
            
            # 处理workflow_name可能是列表的情况
            if isinstance(record_workflow, list):
                workflow_match = workflow_name in record_workflow
            else:
                workflow_match = (
                    record_workflow == workflow_name or
                    workflow_name in record_workflow or
                    record_workflow in workflow_name
                )
            
            # 如果所有条件都匹配
            if commit_match and job_match and workflow_match:
                print(f"✅ 找到匹配记录 (行号: {record.get('_line_number', '?')})")
                print(f"   Commit: {record_commit}")
                print(f"   Job: {record_job}")
                print(f"   Workflow: {record_workflow}")
                return record
        
        # 如果没有找到完全匹配的，尝试只匹配commit
        print(f"⚠️ 未找到完全匹配的记录，尝试只匹配commit...")
        for record in records:
            record_commit = record.get('failure_commit', '')
            if record_commit and failure_commit:
                if len(failure_commit) >= 7 and len(record_commit) >= 7:
                    if record_commit.startswith(failure_commit[:7]) or failure_commit.startswith(record_commit[:7]):
                        print(f"⚠️ 找到commit匹配记录 (行号: {record.get('_line_number', '?')})")
                        print(f"   Commit: {record_commit}")
                        print(f"   Job: {record.get('job_name', '')}")
                        print(f"   Workflow: {record.get('workflow_name', '')}")
                        return record
        
        return None
    
    def extract_error_lines(self, record: Dict) -> List[Dict]:
        """
        从记录中提取错误行信息
        
        Args:
            record: 元数据记录
            
        Returns:
            错误行信息列表
        """
        error_lines = []
        
        # 尝试不同的字段名来获取错误信息
        possible_fields = [
            'error_lines',
            'error_details', 
            'compilation_errors',
            'errors',
            'failure_details'
        ]
        
        for field in possible_fields:
            if field in record:
                field_value = record[field]
                
                if isinstance(field_value, list):
                    # 如果是列表，直接使用
                    for item in field_value:
                        if isinstance(item, str):
                            error_lines.append({'error_text': item})
                        elif isinstance(item, dict):
                            error_lines.append(item)
                elif isinstance(field_value, str):
                    # 如果是字符串，包装成字典
                    error_lines.append({'error_text': field_value})
                elif isinstance(field_value, dict):
                    # 如果是字典，可能包含嵌套的错误信息
                    if 'error_lines' in field_value:
                        nested_errors = field_value['error_lines']
                        if isinstance(nested_errors, list):
                            for item in nested_errors:
                                if isinstance(item, str):
                                    error_lines.append({'error_text': item})
                                elif isinstance(item, dict):
                                    error_lines.append(item)
                
                # 如果找到了错误信息，就不再检查其他字段
                if error_lines:
                    break
        
        return error_lines
    
    def load_expected_errors(self, project_name: str, failure_commit: str, 
                           job_name: str, workflow_name: str) -> List[Dict]:
        """
        加载预期的错误信息
        
        Args:
            project_name: Project name
            failure_commit: 失败的commit SHA
            job_name: 作业名称
            workflow_name: workflow名称
            
        Returns:
            预期错误信息列表
        """
        print(f"📖 加载项目 {project_name} 的预期错误信息...")
        
        # 获取元数据文件路径
        metadata_file = self.get_metadata_file_path(project_name)
        if not metadata_file:
            print(f"❌ 未找到项目 {project_name} 的元数据文件")
            return []
        
        print(f"📁 使用元数据文件: {metadata_file}")
        
        # 加载所有记录
        records = self.load_metadata_records(metadata_file)
        if not records:
            print("❌ 未加载到任何记录")
            return []
        
        # 查找匹配的记录
        matching_record = self.find_matching_record(
            records, failure_commit, job_name, workflow_name
        )
        
        if not matching_record:
            print("❌ 未找到匹配的记录")
            return []
        
        # 提取错误行信息
        error_lines = self.extract_error_lines(matching_record)
        
        if error_lines:
            print(f"✅ 提取到 {len(error_lines)} 条预期错误")
        else:
            print("⚠️ 匹配记录中未找到错误行信息")
        
        return error_lines
    
    def list_available_projects(self) -> List[str]:
        """
        列出所有可用的项目
        
        Returns:
            项目名称列表
        """
        projects = []
        
        if not self.find_repair_patch_dir.exists():
            return projects
        
        # 查找所有*_repair_analysis.jsonl文件
        for file_path in self.find_repair_patch_dir.glob("*_repair_analysis.jsonl"):
            project_name = file_path.stem.replace("_repair_analysis", "")
            projects.append(project_name)
        
        return sorted(projects)
    
    def get_project_statistics(self, project_name: str) -> Dict[str, any]:
        """
        获取项目的统计信息
        
        Args:
            project_name: Project name
            
        Returns:
            统计信息字典
        """
        metadata_file = self.get_metadata_file_path(project_name)
        if not metadata_file:
            return {'error': '未找到元数据文件'}
        
        records = self.load_metadata_records(metadata_file)
        
        if not records:
            return {'error': '未加载到记录'}
        
        # 收集统计信息
        unique_commits = set()
        unique_jobs = set()
        unique_workflows = set()
        total_errors = 0
        
        for record in records:
            if 'failure_commit' in record:
                unique_commits.add(record['failure_commit'])
            if 'job_name' in record:
                unique_jobs.add(record['job_name'])
            if 'workflow_name' in record:
                unique_workflows.add(record['workflow_name'])
            
            # 统计错误数量
            error_lines = self.extract_error_lines(record)
            total_errors += len(error_lines)
        
        return {
            'total_records': len(records),
            'unique_commits': len(unique_commits),
            'unique_jobs': len(unique_jobs),
            'unique_workflows': len(unique_workflows),
            'total_errors': total_errors,
            'metadata_file': str(metadata_file)
        } 