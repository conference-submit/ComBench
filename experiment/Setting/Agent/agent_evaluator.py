#!/usr/bin/env python3
"""
智能代理评估系统 (Agent Evaluation System)

该模块实现了编译错误修复任务的评估框架，支持多种评估指标和基线模型。
"""

import json
import time
from typing import Dict, List, Optional, Tuple, Any, Union, NamedTuple
from dataclasses import dataclass, field
from pathlib import Path
import logging
from datetime import datetime
import os
import shutil
import subprocess
import sys
import tempfile
import re

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from path_config import get_path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing


logger = logging.getLogger(__name__)





@dataclass
class CompilerError:
    """编译错误信息"""
    error_line: str
    error_details: List[str] = field(default_factory=list)
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None


@dataclass
class RepairPatch:
    """修复补丁"""
    error_line: str
    fixed_code: str
    file_path: str
    
    def to_dict(self) -> Dict:
        return {
            'error_line': self.error_line,
            'fixed_code': self.fixed_code,
            'file_path': self.file_path
        }




@dataclass
class CompilerErrorInstance:
    """编译错误实例 - 完全对应 openssl.jsonl 的字段结构"""
    instance_index: int
    error_index: int
    failure_commit: str
    repair_commit: str
    error_line: str
    error_detail: str
    compilation_related_path: Dict
    original_error_lines: List[str]
    workflow_name: str
    job_name: List[str]

    
    def to_dict(self) -> Dict:
        return {
            'instance_index': self.instance_index,
            'error_index': self.error_index,
            'failure_commit': self.failure_commit,
            'repair_commit': self.repair_commit,
            'error_line': self.error_line,
            'error_detail': self.error_detail,
            'compilation_related_path': self.compilation_related_path,
            'original_error_lines': self.original_error_lines,
            'workflow_name': self.workflow_name,
            'job_name': self.job_name
        }


@dataclass
class EvaluationResult:
    """评估结果"""
    model_name: str
    total_instances: int
    total_errors: int
    successful_fixes: int
    exact_matches: int
    valid_patches: int
    total_time: float
    average_time: float
    success_rate: float
    exact_match_rate: float
    patch_validity_rate: float
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'total_instances': self.total_instances,
            'total_errors': self.total_errors,
            'successful_fixes': self.successful_fixes,
            'exact_matches': self.exact_matches,
            'valid_patches': self.valid_patches,
            'total_time': self.total_time,
            'average_time': self.average_time,
            'success_rate': self.success_rate,
            'exact_match_rate': self.exact_match_rate,
            'patch_validity_rate': self.patch_validity_rate
        }


class DiffContent(NamedTuple):
    """Diff内容结构"""
    removed: List[str]  # 被删除的行
    added: List[str]    # 新增的行
    context: List[str]  # 上下文行
    original: str       # 原始diff内容


class PerfectLocalizationDirectRepair:
    """完美定位直接修复器"""
    
    def __init__(self, model_name: str = "claude-sonnet-4"):
        self.model_name = model_name
        logger.info(f"初始化完美定位直接修复器，使用模型: {model_name}")
    
    def generate_fix_for_error(self, instance: CompilerErrorInstance, error_idx: int, repo_path: Optional[str] = None) -> str:
        """生成修复补丁，使用 OpenHands 调用，返回原始 diff 文本"""
        try:
            # 构建 prompt
            prompt = self.create_repair_prompt_text(instance, repo_path)
            
            # 调用 OpenHands，传递临时仓库路径，返回原始 diff 文本
            diff_text = self._call_openhands(prompt, instance, repo_path)
            
            return diff_text
            
        except Exception as e:
            logger.error(f"生成修复补丁时发生错误: {e}")
            return ""
    
    def create_repair_prompt_text(self, instance: CompilerErrorInstance, repo_path: Optional[str] = None) -> str:
        """创建修复提示文本"""
        try:
            error_line = instance.error_line
            error_detail = instance.error_detail
            
            # 构建基本的错误信息
            prompt = f"""请修复以下编译错误：

错误信息：
{error_line}

详细错误：
{error_detail}

请提供修复方案。"""
            
            return prompt
            
        except Exception as e:
            logger.error(f"创建修复提示文本时发生错误: {e}")
            return ""
    
    def _call_openhands(self, prompt: str, instance: CompilerErrorInstance, temp_repo_path: str = None) -> str:
        """调用 OpenHands 生成修复补丁，返回原始 diff 文本"""
        try:
            import subprocess
            import json
            import git
            
            if not temp_repo_path:
                logger.error("需要提供临时仓库路径")
                return ""
            
            # OpenHands 路径
            openhands_path = str(get_path('openhands'))
            
            # 记录调用 OpenHands 前的状态
            repo = git.Repo(temp_repo_path)
            before_commit = repo.head.commit.hexsha
            
            # 创建临时配置文件
            temp_config_file = self._create_temp_config_file(openhands_path, temp_repo_path)
            
            try:
                # 构建命令
                cmd = [
                    "poetry", "run", "python", "-m", "openhands.core.main", 
                    "-t", prompt,
                    "--config-file", temp_config_file
                ]
                
                logger.info(f"调用 OpenHands: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    cwd=openhands_path,
                    capture_output=True,
                    text=True,
                    timeout=36000  # 1小时超时
                )
                
                if result.returncode != 0:
                    logger.error(f"OpenHands 调用失败: {result.stderr}")
                    return ""
                
                # 检查是否有文件被修改
                repo = git.Repo(temp_repo_path)
                if repo.is_dirty() or repo.untracked_files:
                    # 有修改，获取原始 diff 文本
                    # 先尝试获取工作区的修改
                    diff_text = repo.git.diff()
                    
                    # 如果工作区没有修改，尝试获取已提交的修改
                    if not diff_text.strip():
                        diff_text = repo.git.diff(before_commit, 'HEAD')
                    
                    # 如果还是没有，尝试获取与初始提交的差异（包括工作区修改）
                    if not diff_text.strip():
                        diff_text = repo.git.diff(before_commit)
                    
                    # 过滤出只包含 C++ 源文件的 diff
                    filtered_diff = self._filter_cpp_diff(diff_text)
                    
                    if filtered_diff:
                        logger.info(f"获取到 C++ 源文件 diff 文本，长度: {len(filtered_diff)} 字符")
                        return filtered_diff
                    else:
                        logger.warning("没有检测到 C++ 源文件的修改")
                        return ""
                else:
                    logger.warning("OpenHands 没有修改任何文件")
                    return ""
                    
            finally:
                # 清理临时配置文件
                if temp_config_file and os.path.exists(temp_config_file):
                    os.remove(temp_config_file)
                    logger.info(f"已清理临时配置文件: {temp_config_file}")
                    
        except subprocess.TimeoutExpired:
            logger.error("OpenHands 调用超时")
            return ""
        except Exception as e:
            logger.error(f"调用 OpenHands 时发生错误: {e}")
            return ""
    
    def _create_temp_config_file(self, openhands_path: str, temp_repo_path: str) -> str:
        """
        创建临时配置文件，设置正确的工作目录和文件过滤规则
        
        Args:
            openhands_path: OpenHands 项目路径
            temp_repo_path: 临时仓库路径
            
        Returns:
            str: 临时配置文件路径，如果创建失败则返回空字符串
        """
        try:
            # 原始配置文件路径
            original_config_file = os.path.join(openhands_path, "config.toml")
            
            if not os.path.exists(original_config_file):
                logger.error(f"OpenHands config.toml 文件不存在: {original_config_file}")
                return ""
            
            # 创建临时配置文件
            temp_config_file = tempfile.mktemp(suffix='.toml', prefix='openhands_config_')
            
            # 读取原始配置文件内容
            with open(original_config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 修改 workspace_base 设置
            import re
            # 查找并替换 workspace_base 设置
            pattern = r'workspace_base\s*=\s*"[^"]*"'
            replacement = f'workspace_base = "{temp_repo_path}"'
            
            if re.search(pattern, content):
                # 如果找到现有的 workspace_base 设置，替换它
                new_content = re.sub(pattern, replacement, content)
                logger.info(f"已更新现有的 workspace_base 设置为: {temp_repo_path}")
            else:
                # 如果没有找到，在 [core] 部分添加
                core_pattern = r'(\[core\])'
                if re.search(core_pattern, content):
                    new_content = re.sub(
                        core_pattern, 
                        f'\\1\nworkspace_base = "{temp_repo_path}"', 
                        content
                    )
                    logger.info(f"已在 [core] 部分添加 workspace_base 设置: {temp_repo_path}")
                else:
                    # 如果没有 [core] 部分，在文件末尾添加
                    new_content = content + f'\n[core]\nworkspace_base = "{temp_repo_path}"\n'
                    logger.info(f"已添加 [core] 部分和 workspace_base 设置: {temp_repo_path}")
            
            
            # 写入临时配置文件
            with open(temp_config_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            logger.info(f"已创建临时配置文件: {temp_config_file}，设置工作目录为: {temp_repo_path}")
            return temp_config_file
            
        except Exception as e:
            logger.error(f"创建临时配置文件时发生错误: {e}")
            return ""

    def _filter_cpp_diff(self, diff_text: str) -> str:
        """
        过滤 diff 文本，只保留 C++ 源文件的修改
        
        Args:
            diff_text: 原始 diff 文本
            
        Returns:
            str: 过滤后的 diff 文本，只包含 C++ 源文件的修改
        """
        if not diff_text.strip():
            return ""
        
        # C++ 源文件扩展名
        cpp_extensions = {'.cpp', '.cc', '.cxx', '.c++', '.c', '.h', '.hpp', '.hxx', '.h++'}
        
        lines = diff_text.split('\n')
        filtered_lines = []
        current_file = None
        in_cpp_file = False
        
        for line in lines:
            # 检查是否是文件头行（以 diff --git 或 +++ 或 --- 开头）
            if line.startswith('diff --git'):
                # 提取文件名
                parts = line.split()
                if len(parts) >= 4:
                    file_path = parts[3]  # 通常是 +++ 后的文件路径
                    # 去掉 a/ 或 b/ 前缀
                    if file_path.startswith('a/') or file_path.startswith('b/'):
                        file_path = file_path[2:]
                    
                    # 检查是否是 C++ 源文件
                    file_ext = os.path.splitext(file_path)[1].lower()
                    in_cpp_file = file_ext in cpp_extensions
                    current_file = file_path
                    
                    if in_cpp_file:
                        filtered_lines.append(line)
                        logger.debug(f"包含 C++ 源文件: {current_file}")
                    else:
                        logger.debug(f"跳过非 C++ 文件: {current_file}")
                        
            elif line.startswith('+++') or line.startswith('---'):
                # 文件路径行
                if in_cpp_file:
                    filtered_lines.append(line)
                    
            elif line.startswith('@@'):
                # 块头行
                if in_cpp_file:
                    filtered_lines.append(line)
                    
            elif line.startswith('+') or line.startswith('-') or line.startswith(' '):
                # 代码行
                if in_cpp_file:
                    filtered_lines.append(line)
                    
            else:
                # 其他行（如索引行、二进制文件提示等）
                if in_cpp_file:
                    filtered_lines.append(line)
        
        filtered_diff = '\n'.join(filtered_lines)
        
        # 检查是否只包含文件权限修改，没有实际的代码改动
        if filtered_diff.strip():
            # 检查是否只包含模式修改行（old mode/new mode）
            lines = filtered_diff.split('\n')
            has_actual_changes = False
            
            for line in lines:
                # 跳过空行和文件头行
                if not line.strip() or line.startswith('diff --git') or line.startswith('index '):
                    continue
                # 跳过模式修改行
                if line.startswith('old mode') or line.startswith('new mode'):
                    continue
                # 如果有其他类型的行（如 +++、---、@@、+、-、空格开头的代码行），说明有实际改动
                if (line.startswith('+++') or line.startswith('---') or 
                    line.startswith('@@') or line.startswith('+') or 
                    line.startswith('-') or line.startswith(' ')):
                    has_actual_changes = True
                    break
            
            if not has_actual_changes:
                logger.info("过滤后只有文件权限修改，没有实际代码改动，跳过")
                return ""
            else:
                logger.info(f"过滤后保留 C++ 源文件 diff，长度: {len(filtered_diff)} 字符")
        else:
            logger.info("过滤后没有 C++ 源文件的修改")
            
        return filtered_diff


class AgentEvaluator:
    """智能代理评估器"""
    
    def __init__(self, similarity_threshold: float = 0.8, output_dir: str = "results", enable_resume: bool = False):
        self.similarity_threshold = similarity_threshold
        self.enable_resume = enable_resume
        
        # 根据enable_resume决定是否添加日期后缀和是否清理目录
        if enable_resume:
            # 断点续跑模式：不添加日期后缀，直接使用指定的输出目录，不清理
            self.output_dir = output_dir if output_dir else "results"
            logger.info(f"断点续跑模式：使用输出目录 {self.output_dir}")
        else:
            # 普通模式：为输出目录追加日期后缀，格式YYYYMMDD，如果目录存在则清理
            date_suffix = datetime.now().strftime('%Y%m%d')
            self.output_dir = f"{output_dir}-{date_suffix}" if output_dir else f"results-{date_suffix}"
            
            # 普通模式下，如果输出目录已存在则先删除
            if os.path.exists(self.output_dir):
                import shutil
                shutil.rmtree(self.output_dir)
                logger.info(f"普通模式：清理输出目录: {self.output_dir}")
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 重新配置日志到指定的输出目录
        self._setup_logging()
        # 初始化 dataset_name
        self.dataset_name = None
        # 初始化结果文件句柄
        self.result_files = {}
    
    def _setup_logging(self):
        """设置日志到指定的输出目录"""
        # 清除现有的处理器
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        # 设置 logger 级别为 INFO
        logger.setLevel(logging.INFO)
        
        # 创建新的日志处理器
        log_file = os.path.join(self.output_dir, 'evaluation.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 设置格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # 添加处理器
        logger.addHandler(file_handler)
        
        # 确保控制台输出仍然存在
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        logger.info(f"日志已配置到: {log_file}")
    
    
    def _get_result_file_path(self, model_name: str, dataset_name: str = None) -> str:
        """获取结果文件路径"""
        output_dir = self.output_dir
        if dataset_name:
            output_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, f"{model_name}_detailed_results.jsonl")
    
    def _load_existing_results(self, model_name: str, dataset_name: str = None) -> Dict[Tuple[int, int], Dict]:
        """
        加载已存在的结果文件，用于断点续跑
        
        Args:
            model_name: 模型名称
            dataset_name: 数据集名称
            
        Returns:
            Dict[Tuple[int, int], Dict]: 已处理的结果，键为(instance_index, error_index)
        """
        existing_results = {}
        file_path = self._get_result_file_path(model_name, dataset_name)
        
        if not os.path.exists(file_path):
            return existing_results
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        instance_index = data.get('instance_index')
                        error_index = data.get('error_index')
                        if instance_index is not None and error_index is not None:
                            key = (instance_index, error_index)
                            existing_results[key] = data
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"加载已存在结果文件失败: {e}")
        
        logger.info(f"加载了 {len(existing_results)} 个已处理的结果")
        return existing_results
    
    def _log_resume_progress(self, existing_results: Dict[Tuple[int, int], Dict], instances: List[CompilerErrorInstance]):
        """
        记录断点续跑的进度信息（按 error index 粒度）
        
        Args:
            existing_results: 已存在的结果
            instances: 所有实例列表
        """
        if not existing_results:
            return
        
        # 统计已处理的实例和错误
        processed_instances = set()
        total_processed_errors = 0
        total_errors = 0
        partially_processed_instances = set()
        
        for instance_idx, instance in enumerate(instances):
            total_errors += 1  # 每个实例只有一个错误
            
            instance_processed_errors = 0
            if (instance.instance_index, instance.error_index) in existing_results:
                instance_processed_errors += 1
                total_processed_errors += 1
            
            if instance_processed_errors == 1:  # 每个实例只有一个错误
                processed_instances.add(instance_idx + 1)
            elif instance_processed_errors > 0:
                partially_processed_instances.add(instance_idx + 1)
        
        # 计算进度
        instance_progress = len(processed_instances) / len(instances) * 100
        error_progress = total_processed_errors / total_errors * 100 if total_errors > 0 else 0
        
        logger.info(f"断点续跑进度统计（按 error index 粒度）:")
        logger.info(f"  已处理错误: {total_processed_errors}/{total_errors} ({error_progress:.1f}%)")
        logger.info(f"  完全处理的实例: {len(processed_instances)}/{len(instances)} ({instance_progress:.1f}%)")
        logger.info(f"  部分处理的实例: {len(partially_processed_instances)}")
        logger.info(f"  完全处理的实例列表: {sorted(processed_instances)}")
        logger.info(f"  部分处理的实例列表: {sorted(partially_processed_instances)}")
        
        return {
            'total_errors': total_errors,
            'total_processed_errors': total_processed_errors,
            'processed_instances': processed_instances,
            'partially_processed_instances': partially_processed_instances
        }
    
    def _validate_existing_results(self, existing_results: Dict[Tuple[int, int], Dict], instances: List[CompilerErrorInstance]):
        """
        验证已处理结果的完整性
        
        Args:
            existing_results: 已存在的结果
            instances: 所有实例列表
        """
        if not existing_results:
            return
        
        # 检查是否有损坏的结果
        corrupted_results = []
        for key, result in existing_results.items():
            instance_index, error_index = key
            if not isinstance(result, dict) or 'error_result' not in result:
                corrupted_results.append(key)
                continue
            
            error_result = result['error_result']
            required_fields = ['is_successful', 'error_line', 'diff_text', 'time']
            missing_fields = [field for field in required_fields if field not in error_result]
            if missing_fields:
                corrupted_results.append(key)
                logger.warning(f"结果 {key} 缺少必要字段: {missing_fields}")
        
        if corrupted_results:
            logger.warning(f"发现 {len(corrupted_results)} 个损坏的结果，将重新处理")
            for key in corrupted_results:
                existing_results.pop(key, None)
        else:
            logger.info("已处理结果验证通过，数据完整")
    
    def _calculate_complete_evaluation_result(self, model_name: str, instances: List[CompilerErrorInstance], existing_results: Dict[Tuple[int, int], Dict], current_total_time: float, dataset_name: str = None) -> EvaluationResult:
        """
        基于完整的结果文件计算评估结果（断点续跑模式）
        
        Args:
            model_name: 模型名称
            instances: 所有实例列表
            existing_results: 已存在的结果
            current_total_time: 本次运行的总时间
            dataset_name: 数据集名称
            
        Returns:
            EvaluationResult: 完整的评估结果
        """
        # 从结果文件中加载所有结果（包括本次新写入的）
        complete_results = self._load_all_results_from_file(model_name, dataset_name)
        
        # 统计所有结果
        total_successful_fixes = 0
        total_exact_matches = 0
        total_valid_patches = 0
        total_errors = 0
        total_time = 0.0
        
        for result_data in complete_results.values():
            error_result = result_data.get('error_result', {})
            if error_result.get('is_successful', False):
                total_successful_fixes += 1
            if error_result.get('is_exact_match', False):
                total_exact_matches += 1
            if error_result.get('is_valid', False):
                total_valid_patches += 1
            total_errors += 1
            total_time += error_result.get('time', 0.0)
        
        # 计算评估指标
        average_time = total_time / total_errors if total_errors > 0 else 0
        success_rate = total_successful_fixes / total_errors if total_errors > 0 else 0
        exact_match_rate = total_exact_matches / total_errors if total_errors > 0 else 0
        patch_validity_rate = total_valid_patches / total_errors if total_errors > 0 else 0
        
        logger.info(f"完整评估结果统计:")
        logger.info(f"  总错误数: {total_errors}")
        logger.info(f"  成功修复数: {total_successful_fixes}")
        logger.info(f"  精确匹配数: {total_exact_matches}")
        logger.info(f"  有效补丁数: {total_valid_patches}")
        logger.info(f"  总时间: {total_time:.1f}秒")
        logger.info(f"  平均时间: {average_time:.1f}秒")
        logger.info(f"  成功率: {success_rate:.2%}")
        logger.info(f"  精确匹配率: {exact_match_rate:.2%}")
        logger.info(f"  补丁有效率: {patch_validity_rate:.2%}")
        
        return EvaluationResult(
            model_name=model_name,
            total_instances=len(instances),
            total_errors=total_errors,
            successful_fixes=total_successful_fixes,
            exact_matches=total_exact_matches,
            valid_patches=total_valid_patches,
            total_time=total_time,
            average_time=average_time,
            success_rate=success_rate,
            exact_match_rate=exact_match_rate,
            patch_validity_rate=patch_validity_rate
        )
    
    def _load_all_results_from_file(self, model_name: str, dataset_name: str = None) -> Dict[Tuple[int, int], Dict]:
        """
        从结果文件中加载所有结果（包括本次新写入的）
        
        Args:
            model_name: 模型名称
            dataset_name: 数据集名称
            
        Returns:
            Dict[Tuple[int, int], Dict]: 所有结果
        """
        all_results = {}
        file_path = self._get_result_file_path(model_name, dataset_name)
        
        if not os.path.exists(file_path):
            return all_results
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        instance_index = data.get('instance_index')
                        error_index = data.get('error_index')
                        if instance_index is not None and error_index is not None:
                            key = (instance_index, error_index)
                            all_results[key] = data
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"加载完整结果文件失败: {e}")
        
        logger.info(f"从结果文件中加载了 {len(all_results)} 个完整结果")
        return all_results
    
    def _append_result(self, model_name: str, result_data: Dict, dataset_name: str = None):
        """追加单条结果到文件"""
        file_path = self._get_result_file_path(model_name, dataset_name)
        
        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False)
                f.write('\n')
            logger.info(f"结果已追加到: {file_path}")
        except Exception as e:
            logger.error(f"追加结果失败: {e}")
    
    def _save_evaluation_summary(self, model_name: str, evaluation_result: EvaluationResult, dataset_name: str = None):
        """保存评估汇总结果"""
        output_dir = self.output_dir
        if dataset_name:
            output_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        
        summary_file = os.path.join(output_dir, f"{model_name}_evaluation_summary.json")
        summary_data = {
            'dataset': dataset_name,
            'model_name': model_name,
            'evaluation_result': evaluation_result.to_dict(),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评估汇总已保存到: {summary_file}")
    
    def _save_batch_summary_results(self, results: Dict[str, EvaluationResult], dataset_name: str = None):
        """保存批量评估汇总结果"""
        output_dir = self.output_dir
        if dataset_name:
            output_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        
        summary_file = os.path.join(output_dir, "batch_evaluation_summary.json")
        summary_data = {
            'dataset': dataset_name,
            'models': {name: result.to_dict() for name, result in results.items()},
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"批量评估汇总已保存到: {summary_file}")
        
        # 如果是在数据集目录下，也在根目录保存一个总结果
        if dataset_name:
            root_summary_file = os.path.join(os.path.dirname(output_dir), "all_datasets_summary.json")
            try:
                if os.path.exists(root_summary_file):
                    with open(root_summary_file, 'r', encoding='utf-8') as f:
                        all_summaries = json.load(f)
                else:
                    all_summaries = {'datasets': {}}
                
                # 更新当前数据集的结果
                all_summaries['datasets'][dataset_name] = summary_data
                all_summaries['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
                
                with open(root_summary_file, 'w', encoding='utf-8') as f:
                    json.dump(all_summaries, f, ensure_ascii=False, indent=2)
                
                logger.info(f"总结果已更新到: {root_summary_file}")
            except Exception as e:
                logger.error(f"保存总结果时发生错误: {e}")
    
    def load_data(self, data_path: str) -> List[CompilerErrorInstance]:
        """加载数据，支持多种数据格式"""
        instances = []
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # 判断数据格式
            # 检查是否是单个JSON对象（新格式）还是JSONL格式
            lines = content.split('\n')
            if len(lines) == 1 and content.startswith('{') and content.endswith('}'):
                # 新格式：单个JSON对象，包含metadata和compiler_errors
                instances = self._load_json_format(content)
            else:
                # JSONL格式：每行一个CompilerErrorInstance
                instances = self._load_jsonl_format(content)
            
            logger.info(f"成功加载{len(instances)}个编译错误实例")
            return instances
            
        except FileNotFoundError:
            logger.error(f"数据文件不存在: {data_path}")
            return []
        except Exception as e:
            logger.error(f"加载数据时发生错误: {e}")
            return []
    
    def _load_json_format(self, content: str) -> List[CompilerErrorInstance]:
        """加载新的JSON格式数据"""
        try:
            data = json.loads(content)
            
            # 检查是否包含compiler_errors字段
            if 'compiler_errors' not in data:
                logger.error("JSON数据中未找到compiler_errors字段")
                return []
            
            instances = []
            compiler_errors = data['compiler_errors']
            
            for error_data in compiler_errors:
                instance = self._convert_compiler_error_to_instance(error_data)
                if instance:
                    instances.append(instance)
            
            return instances
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            return []
        except Exception as e:
            logger.error(f"处理JSON格式数据失败: {e}")
            return []
    
    def _load_jsonl_format(self, content: str) -> List[CompilerErrorInstance]:
        """加载JSONL格式数据，每行对应一个错误实例"""
        instances = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                instance = self._parse_instance(data)
                if instance:
                    instances.append(instance)
            except json.JSONDecodeError as e:
                logger.warning(f"第{line_num}行JSON解析失败: {e}")
                continue
            except Exception as e:
                logger.warning(f"第{line_num}行数据处理失败: {e}")
                continue
        
        return instances
    
    def _convert_compiler_error_to_instance(self, error_data: Dict) -> Optional[CompilerErrorInstance]:
        """将新格式的compiler_error转换为CompilerErrorInstance"""
        try:
            # 新格式数据结构：
            # {
            #   "commit_sha": "...",
            #   "workflow_name": "...",
            #   "job_name": "...",
            #   "workflow_id": 123,
            #   "job_id": 456,
            #   "error_lines": ["error1", "error2"],
            #   "error_details": ["detail1", "detail2"],
            #   ...
            # }
            
            commit_sha = error_data.get('commit_sha', '')
            workflow_name = error_data.get('workflow_name', '')
            job_name = error_data.get('job_name', '')
            workflow_id = error_data.get('workflow_id', 0)
            job_id = error_data.get('job_id', 0)
            error_lines = error_data.get('error_lines', [])
            error_details = error_data.get('error_details', [])
            
            # 由于新格式没有repair相关信息，我们创建空的diffs
            # 这些实例主要用于错误检测和分析，而不是修复评估
            diffs = []
            for i, error_line in enumerate(error_lines):
                diff_data = {
                    'error_line': error_line,
                    'relevant_repairs': []  # 新格式没有修复信息
                }
                diffs.append(diff_data)
            
            return CompilerErrorInstance(
                failure_commit=commit_sha,
                repair_commit='',  # 新格式没有repair_commit
                error_lines=error_lines,
                error_details=error_details,
                workflow_id=workflow_id,
                job_id=job_id,
                workflow_name=workflow_name,
                job_name=job_name,
                diffs=diffs,
                repair_source='extracted',  # 标记为提取的数据
                compilation_related_paths=error_data.get('compilation_related_paths', {}),
                compilation_related_paths_details=error_data.get('compilation_related_paths_details', [])
            )
            
        except Exception as e:
            logger.warning(f"转换编译错误数据失败: {e}")
            return None
    
    def _parse_instance(self, data: Dict) -> Optional[CompilerErrorInstance]:
        """解析单个实例（每行数据对应一个错误实例）"""
        try:
            # 支持 reproduced_data 格式（openssl.jsonl）
            if 'error_line' in data and 'error_detail' in data:
                # 处理 job_name，确保是列表
                job_name = data.get('job_name', [])
                if isinstance(job_name, str):
                    job_name = [job_name]
                
                return CompilerErrorInstance(
                    instance_index=data.get('instance_index', 0),
                    error_index=data.get('error_index', 0),
                    failure_commit=data.get('failure_commit', ''),
                    repair_commit=data.get('repair_commit', ''),
                    error_line=data.get('error_line', ''),
                    error_detail=data.get('error_detail', ''),
                    compilation_related_path=data.get('compilation_related_path', {}),
                    original_error_lines=data.get('original_error_lines', []),
                    workflow_name=data.get('workflow_name', ''),
                    job_name=job_name
                )
            else:
                # 原格式：支持多个错误（需要转换为单个错误实例）
                error_lines = data.get('error_lines', [])
                error_details = data.get('error_details', [])
                
                # 为每个错误创建单独的实例
                instances = []
                for i, (error_line, error_detail) in enumerate(zip(error_lines, error_details)):
                    instance = CompilerErrorInstance(
                        instance_index=data.get('instance_index', 0),
                        error_index=i + 1,
                        failure_commit=data.get('failure_commit', ''),
                        repair_commit=data.get('repair_commit', ''),
                        error_line=error_line,
                        error_detail=error_detail,
                        compilation_related_path=data.get('compilation_related_paths', {}),
                        original_error_lines=[error_line],
                        workflow_name=data.get('workflow_name', ''),
                        job_name=data.get('job_name', []) if isinstance(data.get('job_name', []), list) else [data.get('job_name', '')]
                    )
                    instances.append(instance)
                
                # 返回第一个实例（如果有的话）
                return instances[0] if instances else None
        except Exception as e:
            logger.warning(f"解析实例失败: {e}")
            return None
    
    def evaluate_model(self, model_name: str, instances: List[CompilerErrorInstance], data_path: str = None, repo_path: str = None, dataset_name: str = None, max_workers: int = None) -> EvaluationResult:
        """评估单个模型"""
        # 设置 dataset_name 和 repo_path
        self.dataset_name = dataset_name
        self.repo_path = repo_path
        
        # 根据enable_resume参数决定是否加载已存在的结果
        existing_results = {}
        if self.enable_resume:
            existing_results = self._load_existing_results(model_name, dataset_name)
            if existing_results:
                logger.info(f"断点续跑模式：发现已存在的结果，将跳过已处理的实例和错误")
                # 统计已处理的结果
                progress_info = self._log_resume_progress(existing_results, instances)
                # 验证已处理结果的完整性
                self._validate_existing_results(existing_results, instances)
        else:
            logger.info("普通模式：每次运行都会重新开始评估")

        
        # 创建修复器
        repairer = PerfectLocalizationDirectRepair(model_name)
        
        # 评估指标
        successful_fixes = 0
        exact_matches = 0
        valid_patches = 0
        total_time = 0.0
        total_errors = 0
        
        # 线程安全的计数器
        successful_fixes_lock = threading.Lock()
        exact_matches_lock = threading.Lock()
        valid_patches_lock = threading.Lock()
        total_errors_lock = threading.Lock()
        
        # 设置线程数，默认为CPU核心数的一半，但不超过8个
        if max_workers is None:
            max_workers = min(max(1, multiprocessing.cpu_count() // 2), 8, len(instances))
        logger.info(f"开始并行评估，使用 {max_workers} 个线程")
        
        # 检查是否有修复数据（新格式没有 diffs 字段，所以设为 False）
        has_repair_data = False
        
        # 使用线程池并行处理instances
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_instance = {
                executor.submit(self._process_single_instance, instance, repairer, has_repair_data, dataset_name, repo_path, existing_results): instance 
                for i, instance in enumerate(instances)
            }
            
            # 收集结果
            for future in as_completed(future_to_instance):
                instance = future_to_instance[future]
                try:
                    result = future.result()
                    if result:
                        instance_successful_fixes, instance_exact_matches, instance_valid_patches, instance_total_errors, instance_time = result
                        
                        # 线程安全地更新全局计数器
                        with successful_fixes_lock:
                            successful_fixes += instance_successful_fixes
                        with exact_matches_lock:
                            exact_matches += instance_exact_matches
                        with valid_patches_lock:
                            valid_patches += instance_valid_patches
                        with total_errors_lock:
                            total_errors += instance_total_errors
                        
                        total_time += instance_time
                        
                        logger.info(f"实例处理完成，累计处理 {total_errors} 个错误")
                        
                except Exception as e:
                    logger.error(f"处理实例时发生错误: {e}")
                    continue
        
        # 计算评估结果
        if self.enable_resume and existing_results:
            # 断点续跑模式：基于完整的结果文件计算评估结果
            evaluation_result = self._calculate_complete_evaluation_result(model_name, instances, existing_results, total_time, dataset_name)
        else:
            # 普通模式：基于本次运行的结果计算
            average_time = total_time / total_errors if total_errors > 0 else 0
            success_rate = successful_fixes / total_errors if total_errors > 0 else 0
            exact_match_rate = exact_matches / total_errors if total_errors > 0 else 0
            patch_validity_rate = valid_patches / total_errors if total_errors > 0 else 0
            
            evaluation_result = EvaluationResult(
                model_name=model_name,
                total_instances=len(instances),
                total_errors=total_errors,
                successful_fixes=successful_fixes,
                exact_matches=exact_matches,
                valid_patches=valid_patches,
                total_time=total_time,
                average_time=average_time,
                success_rate=success_rate,
                exact_match_rate=exact_match_rate,
                patch_validity_rate=patch_validity_rate
            )
        
        # 如果有断点续跑，记录总结信息
        if existing_results and self.enable_resume and 'progress_info' in locals():
            # 使用progress_info计算节省时间
            total_errors = progress_info['total_errors']
            skipped_errors = progress_info['total_processed_errors']
            estimated_saved_time = skipped_errors * evaluation_result.average_time
            
            logger.info(f"断点续跑总结:")
            logger.info(f"  跳过的错误比例: {skipped_errors/total_errors*100:.1f}%")
            logger.info(f"  估计节省时间: {estimated_saved_time:.1f}秒 ({estimated_saved_time/60:.1f}分钟)")
            logger.info(f"  实际处理错误数: {evaluation_result.total_errors}")
            logger.info(f"  实际处理时间: {evaluation_result.total_time:.1f}秒")
        
        # 保存评估汇总结果
        self._save_evaluation_summary(model_name, evaluation_result, dataset_name)
        
        return evaluation_result
    

    

    def _evaluate_openhands_fix_success_from_diff(self, diff_text: str, instance: CompilerErrorInstance, error_idx: int, line_number: int, all_errors_before: List[str], temp_repo_path: str = None) -> Tuple[Optional[bool], List[str], str]:
        """
        从 diff 文本评估 OpenHands 修复成功性
        
        Args:
            diff_text: 原始 diff 文本
            instance: 编译错误实例
            error_idx: 错误索引
            line_number: 行号（用于编译项目）
            all_errors_before: 修复前的所有错误列表
            temp_repo_path: 临时仓库路径
            
        Returns:
            Tuple[Optional[bool], List[str], str]: (修复是否成功, 修复后提取到的错误列表, diff文本)。
            当无法继续验证时，返回 (False, [], "")；如需跳过（未检测到指定错误）应返回 (None, errors_after, "")。
        """
        # 导入错误提取器
        try:
            # 使用相对路径 - 修复路径计算
            reproduce_path = Path(__file__).parent.parent.parent.parent / "experiment" / "reproduce"
            if str(reproduce_path) not in sys.path:
                sys.path.insert(0, str(reproduce_path))
            from error_extractor import ErrorExtractor
        except ImportError as e:
            logger.error(f"无法导入错误提取器: {e}")
            return False, [], ""
        
        # 从dataset_name中提取项目名
        project_name = self.dataset_name.split('_')[0] if self.dataset_name else None
        if not project_name:
            logger.error("无法确定项目名称")
            return False, [], ""
        
        # 获取错误信息
        error_line = instance.error_line
        error_detail = instance.error_detail
        
        # 获取仓库路径
        if temp_repo_path:
            # 使用临时仓库路径
            repo_path = Path(temp_repo_path)
        elif hasattr(self, 'repo_path') and self.repo_path:
            # 使用原始仓库路径
            repo_path = Path(self.repo_path)
        else:
            logger.error("未设置仓库路径")
            return False, [], ""
        
        if not repo_path.exists():
            logger.error(f"仓库路径不存在: {repo_path}")
            return False, [], ""
        
        try:
            # 从diff计算行号容差
            line_tolerance = self._calculate_line_tolerance_from_diff(diff_text)
            
            # 创建错误提取器
            extractor = ErrorExtractor()
            
            # 步骤3: 重新编译项目，检查错误是否消失
            success_after, output_after = self._compile_project(
                project_name, line_number, no_switch=True, temp_repo_path=temp_repo_path
            )
            
            # 从编译输出中提取修复后的错误列表
            if output_after:
                errors_after = extractor.extract_errors(output_after)
            else:
                errors_after = []
            
            # 使用动态计算的line_tolerance进行错误检查
            # 注意：这里应该使用error_detail而不是error_line作为错误消息
            if self._check_error_fixed(all_errors_before, errors_after, error_detail, line_tolerance):
                return True, errors_after, diff_text
            else:
                return False, errors_after, diff_text
                    
        except Exception as e:
            logger.error(f"验证修复时发生错误: {e}")
            return False, [], ""


    def _apply_patches_with_regeneration(self, repo_path: Path, instance: CompilerErrorInstance, error_idx: int, temp_repo_path: str = None) -> Tuple[bool, str]:
        """
        应用修复；失败则调用模型重新生成并重试一次。
        返回 (是否全部应用成功, diff文本)。
        """
        # 调用当前模型生成修复
        try:
            current_model = get_current_model()
            repairer = PerfectLocalizationDirectRepair(current_model)
            diff_text = repairer.generate_fix_for_error(instance, error_idx, temp_repo_path)
        except Exception:
            diff_text = ""
        
        if not diff_text:
            return False, ""
        
        # OpenHands 已经直接修改了文件，直接返回成功
        logger.info("OpenHands 已直接修改文件")
        return True, diff_text


    
    def _compile_project(self, project_name: str, line_number: int, dry_run: bool = False, no_switch: bool = False, temp_repo_path: str = None) -> Tuple[bool, str]:
        """
        编译项目
        
        Args:
            project_name: 项目名称
            line_number: 记录行号
            dry_run: 是否仅模拟运行，不实际编译
            no_switch: 是否不切换到指定的commit
            temp_repo_path: 临时仓库路径，如果提供则使用此路径
            
        Returns:
            Tuple[bool, str]: (是否成功, 输出内容)
        """
        # 在每次编译前尝试清理可能残留的容器，避免环境污染
        # 仅当输出目录名或dataset包含bitcoin时才清理
        try:
            output_basename = os.path.basename(self.output_dir) if hasattr(self, 'output_dir') else ''
            dataset_name_lower = (self.dataset_name or '').lower() if hasattr(self, 'dataset_name') else ''
            should_cleanup_container = ('bitcoin' in output_basename.lower()) or ('bitcoin' in dataset_name_lower)
            if should_cleanup_container:
                cleanup = subprocess.run(
                    ["docker", "rm", "-f", "/ci_win64"],
                    check=False,
                    capture_output=True,
                    text=True
                )
                if cleanup.returncode == 0:
                    logger.info("已清理容器 /ci_win64（bitcoin 数据集/输出目录）")
                else:
                    logger.debug(f"容器 /ci_win64 清理非0返回码: {cleanup.returncode}")
            else:
                logger.debug("跳过容器 /ci_win64 清理：仅在 bitcoin 数据集/输出目录时执行")
        except Exception as e:
            logger.debug(f"清理容器 /ci_win64 逻辑执行异常: {e}")
        
        # 使用extract_and_reproduce.py脚本编译项目
        script_path = Path(__file__).parent.parent.parent.parent / "experiment" / "extract_and_reproduce.py"
        
        # 传入日志目录而非具体文件，具体文件名由 extract_and_reproduce.py 内部构造
        reuse_log_dir = Path(__file__).parent / "logs"
        cmd = [
            "python3", 
            str(script_path),
            "--project", project_name,
            "--line-number", str(line_number)
        ]
        
        # 如果是模拟运行，添加--dry-run参数
        if dry_run:
            cmd.append("--dry-run")
        
        # 如果不切换到指定的commit，添加--no-switch参数
        if no_switch:
            cmd.append("--no-switch")
        
        # 如果提供了临时仓库路径，添加--repo-path参数
        if temp_repo_path:
            cmd.extend(["--repo-path", temp_repo_path])
        
        logger.info(f"运行编译命令: {' '.join(cmd)}")
        
        # 添加重试机制
        max_retries = 2
        for attempt in range(max_retries):
            try:
                result = subprocess.run(
                    cmd, 
                    check=False, 
                    capture_output=True, 
                    text=True
                )
                return result.returncode == 0, result.stdout + result.stderr

            except Exception as e:
                logger.error(f"编译失败: {e}")
                if attempt < max_retries - 1:
                    logger.info("等待5秒后重试...")
                    time.sleep(5)
                else:
                    return False, str(e)
        
        return False, "编译失败，所有重试都失败"
    

    
    
    def _calculate_line_tolerance_from_diff(self, diff_text: str) -> int:
        """
        从diff文本计算行号容差
        
        Args:
            diff_text: diff文本
            
        Returns:
            int: 计算出的行号容差
        """
        if not diff_text:
            return 5  # 默认容差
        
        try:
            # 解析diff，计算修改的行数
            lines = diff_text.split('\n')
            max_line_change = 0
            
            for line in lines:
                if line.startswith('@@'):
                    # 解析@@ -start,count +start,count @@格式
                    parts = line.split('@@')
                    if len(parts) >= 2:
                        hunk_info = parts[1].strip()
                        # 提取+start,count部分
                        if '+' in hunk_info:
                            plus_part = hunk_info.split('+')[1].split()[0]
                            if ',' in plus_part:
                                count = int(plus_part.split(',')[1])
                                max_line_change = max(max_line_change, count)
            
            # 行号容差 = 修改行数 + 2（给一些缓冲）
            line_tolerance = max_line_change + 2
            logger.info(f"从diff计算出行号容差: {line_tolerance} (最大修改行数: {max_line_change})")
            return line_tolerance
            
        except Exception as e:
            logger.warning(f"解析diff计算行号容差失败: {e}，使用默认值5")
            return 5

    def _check_error_fixed(self, errors_before: List[str], errors_after: List[str], error_msg: str, line_tolerance: int = 5) -> bool:
        """
        检查指定错误是否被成功修复（直接通过错误消息判定）
        
        Args:
            errors_before: 修复前提取到的错误列表
            errors_after: 修复后提取到的错误列表
            error_msg: 要检查的错误信息
            line_tolerance: 行号容差，用于错误匹配
            
        Returns:
            bool: 错误是否被成功修复
        """
        if not errors_before or not error_msg:
            logger.warning("输入参数不完整，无法进行错误修复检查")
            return False
        
        try:
            # 导入错误匹配器
            reproduce_path = Path(__file__).parent.parent.parent.parent / "experiment" / "reproduce"
            if str(reproduce_path) not in sys.path:
                sys.path.insert(0, str(reproduce_path))
            from error_matcher import ErrorMatcher
            
            # 初始化错误匹配器
            error_matcher = ErrorMatcher()
            
            # 1. 检查指定错误是否仍然存在
            expected_errors_after = [{'error_lines': [error]} for error in errors_after]
            is_matched_after, matched_error_after = error_matcher.match_single_error(
                error_msg, expected_errors_after, line_tolerance=line_tolerance
            )
            
            if is_matched_after:
                logger.warning(f"修复后仍然检测到指定错误: {error_msg} (line_tolerance={line_tolerance})")
                return False
            
            # 2. 检查是否引入了新错误（通过错误消息对比）
            new_errors = self._detect_new_errors_by_message(errors_before, errors_after)
            if new_errors:
                logger.warning(f"修复引入了新错误: {new_errors}")
                return False
            
            # 3. 只有指定错误消失且没有引入新错误才算成功
            logger.info(f"✅ 指定错误已成功修复: {error_msg} (line_tolerance={line_tolerance})")
            return True
            
        except Exception as e:
            logger.error(f"错误修复检查失败: {e}")
            return False

    def _detect_new_errors_by_message(self, errors_before: List[str], errors_after: List[str]) -> List[str]:
        """
        通过错误消息检测新引入的错误
        
        Args:
            errors_before: 修复前错误列表
            errors_after: 修复后错误列表
            
        Returns:
            List[str]: 新引入的错误列表
        """
        # 提取错误消息的关键部分（文件:行号:错误类型）
        before_messages = set(self._extract_error_message(error) for error in errors_before)
        after_messages = set(self._extract_error_message(error) for error in errors_after)
        
        # 找出新增的错误消息
        new_messages = after_messages - before_messages
        new_errors = [error for error in errors_after 
                      if self._extract_error_message(error) in new_messages]
        
        return new_errors

    def _extract_error_message(self, error: str) -> str:
        """
        提取错误消息的关键部分（只提取error:后面的错误描述，忽略分号后的提示信息）
        
        Args:
            error: 错误消息，例如: "apps/s_client.c:945:34: error: 'ReadOptions' does not name a type; did you mean 'DBOptions'?"
            
        Returns:
            str: 错误消息的关键部分，例如: "'ReadOptions' does not name a type"
        """
        # 查找 "error:" 的位置
        error_index = error.find("error:")
        if error_index != -1:
            # 提取 error: 后面的部分
            error_desc = error[error_index + 6:].strip()  # 6 是 "error:" 的长度
            
            # 如果包含分号，只取分号前的部分（忽略提示信息）
            semicolon_index = error_desc.find(';')
            if semicolon_index != -1:
                error_desc = error_desc[:semicolon_index].strip()
            
            return error_desc
        
        # 如果没有找到 "error:"，返回整个错误消息
        return error
    
    
    
    

    
    
    def _save_summary_results(self, results: Dict[str, EvaluationResult], dataset_name: str = None):
        """保存汇总结果"""
        output_dir = self.output_dir
        if dataset_name:
            output_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        
        summary_file = os.path.join(output_dir, "evaluation_summary.json")
        summary_data = {
            'dataset': dataset_name,
            'models': {name: result.to_dict() for name, result in results.items()},
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"汇总结果已保存到: {summary_file}")
        
        # 如果是在数据集目录下，也在根目录保存一个总结果
        if dataset_name:
            # 获取根目录路径（去掉dataset_name部分）
            root_output_dir = self.output_dir
            if dataset_name in root_output_dir:
                root_output_dir = root_output_dir.replace(f"/{dataset_name}", "").replace(f"\\{dataset_name}", "")
            root_summary_file = os.path.join(root_output_dir, "all_datasets_summary.json")
            try:
                if os.path.exists(root_summary_file):
                    with open(root_summary_file, 'r', encoding='utf-8') as f:
                        all_summaries = json.load(f)
                else:
                    all_summaries = {'datasets': {}}
                
                # 更新当前数据集的结果
                all_summaries['datasets'][dataset_name] = summary_data
                all_summaries['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
                
                with open(root_summary_file, 'w', encoding='utf-8') as f:
                    json.dump(all_summaries, f, ensure_ascii=False, indent=2)
                
                logger.info(f"总结果已更新到: {root_summary_file}")
            except Exception as e:
                logger.error(f"保存总结果时发生错误: {e}")
    

    def batch_evaluate(self, model_names: List[str], instances: List[CompilerErrorInstance], repo_path: str = None, dataset_name: str = None, max_workers: int = None) -> Dict[str, EvaluationResult]:
        """批量评估多个模型"""
        logger.info(f"开始批量评估{len(model_names)}个模型")
        
        results = {}
        
        for model_name in model_names:
            try:
                result = self.evaluate_model(
                    model_name, 
                    instances, 
                    repo_path=repo_path,
                    dataset_name=dataset_name,
                    max_workers=max_workers
                )
                results[model_name] = result
                logger.info(f"模型 {model_name} 评估完成")
            except Exception as e:
                logger.error(f"评估模型 {model_name} 时发生错误: {e}")
                continue
        
        # 保存批量评估汇总结果
        self._save_batch_summary_results(results, dataset_name)
        
        return results


    
    def get_available_models(self) -> List[str]:
        """获取可用的模型列表"""
        return get_available_models()
    
    def print_evaluation_summary(self, results: Dict[str, EvaluationResult]):
        """打印评估汇总"""
        print("\n" + "="*80)
        print("智能代理评估结果汇总")
        print("="*80)
        
        for model_name, result in results.items():
            print(f"\n模型: {model_name}")
            print(f"总实例数: {result.total_instances}")
            print(f"错误总行数: {result.total_errors}")
            print(f"修复成功数: {result.successful_fixes}")
            print(f"精确匹配数: {result.exact_matches}")
            print(f"有效补丁数: {result.valid_patches}")
            print(f"修复成功率: {result.success_rate:.2%}")
            print(f"精确匹配率: {result.exact_match_rate:.2%}")
            print(f"补丁有效率: {result.patch_validity_rate:.2%}")
            print(f"平均修复时间: {result.average_time:.2f}秒")

    def _checkout_to_commit(self, repo_path: Path, commit_sha: str) -> bool:
        """
        切换到指定的commit
        
        Args:
            repo_path: 仓库路径
            commit_sha: commit的SHA值
            
        Returns:
            bool: 是否成功切换
        """
        try:
            logger.info(f"正在切换到commit: {commit_sha}")
            
            # 检查是否为Git仓库
            if not (repo_path / '.git').exists():
                logger.error(f"不是Git仓库: {repo_path}")
                return False
            
            # 第一步：修复仓库权限
            logger.info("修复仓库权限")
            if not self._fix_permission(str(repo_path)):
                logger.warning("仓库权限修复失败，但继续尝试checkout")
            
            # 清理Git锁定文件
            self._cleanup_git_locks(repo_path)
            
            # 配置Git安全目录（使用通配符）
            try:
                # 检查是否已经配置了通配符
                result = subprocess.run([
                    "git", "config", "--global", "--get-all", "safe.directory"
                ], capture_output=True, text=True)
                
                safe_directories = result.stdout.strip().split('\n') if result.stdout.strip() else []
                
                # 如果没有通配符，添加通配符配置
                if '*' not in safe_directories:
                    subprocess.run([
                        "git", "config", "--global", "--add", "safe.directory", "*"
                    ], check=True, capture_output=True)
                    logger.info("已添加通配符安全目录配置")
            except Exception as e:
                logger.warning(f"配置Git安全目录失败: {e}")
            
            # 设置环境变量
            env = os.environ.copy()
            env['GIT_SAFETY_CHECKS'] = 'disabled'
            
            # 执行checkout，带重试
            for attempt in range(3):
                try:
                    # 清理锁定文件
                    self._cleanup_git_locks(repo_path)
                    
                    # 尝试checkout
                    result = subprocess.run(
                        ["git", "-C", str(repo_path), "checkout", "-f", commit_sha],
                        capture_output=True,
                        text=True,
                        env=env
                    )
                    
                    if result.returncode == 0:
                        logger.info(f"成功切换到commit: {commit_sha}")
                        return True
                    
                    logger.warning(f"第{attempt+1}次checkout失败: {result.stderr}")
                    if attempt < 2:
                        time.sleep(3)
                        

                except Exception as e:
                    logger.warning(f"第{attempt+1}次checkout异常: {e}")
                    if attempt < 2:
                        time.sleep(3)
            
            logger.error(f"无法切换到commit: {commit_sha}")
            return False
                
        except Exception as e:
            logger.error(f"切换commit时发生错误: {e}")
            return False
    
    def _cleanup_git_locks(self, repo_path: Path):
        """
        清理Git锁定文件
        
        Args:
            repo_path: 仓库路径
        """
        try:
            # 常见的Git锁定文件
            lock_files = [
                '.git/index.lock',
                '.git/MERGE_HEAD.lock',
                '.git/MERGE_MODE.lock',
                '.git/MERGE_MSG.lock',
                '.git/ORIG_HEAD.lock',
                '.git/CHERRY_PICK_HEAD.lock',
                '.git/REVERT_HEAD.lock',
                '.git/REBASE_HEAD.lock',
                '.git/REBASE_MERGE.lock',
                '.git/REBASE_APPLY.lock'
            ]
            
            for lock_file in lock_files:
                lock_path = repo_path / lock_file
                if lock_path.exists():
                    try:
                        # 使用sudo删除锁定文件，避免权限问题
                        subprocess.run([
                            'sudo', 'rm', '-f', str(lock_path)
                        ], check=True, capture_output=True)
                        logger.info(f"已删除锁定文件: {lock_file}")
                    except Exception as e:
                        logger.warning(f"删除锁定文件失败 {lock_file}: {e}")
            
            # 清理可能存在的临时文件
            temp_patterns = ['*.tmp', '*.lock', '*.pid']
            for pattern in temp_patterns:
                for temp_file in repo_path.glob(pattern):
                    try:
                        # 使用sudo删除临时文件
                        subprocess.run([
                            'sudo', 'rm', '-f', str(temp_file)
                        ], check=True, capture_output=True)
                        logger.info(f"已删除临时文件: {temp_file}")
                    except Exception as e:
                        logger.warning(f"删除临时文件失败 {temp_file}: {e}")
                        
        except Exception as e:
            logger.warning(f"清理Git锁定文件时发生错误: {e}")
    


    def _process_single_instance(self, instance: CompilerErrorInstance, repairer: PerfectLocalizationDirectRepair, has_repair_data: bool, dataset_name: str = None, repo_path: str = None, existing_results: Dict[Tuple[int, int], Dict] = None) -> Optional[Tuple[int, int, int, int, float]]:
        """
        处理单个错误实例 - 用于并行处理
        
        Args:
            instance: 编译错误实例（每个实例对应一个错误）
            repairer: 修复器
            has_repair_data: 是否有修复数据
            dataset_name: 数据集名称
            repo_path: 仓库路径
            existing_results: 已存在的结果
            
        Returns:
            Optional[Tuple[int, int, int, int, float]]: (successful_fixes, exact_matches, valid_patches, total_errors, total_time)
        """
        temp_repo_path = None
        try:
            # 检查断点续跑：跳过已处理的错误
            if existing_results:
                if (instance.instance_index, instance.error_index) in existing_results:
                    logger.info(f"实例 {instance.instance_index} 错误 {instance.error_index} 已处理过，跳过")
                    return (0, 0, 0, 0, 0.0)
            
            logger.info(f"评估实例 {instance.instance_index} 错误 {instance.error_index}")
            
            # 第一步：选择合适的job
            project_name = self.dataset_name
            if not project_name:
                logger.error("无法确定项目名称")
                return None
            
            # 加载项目配置
            project_configs_path = Path(__file__).parent.parent.parent.parent / "project_configs.json"
            try:
                with open(project_configs_path, 'r', encoding='utf-8') as f:
                    project_configs = json.load(f)
                project_config = project_configs.get(project_name, {})
            except Exception as e:
                logger.warning(f"读取project_configs.json失败: {e}")
                project_config = {}
            
            # 获取job名称
            job_names = instance.job_name
            
            # 选择合适的job
            job_name_str = self._select_compilable_job(job_names, project_config)
            
            # 如果没有找到可编译的job，跳过此条记录
            if job_name_str is None:
                logger.warning(f"⚠️ 跳过修复: 没有找到可编译的job")
                return None

            # 创建临时仓库
            temp_repo_path = self._create_temp_repo(instance.instance_index, repo_path, dataset_name)
            if not temp_repo_path:
                logger.error(f"无法为实例 {instance.instance_index} 创建临时仓库")
                return None
            
            # 第一步：在临时仓库中 checkout 到 failure commit
            if not self._checkout_to_commit(Path(temp_repo_path), instance.failure_commit):
                logger.error(f"无法 checkout 到 failure commit: {instance.failure_commit}")
                return None
            
            # 处理单个错误
            start_time = time.time()
            error_line = instance.error_line
            error_detail = instance.error_detail
            
            logger.info(f"处理错误: {error_line}")
            
            # 生成修复补丁（带重试机制）
            diff_text = repairer.generate_fix_for_error(instance, 0, temp_repo_path)
            
            # 如果第一次没有生成补丁，进行重试
            if not diff_text:
                logger.warning(f"第一次未生成修复补丁，进行重试...")
                diff_text = repairer.generate_fix_for_error(instance, 0, temp_repo_path)
            
            end_time = time.time()
            error_time = end_time - start_time
            
            # 如果重试后仍然没有生成补丁，标记为修复失败
            if not diff_text:
                logger.error(f"重试后仍未生成修复补丁，标记为修复失败")
                
                # 读取临时目录中的 trajectories（即使修复失败也可能有轨迹信息）
                trajectories = self._read_trajectories_from_temp_repo(temp_repo_path)
                
                # 构建失败的结果数据
                error_result = {
                    'is_successful': False,
                    'is_exact_match': False,
                    'is_valid': False,
                    'error_line': error_line,
                    'diff_text': "",  # 没有生成 diff
                    'errors_before': instance.original_error_lines,
                    'errors_after': instance.original_error_lines,  # 修复失败，错误列表不变
                    'error_detail': error_detail,
                    'time': error_time,
                    'evaluation_mode': 'full' if has_repair_data else 'patch_generation_only',
                    'trajectories': trajectories
                }
                
                # 立即保存失败结果
                result_data = {
                    'instance_index': instance.instance_index,
                    'error_index': instance.error_index,
                    'error_result': error_result,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                self._append_result(repairer.model_name, result_data, dataset_name)
                
                logger.info(f"❌ 实例 {instance.instance_index} 错误 {instance.error_index} 修复失败：未生成补丁")
                return (0, 0, 0, 1, error_time)
            
            # 构建 all_errors_before
            all_errors_before = instance.original_error_lines
            
            # 评估修复成功性（OpenHands 已直接修改文件，直接编译验证）
            is_successful, errors_after, applied_diff = self._evaluate_openhands_fix_success_from_diff(
                diff_text, instance, 0, instance.instance_index, all_errors_before, temp_repo_path
            )
            
            # 如果返回None，表示未检测到指定错误，跳过该记录
            if is_successful is None:
                logger.info(f"跳过错误，未检测到指定错误")
                return (0, 0, 0, 0, error_time)
            
            logger.info(f"实例 {instance.instance_index} 错误 {instance.error_index} 修复的错误 {error_line}")
            if is_successful:
                logger.info(f"✅ 实例 {instance.instance_index} 错误 {instance.error_index} 成功修复编译错误")
            else:
                logger.info(f"❌ 实例 {instance.instance_index} 错误 {instance.error_index} 未修复编译错误")

            logger.info(f"实例 {instance.instance_index} 错误 {instance.error_index} 修复前错误列表 (共{len(all_errors_before)}个):")
            for i, error in enumerate(all_errors_before, 1):
                logger.info(f"  {i}. {error}")
        
            logger.info(f"实例 {instance.instance_index} 错误 {instance.error_index} 修复后错误列表 (共{len(errors_after)}个):")
            for i, error in enumerate(errors_after, 1):
                logger.info(f"  {i}. {error}")
            # 打印 diff 信息用于调试
            logger.info(f"📋 实例 {instance.instance_index} 错误 {instance.error_index} 生成的 diff:")
            logger.info(f"  Diff 长度: {len(diff_text)} 字符")
            logger.info(f"  Diff 内容:")
            logger.info(f"    {'='*50}")
            logger.info(diff_text)
            logger.info(f"    {'='*50}")
            
            # 评估结果
            is_exact_match = False
            is_valid = False
            
            successful_fixes = 1 if is_successful else 0
            
            # 如果有修复数据，进行精确匹配和有效性评估
            if has_repair_data:
                # 评估精确匹配
                #is_exact_match = self._evaluate_exact_match(patches, instance, 0)
                #if is_exact_match:
                    #exact_matches = 1
                
                # 评估补丁有效性
                #is_valid = self._evaluate_patch_validity(patches, instance, 0)
                #if is_valid:
                #    valid_patches = 1
                pass
            
            # 读取临时目录中的 trajectories
            trajectories = self._read_trajectories_from_temp_repo(temp_repo_path)
            
            # 构建单条结果数据
            error_result = {
                'is_successful': is_successful,
                'is_exact_match': is_exact_match,
                'is_valid': is_valid,
                'error_line': error_line,
                'diff_text': diff_text,  # 原始 diff 文本
                'errors_before': instance.original_error_lines,
                'errors_after': errors_after,
                'error_detail': error_detail,
                'time': error_time,
                'evaluation_mode': 'full' if has_repair_data else 'patch_generation_only',
                'trajectories': trajectories
            }
            
            # 立即保存单条结果
            result_data = {
                'instance_index': instance.instance_index,
                'error_index': instance.error_index,
                'error_result': error_result,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            self._append_result(repairer.model_name, result_data, dataset_name)
        
            logger.info(f"实例 {instance.instance_index} 错误 {instance.error_index} 处理完成")
            
            return (successful_fixes, 0, 0, 1, error_time)
            
        except Exception as e:
            logger.error(f"处理实例 {instance.instance_index} 时发生错误: {e}")
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
            return None
        finally:
            # 清理临时仓库
            if temp_repo_path:
                try:
                    self._cleanup_temp_repo(temp_repo_path)
                    logger.info(f"实例 {instance.instance_index} 临时仓库清理完成")
                except Exception as cleanup_e:
                    logger.warning(f"清理临时仓库时发生错误: {cleanup_e}")

    def _select_compilable_job(self, job_names, project_config):
        """根据项目配置选择可编译的job"""
        if not project_config:
            # 如果没有项目配置，返回第一个job
            if isinstance(job_names, list) and job_names:
                return job_names[0]
            return job_names
        
        reproducible_jobs = project_config.get('reproducible_jobs', [])
        
        if not reproducible_jobs:
            logger.warning("项目配置中没有可复现的job列表，使用第一个job")
            if isinstance(job_names, list) and job_names:
                return job_names[0]
            return job_names
        
        # 如果job_names是单个字符串，转换为列表
        if isinstance(job_names, str):
            job_names = [job_names]
        
        # 优先选择在reproducible_jobs中的job
        for job in job_names:
            if job in reproducible_jobs:
                logger.info(f"选择可编译的job: {job}")
                return job
        
        # 如果没有找到可编译的job，返回None
        logger.error("没有找到可编译的job，跳过此记录")
        return None
    
    def _create_temp_repo(self, instance_index: int, source_repo_path: str, dataset_name: str) -> Optional[str]:
        """
        为指定实例创建临时仓库副本
        
        Args:
            instance_index: 实例索引
            source_repo_path: 源仓库路径
            dataset_name: 数据集名称
            
        Returns:
            Optional[str]: 临时仓库路径，如果创建失败则返回None
        """
        try:
            if not os.path.exists(source_repo_path):
                logger.error(f"源仓库路径不存在: {source_repo_path}")
                return None
            
            # 创建临时目录
            temp_dir = tempfile.mkdtemp(prefix=f"{dataset_name}_repo_temp_{instance_index+1}_")
            
            # 复制仓库
            try:
                rsync_cmd = [
                    'sudo',
                    'rsync', 
                    '-a',  # 归档模式
                    '--exclude=.git/index.lock',  # 排除锁定文件
                    '--exclude=build/',  # 排除build目录
                    '--exclude=artifacts/',  # 排除artifacts目录
                    '--exclude=*.tmp',  # 排除临时文件
                    '--exclude=*.log',  # 排除日志文件
                    f"{source_repo_path}/", 
                    temp_dir
                ]
                
                rsync_process = subprocess.run(rsync_cmd, capture_output=True, text=True)
                
                if rsync_process.returncode != 0:
                    logger.error(f"rsync复制失败: {rsync_process.stderr}")
                    raise Exception("rsync复制失败")
                
                logger.info("rsync复制仓库成功")
                
                # 修复临时目录权限
                self._fix_permission(temp_dir)
                

            except FileNotFoundError:
                logger.error("rsync命令不可用")
                raise Exception("rsync命令不可用")
            except Exception as e:
                logger.error(f"rsync复制异常: {e}")
                raise e
            
            # 清理Git锁定文件
            self._cleanup_git_locks(Path(temp_dir))
            
            # 配置Git安全目录
            self._setup_git_safe_directory(temp_dir)
            
            logger.info(f"临时仓库创建成功: {temp_dir}")
            return temp_dir
            
        except Exception as e:
            logger.error(f"创建临时仓库失败: {e}")
            # 如果创建失败，尝试清理已创建的临时目录
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception:
                    pass
            return None
    
    def _fix_permission(self, directory_path: str) -> bool:
        """
        修复目录权限，设置777权限和ubuntu所有权
        
        Args:
            directory_path: 需要修复权限的目录路径
            
        Returns:
            bool: 是否成功修复权限
        """
        try:
            logger.info(f"开始修复目录权限: {directory_path}")
            
            # 步骤1: 设置777权限
            logger.info("设置777权限...")
            chmod_result = subprocess.run([
                'sudo', 'chmod', '-R', '777', directory_path
            ], check=True, capture_output=True, text=True)
            
            if chmod_result.returncode == 0:
                logger.info("777权限设置成功")
            else:
                logger.warning(f"777权限设置失败: {chmod_result.stderr}")
                return False
            
            # 步骤2: 设置ubuntu所有权
            logger.info("设置ubuntu所有权...")
            chown_result = subprocess.run([
                'sudo', 'chown', '-R', 'ubuntu:ubuntu', directory_path
            ], check=True, capture_output=True, text=True)
            
            if chown_result.returncode == 0:
                logger.info("ubuntu所有权设置成功")
            else:
                logger.warning(f"ubuntu所有权设置失败: {chown_result.stderr}")
                return False
            
            logger.info(f"目录权限修复完成: {directory_path}")
            return True
            

        except Exception as e:
            logger.error(f"权限修复异常: {e}")
            return False

    def _setup_git_safe_directory(self, directory_path: str) -> bool:
        """
        设置Git安全目录，使用通配符简化配置
        
        Args:
            directory_path: 需要添加到安全目录的路径
            
        Returns:
            bool: 是否成功设置
        """
        try:
            # 检查是否已经配置了通配符
            result = subprocess.run([
                "git", "config", "--global", "--get-all", "safe.directory"
            ], capture_output=True, text=True)
            
            safe_directories = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            # 如果已经有通配符配置，直接返回
            if '*' in safe_directories:
                logger.debug("Git已配置通配符安全目录(*)，无需添加具体目录")
                return True
            
            # 如果没有通配符，添加通配符配置
            subprocess.run([
                "git", "config", "--global", "--add", "safe.directory", "*"
            ], check=True, capture_output=True)
            
            logger.info("已添加通配符安全目录配置")
            return True
            
        except Exception as e:
            logger.warning(f"设置Git安全目录失败: {e}")
            return False

    def _read_trajectories_from_temp_repo(self, temp_repo_path: str) -> str:
        """
        读取临时仓库中的 trajectories 文件内容
        
        Args:
            temp_repo_path: 临时仓库路径
            
        Returns:
            str: trajectories 文件的内容，如果不存在则返回空字符串
        """
        try:
            if not temp_repo_path or not os.path.exists(temp_repo_path):
                logger.warning(f"临时仓库路径不存在: {temp_repo_path}")
                return ""
            
            trajectories_path = os.path.join(temp_repo_path, 'trajectories')
            if not os.path.exists(trajectories_path):
                logger.info(f"trajectories 文件不存在: {trajectories_path}")
                return ""
            
            # 读取 trajectories 文件内容
            with open(trajectories_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"成功读取 trajectories 文件，内容长度: {len(content)} 字符")
            return content
            
        except Exception as e:
            logger.error(f"读取 trajectories 文件时发生错误: {e}")
            return ""

    def _cleanup_temp_repo(self, temp_repo_path: str):
        """
        清理临时仓库
        
        Args:
            temp_repo_path: 临时仓库路径
        """
        try:
            if temp_repo_path and os.path.exists(temp_repo_path):
                # 先清理Git锁定文件
                self._cleanup_git_locks(Path(temp_repo_path))
                
                # 直接使用sudo删除
                subprocess.run([
                    'sudo', 'rm', '-rf', temp_repo_path
                ], check=True, capture_output=True, text=True)
                logger.info(f"临时仓库清理完成: {temp_repo_path}")
            elif temp_repo_path:
                logger.warning(f"临时仓库路径不存在: {temp_repo_path}")
        except Exception as e:
            logger.warning(f"清理临时仓库失败: {e}")            elif temp_repo_path:
                logger.warning(f"临时仓库路径不存在: {temp_repo_path}")
        except Exception as e:
            logger.warning(f"清理临时仓库失败: {e}")
            elif temp_repo_path:
                logger.warning(f"临时仓库路径不存在: {temp_repo_path}")
        except Exception as e:
            logger.warning(f"清理临时仓库失败: {e}")
