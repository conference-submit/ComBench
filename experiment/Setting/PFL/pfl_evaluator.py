#!/usr/bin/env python3
"""
Perfect Localization Direct Repair (PFL) evaluation system

This module implements an evaluation framework for compilation error repair tasks, supporting multiple evaluation metrics and baseline models.
"""

import json
import time
import difflib
from typing import Dict, List, Optional, Tuple, Any, Union, NamedTuple
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod
import logging
from datetime import datetime
import os
import shutil
import subprocess
import whatthepatch
import sys
import tempfile
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import multiprocessing
import tiktoken

from model.common import call_llm, set_model, get_current_model
from model.register import register_all_models, get_available_models, get_model_info
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from patch_utils import (
    extract_json_from_response, 
    convert_to_diff_format, 
    extract_original_code_from_file,
    apply_search_replace_patch,
    get_file_content_from_git,
    get_file_content_from_git_with_path,
    get_file_content_from_local
)

logger = logging.getLogger(__name__)





@dataclass
class CompilerError:
    """Compilation error information"""
    error_line: str
    error_details: List[str] = field(default_factory=list)
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None


@dataclass
class RepairPatch:
    """Fix patch"""
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
class SearchReplacePatch:
    """Search-Replace format fix patch"""
    error_line: str
    file_path: str
    start_line: int
    end_line: int
    original_code: str
    fixed_code: str
    confidence: float = 0.5
    explanation: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'error_line': self.error_line,
            'file_path': self.file_path,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'original_code': self.original_code,
            'fixed_code': self.fixed_code,
            'confidence': self.confidence,
            'explanation': self.explanation
        }


@dataclass
class CompilerErrorInstance:
    """Compilation error instance"""
    failure_commit: str
    repair_commit: str
    error_lines: List[str]
    error_details: List[str]
    workflow_id: int
    job_id: int
    workflow_name: str
    job_name: str
    diffs: List[Dict]
    repair_source: str
    # Additional compilation-related path information (from data files, such as openssl_repair_analysis.jsonl)
    compilation_related_paths: Dict = field(default_factory=dict)
    compilation_related_paths_details: List[Dict] = field(default_factory=list)

    
    def to_dict(self) -> Dict:
        return {
            'failure_commit': self.failure_commit,
            'repair_commit': self.repair_commit,
            'error_lines': self.error_lines,
            'error_details': self.error_details,
            'workflow_id': self.workflow_id,
            'job_id': self.job_id,
            'workflow_name': self.workflow_name,
            'job_name': self.job_name,
            'diffs': self.diffs,
            'repair_source': self.repair_source,
            'compilation_related_paths': self.compilation_related_paths,
            'compilation_related_paths_details': self.compilation_related_paths_details
        }


@dataclass
class EvaluationResult:
    """Evaluation result"""
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
    """Diff content structure"""
    removed: List[str]  # Removed lines
    added: List[str]    # Added lines
    context: List[str]  # Context lines
    original: str       # Original diff content


class PerfectLocalizationDirectRepair:
    """Perfect Localization Direct Repair evaluator"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        set_model(model_name)
        logger.info(f"Initializing Perfect Localization Direct Repair evaluator, using model: {model_name}")
    
    def generate_fix_for_error(self, instance: CompilerErrorInstance, error_idx: int, repo_path: Optional[str] = None) -> List[SearchReplacePatch]:
        """Generate fix patches, may return multiple patches"""
        try:
            messages = self.create_repair_prompt(instance, error_idx, repo_path)
            
            # Call large model to generate fixes (with exponential backoff retry mechanism)
            max_retries = 3  # Maximum retry count
            base_delay = 3.0  # Base delay time (seconds)
            response = None
            json_response = None
            
            for attempt in range(max_retries + 1):  # 0 to 3, total 4 attempts
                if attempt == 0:
                    logger.info(f"Error {error_idx + 1} attempt {attempt + 1} calling large model...")
                else:
                    # Calculate exponential backoff delay time
                    delay = base_delay * (2 ** (attempt - 1))  # 1s, 2s, 4s
                    logger.warning(f"Error {error_idx + 1} retry {attempt + 1} calling large model, waiting {delay:.1f} seconds...")
                    time.sleep(delay)
                
                response = call_llm(messages, model=self.model_name, temperature=0.1)
                
                if not response:
                    if attempt < max_retries:
                        logger.warning(f"❌ Error {error_idx + 1} attempt {attempt + 1} no response received, preparing to retry...")
                    else:
                        logger.error(f"❌ Error {error_idx + 1} still no response after {max_retries + 1} attempts")
                    continue
                
                # Extract fixed code
                json_response = extract_json_from_response(response)
                if json_response and 'locations' in json_response and json_response['locations']:
                    logger.info(f"✅ Error {error_idx + 1} attempt {attempt + 1} successfully extracted valid patch")
                    break
                else:
                    if attempt < max_retries:
                        logger.warning(f"❌ Error {error_idx + 1} attempt {attempt + 1} no valid patch extracted, preparing to retry...")
                    else:
                        logger.error(f"❌ Error {error_idx + 1} still no valid patch extracted after {max_retries + 1} attempts")
            
            if not response:
                logger.error(f"Large model call failed")
                return []
            
            if not json_response or 'locations' not in json_response:
                logger.warning("Unable to extract fix code from response")
                return []
            
            # Get error information
            diff = instance.diffs[error_idx]
            error_line = diff['error_line']
            
            # Create patch for each fix
            patches = []
            for location in json_response.get('locations', []):
                file_path = location.get('file_path', '')
                start_line = location.get('start_line', 1)
                end_line = location.get('end_line', 1)
                original_code = location.get('original_code', '')
                fixed_code = location.get('fixed_code', '')
                confidence = location.get('confidence', 0.5)
                explanation = location.get('explanation', '')
                
                if not original_code:
                    logger.warning("original_code 为空，将根据行号插入新代码")

                patch = SearchReplacePatch(
                    error_line=instance.error_lines[error_idx],
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    original_code=original_code,
                    fixed_code=fixed_code,
                    confidence=confidence,
                    explanation=explanation
                )
                patches.append(patch)
            
            # 对patches进行去重
            unique_patches = self._deduplicate_patches(patches)
            
            # 记录去重统计信息
            original_count = len(patches)
            unique_count = len(unique_patches)
            if original_count != unique_count:
                logger.info(f"PFL patches去重: {original_count} -> {unique_count} (减少 {original_count - unique_count} 个重复)")
            
            return unique_patches
            
        except Exception as e:
            logger.error(f"生成修复补丁时发生错误: {e}")
            return []
    
    def _deduplicate_patches(self, patches: List[SearchReplacePatch]) -> List[SearchReplacePatch]:
        """对patches进行去重，基于patch内容完全相同"""
        seen_patches = set()
        unique_patches = []
        
        for patch in patches:
            # 创建patch的唯一标识，对代码内容进行normalize（去除空格）
            patch_key = (
                patch.file_path,
                self._normalize_code(patch.original_code),
                self._normalize_code(patch.fixed_code),
                patch.start_line,
                patch.end_line
            )
            
            # 去重：如果patch内容完全相同，只保留第一个
            if patch_key not in seen_patches:
                seen_patches.add(patch_key)
                unique_patches.append(patch)
                logger.debug(f"保留patch: {patch.file_path} - {patch.original_code} -> {patch.fixed_code}")
            else:
                logger.debug(f"跳过重复patch: {patch.file_path} - {patch.original_code} -> {patch.fixed_code}")
        
        return unique_patches
    
    def _normalize_code(self, code: str) -> str:
        """
        规范化代码：
        1. 去除所有空白字符（空格、制表符、换行符等）
        
        Args:
            code: 要处理的代码
            
        Returns:
            处理后的代码
        """
        import re
        
        # 去除所有空白字符（空格、制表符、换行符等）
        return re.sub(r'\s+', '', code) if code else ''
    

    
    def create_repair_prompt(self, instance: CompilerErrorInstance, error_idx: int, repo_path: Optional[str] = None) -> List[Dict]:
        """创建修复提示"""
        error_line = instance.error_lines[error_idx]
        error_detail = instance.error_details[error_idx]
        diff = instance.diffs[error_idx]
        
        logger.debug(f"创建修复提示 - 错误行: {error_line}")
        logger.debug(f"错误详情: {error_detail}")
        logger.debug(f"仓库路径: {repo_path}")
        
        # 获取当前模型的max_input_tokens限制
        current_model = get_current_model()
        model_info = get_model_info(current_model)
        max_input_tokens = model_info.get('max_input_tokens', 128000)  # 默认值
        logger.info(f"当前模型 {current_model} 的max_input_tokens限制: {max_input_tokens}")
        
        # 收集所有需要修改的文件及其内容
        files_content = []
        
        # 本地辅助函数：尝试添加文件内容到 files_content，避免重复
        def _try_add_file(file_name: str, priority: str = "unknown"):
            try:
                if not file_name:
                    return False
                if file_name in [f[0] for f in files_content]:
                    return False
                
                content = None
                real_file_path = file_name  # 默认使用原始文件名
                if repo_path:
                    logger.debug(f"尝试本地查找文件: {file_name} (优先级: {priority})")
                    content, found_path = get_file_content_from_local(repo_path, file_name)
                    if content and found_path and found_path != file_name:
                        real_file_path = found_path
                        logger.info(f"本地找到真实文件路径: {file_name} -> {real_file_path}")
                    if not content:
                        logger.debug(f"本地查找失败，尝试git方式: {file_name}")
                        content, real_file_path = get_file_content_from_git_with_path(
                            repo_path,
                            instance.failure_commit,
                            file_name
                        )
                        if content and real_file_path and real_file_path != file_name:
                            logger.info(f"Git找到真实文件路径: {file_name} -> {real_file_path}")
                
                if content:
                    lines = content.splitlines()
                    numbered_content = '\n'.join(f"{i+1:4d}: {line}" for i, line in enumerate(lines))
                    
                    # 检查实际内容是否会超过token限制
                    actual_content = f"\n文件: {real_file_path}\n```cpp\n{numbered_content}\n```\n"
                    if not check_token_limit(actual_content):
                        logger.info(f"实际文件内容 {real_file_path} 会超过token限制，跳过")
                        return False
                    
                    files_content.append((real_file_path, numbered_content))
                    logger.info(f"成功获取文件内容: {real_file_path} (优先级: {priority}), 长度: {len(content)}")
                    return True
                else:
                    # 按需求：无法获取内容也需要追加文件占位，内容为空串
                    # 预测添加文件后的token数量（空内容）
                    predicted_content = f"\n文件: {real_file_path}\n```cpp\n\n```\n"
                    if not check_token_limit(predicted_content):
                        logger.info(f"添加空文件 {real_file_path} 后会超过token限制，跳过")
                        return False
                    
                    files_content.append((real_file_path, ''))
                    logger.warning(f"无法获取文件内容: {real_file_path}，已使用空内容占位")
                    return True
            except Exception as _e:
                logger.warning(f"添加文件失败 {file_name}: {_e}")
                return False
        
        # 使用tiktoken计算token数量
        def count_tokens(text: str) -> int:
            if current_model.startswith("gpt-") or current_model.startswith("claude-") or current_model.startswith("gemini-"):
                encoding = tiktoken.get_encoding("o200k_base")
                return len(encoding.encode(text))
            
            # Qwen模型
            elif current_model.startswith("qwen"):
                from transformers import AutoTokenizer
                # 使用正确的模型标识符
                model_name = current_model.replace("qwen", "Qwen/Qwen")
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                tokens = tokenizer.encode(text)
                return len(tokens)
            
            # DeepSeek模型
            elif current_model.startswith("deepseek-"):
                from transformers import AutoTokenizer
                # 使用transformers处理DeepSeek模型
                model_name = current_model.replace("deepseek-", "deepseek-ai/deepseek-")
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                tokens = tokenizer.encode(text)
                return len(tokens)
            
            # Kimi-K2模型
            elif current_model.startswith("Kimi-K2"):
                from transformers import AutoTokenizer
                # 使用transformers处理Kimi-K2模型
                model_name = current_model.replace("Kimi-K2", "moonshotai/Kimi-K2-Instruct")
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                tokens = tokenizer.encode(text)
                return len(tokens)
            
            # 其他模型
            else:
                raise NotImplementedError(f"未知模型 {current_model} 不支持token计算")
        
        # 定义统一的prompt模板
        PROMPT_TEMPLATE = """You are a professional C/C++ compilation error repair expert. Please fix the code errors based on the following information:

**Error Information:**
{error_line}

**Error Location:**
Files to be modified:
{files_str}

**Error Details:**
{error_detail}

**Repair Requirements:**
1. Only fix the code that causes compilation errors
2. Maintain the original logic and functionality of the code
3. Use minimal modifications
4. Ensure the fixed code can compile successfully
5. For each file that needs modification, provide the corresponding repair code

**Important: Please provide the repair code strictly in the following JSON format:**

```json
{{
    "locations": [
        {{
            "file_path": "file path",
            "start_line": start line number,
            "end_line": end line number,
            "original_code": "original code to be replaced (use \\n for multiple lines)",
            "fixed_code": "fixed code (use \\n for multiple lines)"
        }}
    ],
    "confidence": 0.8,
    "explanation": "repair explanation",
    "fix_description": "detailed repair description"
}}
```

**Format Description:**
1. file_path: The file path to be modified
2. start_line/end_line: The line number range to be modified
3. original_code: The original code to be replaced, must exactly match the code in the file (without line numbers)
4. fixed_code: The fixed code (without line numbers)
5. confidence: Confidence level (0.0-1.0)
6. explanation: Brief repair explanation
7. fix_description: Detailed repair description

**Important Notes:**
1. original_code must exactly match the actual code in the file
2. If modifying multiple lines, both original_code and fixed_code should contain complete lines
3. If inserting new code (such as adding headers, functions, etc.), you can set original_code to an empty string ""
4. Line number ranges should be accurate and include all lines to be modified
5. Only provide repair-related code, do not include explanations or other content
6. Ensure each modification has sufficient context information

**Code Insertion Examples:**
- Add header file at the beginning of file: start_line=1, end_line=0, original_code="", fixed_code="#include <string>"
- Add variable inside function: start_line=5, end_line=5, original_code="", fixed_code="    int x = 42;"
- Add function at the end of file: start_line=10, end_line=10, original_code="", fixed_code="void helper() {{ }}"
"""

        # 检查是否超过token限制
        def check_token_limit(additional_content: str = "") -> bool:
            """
            检查添加指定内容后是否会超过token限制
            
            Args:
                additional_content: 要添加的额外内容
                
            Returns:
                bool: 是否在限制范围内
            """
            current_files_str = ""
            for file_name, content in files_content:
                current_files_str += f"\n文件: {file_name}\n```cpp\n{content}\n```\n"
            
            # 添加额外内容
            if additional_content:
                current_files_str += additional_content
            
            # 使用统一的模板
            full_prompt = PROMPT_TEMPLATE.format(
                error_line=error_line,
                files_str=current_files_str,
                error_detail=error_detail
            )
            token_count = count_tokens(full_prompt)
            logger.debug(f"当前token数量: {token_count}, 限制: {max_input_tokens}")
            return token_count < max_input_tokens
        
        # 按优先级从compilation_related_paths_details/compilation_related_paths中获取文件
        def collect_files_by_priority():
            """按优先级收集文件，在达到token限制时停止"""
            priority_order = ['error_file', 'direct_includes', 'indirect_includes']
            candidate_files_by_priority = {'error_file': set(), 'direct_includes': set(), 'indirect_includes': set()}
            
            try:
                details = getattr(instance, 'compilation_related_paths_details', None)
                if details and isinstance(details, list):
                    # 根据error_idx获取对应的item，而不是遍历所有item
                    if error_idx < len(details):
                        item = details[error_idx]
                        if isinstance(item, dict):
                            for key in priority_order:
                                val = item.get(key)
                                if isinstance(val, list):
                                    for v in val:
                                        if isinstance(v, str) and v.strip():
                                            candidate_files_by_priority[key].add(v.strip())
                                elif isinstance(val, str) and val.strip():
                                    candidate_files_by_priority[key].add(val.strip())
                        logger.info(f"使用compilation_related_paths_details[{error_idx}]获取文件信息")
                    else:
                        logger.warning(f"error_idx {error_idx} 超出compilation_related_paths_details长度 {len(details)}")
            except Exception as e:
                logger.warning(f"解析compilation_related_paths失败: {e}")
            
            # 按优先级顺序添加文件
            for priority in priority_order:
                candidate_files = candidate_files_by_priority[priority]
                if candidate_files:
                    logger.info(f"按优先级 {priority} 添加文件: {candidate_files}")
                    for file_name in candidate_files:
                        # 预测添加文件后的token数量
                        if _try_add_file(file_name, priority):
                            # 检查是否超过token限制
                            if not check_token_limit():
                                logger.info(f"已达到token限制，停止添加 {priority} 优先级文件")
                                return
                    logger.info(f"完成添加 {priority} 优先级文件，当前文件数量: {len(files_content)}")
        
        # 执行按优先级收集文件
        collect_files_by_priority()
        
        # 如果按优先级收集文件失败，尝试从diff中获取文件
        if not files_content:
            logger.debug(f"按优先级收集文件失败，尝试从diff中获取文件，relevant_repairs: {diff.get('relevant_repairs', [])}")
            for repair in diff.get('relevant_repairs', []):
                if repair.get('modified_file'):
                    file_name = repair['modified_file']
                    logger.debug(f"尝试获取文件: {file_name}")
                    _try_add_file(file_name, "diff")
        
        # 如果仍然没有文件，尝试从错误信息中提取文件路径
        if not files_content and repo_path:
            logger.debug("前两步没有找到文件，尝试从错误信息中提取文件路径")
            # 从错误信息中提取文件路径
            import re
            # 匹配常见的文件路径模式
            file_patterns = [
                r'([^:\s]+\.(?:h|hpp|c|cpp|cc|cxx))',  # 匹配 .h, .hpp, .c, .cpp, .cc, .cxx 文件
                r'([^:\s]+/[^:\s]+\.(?:h|hpp|c|cpp|cc|cxx))',  # 匹配带路径的文件
                r'([^:\s]+\\[^:\s]+\.(?:h|hpp|c|cpp|cc|cxx))',  # 匹配Windows路径
            ]
            
            extracted_files = set()
            for pattern in file_patterns:
                matches = re.findall(pattern, error_line + " " + error_detail)
                logger.debug(f"正则匹配结果: {matches}")
                for match in matches:
                    # 清理文件路径
                    file_path = match.strip()
                    # 移除开头的 ./
                    if file_path.startswith('./'):
                        file_path = file_path[2:]
                    # 移除开头的 /__w/rocksdb/rocksdb/ 等CI路径前缀
                    if '/__w/' in file_path:
                        parts = file_path.split('/__w/')
                        if len(parts) > 1:
                            file_path = parts[1].split('/', 2)[-1] if len(parts[1].split('/', 2)) > 2 else parts[1]
                    
                    logger.info(f"提取的文件路径: {file_path}")
                    if file_path and file_path not in extracted_files:
                        extracted_files.add(file_path)
                        
                        # 获取文件内容
                        _try_add_file(file_path, "extracted")
                        if files_content:
                            break  # 找到第一个有效文件就停止
        
        # 构建文件列表和内容字符串
        files_str = ""
        for file_name, content in files_content:
            files_str += f"\n文件: {file_name}\n```cpp\n{content}\n```\n"
        if not files_str:
            files_str = "无法获取文件内容"
            logger.warning("最终无法获取任何文件内容")
        else:
            # 计算最终使用的token数量
            final_prompt = PROMPT_TEMPLATE.format(
                error_line=error_line,
                files_str=files_str,
                error_detail=error_detail
            )
            final_tokens = count_tokens(final_prompt)
            logger.info(f"成功获取了 {len(files_content)} 个文件的内容，最终使用token数量: {final_tokens}/{max_input_tokens} ({final_tokens/max_input_tokens*100:.1f}%)")
        
        # 使用统一的模板构建最终的prompt
        prompt = PROMPT_TEMPLATE.format(
            error_line=error_line,
            files_str=files_str,
            error_detail=error_detail
        )

        return [{"role": "user", "content": prompt}]
    
    def _extract_code_from_response(self, response: str) -> Optional[Dict[str, str]]:
        """从响应中提取代码，返回文件名到代码的映射"""
        import re
        import json
        
        # 首先尝试提取JSON响应
        json_response = extract_json_from_response(response)
        if json_response and 'locations' in json_response:
            # 将JSON格式转换为diff格式
            fixes = {}
            for location in json_response.get('locations', []):
                file_path = location.get('file_path', '')
                start_line = location.get('start_line', 1)
                end_line = location.get('end_line', 1)
                original_code = location.get('original_code', '')
                fixed_code = location.get('fixed_code', '')
                
                # 转换为diff格式
                diff_content = convert_to_diff_format(start_line, end_line, original_code, fixed_code)
                fixes[file_path] = diff_content
            
            return fixes
        
        # 如果JSON解析失败，尝试查找diff代码块（向后兼容）
        pattern = r'```diff:(.*?)\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        if not matches:
            # 尝试查找普通代码块
            code_pattern = r'```(?:diff|cpp|c\+\+|c)?\s*\n(.*?)\n```'
            match = re.search(code_pattern, response, re.DOTALL)
            if match:
                # 如果只找到一个代码块，假设它是第一个文件的修复
                return {'': match.group(1).strip()}
            return None
        
        # 构建文件名到规范化diff内容的映射
        fixes = {}
        for file_name, code in matches:
            # 规范化diff内容
            normalized_code = self._normalize_diff_format(code.strip(), file_name.strip())
            fixes[file_name.strip()] = normalized_code
        
        return fixes
    

    
    def _normalize_diff_format(self, diff_content: str, file_name: str) -> str:
        """
        规范化diff格式，确保符合git apply的要求
        
        Args:
            diff_content: 原始diff内容
            file_name: 文件名
            
        Returns:
            规范化后的diff内容
        """
        lines = diff_content.split('\n')
        normalized_lines = []
        
        # 查找第一个@@行
        hunk_started = False
        for i, line in enumerate(lines):
            if line.startswith('@@'):
                hunk_started = True
                # 确保@@行格式正确
                if not re.match(r'^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@', line):
                    # 尝试修复@@行格式
                    match = re.search(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
                    if match:
                        old_start = match.group(1)
                        old_count = match.group(2) or '1'
                        new_start = match.group(3)
                        new_count = match.group(4) or '1'
                        line = f'@@ -{old_start},{old_count} +{new_start},{new_count} @@'
                normalized_lines.append(line)
            elif hunk_started:
                # 确保修改行格式正确
                if line.startswith('+') or line.startswith('-') or line.startswith(' '):
                    normalized_lines.append(line)
                else:
                    # 如果不是以+/-/ 开头，添加空格作为上下文
                    normalized_lines.append(' ' + line)
            else:
                # 在第一个@@之前，跳过或作为注释处理
                continue
        
        # 如果没有找到@@行，尝试从内容推断
        if not hunk_started and lines:
            # 分析内容，确定是添加还是删除
            added_lines = []
            removed_lines = []
            context_lines = []
            
            for line in lines:
                if line.strip():
                    if line.startswith('+'):
                        added_lines.append(line)
                    elif line.startswith('-'):
                        removed_lines.append(line)
                    else:
                        context_lines.append(line)
            
            # 根据内容生成合适的@@行
            if added_lines and removed_lines:
                # 既有添加又有删除
                normalized_lines = ['@@ -1,1 +1,1 @@']
            elif added_lines:
                # 只有添加
                normalized_lines = ['@@ -0,0 +1,1 @@']
            elif removed_lines:
                # 只有删除
                normalized_lines = ['@@ -1,1 +0,0 @@']
            else:
                # 只有上下文
                normalized_lines = ['@@ -1,1 +1,1 @@']
            
            # 添加所有行
            for line in lines:
                if line.strip():
                    if line.startswith('+') or line.startswith('-'):
                        normalized_lines.append(line)
                    else:
                        normalized_lines.append(' ' + line)
        
        result = '\n'.join(normalized_lines)
        
        # 确保结果不为空
        if not result.strip():
            # 如果结果为空，生成一个默认的diff
            result = """@@ -1,1 +1,1 @@
- 
+ """
        
        return result




class PFLEvaluator:
    """PFL评估器"""
    
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
        # 注册所有可用模型
        register_all_models()
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
        记录断点续跑的进度信息（基于去重后的错误）
        
        Args:
            existing_results: 已存在的结果
            instances: 所有实例列表
        """
        if not existing_results:
            return
        
        # 统计已处理的实例和唯一错误
        processed_instances = set()
        total_processed_unique_errors = 0
        total_unique_errors = 0
        
        for instance_idx, instance in enumerate(instances):
            # 对每个实例进行去重
            unique_errors = self._deduplicate_errors(instance.error_lines, instance.error_details)
            total_unique_errors += len(unique_errors)
            
            instance_processed_unique_errors = 0
            for unique_error in unique_errors:
                error_idx = unique_error['original_idx']
                if (instance_idx + 1, error_idx + 1) in existing_results:
                    instance_processed_unique_errors += 1
                    total_processed_unique_errors += 1
            
            if instance_processed_unique_errors == len(unique_errors):
                processed_instances.add(instance_idx + 1)
        
        # 计算进度
        instance_progress = len(processed_instances) / len(instances) * 100
        error_progress = total_processed_unique_errors / total_unique_errors * 100 if total_unique_errors > 0 else 0
        
        logger.info(f"断点续跑进度统计（基于去重后的错误）:")
        logger.info(f"  已处理实例: {len(processed_instances)}/{len(instances)} ({instance_progress:.1f}%)")
        logger.info(f"  已处理唯一错误: {total_processed_unique_errors}/{total_unique_errors} ({error_progress:.1f}%)")
        logger.info(f"  完全处理的实例: {sorted(processed_instances)}")
        
        return {
            'total_unique_errors': total_unique_errors,
            'total_processed_unique_errors': total_processed_unique_errors,
            'processed_instances': processed_instances
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
            required_fields = ['is_successful', 'error_line', 'patches', 'time']
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
            if content.startswith('{') and not content.startswith('{"failure_commit"'):
                # 新格式：单个JSON对象，包含metadata和compiler_errors
                instances = self._load_json_format(content)
            else:
                # 原格式：JSONL格式，每行一个CompilerErrorInstance
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
        """加载原有的JSONL格式数据"""
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
        """解析单个实例"""
        try:
            return CompilerErrorInstance(
                failure_commit=data.get('failure_commit', ''),
                repair_commit=data.get('repair_commit', ''),
                error_lines=data.get('error_lines', []),
                error_details=data.get('error_details', []),
                workflow_id=data.get('workflow_id', 0),
                job_id=data.get('job_id', 0),
                workflow_name=data.get('workflow_name', ''),
                job_name=data.get('job_name', ''),
                diffs=data.get('diffs', []),
                repair_source=data.get('repair_source', ''),
                compilation_related_paths=data.get('compilation_related_paths', {}),
                compilation_related_paths_details=data.get('compilation_related_paths_details', [])
            )
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
        
        # 检查数据类型，确定评估模式
        has_repair_data = self._check_repair_data_availability(instances)
        if not has_repair_data:
            logger.info("检测到数据没有修复信息，将使用错误检测和补丁生成模式")
        
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
        
        # 使用线程池并行处理instances
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_instance = {
                executor.submit(self._process_single_instance, instance, i, repairer, has_repair_data, dataset_name, repo_path, existing_results): instance 
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
            total_unique_errors = progress_info['total_unique_errors']
            skipped_unique_errors = progress_info['total_processed_unique_errors']
            estimated_saved_time = skipped_unique_errors * evaluation_result.average_time
            
            logger.info(f"断点续跑总结:")
            logger.info(f"  跳过的错误比例: {skipped_unique_errors/total_unique_errors*100:.1f}%")
            logger.info(f"  估计节省时间: {estimated_saved_time:.1f}秒 ({estimated_saved_time/60:.1f}分钟)")
            logger.info(f"  实际处理错误数: {evaluation_result.total_errors}")
            logger.info(f"  实际处理时间: {evaluation_result.total_time:.1f}秒")
        
        # 保存评估汇总结果
        self._save_evaluation_summary(model_name, evaluation_result, dataset_name)
        
        return evaluation_result
    
    def _check_repair_data_availability(self, instances: List[CompilerErrorInstance]) -> bool:
        """检查数据是否包含修复信息"""
        if not instances:
            return False
        
        # 检查前几个实例是否有修复数据
        sample_size = min(5, len(instances))
        for instance in instances[:sample_size]:
            # 检查是否有repair_commit
            if instance.repair_commit:
                return True
            
            # 检查是否有相关修复
            for diff in instance.diffs:
                if diff.get('relevant_repairs') and len(diff['relevant_repairs']) > 0:
                    return True
        
        return False
    
    def _evaluate_fix_success_with_compiled_output(self, patches: List[SearchReplacePatch], instance: CompilerErrorInstance, error_idx: int, line_number: int, pre_extracted_errors: List[str], temp_repo_path: str = None) -> Tuple[Optional[bool], List[str], List[SearchReplacePatch]]:
        """
        评估修复成功性（使用已编译的输出，避免重复编译）
        
        Args:
            patches: 修复补丁列表
            instance: 编译错误实例
            error_idx: 错误索引
            line_number: 行号（用于编译项目）
            pre_extracted_errors: 预提取的错误列表
            temp_repo_path: 临时仓库路径，如果提供则使用此路径
            
        Returns:
            Tuple[Optional[bool], List[str], List[SearchReplacePatch]]: (修复是否成功, 修复后提取到的错误列表, 实际应用的补丁列表)。
            当无法继续验证时，返回 (False, [], patches)；如需跳过（未检测到指定错误）应返回 (None, errors_after, patches)。
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
            return False, [], patches
        
        # 从dataset_name中提取项目名
        project_name = self.dataset_name.split('_')[0] if self.dataset_name else None
        if not project_name:
            logger.error("无法确定项目名称")
            return False, [], patches
        
        # 获取错误信息
        error_line = instance.error_lines[error_idx]
        error_detail = instance.error_details[error_idx]
        
        # 获取仓库路径
        if temp_repo_path:
            # 使用临时仓库路径
            repo_path = Path(temp_repo_path)
        elif hasattr(self, 'repo_path') and self.repo_path:
            # 使用原始仓库路径
            repo_path = Path(self.repo_path)
        else:
            logger.error("未设置仓库路径")
            return False, [], patches
        
        if not repo_path.exists():
            logger.error(f"仓库路径不存在: {repo_path}")
            return False, [], patches
        
        try:
            
            # 步骤1: 先checkout到failure_commit状态
            if not self._checkout_to_commit(repo_path, instance.failure_commit):
                logger.error(f"无法切换到failure_commit: {instance.failure_commit}")
                return False, [], patches
            
            # 步骤2: 应用修复补丁（失败则触发重生成并重试）
            applied_ok, used_patches = self._apply_patches_with_regeneration(
                repo_path, instance, error_idx, patches, temp_repo_path
            )
            if not applied_ok:
                logger.error("应用补丁失败（包含重试）")
                return False, [], used_patches
            
            # 步骤3: 重新编译项目，检查错误是否消失
            success_after, output_after = self._compile_project(
                project_name, line_number, no_switch=True, temp_repo_path=temp_repo_path
            )
            
            # 提取修复后的错误列表
            try:
                from error_extractor import ErrorExtractor
                error_extractor = ErrorExtractor()
                errors_after = error_extractor.extract_errors(output_after) if output_after else []
            except ImportError as e:
                logger.error(f"无法导入错误提取器: {e}")
                return False, []
            
            # 计算动态line tolerance
            modification_lines = self._calculate_patch_modification_lines(patches)
            dynamic_line_tolerance = self._calculate_dynamic_line_tolerance(modification_lines)
            
            # 使用新的校验逻辑检查指定错误是否被成功修复
            if self._check_error_fixed(pre_extracted_errors, errors_after, error_line, dynamic_line_tolerance):
                return True, errors_after, used_patches
            else:
                return False, errors_after, used_patches
                    
        except Exception as e:
            logger.error(f"验证补丁时发生错误: {e}")
            return False, [], patches

    def _apply_patches_with_regeneration(self, repo_path: Path, instance: CompilerErrorInstance, error_idx: int, patches: List[SearchReplacePatch], temp_repo_path: str = None) -> Tuple[bool, List[SearchReplacePatch]]:
        """
        应用补丁；若有任一补丁应用失败，则调用模型重新生成补丁并重试一次。
        返回 (是否全部应用成功, 实际使用的补丁列表)。
        """
        # 尝试首次应用
        if self._apply_patches_sequence(repo_path, patches):
            return True, patches
        # 失败则回滚到 failure_commit 并重试（重生成人工）
        try:
            self._checkout_to_commit(repo_path, instance.failure_commit)
        except Exception:
            pass
        # 调用当前模型重新生成补丁
        try:
            current_model = get_current_model()
            repairer = PerfectLocalizationDirectRepair(current_model)
            regenerated = repairer.generate_fix_for_error(instance, error_idx, temp_repo_path)
        except Exception:
            regenerated = []
        if not regenerated:
            return False, patches
        # 再次尝试应用新补丁
        if self._apply_patches_sequence(repo_path, regenerated):
            return True, regenerated
        return False, regenerated

    def _apply_patches_sequence(self, repo_path: Path, patches: List[SearchReplacePatch]) -> bool:
        """按顺序应用一组补丁，任一失败则返回False。"""
        for patch in patches:
            try:
                if isinstance(patch, SearchReplacePatch):
                    if not apply_search_replace_patch(repo_path, patch, logger):
                        logger.error(f"应用Search-Replace补丁失败: {patch.file_path}")
                        return False
                else:
                    if not self._apply_patch(repo_path, patch.file_path, patch.fixed_code):
                        logger.error(f"应用补丁失败: {patch.file_path}")
                        return False
            except Exception as e:
                logger.error(f"应用补丁异常: {e}")
                return False
        return True

    def _evaluate_fix_success(self, patches: List[SearchReplacePatch], instance: CompilerErrorInstance, error_idx: int, line_number: int = None) -> Optional[bool]:
        """
        评估修复成功性（通过实际编译检查）- 向后兼容方法
        
        Args:
            patches: 修复补丁列表
            instance: 编译错误实例
            error_idx: 错误索引
            line_number: 行号（用于编译项目），如果为None则从instance获取
            
        Returns:
            Optional[bool]: 修复是否成功，None表示未检测到指定错误，应跳过该记录
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
            return False
        
        # 从dataset_name中提取项目名
        project_name = self.dataset_name.split('_')[0] if self.dataset_name else None
        if not project_name:
            logger.error("无法确定项目名称")
            return False
        
        # 获取错误信息
        error_line = instance.error_lines[error_idx]
        error_detail = instance.error_details[error_idx]
        
        # 获取仓库路径
        if temp_repo_path:
            # 使用临时仓库路径
            repo_path = Path(temp_repo_path)
        elif hasattr(self, 'repo_path') and self.repo_path:
            # 使用原始仓库路径
            repo_path = Path(self.repo_path)
        else:
            logger.error("未设置仓库路径")
            return False
        
        if not repo_path.exists():
            logger.error(f"仓库路径不存在: {repo_path}")
            return False
        
        try:
            # 步骤1: 编译项目，确认错误存在
            logger.info("🔍 验证错误是否存在...")
            success_before, output_before = self._compile_project(
                project_name, line_number,
                temp_repo_path=temp_repo_path
            )
            
            # 检查编译输出是否包含指定的错误
            error_exists, errors_before = self._check_error_in_output(output_before, error_line)
            logger.info(f"修复前错误列表 (共{len(errors_before)}个):")
            for i, error in enumerate(errors_before, 1):
                logger.info(f"  {i}. {error}")
            if not error_exists:
                logger.warning("未检测到指定的错误")
                logger.info("跳过该记录，不保存结果")
                return None
            else:
                logger.info("✅ 确认错误存在")
            
            # 步骤2: 先checkout到failure_commit状态
            logger.info(f"🔧 切换到failure_commit: {instance.failure_commit}")
            if not self._checkout_to_commit(repo_path, instance.failure_commit):
                logger.error(f"无法切换到failure_commit: {instance.failure_commit}")
                return False
            
            # 步骤3: 应用修复补丁
            logger.info("🔧 应用修复补丁...")
            for patch in patches:
                if isinstance(patch, SearchReplacePatch):
                    # 应用Search-Replace格式的补丁
                    if not apply_search_replace_patch(repo_path, patch, logger):
                        logger.error(f"应用Search-Replace补丁失败: {patch.file_path}")
                        return False
                else:
                    # 应用传统的diff格式补丁（向后兼容）
                    if not self._apply_patch(repo_path, patch.file_path, patch.fixed_code):
                        logger.error(f"应用补丁失败: {patch.file_path}")
                        return False
            
            # 步骤4: 重新编译项目，检查错误是否消失
            logger.info("🔍 验证错误是否修复...")
            success_after, output_after = self._compile_project(
                project_name, line_number, no_switch=True, temp_repo_path=temp_repo_path
            )
            
            # 提取修复后的错误列表
            _, errors_after = self._check_error_in_output(output_after, error_line)
            
            logger.info(f"修复后错误列表 (共{len(errors_after)}个):")
            for i, error in enumerate(errors_after, 1):
                logger.info(f"  {i}. {error}")
            
            # 计算动态line tolerance
            modification_lines = self._calculate_patch_modification_lines(patches)
            dynamic_line_tolerance = self._calculate_dynamic_line_tolerance(modification_lines)
            
            # 使用新的校验逻辑检查指定错误是否被成功修复
            if self._check_error_fixed(errors_before, errors_after, error_line, dynamic_line_tolerance):
                logger.info("✅ 指定错误已成功修复，补丁有效")
                return True
            else:
                logger.warning("❌ 应用补丁后指定错误仍然存在，补丁无效")
                return False
                    
        except Exception as e:
            logger.error(f"验证补丁时发生错误: {e}")
            return False
    
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
    

    
    def _check_error_in_output(self, output: str, error_msg: str, pre_extracted_errors: List[str] = None) -> Tuple[bool, List[str]]:
        """
        检查编译输出是否包含指定的错误，并返回提取到的错误列表
        
        Args:
            output: 编译输出内容
            error_msg: 错误信息
            pre_extracted_errors: 预提取的错误列表（可选）
            
        Returns:
            Tuple[bool, List[str]]: (是否包含指定错误, 提取到的错误列表)
        """
        if not error_msg:
            return False, []
        
        try:
            # 导入错误提取器和错误匹配器
            reproduce_path = Path(__file__).parent.parent.parent.parent / "experiment" / "reproduce"
            if str(reproduce_path) not in sys.path:
                sys.path.insert(0, str(reproduce_path))
            from error_extractor import ErrorExtractor
            from error_matcher import ErrorMatcher
            
            # 初始化错误提取器和匹配器
            error_extractor = ErrorExtractor()
            error_matcher = ErrorMatcher()
            
            # 提取编译错误
            if pre_extracted_errors is not None:
                # 使用预提取的错误列表
                extracted_errors = pre_extracted_errors
            elif output:
                # 从输出中提取错误
                extracted_errors = error_extractor.extract_errors(output)
            else:
                # 没有输出也没有预提取错误
                return False, []
            
            # 将error_msg作为实际错误，extracted_errors作为预期错误列表
            # 构建预期错误列表，每个提取的错误作为一个预期错误
            expected_errors = [{'error_lines': [extracted_error]} for extracted_error in extracted_errors]
            
            # 使用match_single_error进行匹配
            is_matched, matched_error = error_matcher.match_single_error(error_msg, expected_errors)
            if is_matched:
                return True, extracted_errors
            
            # 如果没有通过错误匹配器找到，尝试直接字符串匹配作为降级方案
            if output and error_msg in output:
                return True, extracted_errors
            
            return False, extracted_errors
        except Exception as e:
            logger.error(f"错误匹配器失败: {e}")
            # 降级到简单的字符串匹配
            if output and error_msg in output:
                return True, []
            return False, []
    
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
    
    def _deduplicate_errors(self, error_lines: List[str], error_details: List[str]) -> List[Dict]:
        """
        根据错误消息内容对错误行进行去重
        
        Args:
            error_lines: 错误行列表
            error_details: 错误详情列表
            
        Returns:
            List[Dict]: 去重后的错误信息列表，每个元素包含：
                - original_idx: 原始索引
                - error_message: 提取的错误消息
                - duplicate_count: 重复次数
        """
        error_message_map = {}  # error_message -> {'indices': [idx], 'count': count}
        
        # 提取每个错误行的错误消息并统计
        for idx, error_line in enumerate(error_lines):
            error_message = self._extract_error_message(error_line)
            
            if error_message not in error_message_map:
                error_message_map[error_message] = {
                    'indices': [idx],
                    'count': 1
                }
            else:
                error_message_map[error_message]['indices'].append(idx)
                error_message_map[error_message]['count'] += 1
        
        # 构建去重后的错误列表，使用第一次出现的索引作为代表
        unique_errors = []
        for error_message, info in error_message_map.items():
            original_idx = info['indices'][0]  # 使用第一次出现的索引
            unique_errors.append({
                'original_idx': original_idx,
                'error_message': error_message,
                'duplicate_count': info['count']
            })
        
        # 按原始索引排序，保持处理顺序的一致性
        unique_errors.sort(key=lambda x: x['original_idx'])
        
        return unique_errors
    
    
    def _apply_patch(self, repo_path: Path, modified_file: str, diff_content: str) -> bool:
        """
        应用修复补丁（直接手动应用）
        
        Args:
            repo_path: 仓库路径
            modified_file: 要修改的文件路径
            diff_content: 补丁内容
            
        Returns:
            bool: 是否成功应用补丁
        """
        logger.info(f"🔧 手动应用补丁到文件: {modified_file}")
        
        # 直接使用手动应用，跳过git apply
        return self._apply_patch_manually(repo_path, modified_file, diff_content)
    
    def _apply_patch_manually(self, repo_path: Path, modified_file: str, diff_content: str) -> bool:
        """
        手动应用补丁（当git apply失败时）
        
        Args:
            repo_path: 仓库路径
            modified_file: 要修改的文件路径
            diff_content: 补丁内容
            
        Returns:
            bool: 是否成功应用补丁
        """
        try:
            file_path = repo_path / modified_file
            # 检查文件是否存在
            if not file_path.exists():
                # 尝试查找文件
                possible_paths = list(repo_path.glob(f"**/{modified_file}"))
                if possible_paths:
                    file_path = possible_paths[0]
                    logger.info(f"找到文件: {file_path}")
                else:
                    logger.error(f"❌ 文件不存在: {modified_file}")
                    return False
            
            # 读取原始文件内容
            with open(file_path, 'r') as f:
                original_content = f.read()
            
            # 解析diff内容并应用修改
            modified_content = self._apply_diff_to_content(original_content, diff_content)
            
            # 写入修改后的内容
            with open(file_path, 'w') as f:
                f.write(modified_content)
            
            logger.info(f"✅ 手动应用补丁成功: {modified_file}")
            return True
        except Exception as e:
            logger.error(f"❌ 手动应用补丁失败: {e}")
            return False
    
    def _apply_diff_to_content(self, original_content: str, diff_content: str) -> str:
        """
        将diff内容应用到原始文件内容
        
        Args:
            original_content: 原始文件内容
            diff_content: diff内容
            
        Returns:
            修改后的文件内容
        """
        try:
            # 使用whatthepatch库解析diff
            patches = list(whatthepatch.parse_patch(diff_content))
            if patches and patches[0].changes:
                return self._apply_diff_with_whatthepatch(original_content, patches[0])
        except Exception as e:
            logger.warning(f"whatthepatch解析失败: {e}")
        
        # 降级到基于@@定位的方法
        return self._apply_diff_with_at_signature(original_content, diff_content)
    
    def _apply_diff_with_whatthepatch(self, original_content: str, patch) -> str:
        """
        使用whatthepatch库应用diff
        """
        original_lines = original_content.split('\n')
        modified_lines = original_lines.copy()
        
        # 按行号排序变更，从后往前应用，避免行号偏移
        changes = sorted(patch.changes, key=lambda x: (x.old or 0, x.new or 0), reverse=True)
        
        for change in changes:
            if change.old is not None and change.new is not None:
                # 修改行
                if 0 <= change.old - 1 < len(modified_lines):
                    modified_lines[change.old - 1] = change.line
            elif change.old is None and change.new is not None:
                # 新增行
                if change.new - 1 <= len(modified_lines):
                    modified_lines.insert(change.new - 1, change.line)
            elif change.old is not None and change.new is None:
                # 删除行
                if 0 <= change.old - 1 < len(modified_lines):
                    del modified_lines[change.old - 1]
        
        return '\n'.join(modified_lines)
    
    def _apply_diff_with_at_signature(self, original_content: str, diff_content: str) -> str:
        """
        使用@@签名定位范围，然后精确应用diff
        """
        original_lines = original_content.split('\n')
        diff_lines = diff_content.strip().split('\n')
        
        # 查找@@行并解析范围
        for i, line in enumerate(diff_lines):
            if line.startswith('@@'):
                # 解析@@行获取大概范围
                match = re.search(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
                if match:
                    old_start = int(match.group(1))
                    old_count = int(match.group(2)) if match.group(2) else 1
                    
                    # 计算大概的范围（上下容错10行）
                    start_idx = max(0, old_start - 1 - 10)
                    end_idx = min(len(original_lines), old_start + old_count + 10)
                    
                    # 提取这个范围内的原始内容
                    target_range = original_lines[start_idx:end_idx]
                    
                    # 提取diff中的变更内容
                    diff_changes = self._extract_diff_changes(diff_lines, i)
                    
                    # 在目标范围内应用精确的diff
                    modified_range = self._apply_changes_in_range(target_range, diff_changes)
                    
                    # 替换原始内容中的对应范围
                    original_lines[start_idx:end_idx] = modified_range
                    break
        
        return '\n'.join(original_lines)
    
    def _extract_diff_changes(self, diff_lines: List[str], start_idx: int) -> List[Dict]:
        """
        从diff行中提取变更信息
        """
        changes = []
        i = start_idx + 1
        
        while i < len(diff_lines) and not diff_lines[i].startswith('@@'):
            line = diff_lines[i]
            if line.startswith('+'):
                changes.append({'type': 'add', 'content': line[1:]})
            elif line.startswith('-'):
                changes.append({'type': 'remove', 'content': line[1:]})
            elif line.startswith(' '):
                changes.append({'type': 'context', 'content': line[1:]})
            i += 1
        
        return changes
    
    def _apply_changes_in_range(self, target_range: List[str], changes: List[Dict]) -> List[str]:
        """
        在指定范围内应用变更
        """
        result = target_range.copy()
        
        # 提取要删除和添加的行
        lines_to_remove = [c['content'] for c in changes if c['type'] == 'remove']
        lines_to_add = [c['content'] for c in changes if c['type'] == 'add']
        context_lines = [c['content'] for c in changes if c['type'] == 'context']
        
        # 删除匹配的行
        for line_to_remove in lines_to_remove:
            for i, line in enumerate(result):
                if self._similar_lines(line, line_to_remove):
                    del result[i]
                    break
        
        # 通过上下文推测插入位置
        if lines_to_add:
            insert_pos = self._infer_insert_position(result, lines_to_remove, context_lines)
            for line_to_add in lines_to_add:
                result.insert(insert_pos, line_to_add)
                insert_pos += 1
        
        return result
    
    def _infer_insert_position(self, target_range: List[str], lines_to_remove: List[str], context_lines: List[str]) -> int:
        """
        通过删除行和上下文来推测插入位置
        """
        # 优先在删除行附近插入
        if lines_to_remove:
            for i, line in enumerate(target_range):
                if any(self._similar_lines(line, remove_line) for remove_line in lines_to_remove):
                    return i
        
        # 通过上下文行定位
        if context_lines:
            context_positions = []
            for context_line in context_lines:
                for i, line in enumerate(target_range):
                    if self._similar_lines(line, context_line):
                        context_positions.append(i)
            
            if context_positions:
                context_positions.sort()
                if len(context_positions) > 1:
                    return (context_positions[0] + context_positions[-1]) // 2
                else:
                    return context_positions[0] + 1
        
        return 0
    

    
    def _similar_lines(self, line1: str, line2: str) -> bool:
        """
        判断两行是否相似
        
        Args:
            line1: 第一行
            line2: 第二行
            
        Returns:
            是否相似
        """
        # 去除空白字符后比较
        line1_clean = line1.strip()
        line2_clean = line2.strip()
        
        if line1_clean == line2_clean:
            return True
        
        # 如果完全匹配失败，尝试部分匹配
        if len(line1_clean) > 10 and len(line2_clean) > 10:
            # 计算相似度
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, line1_clean, line2_clean).ratio()
            return similarity > 0.8
        
        return False
    
    def _extract_diff_content(self, diff_text: str, include_context: bool = False) -> DiffContent:
        """
        从diff文本中提取修改的代码内容
        
        Args:
            diff_text: diff格式的文本
            include_context: 是否包含上下文行
            
        Returns:
            DiffContent对象，包含删除的行、新增的行和上下文行
        """
        removed_lines = []
        added_lines = []
        context_lines = []
        
        try:
            # 确保diff_text是字符串
            if not isinstance(diff_text, str):
                diff_text = str(diff_text)
            
            # 使用whatthepatch解析diff
            patches = list(whatthepatch.parse_patch(diff_text))
            if patches:
                diff = patches[0]
                
                if diff.changes:
                    for change in diff.changes:
                        if change.old is None and change.new is not None:
                            # 新增的行
                            added_lines.append(change.line)
                        elif change.old is not None and change.new is None:
                            # 删除的行
                            removed_lines.append(change.line)
                        elif include_context:
                            # 上下文行
                            context_lines.append(change.line)
                    
                    # 如果whatthepatch解析成功但没有识别出任何删除/添加行，触发降级处理
                    if not added_lines and not removed_lines and ('+' in diff_text or '-' in diff_text):
                        raise ValueError("No add/remove lines found despite +/- in diff")
                else:
                    # 如果没有解析到changes，可能是普通代码
                    added_lines = [diff_text.strip()]
            else:
                # 如果没有解析到任何补丁，降级处理
                raise ValueError("No patches found")
                    
        except Exception as e:
            logger.warning(f"解析diff时发生错误: {e}, diff内容: {diff_text[:200]}...")
            # 降级为简单的行分析
            lines = diff_text.strip().split('\n')
            for line in lines:
                if line.startswith('+') and not line.startswith('+++'):
                    added_lines.append(line[1:].rstrip())
                elif line.startswith('-') and not line.startswith('---'):
                    removed_lines.append(line[1:].rstrip())
                elif include_context and not line.startswith('@@'):
                    context_lines.append(line.rstrip())
        
        # 移除空行和重复行
        added_lines = list(filter(None, dict.fromkeys(added_lines)))
        removed_lines = list(filter(None, dict.fromkeys(removed_lines)))
        context_lines = list(filter(None, dict.fromkeys(context_lines)))
        
        return DiffContent(removed_lines, added_lines, context_lines, diff_text.strip())

    def _normalize_code(self, code: str) -> str:
        """
        规范化代码：
        1. 去除所有空白字符
        2. 去除单行和多行注释
        3. 去除空行
        
        Args:
            code: 要处理的代码
            
        Returns:
            处理后的代码
        """
        import re
        
        # 去除多行注释 /* ... */
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        # 去除单行注释 // ...
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        
        # 去除所有空白字符（空格、制表符、换行符等）
        code = re.sub(r'\s+', '', code)
        
        return code

    def _normalize_diff_content(self, content: DiffContent) -> DiffContent:
        """
        规范化DiffContent中的代码内容
        
        Args:
            content: DiffContent对象
            
        Returns:
            处理后的DiffContent对象
        """
        return DiffContent(
            removed=[self._normalize_code(line) for line in content.removed if self._normalize_code(line)],
            added=[self._normalize_code(line) for line in content.added if self._normalize_code(line)],
            context=[self._normalize_code(line) for line in content.context if self._normalize_code(line)],
            original=content.original
        )

    def _calculate_diff_similarity(self, diff1: str, diff2: Union[str, DiffContent]) -> float:
        """
        计算两个diff的相似度
        
        使用更智能的比较方法：
        1. 分别提取两个diff的内容
        2. 比较添加和删除的内容
        3. 考虑行的顺序和内容
        """
        # 如果diff2已经是DiffContent对象，直接使用
        content1 = self._extract_diff_content(diff1)
        content2 = diff2 if isinstance(diff2, DiffContent) else self._extract_diff_content(diff2)
        
        # 规范化内容
        content1 = self._normalize_diff_content(content1)
        if isinstance(content2, DiffContent):
            content2 = self._normalize_diff_content(content2)
        
        # 如果没有修改内容，返回0
        if not (content1.added or content1.removed) or not (content2.added or content2.removed):
            return 0.0
        
        # 计算添加行的相似度
        added_sim = difflib.SequenceMatcher(
            None,
            '\n'.join(content1.added),
            '\n'.join(content2.added)
        ).ratio() if content1.added and content2.added else 0.0
        
        # 计算删除行的相似度
        removed_sim = difflib.SequenceMatcher(
            None,
            '\n'.join(content1.removed),
            '\n'.join(content2.removed)
        ).ratio() if content1.removed and content2.removed else 0.0
        
        # 如果都没有相应的修改类型，返回0
        if not (content1.added or content2.added) and not (content1.removed or content2.removed):
            return 0.0
        
        # 如果只有一种修改类型，返回那种类型的相似度
        if not (content1.added or content2.added):
            return removed_sim
        if not (content1.removed or content2.removed):
            return added_sim
        
        # 加权平均，可以根据需要调整权重
        return (added_sim * 0.7 + removed_sim * 0.3)

    def _evaluate_exact_match(self, patches: List[SearchReplacePatch], instance: CompilerErrorInstance, error_idx: int) -> bool:
        """评估精确匹配"""
        diff = instance.diffs[error_idx]
        if not diff['relevant_repairs'] or not patches:
            return False
        
        # 如果补丁数量不匹配，直接返回False
        if len(patches) != len(diff['relevant_repairs']):
            return False
        
        # 提取所有正确答案的代码
        expected_codes = []
        for repair in diff.get('relevant_repairs', []):
            if 'diff' not in repair:
                return False
            expected_content = self._extract_diff_content(repair['diff'])
            # 规范化内容
            expected_content = self._normalize_diff_content(expected_content)
            modified_file = repair.get('modified_file', 'unknown_file')
            expected_codes.append({
                'content': expected_content,
                'file': modified_file,
                'basename': os.path.basename(modified_file),
                'matched': False
            })
        
        # logger.info("\n" + "="*50)
        # logger.info(f"开始比较 {len(patches)} 个补丁")
        # logger.info("="*50)
        
        # 遍历每个补丁，尝试找到匹配的正确答案
        for patch_idx, patch in enumerate(patches, 1):
            found_match = False
            # 解析补丁内容并规范化
            patch_content = self._extract_diff_content(patch.fixed_code)
            patch_content = self._normalize_diff_content(patch_content)
            patch_basename = os.path.basename(patch.file_path)
            
            # logger.info(f"\n补丁 #{patch_idx}:")
            # logger.info(f"文件名: {patch_basename}")
            # logger.info(f"LLM输出:\n{patch_content.original}")
            # logger.info("-"*30)
            
            # 在所有未匹配的正确答案中查找
            for expected_idx, expected in enumerate(expected_codes, 1):
                if expected['matched']:
                    continue
                
                # logger.info(f"与正确答案 #{expected_idx} 比较:")
                # logger.info(f"文件名: {expected['basename']}")
                # logger.info(f"正确答案:\n{expected['content'].original}")
                
                # 比较文件名是否匹配
                basename_match = patch_basename == expected['basename']
                if not basename_match:
                    # logger.info("✗ 文件名不匹配")
                    # logger.info("-"*30)
                    continue
                
                # 比较添加和删除的内容是否完全匹配
                added_match = set(patch_content.added) == set(expected['content'].added)
                removed_match = set(patch_content.removed) == set(expected['content'].removed)
                
                if added_match and removed_match:
                    expected['matched'] = True
                    found_match = True
                    # logger.info("✓ 完全匹配!")
                    break
                # else:
                    # logger.info("✗ 内容不匹配")
                # logger.info("-"*30)
            
            if not found_match:
                return False
            
            # logger.info("="*50)
        
        # 检查是否所有正确答案都找到了匹配
        return all(expected['matched'] for expected in expected_codes)
    
    def _evaluate_patch_validity(self, patches: List[SearchReplacePatch], instance: CompilerErrorInstance, error_idx: int) -> bool:
        """评估补丁有效性"""
        diff = instance.diffs[error_idx]
        if not diff['relevant_repairs'] or not patches:
            return False
        
        # 如果补丁数量不匹配，直接返回False
        if len(patches) != len(diff['relevant_repairs']):
            return False
        
        # 提取所有正确答案的代码
        expected_codes = []
        for repair in diff.get('relevant_repairs', []):
            if 'diff' not in repair:
                continue
            expected_content = self._extract_diff_content(repair['diff'])
            modified_file = repair.get('modified_file', 'unknown_file')
            expected_codes.append({
                'content': expected_content,
                'file': modified_file,
                'basename': os.path.basename(modified_file),
                'matched': False
            })
        
        logger.info("\n" + "="*50)
        logger.info(f"开始比较 {len(patches)} 个补丁的有效性（相似度阈值: {self.similarity_threshold}）")
        logger.info("="*50)
        
        # 遍历每个补丁，尝试找到匹配的正确答案
        for patch_idx, patch in enumerate(patches, 1):
            found_match = False
            patch_basename = os.path.basename(patch.file_path)
            
            logger.info(f"\n补丁 #{patch_idx}:")
            logger.info(f"文件名: {patch_basename}")
            logger.info(f"行号范围: {patch.start_line}-{patch.end_line}")
            logger.info(f"置信度: {patch.confidence:.2f}")
            logger.info(f"原始代码:\n{patch.original_code}")
            logger.info(f"修复代码:\n{patch.fixed_code}")
            if hasattr(patch, 'explanation') and patch.explanation:
                logger.info(f"说明: {patch.explanation}")
            logger.info("-"*30)
            
            # 在所有未匹配的正确答案中查找
            for expected_idx, expected in enumerate(expected_codes, 1):
                if expected['matched']:
                    continue
                
                logger.info(f"与正确答案 #{expected_idx} 比较:")
                logger.info(f"文件名: {expected['basename']}")
                logger.info(f"正确答案:\n{expected['content'].original}")
                
                # 比较文件名是否匹配
                basename_match = patch_basename == expected['basename']
                if not basename_match:
                    logger.info("✗ 文件名不匹配")
                    logger.info("-"*30)
                    continue
                
                # 计算相似度
                similarity = self._calculate_diff_similarity(patch.fixed_code, expected['content'])
                logger.info(f"相似度: {similarity:.2%}")
                
                if similarity >= self.similarity_threshold:
                    expected['matched'] = True
                    found_match = True
                    logger.info("✓ 相似度达到阈值!")
                    break
                else:
                    logger.info("✗ 相似度未达到阈值")
                logger.info("-"*30)
            
            if not found_match:
                return False
            
            logger.info("="*50)
        
        # 检查是否所有正确答案都找到了匹配
        return all(expected['matched'] for expected in expected_codes)
    
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
    
    def _calculate_patch_modification_lines(self, patches: List[SearchReplacePatch]) -> int:
        """
        计算patch的总修改行数
        
        Args:
            patches: 修复补丁列表
            
        Returns:
            int: 总修改行数（添加行数 + 删除行数）
        """
        total_modifications = 0
        
        for patch in patches:
            try:
                if isinstance(patch, SearchReplacePatch):
                    # 对于Search-Replace格式，计算行数差异
                    original_lines = len(patch.original_code.split('\n')) if patch.original_code else 0
                    fixed_lines = len(patch.fixed_code.split('\n')) if patch.fixed_code else 0
                    modifications = max(original_lines, fixed_lines)  # 取较大值作为修改行数
                    total_modifications += modifications
                    logger.debug(f"Search-Replace补丁 {patch.file_path} 修改行数: {modifications} (原始: {original_lines}, 修复: {fixed_lines})")
                else:
                    # 对于传统的diff格式补丁
                    diff_content = self._extract_diff_content(patch.fixed_code)
                    modifications = len(diff_content.added) + len(diff_content.removed)
                    total_modifications += modifications
                    logger.debug(f"Diff补丁 {patch.file_path} 修改行数: {modifications} (添加: {len(diff_content.added)}, 删除: {len(diff_content.removed)})")
            except Exception as e:
                logger.warning(f"计算补丁修改行数时发生错误: {e}")
                # 如果无法解析，使用默认值
                total_modifications += 1
        
        logger.info(f"总修改行数: {total_modifications}")
        return total_modifications
    
    def _calculate_dynamic_line_tolerance(self, modification_lines: int) -> int:
        """
        根据修改行数动态计算line tolerance
        
        Args:
            modification_lines: 修改行数
            
        Returns:
            int: 动态计算的line tolerance
        """
        # 基础tolerance
        base_tolerance = 5
        
        # 根据修改行数调整tolerance
        if modification_lines <= 1:
            # 小修改，使用较小的tolerance
            tolerance = max(3, base_tolerance - 2)
        elif modification_lines <= 3:
            # 中等修改，使用基础tolerance
            tolerance = base_tolerance
        elif modification_lines <= 10:
            # 较大修改，适当增加tolerance
            tolerance = base_tolerance + 2
        elif modification_lines <= 20:
            # 大修改，进一步增加tolerance
            tolerance = base_tolerance + 5
        else:
            # 超大修改，使用较大的tolerance
            tolerance = base_tolerance + 10
        
        logger.info(f"修改行数: {modification_lines}, 动态计算line tolerance: {tolerance}")
        return tolerance

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
        print("PFL评估结果汇总")
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
    


    def _process_single_instance(self, instance: CompilerErrorInstance, instance_index: int, repairer: PerfectLocalizationDirectRepair, has_repair_data: bool, dataset_name: str = None, repo_path: str = None, existing_results: Dict[Tuple[int, int], Dict] = None) -> Optional[Tuple[int, int, int, int, float]]:
        """
        处理单个实例 - 用于并行处理
        
        Args:
            instance: 编译错误实例
            instance_index: 实例索引
            repairer: 修复器
            has_repair_data: 是否有修复数据
            dataset_name: 数据集名称
            
        Returns:
            Optional[Tuple[int, int, int, int, float]]: (successful_fixes, exact_matches, valid_patches, total_errors, total_time)
        """
        temp_repo_path = None
        try:
            # 第一步：对错误行按error后的内容进行去重
            unique_errors = self._deduplicate_errors(instance.error_lines, instance.error_details)
            
            # 检查断点续跑：如果实例的所有错误都已处理完成，直接跳过
            if existing_results:
                unique_errors_processed = 0
                for unique_error in unique_errors:
                    error_idx = unique_error['original_idx']
                    if (instance_index + 1, error_idx + 1) in existing_results:
                        unique_errors_processed += 1
                logger.info(f"实例 {instance_index+1} 已处理 {unique_errors_processed}/{len(unique_errors)} 个唯一错误")
                if unique_errors_processed == len(unique_errors):
                    logger.info(f"实例 {instance_index+1} 的所有唯一错误已处理完成，跳过")
                    return (0, 0, 0, 0, 0.0)
            
            logger.info(f"评估实例 {instance_index+1}")
            
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
            if isinstance(job_names, str):
                job_names = [job_names]
            
            # 选择合适的job
            job_name_str = self._select_compilable_job(job_names, project_config)
            
            # 如果没有找到可编译的job，跳过此条记录
            if job_name_str is None:
                logger.warning(f"⚠️ 跳过修复: 没有找到可编译的job")
                return None

            # 创建临时仓库
            temp_repo_path = self._create_temp_repo(instance_index, repo_path, dataset_name)
            if not temp_repo_path:
                logger.error(f"无法为实例 {instance_index+1} 创建临时仓库")
                return None
            
            instance_results = []
            instance_time = 0.0
            instance_successful_fixes = 0
            instance_exact_matches = 0
            instance_valid_patches = 0
            instance_total_errors = 0
            
            logger.info(f"实例 {instance_index+1} 总错误行数: {len(instance.error_lines)}, 去重后唯一错误数: {len(unique_errors)}")
            
            # 过滤掉已处理的错误，生成最终要处理的错误列表
            final_unique_errors = []
            if existing_results:
                unique_errors_processed = 0
                for unique_error in unique_errors:
                    error_idx = unique_error['original_idx']
                    if (instance_index + 1, error_idx + 1) in existing_results:
                        unique_errors_processed += 1
                        logger.info(f"错误 {error_idx + 1} 已处理过，跳过")
                    else:
                        final_unique_errors.append(unique_error)
                
                if unique_errors_processed > 0:
                    logger.info(f"实例 {instance_index+1} 已处理 {unique_errors_processed}/{len(unique_errors)} 个唯一错误，继续处理剩余 {len(final_unique_errors)} 个错误")
                else:
                    final_unique_errors = unique_errors
            else:
                final_unique_errors = unique_errors
            
            logger.info(f"实例 {instance_index+1} 最终需要处理的唯一错误数: {len(final_unique_errors)}")
            
            # 步骤1: 验证错误是否存在（每个instance只验证一次）
            logger.info(f"🔍 验证实例 {instance_index+1} 的错误是否存在...")
            
            # 编译项目，确认错误存在
            success_before, output_before = self._compile_project(
                project_name, instance_index + 1, temp_repo_path=temp_repo_path
            )
            
            # 步骤1: 一次性提取所有错误，避免重复编译
            logger.info(f"🔍 一次性提取实例 {instance_index+1} 的所有错误...")
            
            # 导入错误提取器与错误匹配器
            try:
                reproduce_path = Path(__file__).parent.parent.parent.parent / "experiment" / "reproduce"
                if str(reproduce_path) not in sys.path:
                    sys.path.insert(0, str(reproduce_path))
                from error_extractor import ErrorExtractor
                from error_matcher import ErrorMatcher
                error_extractor = ErrorExtractor()
                error_matcher = ErrorMatcher()
            except ImportError as e:
                logger.error(f"无法导入错误提取器: {e}")
                return None
            
            # 一次性提取所有编译错误
            all_errors_before = error_extractor.extract_errors(output_before) if output_before else []

            
            # 验证每个错误是否在提取的错误列表中（使用错误匹配器，支持路径后缀一致匹配）
            valid_error_indices = []
            expected_errors_for_match = [{'error_lines': [extracted_error]} for extracted_error in all_errors_before]
            for unique_error in final_unique_errors:
                error_idx = unique_error['original_idx']
                error_line = instance.error_lines[error_idx]
                
                # 使用错误匹配器进行相似度匹配（行号需一致，路径允许前缀不同）
                is_matched, _ = error_matcher.match_single_error(error_line, expected_errors_for_match)
                error_exists = bool(is_matched)
                if error_exists:
                    valid_error_indices.append(error_idx)
                    logger.info(f"✅ 实例 {instance_index+1} 确认错误存在: {error_idx+1}: {error_line}")
                else:
                    logger.warning(f"❌ 实例 {instance_index+1} 未检测到错误: {error_idx+1}: {error_line}")
            
            if not valid_error_indices:
                logger.warning(f"实例 {instance_index+1} 中未检测到任何有效错误，跳过该实例")
                return None
            
            logger.info(f"实例 {instance_index+1} 有效错误数: {len(valid_error_indices)}")
            
            # 步骤2: 对每个有效错误进行修复
            for unique_error in final_unique_errors:
                error_idx = unique_error['original_idx']
                
                # 只处理验证过的有效错误
                if error_idx not in valid_error_indices:
                    logger.info(f"跳过未验证的错误 {error_idx + 1}")
                    continue
                
                error_message = unique_error['error_message']
                duplicate_count = unique_error['duplicate_count']
                
                start_time = time.time()
                error_line = instance.error_lines[error_idx]
                
                logger.info(f"处理唯一错误: {error_message} (出现 {duplicate_count} 次，使用第 {error_idx + 1} 行作为代表)")
                
                # 生成修复补丁（带重试机制）
                patches = repairer.generate_fix_for_error(instance, error_idx, temp_repo_path)
                
                # 如果第一次没有生成补丁，进行重试
                if not patches:
                    logger.warning(f"错误 {error_idx + 1} 第一次未生成修复补丁，进行重试...")
                    patches = repairer.generate_fix_for_error(instance, error_idx, temp_repo_path)
                
                end_time = time.time()
                error_time = end_time - start_time
                
                # 如果重试后仍然没有生成补丁，标记为修复失败而不是跳过
                if not patches:
                    logger.error(f"错误 {error_idx + 1} 重试后仍未生成修复补丁，标记为修复失败")
                    
                    # 构建失败的结果数据
                    error_result = {
                        'is_successful': False,
                        'is_exact_match': False,
                        'is_valid': False,
                        'error_line': error_line,
                        'patches': [],
                        'errors_before': all_errors_before,
                        'errors_after': all_errors_before,  # 修复失败，错误列表不变
                        'error_detail': instance.error_details[error_idx],
                        'time': error_time,
                        'evaluation_mode': 'full' if has_repair_data else 'patch_generation_only',
                        'failure_reason': 'no_patches_generated'
                    }
                    
                    # 计入统计
                    instance_time += error_time
                    instance_total_errors += 1
                    
                    # 立即保存失败结果
                    result_data = {
                        'instance_index': instance_index + 1,
                        'error_index': error_idx + 1,
                        'error_result': error_result,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    self._append_result(repairer.model_name, result_data, dataset_name)
                    
                    logger.info(f"❌ 实例 {instance_index+1} 错误 {error_idx + 1} 修复失败：未生成补丁")
                    continue
                
                # 评估修复成功性（使用已编译的输出）
                is_successful, errors_after, used_patches = self._evaluate_fix_success_with_compiled_output(
                    patches, instance, error_idx, instance_index + 1, all_errors_before, temp_repo_path
                )
                
                # 如果返回None，表示未检测到指定错误，跳过该记录
                if is_successful is None:
                    logger.info(f"跳过错误 {error_idx + 1}，未检测到指定错误")
                    continue
                
                logger.info(f"实例 {instance_index+1} 错误 {error_idx + 1} 修复的错误 {error_line}")
                if is_successful:
                    logger.info(f"✅ 实例 {instance_index+1} 错误 {error_idx + 1} 成功修复编译错误")
                else:
                    logger.info(f"❌ 实例 {instance_index+1} 错误 {error_idx + 1} 未修复编译错误")

                logger.info(f"实例 {instance_index+1} 错误 {error_idx + 1} 修复前错误列表 (共{len(all_errors_before)}个):")
                for i, error in enumerate(all_errors_before, 1):
                    logger.info(f"  {i}. {error}")
            
                logger.info(f"实例 {instance_index+1} 错误 {error_idx + 1} 修复后错误列表 (共{len(errors_after)}个):")
                for i, error in enumerate(errors_after, 1):
                    logger.info(f"  {i}. {error}")
                # 打印补丁信息用于调试
                logger.info(f"📋 实例 {instance_index+1} 错误 {error_idx + 1} 生成的补丁:")
                for patch_idx, patch in enumerate(used_patches, 1):
                    logger.info(f"  补丁 {patch_idx}:")
                    logger.info(f"    文件: {patch.file_path}")
                    logger.info(f"    行号范围: {patch.start_line}-{patch.end_line}")
                    logger.info(f"    置信度: {patch.confidence:.2f}")
                    logger.info(f"    错误: {patch.error_line}")
                    logger.info(f"    原始代码:\n {patch.original_code}")
                    logger.info(f"    修复代码:\n {patch.fixed_code}")
                    if patch.explanation:
                        logger.info(f"    说明: {patch.explanation}")
                    logger.info(f"    {'='*50}")
                
                # 只有通过验证的记录才计入统计
                instance_time += error_time
                instance_total_errors += 1
                
                # 评估结果
                is_exact_match = False
                is_valid = False
                
                if is_successful:
                    instance_successful_fixes += 1
                
                # 如果有修复数据，进行精确匹配和有效性评估
                if has_repair_data:
                    # 评估精确匹配
                    #is_exact_match = self._evaluate_exact_match(patches, instance, error_idx)
                    #if is_exact_match:
                        #instance_exact_matches += 1
                    
                    # 评估补丁有效性
                    #is_valid = self._evaluate_patch_validity(patches, instance, error_idx)
                    #if is_valid:
                    #    instance_valid_patches += 1
                    pass
                
                # 构建单条结果数据
                error_result = {
                    'is_successful': is_successful,
                    'is_exact_match': is_exact_match,
                    'is_valid': is_valid,
                    'error_line': error_line,
                    'patches': [patch.to_dict() for patch in used_patches],
                    'errors_before': all_errors_before,
                    'errors_after': errors_after,
                    'error_detail': instance.error_details[error_idx],
                    'time': error_time,
                    'evaluation_mode': 'full' if has_repair_data else 'patch_generation_only'
                }
                
                instance_results.append(error_result)
                
                # 立即保存单条结果
                result_data = {
                    'instance_index': instance_index + 1,
                    'error_index': error_idx + 1,
                    'error_result': error_result,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                self._append_result(repairer.model_name, result_data, dataset_name)
            
            logger.info(f"实例 {instance_index+1} 处理完成，处理了 {instance_total_errors} 个错误")
            
            return (instance_successful_fixes, instance_exact_matches, instance_valid_patches, instance_total_errors, instance_time)
            
        except Exception as e:
            logger.error(f"处理实例 {instance_index+1} 时发生错误: {e}")
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
            return None
        finally:
            # 清理临时仓库
            if temp_repo_path:
                try:
                    self._cleanup_temp_repo(temp_repo_path)
                    logger.info(f"线程 {instance_index+1} 临时仓库清理完成")
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
            logger.warning(f"清理临时仓库失败: {e}")