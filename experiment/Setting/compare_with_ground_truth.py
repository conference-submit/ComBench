#!/usr/bin/env python3

import json
import os
import sys
import subprocess
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import argparse
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time
import logging

from patch_utils import get_file_content_from_local, normalize_file_path_for_git, get_file_content_from_git, get_file_content_from_git_with_path

sys.path.append(str(Path(__file__).parent.parent.parent.parent / "compilation_error"))
from error_classifier import ErrorClassifier, ErrorType

@dataclass
class PFLResult:
    instance_index: int
    error_index: int
    is_successful: bool
    error_line: str
    patches: List[Dict[str, Any]]
    errors_before: List[str]
    errors_after: List[str]
    error_detail: str
    failure_commit: str = ''
    repair_commit: str = ''
    diff_text: str = ''
    error_classification: Dict[str, str] = None

@dataclass
class GroundTruthData:
    failure_commit: str
    repair_commit: str
    error_lines: List[str]
    compilation_related_paths_details: List[Dict[str, Any]]
    error_details: List[str]

def apply_search_replace_patch_in_memory(content: str, patch, logger_instance=None) -> str:
    log = logger_instance or print
    
    original_code = patch.get('original_code', '')
    fixed_code = patch.get('fixed_code', '')
    start_line = patch.get('start_line', 1)
    end_line = patch.get('end_line', 1)
    
    if not original_code and not fixed_code:
        log.warning("Invalid patch: both original_code and fixed_code are empty")
        return content
    
    orig_lines = content.splitlines(keepends=True)
    
    start_line_idx = start_line - 1
    end_line_idx = min(end_line, len(orig_lines))
    
    if original_code:
        range_content = ''.join(orig_lines[start_line_idx:end_line_idx])
        orig_code = original_code
        fixed_code = fixed_code
        
        normalized_range_content = ' '.join(range_content.split())
        normalized_orig_code = ' '.join(orig_code.split())
        
        if normalized_orig_code in normalized_range_content:
            from patch_utils import replace_ignoring_whitespace
            new_range_content = replace_ignoring_whitespace(range_content, orig_code, fixed_code)
            
            new_content = ''.join(orig_lines[:start_line_idx]) + new_range_content + ''.join(orig_lines[end_line_idx:])
            
            log.info(f"✅ Search and replace successful within specified line range")
            return new_content
        else:
            log.warning(f"⚠️ Original code not found within specified line range, trying to search in entire file range")
            
            range_content = ''.join(orig_lines)
            normalized_range_content = ' '.join(range_content.split())
            
            if normalized_orig_code in normalized_range_content:
                from patch_utils import replace_ignoring_whitespace
                new_content = replace_ignoring_whitespace(range_content, orig_code, fixed_code)
                
                log.info(f"✅ Search and replace successful in entire file range")
                return new_content
            else:
                log.error(f"❌ Original code not found in entire file range either: {orig_code[:100]}...")
                return content
    else:
        log.info(f"Original code is empty, inserting fix code after line {end_line_idx}")
        
        fixed_code_lines = fixed_code.split('\n')
        
        orig_indent = 0
        if start_line_idx < len(orig_lines) and orig_lines[start_line_idx].strip():
            orig_indent = len(orig_lines[start_line_idx]) - len(orig_lines[start_line_idx].lstrip())
        
        new_lines = []
        
        if end_line_idx == 0:
            log.info(f"Inserting fix code at beginning of file")
            new_lines = []
        else:
            new_lines.extend(orig_lines[:end_line_idx])
        
        for i, line in enumerate(fixed_code_lines):
            if line.strip():
                new_lines.append(' ' * orig_indent + line + '\n')
            else:
                new_lines.append('\n')
        
        new_lines.extend(orig_lines[end_line_idx:])
        
        log.info(f"✅ Successfully inserted fix code after line {end_line_idx}")
        return ''.join(new_lines)



class GroundTruthComparator:
    def __init__(self, pfl_results_file: str, ground_truth_file: str, 
                 repo_path: str, output_file: str = None, 
                 auto_load: bool = True, max_workers: int = None, 
                 use_parallel: bool = True):
        self.pfl_results_file = pfl_results_file
        self.ground_truth_file = ground_truth_file
        self.repo_path = repo_path
        self.output_file = output_file or "pfl_ground_truth_comparison.jsonl"
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.use_parallel = use_parallel
        
        self.output_dir = Path(self.output_file).parent
        output_filename = Path(self.output_file).stem
        self.log_file = self.output_dir / f"{output_filename}.log"
        
        self._setup_logging()
        
        self.error_classifier = ErrorClassifier()
        
        self.pfl_results = []
        self.ground_truth_data = {}
        
        if auto_load:
            self.pfl_results = self._load_pfl_results()
            self.ground_truth_data = self._load_ground_truth_data()
    
    def _setup_logging(self):
        import logging
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger = logging.getLogger("PFLComparator")
        logger.setLevel(logging.INFO)
        
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        self.logger = logger
        self.logger.info("Ground Truth comparator initialization completed")
        self.logger.info(f"Log file: {self.log_file}")
        
    def _load_pfl_results(self) -> List[PFLResult]:
        results = []
        
        try:
            with open(self.pfl_results_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if data.get('error_result', {}).get('is_successful', False):
                                instance_info = data.get('instance', {})
                                failure_commit = instance_info.get('failure_commit', '')
                                repair_commit = instance_info.get('repair_commit', '')
                                
                                error_line = data['error_result']['error_line']
                                main_type, detailed_type = self.error_classifier.identify_error_type(error_line)
                                error_classification = {
                                    'main_type': main_type.value,
                                    'detailed_type': detailed_type.value
                                }
                                
                                result = PFLResult(
                                    instance_index=data['instance_index'],
                                    error_index=data['error_index'],
                                    is_successful=data['error_result']['is_successful'],
                                    error_line=error_line,
                                    patches=data['error_result'].get('patches', []),
                                    errors_before=data['error_result'].get('errors_before', []),
                                    errors_after=data['error_result'].get('errors_after', []),
                                    error_detail=data['error_result'].get('error_detail', ''),
                                    failure_commit=failure_commit,
                                    repair_commit=repair_commit,
                                    diff_text=data['error_result'].get('diff_text', ''),
                                    error_classification=error_classification
                                )
                                results.append(result)
                        except json.JSONDecodeError as e:
                            self.logger.error(f"Failed to parse JSON on line {line_num+1} of PFL results: {e}")
                            continue
                        except KeyError as e:
                            self.logger.error(f"Line {line_num+1} of PFL results missing required fields: {e}")
                            continue
            
            self.logger.info(f"Loaded {len(results)} successful PFL results, will use instance_index for matching")
        except Exception as e:
            self.logger.error(f"Failed to load PFL results file: {e}")
            raise
        
        return results
    

    
    def _load_ground_truth_data(self) -> Dict[str, GroundTruthData]:
        ground_truth = {}
        try:
            with open(self.ground_truth_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line.strip():
                        try:
                            data = json.loads(line)
                            ground_truth[line_num] = GroundTruthData(
                                failure_commit=data['failure_commit'],
                                repair_commit=data['repair_commit'],
                                error_lines=data['error_lines'],
                                compilation_related_paths_details=data.get('compilation_related_paths_details', []),
                                error_details=data.get('error_details', [])
                            )
                        except json.JSONDecodeError as e:
                            self.logger.error(f"Failed to parse JSON on line {line_num+1} of ground truth: {e}")
                            continue
                        except KeyError as e:
                            self.logger.error(f"ground truth第{line_num+1}行缺少必要字段: {e}")
                            continue
            self.logger.info(f"成功加载 {len(ground_truth)} 个ground truth记录")
        except Exception as e:
            self.logger.error(f"加载ground truth文件失败: {e}")
            raise
        return ground_truth
    
    def _find_matching_ground_truth_by_error_pattern(self, pfl_result: PFLResult) -> Optional[GroundTruthData]:
        gt_index = pfl_result.instance_index - 1
        
        if gt_index in self.ground_truth_data:
            gt_data = self.ground_truth_data[gt_index]
            self.logger.info(f"✓ 通过instance_index匹配 - instance_index {pfl_result.instance_index} -> ground truth line {gt_index + 1}")
            return gt_data
        else:
            self.logger.warning(f"instance_index {pfl_result.instance_index} 超出ground truth范围 (0-{len(self.ground_truth_data)-1})")
            return None

    
    def fetch_remote_commit(self, commit_sha: str) -> bool:
        def sanitize_commit_sha(sha: str) -> str:
            return sha.strip() if sha else ""
        
        commit_sha = sanitize_commit_sha(commit_sha)
        if not commit_sha:
            return False
        
        try:
            try:
                remotes_out = subprocess.run(
                    ['git', 'remote'], cwd=self.repo_path, capture_output=True, text=True, check=True
                ).stdout.split()
                base_list = remotes_out or []
            except subprocess.CalledProcessError:
                base_list = []
            
            if not base_list:
                remotes = ['origin']
            else:
                remotes = []
                if 'origin' in base_list:
                    remotes.append('origin')
                for r in base_list:
                    if r != 'origin' and r not in remotes:
                        remotes.append(r)

            for remote in remotes:
                for args in (
                    ['git', 'fetch', '--depth', '1', remote, commit_sha],
                    ['git', 'fetch', remote, commit_sha],
                ):
                    try:
                        self.logger.info(f"尝试从远端 {remote} 抓取对象 {commit_sha}")
                        r = subprocess.run(args, cwd=self.repo_path, capture_output=True, text=True, timeout=30)
                        if r.returncode == 0:
                            return True
                        err = (r.stderr or '').strip()
                        if err:
                            self.logger.debug(f"fetch 失败({remote}): {err[:200]}")
                    except Exception as e:
                        self.logger.debug(f"fetch 异常({remote}): {e}")
                        continue
        except Exception as e:
            self.logger.error(f"获取远程commit失败: {e}")
            return False
        return False

    def get_git_diff(self, failure_commit: str, repair_commit: str, 
                    file_paths: List[str]) -> str:
        import difflib
        
        try:
            original_cwd = os.getcwd()
            os.chdir(self.repo_path)
            
            self.logger.info(f"Fetching commits: {failure_commit} and {repair_commit}")
            
            failure_fetched = self.fetch_remote_commit(failure_commit)
            repair_fetched = self.fetch_remote_commit(repair_commit)
            
            if failure_fetched:
                self.logger.info(f"成功fetch failure commit: {failure_commit}")
            else:
                self.logger.warning(f"Failed to fetch failure commit: {failure_commit}")
            
            if repair_fetched:
                self.logger.info(f"成功fetch repair commit: {repair_commit}")
            else:
                self.logger.warning(f"Failed to fetch repair commit: {repair_commit}")
            
            all_diffs = []
            for file_path in file_paths:
                try:
                    failure_content, failure_path = get_file_content_from_git_with_path(self.repo_path, failure_commit, file_path)
                    repair_content, repair_path = get_file_content_from_git_with_path(self.repo_path, repair_commit, file_path)
                    
                    if failure_content is None and repair_content is None:
                        self.logger.warning(f"跳过文件: {file_path} (在两个commit中都无法找到)")
                        continue
                    
                    actual_file_path = repair_path or failure_path or file_path
                    
                    is_new_file = failure_content is None and repair_content is not None
                    is_deleted_file = failure_content is not None and repair_content is None
                    
                    fromfile = f'a/{actual_file_path}'
                    tofile = f'b/{actual_file_path}'
                    
                    if is_new_file:
                        self.logger.info(f"检测到新增文件: {file_path}")
                        fromfile = '/dev/null'
                        failure_content = ""
                    elif is_deleted_file:
                        self.logger.info(f"检测到删除文件: {file_path}")
                        repair_content = ""
                    
                    diff_lines = list(difflib.unified_diff(
                        failure_content.splitlines(keepends=True) if failure_content else [],
                        repair_content.splitlines(keepends=True) if repair_content else [],
                        fromfile=fromfile,
                        tofile=tofile,
                        n=3
                    ))
                    
                    if diff_lines:
                        diff_content = ''.join(diff_lines)
                        all_diffs.append(diff_content)
                        self.logger.info(f"✅ 成功生成diff: {file_path}")
                    else:
                        self.logger.info(f"跳过文件: {file_path} (没有差异)")
                        
                except Exception as e:
                    self.logger.error(f"处理文件 {file_path} 时发生异常: {e}")
                    continue
            
            return '\n'.join(all_diffs)
                
        except Exception as e:
            self.logger.error(f"Error getting git diff: {e}")
            return ""
        finally:
            os.chdir(original_cwd)
    
    def extract_file_paths_from_pfl_patches(self, patches: List[Dict[str, Any]]) -> List[str]:
        file_paths = []
        for patch in patches:
            file_path = patch.get('file_path', '')
            if file_path and file_path not in file_paths:
                normalized_path = normalize_file_path_for_git(file_path)
                if normalized_path:
                    file_paths.append(normalized_path)
        return file_paths
    
    def extract_file_paths_from_ground_truth(self, ground_truth: GroundTruthData, error_index: int = None) -> List[str]:
        file_paths = []
        
        if error_index is not None and ground_truth.compilation_related_paths_details:
            detail_index = error_index - 1
            if 0 <= detail_index < len(ground_truth.compilation_related_paths_details):
                path_detail = ground_truth.compilation_related_paths_details[detail_index]
                
                error_files = path_detail.get('error_file', [])
                for error_file in error_files:
                    normalized_path = normalize_file_path_for_git(error_file)
                    if normalized_path:
                        file_paths.append(normalized_path)
                
                direct_includes = path_detail.get('direct_includes', [])
                for include_file in direct_includes:
                    normalized_path = normalize_file_path_for_git(include_file)
                    if normalized_path:
                        file_paths.append(normalized_path)
                
                indirect_includes = path_detail.get('indirect_includes', [])
                for include_file in indirect_includes:
                    normalized_path = normalize_file_path_for_git(include_file)
                    if normalized_path:
                        file_paths.append(normalized_path)
                
                self.logger.info(f"使用error_index {error_index} 从compilation_related_paths_details[{detail_index}] 提取到 {len(file_paths)} 个文件")
            else:
                self.logger.warning(f"error_index {error_index} 超出compilation_related_paths_details范围 (0-{len(ground_truth.compilation_related_paths_details)-1})")
        
        if not file_paths:
            self.logger.info("回退到原来的文件提取逻辑")
            for path_detail in ground_truth.compilation_related_paths_details:
                error_files = path_detail.get('error_file', [])
                for error_file in error_files:
                    normalized_path = normalize_file_path_for_git(error_file)
                    if normalized_path:
                        file_paths.append(normalized_path)
        
        return file_paths
    
    def apply_pfl_patches_and_generate_diff(self, pfl_result: PFLResult, 
                                         ground_truth: GroundTruthData) -> str:
        import difflib
        
        file_patches = {}
        seen_patches = set()
        
        for patch in pfl_result.patches:
            file_path = patch.get('file_path', '')
            if not file_path:
                continue
                
            file_path = normalize_file_path_for_git(file_path)
            
            patch_key = (
                file_path,
                patch.get('original_code', ''),
                patch.get('fixed_code', ''),
                patch.get('start_line', 0),
                patch.get('end_line', 0)
            )
            
            if patch_key not in seen_patches:
                seen_patches.add(patch_key)
                if file_path not in file_patches:
                    file_patches[file_path] = []
                file_patches[file_path].append(patch)
            else:
                self.logger.info(f"跳过重复patch: {file_path} (已存在相同内容的patch)")
        
        total_patches = len(pfl_result.patches)
        unique_patches = sum(len(patches) for patches in file_patches.values())
        self.logger.info(f"原始patches数量: {total_patches}")
        self.logger.info(f"去重后patches数量: {unique_patches}")
        self.logger.info(f"去重减少: {total_patches - unique_patches} 个重复patch")
        self.logger.info(f"收集到 {len(file_patches)} 个需要修改的文件")
        
        file_contents = {}
        
        for file_path, patches in file_patches.items():
            original_content, found_path = get_file_content_from_git_with_path(self.repo_path, ground_truth.failure_commit, file_path)
            
            if original_content is None:
                self.logger.warning(f"File not found for patching: {file_path}")
                continue
            
            actual_file_path = found_path if found_path else file_path
            patched_content = original_content
            
            for patch in patches:
                if not patch.get('original_code') and not patch.get('fixed_code'):
                    self.logger.warning(f"Invalid PFL patch format: {patch}")
                    continue
                
                patched_content = apply_search_replace_patch_in_memory(patched_content, patch, self.logger)
            
            file_contents[file_path] = (original_content, patched_content, actual_file_path)
        
        all_diffs = []
        
        for file_path, (original_content, patched_content, actual_file_path) in file_contents.items():
            diff_lines = list(difflib.unified_diff(
                original_content.splitlines(keepends=True),
                patched_content.splitlines(keepends=True),
                fromfile=f'a/{actual_file_path}',
                tofile=f'b/{actual_file_path}',
                n=3
            ))
            
            if diff_lines:
                diff_content = ''.join(diff_lines)
                all_diffs.append(diff_content)
                self.logger.info(f"✅ 成功生成diff: {actual_file_path}")
            else:
                self.logger.info(f"跳过文件: {actual_file_path} (没有差异)")
        
        return '\n'.join(all_diffs)
    

    





    
    def compare_patches_with_ground_truth(self, pfl_result: PFLResult, 
                                        ground_truth: GroundTruthData, pfl_diff: str = "") -> Dict[str, Any]:
        comparison_result = {
            'instance_index': pfl_result.instance_index,
            'error_index': pfl_result.error_index,
            'failure_commit': ground_truth.failure_commit,
            'repair_commit': ground_truth.repair_commit,
            'pfl_patches': pfl_result.patches,
            'pfl_patches_diff': pfl_diff,
            'ground_truth_diff': '',
            'file_paths_pfl': [],
            'file_paths_ground_truth': [],
            'em_analysis': {},  
            'error_classification': pfl_result.error_classification
        }
        
        if pfl_result.diff_text and pfl_result.diff_text.strip():
            self.logger.info(f"使用PFL结果中的diff_text，跳过patch到diff转换")
            final_pfl_diff = pfl_result.diff_text
            comparison_result['pfl_patches_diff'] = final_pfl_diff
        else:
            self.logger.info(f"PFL结果中没有diff_text，使用patches生成diff")
            final_pfl_diff = pfl_diff
        
        pfl_file_paths = self.extract_file_paths_from_pfl_patches(pfl_result.patches)
        gt_file_paths = self.extract_file_paths_from_ground_truth(ground_truth, pfl_result.error_index)
        
        comparison_result['file_paths_pfl'] = pfl_file_paths
        comparison_result['file_paths_ground_truth'] = gt_file_paths
        
        all_file_paths = list(set(pfl_file_paths + gt_file_paths))
        if all_file_paths:
            git_diff = self.get_git_diff(ground_truth.failure_commit, 
                                       ground_truth.repair_commit, 
                                       all_file_paths)
            comparison_result['ground_truth_diff'] = git_diff
        
        comparison_result['em_analysis'] = self.analyze_containment_with_string_match(
            pfl_result, ground_truth, git_diff, final_pfl_diff
        )
        
        return comparison_result
    
    def analyze_containment_with_string_match(self, pfl_result: PFLResult, ground_truth: GroundTruthData, 
                                            git_diff: str, pfl_diff: str = "") -> Dict[str, Any]:
        if not pfl_diff or not git_diff:
            return {
                'patches_contained_in_ground_truth': False,
                'containment_score': 0.0,
                'matched_lines': 0,
                'total_pfl_lines': 0,
                'detailed_match_info': '无法进行字符串匹配分析（缺少diff信息）'
            }
        
        def normalize_diff(diff_content):
            lines = diff_content.split('\n')
            normalized_lines = []
            
            for line in lines:
                if line.startswith('diff --git') or line.startswith('index ') or line.startswith('---') or line.startswith('+++'):
                    continue
                if line.startswith('@@'):
                    continue
                if line.startswith('+') or line.startswith('-'):
                    code_line = line[1:].strip()
                    if code_line and not code_line.isdigit():
                        normalized_lines.append(code_line)
            
            return '\n'.join(normalized_lines)
        
        normalized_pfl_diff = normalize_diff(pfl_diff)
        normalized_gt_diff = normalize_diff(git_diff)
        
        if not normalized_pfl_diff:
            return {
                'patches_contained_in_ground_truth': False,
                'containment_score': 0.0,
                'matched_lines': 0,
                'total_pfl_lines': 0,
                'detailed_match_info': 'PFL diff为空，无法进行匹配'
            }
        
        pfl_lines = [line.strip() for line in normalized_pfl_diff.split('\n') if line.strip()]
        gt_lines = [line.strip() for line in normalized_gt_diff.split('\n') if line.strip()]
        
        total_pfl_lines = len(pfl_lines)
        matched_lines = 0
        
        for pfl_line in pfl_lines:
            for gt_line in gt_lines:
                if pfl_line == gt_line:
                    matched_lines += 1
                    break
        
        containment_score = matched_lines / total_pfl_lines if total_pfl_lines > 0 else 0.0
        
        patches_contained = containment_score == 1.0
        
        return {
            'patches_contained_in_ground_truth': patches_contained,
            'containment_score': containment_score,
            'matched_lines': matched_lines,
            'total_pfl_lines': total_pfl_lines,
            'detailed_match_info': f'匹配了{matched_lines}/{total_pfl_lines}行，包含性分数: {containment_score:.3f}'
        }
    

    
    

    
    def _deduplicate_patches(self, patches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen_patches = set()
        unique_patches = []
        
        for patch in patches:
            patch_key = (
                patch.get('file_path', ''),
                patch.get('original_code', ''),
                patch.get('fixed_code', ''),
                patch.get('start_line', 0),
                patch.get('end_line', 0)
            )
            
            if patch_key not in seen_patches:
                seen_patches.add(patch_key)
                unique_patches.append(patch)
        
        return unique_patches
    
    
    def run_comparison(self):
        print("进入 run_comparison 方法")
        total_count = len(self.pfl_results)
        gt_count = len(self.ground_truth_data)
        
        print(f"PFL结果数量: {total_count}, Ground truth数量: {gt_count}")
        self.logger.info(f"开始分析 {total_count} 个PFL结果...")
        self.logger.info(f"Ground truth数据数量: {gt_count}")
        self.logger.info(f"并行处理: {'启用' if self.use_parallel else '禁用'}")
        if self.use_parallel:
            self.logger.info(f"工作进程数: {self.max_workers}")
        
        print("开始准备匹配数据对...")
        
        matched_pairs = []
        matched_count = 0
        
        self.logger.info("开始匹配PFL结果与ground truth...")
        for i, pfl_result in enumerate(self.pfl_results):
            self.logger.info(f"处理第 {i+1}/{total_count} 个PFL结果 (instance_index: {pfl_result.instance_index})")
            ground_truth = self._find_matching_ground_truth_by_error_pattern(pfl_result)
            
            if ground_truth:
                matched_count += 1
                matched_pairs.append((pfl_result, ground_truth, matched_count - 1))
                self.logger.info(f"✓ 找到匹配的ground truth (instance_index {pfl_result.instance_index})")
            else:
                if pfl_result.failure_commit:
                    self.logger.warning(f"⚠️  未找到匹配的ground truth (instance_index {pfl_result.instance_index}, commit: {pfl_result.failure_commit[:8]}...)")
                else:
                    self.logger.warning(f"⚠️  未找到匹配的ground truth (instance_index {pfl_result.instance_index}, 无commit信息)")
        
        self.logger.info(f"匹配完成，准备处理 {matched_count} 个匹配的数据对")
        
        if self.use_parallel and matched_count > 1:
            comparison_results = self._run_parallel_comparison(matched_pairs)
        else:
            comparison_results = self._run_serial_comparison(matched_pairs)
        
        self.logger.info(f"开始保存结果到文件: {self.output_file}")
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for result in comparison_results:
                if result:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
        self.logger.info(f"结果保存完成，共保存 {len([r for r in comparison_results if r is not None])} 个结果")
        
        successful_results = [r for r in comparison_results if r is not None]
        
        self.logger.info(f"\n=== 分析完成 ===")
        self.logger.info(f"总PFL结果数: {total_count}")
        self.logger.info(f"成功匹配数: {matched_count}")
        self.logger.info(f"成功处理数: {len(successful_results)}")
        self.logger.info(f"匹配率: {matched_count/total_count*100:.1f}%")
        self.logger.info(f"处理成功率: {len(successful_results)/matched_count*100:.1f}%" if matched_count > 0 else "处理成功率: 0%")
        self.logger.info(f"结果保存到: {self.output_file}")
        
        return successful_results
    
    def _run_serial_comparison(self, matched_pairs: List[Tuple[PFLResult, GroundTruthData, int]]) -> List[Optional[Dict[str, Any]]]:
        self.logger.info("使用串行处理模式")
        comparison_results = []
        
        for pfl_result, ground_truth, index in tqdm(matched_pairs, desc="处理PFL结果"):
            try:
                if pfl_result.diff_text and pfl_result.diff_text.strip():
                    self.logger.info(f"使用PFL结果中的diff_text，跳过patch到diff转换")
                    pfl_diff = ""
                else:
                    self.logger.info(f"PFL结果中没有diff_text，使用patches生成diff")
                    pfl_diff = self.apply_pfl_patches_and_generate_diff(pfl_result, ground_truth)
                
                comparison_result = self.compare_patches_with_ground_truth(
                    pfl_result, ground_truth, pfl_diff
                )
                comparison_results.append(comparison_result)
                
            except Exception as e:
                self.logger.error(f"处理 instance_index {pfl_result.instance_index} 时发生错误: {e}")
                comparison_results.append(None)
        
        return comparison_results
    
    def _run_parallel_comparison(self, matched_pairs: List[Tuple[PFLResult, GroundTruthData, int]]) -> List[Optional[Dict[str, Any]]]:
        self.logger.info(f"使用并行处理模式，工作进程数: {self.max_workers}")
        self.logger.info(f"准备处理 {len(matched_pairs)} 个匹配的数据对")
        
        comparison_results = [None] * len(matched_pairs)
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {}
            for i, (pfl_result, ground_truth, _) in enumerate(matched_pairs):
                future = executor.submit(process_single_comparison_worker, 
                                       (pfl_result, ground_truth, i, self.repo_path, self.output_file))
                future_to_index[future] = i
            
            with tqdm(total=len(matched_pairs), desc="并行处理PFL结果") as pbar:
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        comparison_results[index] = result
                        if result:
                            pbar.set_postfix({
                                'instance': result.get('instance_index', 'unknown'),
                                'success': '✓'
                            })
                        else:
                            pbar.set_postfix({
                                'instance': 'unknown',
                                'success': '✗'
                            })
                    except Exception as e:
                        self.logger.error(f"并行处理任务 {index} 失败: {e}")
                        comparison_results[index] = None
                        pbar.set_postfix({
                            'instance': 'error',
                            'success': '✗'
                        })
                    finally:
                        pbar.update(1)
        
        self.logger.info(f"并行处理完成，共处理 {len(comparison_results)} 个结果")
        successful_count = sum(1 for r in comparison_results if r is not None)
        self.logger.info(f"成功处理 {successful_count} 个结果")
        
        return comparison_results

def process_single_comparison_worker(args_tuple: Tuple[PFLResult, GroundTruthData, int, str, str]) -> Optional[Dict[str, Any]]:
    pfl_result, ground_truth, index, repo_path, output_file = args_tuple
    
    try:
        os.environ['USE_CUSTOM_OPENAI_API'] = 'true'
        
        temp_comparator = GroundTruthComparator(
            pfl_results_file="",
            ground_truth_file="",
            repo_path=repo_path,
            output_file=output_file,
            auto_load=False
        )
        
        output_file = temp_comparator.output_file
        log_file = str(Path(output_file).parent / f"{Path(output_file).stem}.log")
        
        import logging
        temp_logger = logging.getLogger(f"PFLComparator_Worker_{index}")
        temp_logger.setLevel(logging.INFO)
        
        for handler in temp_logger.handlers[:]:
            temp_logger.removeHandler(handler)
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(f'[Worker-{index}] %(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        temp_logger.addHandler(file_handler)
        
        temp_comparator.logger = temp_logger
        
        temp_comparator.logger.info(f"[Worker-{index}] 开始处理 instance_index {pfl_result.instance_index}")
        
        if pfl_result.diff_text and pfl_result.diff_text.strip():
            temp_comparator.logger.info(f"[Worker-{index}] 使用PFL结果中的diff_text，跳过patch到diff转换")
            pfl_diff = ""
        else:
            temp_comparator.logger.info(f"[Worker-{index}] PFL结果中没有diff_text，使用patches生成diff")
            pfl_diff = temp_comparator.apply_pfl_patches_and_generate_diff(pfl_result, ground_truth)
        
        comparison_result = temp_comparator.compare_patches_with_ground_truth(
            pfl_result, ground_truth, pfl_diff
        )
        
        temp_comparator.logger.info(f"[Worker-{index}] 完成处理 instance_index {pfl_result.instance_index}")
        return comparison_result
        
    except Exception as e:
        import logging
        temp_logger = logging.getLogger(f"PFLComparator_Worker_{index}")
        temp_logger.error(f"处理 instance_index {pfl_result.instance_index} 时发生错误: {e}")
        return None
    

def main():
    parser = argparse.ArgumentParser(description='Compare PFL results with ground truth')
    parser.add_argument('--pfl-results', required=True, 
                       help='Path to PFL detailed results JSONL file')
    parser.add_argument('--ground-truth', required=True,
                       help='Path to ground truth repair analysis JSONL file')
    parser.add_argument('--repo-path', required=True,
                       help='Path to repository')
    parser.add_argument('--output', default='pfl_ground_truth_comparison.jsonl',
                       help='Output file path')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of parallel workers (default: min(CPU cores, 8))')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing (use serial mode)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pfl_results):
        print(f"Error: PFL results file not found: {args.pfl_results}")
        sys.exit(1)
    
    if not os.path.exists(args.ground_truth):
        print(f"Error: Ground truth file not found: {args.ground_truth}")
        sys.exit(1)
    
    if not os.path.exists(args.repo_path):
        print(f"Error: Repository not found: {args.repo_path}")
        sys.exit(1)
    
    comparator = GroundTruthComparator(
        pfl_results_file=args.pfl_results,
        ground_truth_file=args.ground_truth,
        repo_path=args.repo_path,
        output_file=args.output,
        max_workers=args.max_workers,
        use_parallel=not args.no_parallel
    )
    
    if args.verbose:
        comparator.logger.setLevel(logging.DEBUG)
        for handler in comparator.logger.handlers:
            handler.setLevel(logging.DEBUG)
    
    comparator.logger.info("=== 配置信息 ===")
    comparator.logger.info(f"PFL结果文件: {args.pfl_results}")
    comparator.logger.info(f"Ground truth文件: {args.ground_truth}")
    comparator.logger.info(f"仓库路径: {args.repo_path}")
    comparator.logger.info(f"输出文件: {args.output}")
    comparator.logger.info(f"并行处理: {'启用' if not args.no_parallel else '禁用'}")
    if not args.no_parallel:
        comparator.logger.info(f"最大工作进程数: {comparator.max_workers}")
    comparator.logger.info(f"详细日志: {'启用' if args.verbose else '禁用'}")
    
    start_time = time.time()
    
    results = comparator.run_comparison()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n=== 分析统计 ===")
    print(f"成功编译的PFL结果数量: {len(comparator.pfl_results)}")
    print(f"找到对应ground truth的数量: {len(results)}")
    print(f"覆盖率: {len(results)/len(comparator.pfl_results)*100:.1f}%")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    print(f"平均每个结果耗时: {elapsed_time/len(results):.2f} 秒" if len(results) > 0 else "平均每个结果耗时: N/A")
    
    comparator.logger.info(f"=== 性能统计 ===")
    comparator.logger.info(f"总耗时: {elapsed_time:.2f} 秒")
    comparator.logger.info(f"平均每个结果耗时: {elapsed_time/len(results):.2f} 秒" if len(results) > 0 else "平均每个结果耗时: N/A")
    if not args.no_parallel and len(results) > 1:
        estimated_serial_time = elapsed_time * comparator.max_workers
        speedup = estimated_serial_time / elapsed_time
        comparator.logger.info(f"预估串行处理时间: {estimated_serial_time:.2f} 秒")
        comparator.logger.info(f"并行加速比: {speedup:.2f}x")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
