#!/usr/bin/env python3
"""
Patch application utility functions module
"""

import json
import re
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from difflib import SequenceMatcher
import subprocess
import os

logger = logging.getLogger(__name__)


def extract_json_from_response(response: str) -> Dict:
    """Extract JSON from model response, referencing patch_agent.py implementation"""
    try:
        # Try to parse response directly as JSON
        return json.loads(response)
    except json.JSONDecodeError:
        # If failed, try to extract JSON part from text
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find parts containing braces
        json_match = re.search(r'({[\s\S]*})', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
    
    logger.warning(f"Unable to extract JSON from response: {response[:200]}...")
    return {}


def convert_to_diff_format(start_line: int, end_line: int, original_code: str, fixed_code: str) -> str:
    """
    Convert search-replace format to diff format
    
    Args:
        start_line: Start line number
        end_line: End line number
        original_code: Original code
        fixed_code: Fixed code
        
    Returns:
        Diff format string
    """
    original_lines = original_code.split('\n') if original_code else []
    fixed_lines = fixed_code.split('\n') if fixed_code else []
    
    # Calculate line count
    old_count = len(original_lines)
    new_count = len(fixed_lines)
    
    # Generate @@ line
    diff_header = f"@@ -{start_line},{old_count} +{start_line},{new_count} @@\n"
    
    # 生成diff内容
    diff_lines = []
    
    # 添加删除的行
    for line in original_lines:
        diff_lines.append(f"-{line}")
    
    # 添加新增的行
    for line in fixed_lines:
        diff_lines.append(f"+{line}")
    
    return diff_header + '\n'.join(diff_lines)


def replace_ignoring_whitespace(text: str, old: str, new: str) -> str:
    """
    忽略空白字符差异的替换函数（先尝试直接replace，失败再归一化匹配）
    参考agent_base.py的实现
    
    Args:
        text: 原始文本
        old: 要替换的文本
        new: 替换后的文本
        
    Returns:
        str: 替换后的文本
    """
    # 先尝试直接替换
    replaced = text.replace(old, new, 1)
    if replaced != text:
        return replaced
    
    # 如果直接替换失败，再归一化后查找
    def normalize(s):
        return ' '.join(s.split())
    
    norm_text = normalize(text)
    norm_old = normalize(old)
    start = norm_text.find(norm_old)
    if start == -1:
        return text
    
    def map_norm_to_raw(norm, raw, norm_start, norm_len):
        raw_i = 0
        norm_i = 0
        raw_start = None
        raw_end = None
        while raw_i < len(raw) and norm_i <= norm_start + norm_len:
            while raw_i < len(raw) and raw[raw_i].isspace():
                raw_i += 1
            while norm_i < len(norm) and norm[norm_i].isspace():
                norm_i += 1
            if norm_i == norm_start and raw_start is None:
                raw_start = raw_i
            if norm_i == norm_start + norm_len:
                raw_end = raw_i
                break
            if raw_i < len(raw) and norm_i < len(norm):
                raw_i += 1
                norm_i += 1
        if raw_end is None:
            raw_end = len(raw)
        return raw_start, raw_end
    
    raw_start, raw_end = map_norm_to_raw(norm_text, text, start, len(norm_old))
    if raw_start is None or raw_end is None:
        return text
    return text[:raw_start] + new + text[raw_end:]


def similar_code(code1: str, code2: str) -> bool:
    """
    判断两段代码是否相似
    
    Args:
        code1: 第一段代码
        code2: 第二段代码
        
    Returns:
        是否相似
    """
    # 去除空白字符后比较
    code1_clean = code1.strip()
    code2_clean = code2.strip()
    
    if code1_clean == code2_clean:
        return True
    
    # 如果完全匹配失败，尝试部分匹配
    if len(code1_clean) > 10 and len(code2_clean) > 10:
        # 计算相似度
        similarity = SequenceMatcher(None, code1_clean, code2_clean).ratio()
        return similarity > 0.8
    
    return False


def extract_original_code_from_file(repo_path: str, file_path: str, start_line: int, end_line: int) -> str:
    """
    从文件中提取指定行号的原始代码
    
    Args:
        repo_path: 仓库路径
        file_path: 文件路径
        start_line: Start line number
        end_line: End line number
        
    Returns:
        提取的原始代码
    """
    try:
        full_path = Path(repo_path) / file_path
        if not full_path.exists():
            # 尝试查找文件
            possible_paths = list(Path(repo_path).glob(f"**/{file_path}"))
            if possible_paths:
                full_path = possible_paths[0]
            else:
                logger.warning(f"文件不存在: {file_path}")
                return ""
        
        with open(full_path, 'r') as f:
            lines = f.readlines()
        
        # 提取指定行号的代码
        if 1 <= start_line <= len(lines) and 1 <= end_line <= len(lines):
            extracted_lines = lines[start_line - 1:end_line]
            return ''.join(extracted_lines).rstrip('\n')
        else:
            logger.warning(f"行号范围无效: {start_line}-{end_line}, 文件总行数: {len(lines)}")
            return ""
            
    except Exception as e:
        logger.error(f"提取原始代码时发生错误: {e}")
        return ""


def get_file_content_from_local(repo_path: str, file_path: str) -> Tuple[Optional[str], str]:
    """
    直接从本地文件系统获取文件内容，使用endswith逻辑匹配文件路径
    参考 get_file_content_from_git_with_path 的实现
    
    Args:
        repo_path: 仓库路径
        file_path: 文件路径
        
    Returns:
        Tuple[Optional[str], str]: (文件内容, 找到的文件路径)，如果获取失败返回(None, "")
    """
    try:
        repo_path_obj = Path(repo_path)
        file_name = os.path.basename(file_path)
        
        # 首先尝试直接路径拼接
        # 标准化file_path，确保使用正确的路径分隔符
        normalized_file_path = file_path.replace('\\', '/')  # 统一使用正斜杠
        target_file = repo_path_obj / normalized_file_path
        if target_file.exists():
            # 尝试以不同的编码读取文件
            encodings = ['utf-8', 'latin1', 'cp1252', 'gbk', 'shift-jis']
            for encoding in encodings:
                try:
                    with open(target_file, 'r', encoding=encoding) as f:
                        return f.read(), normalized_file_path
                except UnicodeDecodeError:
                    continue
                except Exception:
                    continue
        
        # 如果直接路径失败，使用endswith逻辑查找文件
        logger.debug(f"直接路径查找失败，尝试endswith匹配: {file_path}")
        
        # 递归查找所有文件
        all_files = []
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, repo_path)
                all_files.append(rel_path.replace(os.sep, '/'))
        
        matching_files = []
        basename_matching_files = []
        
        for found_file_path in all_files:
            # 检查file_path的后缀是否等于found_file_path的全部
            if file_path.endswith(found_file_path):
                # 计算匹配长度（file_path的路径部分数量）
                match_length = len(found_file_path)
                matching_files.append((found_file_path, match_length))
            
            # 同时检查basename匹配
            if found_file_path.endswith(f'/{file_name}'):
                basename_matching_files.append(found_file_path)
        
        # 优先使用最长路径后缀匹配的结果
        if matching_files:
            # 按匹配长度排序，选择最长的匹配
            matching_files.sort(key=lambda x: x[1], reverse=True)
            best_match = matching_files[0][0]
            match_length = matching_files[0][1]
            logger.debug(f"最长路径后缀匹配成功: {file_path} -> {best_match} (匹配长度: {match_length})")
        # 回退到basename匹配，同样按路径长度排序
        elif basename_matching_files:
            # 对basename匹配的文件也按路径长度排序，选择最长的
            basename_matching_files.sort(key=lambda x: len(x), reverse=True)
            best_match = basename_matching_files[0]
            logger.debug(f"basename匹配成功: {file_name} -> {best_match} (路径长度: {len(best_match)})")
        else:
            logger.warning(f"未找到匹配的文件: {file_path}")
            return None, ""
        
        # 构建完整路径并读取文件
        target_file = repo_path_obj / best_match.replace('/', os.sep)
        
        # 尝试以不同的编码读取文件
        encodings = ['utf-8', 'latin1', 'cp1252', 'gbk', 'shift-jis']
        for encoding in encodings:
            try:
                with open(target_file, 'r', encoding=encoding) as f:
                    return f.read(), best_match
            except UnicodeDecodeError:
                continue
            except Exception:
                continue
        
        return None, best_match
        
    except Exception as e:
        logger.error(f"获取文件内容时发生错误: {file_path} - {e}")
        return None, ""


def apply_search_replace_patch(repo_path: Path, patch, logger_instance=None) -> bool:
    """
    应用Search-Replace格式的补丁，参考agent_base.py的_apply_fix方法
    
    Args:
        repo_path: 仓库路径
        patch: SearchReplacePatch对象
        logger_instance: 日志记录器实例
        
    Returns:
        bool: 是否成功应用补丁
    """
    log = logger_instance or logger
    
    try:
        # 使用get_file_content_from_local获取文件内容和真实路径
        file_content, found_path = get_file_content_from_local(str(repo_path), str(patch.file_path))
        
        # 如果文件不存在，尝试创建新文件
        if file_content is None:
            log.warning(f"文件不存在，尝试创建新文件: {patch.file_path}")
            try:
                # 构建目标文件路径
                target_file = Path(repo_path) / patch.file_path
                target_file.parent.mkdir(parents=True, exist_ok=True)
                with open(target_file, 'w', encoding='utf-8') as f:
                    f.write(patch.fixed_code)
                log.info(f"✅ 创建新文件成功: {target_file}")
                return True
            except Exception as e:
                log.error(f"❌ 创建新文件失败: {patch.file_path}, 错误: {e}")
                return False
        
        # 使用找到的真实路径构建目标文件路径
        target_file = Path(repo_path) / found_path
        log.info(f"使用找到的文件路径: {found_path} -> {target_file}")
        
        orig_lines = file_content.splitlines(keepends=True)
        
        # 获取需要替换的行范围
        start_line = patch.start_line - 1  # 转换为0索引
        end_line = min(patch.end_line, len(orig_lines))  # 确保不越界
        
        if patch.original_code:
            # 有原始代码，进行精确替换
            range_content = ''.join(orig_lines[start_line:end_line])
            orig_code = patch.original_code
            fixed_code = patch.fixed_code
            
            # 检查原始代码是否存在于指定范围内（忽略空白字符）
            normalized_range_content = ' '.join(range_content.split())
            normalized_orig_code = ' '.join(orig_code.split())
            
            if normalized_orig_code in normalized_range_content:
                # 使用忽略空白字符的替换函数
                new_range_content = replace_ignoring_whitespace(range_content, orig_code, fixed_code)
                
                # 重建文件内容
                new_content = ''.join(orig_lines[:start_line]) + new_range_content + ''.join(orig_lines[end_line:])
                
                # 写入修复后的文件
                with open(target_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                log.info(f"✅ 在指定行范围内搜索替换成功: {target_file}")
                return True
            else:
                log.warning(f"⚠️ 在指定行范围内没找到原始代码，尝试在整个文件范围内搜索")
                
                # 在整个文件范围内搜索替换
                range_content = ''.join(orig_lines)
                normalized_range_content = ' '.join(range_content.split())
                
                if normalized_orig_code in normalized_range_content:
                    # 使用忽略空白字符的替换函数
                    new_content = replace_ignoring_whitespace(range_content, orig_code, fixed_code)
                    
                    # 写入修复后的文件
                    with open(target_file, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    
                    log.info(f"✅ 在整个文件范围内搜索替换成功: {target_file}")
                    return True
                else:
                    log.error(f"❌ 在整个文件范围内也没找到原始代码: {orig_code}")
                    return False
        else:
            # 如果original_code为空，则根据行号在指定行后插入fixed_code
            log.info(f"原始代码为空，在行 {end_line} 后插入修复代码: {target_file}")
            
            # 分割修复代码为行
            fixed_code_lines = patch.fixed_code.split('\n')
            
            # 计算原始行的缩进（如果存在）
            orig_indent = 0
            if start_line < len(orig_lines) and orig_lines[start_line].strip():
                orig_indent = len(orig_lines[start_line]) - len(orig_lines[start_line].lstrip())
            
            # 构建新的文件内容
            new_lines = []
            
            # 处理不同的插入场景
            if end_line == 0:
                # 在文件开头插入
                log.info(f"在文件开头插入修复代码")
                new_lines = []
            else:
                # 在指定行后插入
                new_lines.extend(orig_lines[:end_line])  # 添加原始行及之前的行
            
            # 添加修复的行（保持原始缩进）
            for i, line in enumerate(fixed_code_lines):
                if line.strip():  # 非空行
                    new_lines.append(' ' * orig_indent + line + '\n')
                else:
                    # 空行，保持换行符
                    new_lines.append('\n')
            
            new_lines.extend(orig_lines[end_line:])  # 添加原始行之后的行
            
            # 写入修复后的文件
            try:
                with open(target_file, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
                log.info(f"✅ 在行 {end_line} 后插入修复代码成功: {target_file}")
                return True
            except Exception as e:
                log.error(f"❌ 在行 {end_line} 后插入修复代码失败: {target_file}, 错误: {e}")
                return False
        
    except Exception as e:
        log.error(f"❌ 应用Search-Replace补丁失败: {e}")
        return False


def normalize_file_path_for_git(file_path: str) -> str:
    """
    将文件路径转换为相对于项目根目录的路径，用于git命令
    
    Args:
        file_path: 原始文件路径（可能是绝对路径或相对路径）
        
    Returns:
        str: 相对于项目根目录的路径
    """
    try:
        # 确保路径使用正斜杠（git期望的格式）
        normalized_path = file_path.replace(os.sep, '/')
        
        # 如果路径包含项目根目录，尝试截取相对部分
        # 这里可以根据需要添加项目特定的路径处理逻辑
        return normalized_path
    except Exception as e:
        logger.warning(f"路径标准化失败: {file_path} - {e}")
        return file_path.replace(os.sep, '/')


def get_file_content_from_git(repo_path: str, commit: str, file_path: str) -> Optional[str]:
    """
    获取指定commit下的文件内容，优先直接路径，失败则用ls-tree查找。
    参考 analyze_ci_fixes_unified.py 的实现
    
    Args:
        repo_path: 仓库路径
        commit: 提交SHA
        file_path: 文件路径
        
    Returns:
        Optional[str]: 文件内容，如果获取失败返回None
    """
    file_name = os.path.basename(file_path)
    
    # 检查仓库路径是否存在
    if not os.path.exists(repo_path):
        logger.warning(f"仓库路径不存在: {repo_path}")
        return None
    
    # 检查是否为Git仓库
    if not os.path.exists(os.path.join(repo_path, '.git')):
        logger.warning(f"不是Git仓库: {repo_path}")
        return None
    
    try:
        result = subprocess.run(
            ['git', 'show', f'{commit}:{file_path}'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8',
            errors='ignore'
        )
        logger.debug(f"直接成功: {commit}:{file_path}")
        return result.stdout
    except subprocess.CalledProcessError:
        logger.debug(f"直接路径查找失败，尝试ls-tree: {commit}:{file_path}")
        try:
            ls_result = subprocess.run(
                ['git', 'ls-tree', '-r', '--name-only', commit],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
                encoding='utf-8',
                errors='ignore'
            )
            
            matching_files = []
            basename_matching_files = []
            file_path_parts = file_path.split('/')

            for found_file_path in ls_result.stdout.splitlines():
                file_path_parts = found_file_path.split('/')
                
                # 检查file_path的后缀是否等于found_file_path的全部
                if file_path.endswith(found_file_path):
                    # 计算匹配长度（file_path的路径部分数量）
                    match_length = len(found_file_path)
                    matching_files.append((found_file_path, match_length))
                
                # 同时检查basename匹配
                if found_file_path.endswith(f'/{file_name}'):
                    basename_matching_files.append(found_file_path)
                        
            # 优先使用最长路径后缀匹配的结果
            if matching_files:
                # 按匹配长度排序，选择最长的匹配
                matching_files.sort(key=lambda x: x[1], reverse=True)
                best_match = matching_files[0][0]
                match_length = matching_files[0][1]
                logger.debug(f"最长路径后缀匹配成功: {file_path} -> {best_match} (匹配长度: {match_length})")
            # 回退到basename匹配
            elif basename_matching_files:
                best_match = basename_matching_files[0]
                logger.debug(f"basename匹配成功: {file_name} -> {best_match}")
            else:
                logger.warning(f"ls-tree中未找到文件: {file_path}@{commit}")
                return None

            result = subprocess.run(
                ['git', 'show', f'{commit}:{best_match}'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
                encoding='utf-8',
                errors='ignore'
            )
            return result.stdout

        except subprocess.TimeoutExpired as e:
            logger.warning(f"获取文件内容超时: {file_path}@{commit} - {e}")
        except Exception as e:
            logger.warning(f"ls-tree回退失败: {file_path}@{commit} - {e}")
    except Exception as e:
        logger.error(f"获取文件内容时发生未知错误: {file_path}@{commit} - {e}")
    return None


def get_file_content_from_git_with_path(repo_path: str, commit: str, file_path: str) -> Tuple[Optional[str], str]:
    """
    获取指定commit下的文件内容，返回文件内容和找到的文件路径
    参考 analyze_ci_fixes_unified.py 的实现
    
    Args:
        repo_path: 仓库路径
        commit: 提交SHA
        file_path: 文件路径
        
    Returns:
        Tuple[Optional[str], str]: (文件内容, 找到的文件路径)
    """
    file_name = os.path.basename(file_path)
    
    # 检查仓库路径是否存在
    if not os.path.exists(repo_path):
        logger.warning(f"仓库路径不存在: {repo_path}")
        return None, ""
    
    # 检查是否为Git仓库
    if not os.path.exists(os.path.join(repo_path, '.git')):
        logger.warning(f"不是Git仓库: {repo_path}")
        return None, ""
    
    try:
        result = subprocess.run(
            ['git', 'show', f'{commit}:{file_path}'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8',
            errors='ignore'
        )
        logger.debug(f"直接成功: {commit}:{file_path}")
        return result.stdout, file_path
    except subprocess.CalledProcessError:
        logger.debug(f"直接路径查找失败，尝试ls-tree: {commit}:{file_path}")
        try:
            ls_result = subprocess.run(
                ['git', 'ls-tree', '-r', '--name-only', commit],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
                encoding='utf-8',
                errors='ignore'
            )
            
            matching_files = []
            basename_matching_files = []
            file_path_parts = file_path.split('/')

            for found_file_path in ls_result.stdout.splitlines():
                file_path_parts = found_file_path.split('/')
                
                # 检查file_path的后缀是否等于found_file_path的全部
                if file_path.endswith(found_file_path):
                    # 计算匹配长度（file_path的路径部分数量）
                    match_length = len(found_file_path)
                    matching_files.append((found_file_path, match_length))
                
                # 同时检查basename匹配
                if found_file_path.endswith(f'/{file_name}'):
                    basename_matching_files.append(found_file_path)
                        
            # 优先使用最长路径后缀匹配的结果
            if matching_files:
                # 按匹配长度排序，选择最长的匹配
                matching_files.sort(key=lambda x: x[1], reverse=True)
                best_match = matching_files[0][0]
                match_length = matching_files[0][1]
                logger.debug(f"最长路径后缀匹配成功: {file_path} -> {best_match} (匹配长度: {match_length})")
            # 回退到basename匹配
            elif basename_matching_files:
                best_match = basename_matching_files[0]
                logger.debug(f"basename匹配成功: {file_name} -> {best_match}")
            else:
                logger.warning(f"ls-tree中未找到文件: {file_path}@{commit}")
                return None, ""

            result = subprocess.run(
                ['git', 'show', f'{commit}:{best_match}'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
                encoding='utf-8',
                errors='ignore'
            )
            return result.stdout, best_match

        except subprocess.TimeoutExpired as e:
            logger.warning(f"获取文件内容超时: {file_path}@{commit} - {e}")
        except Exception as e:
            logger.warning(f"ls-tree回退失败: {file_path}@{commit} - {e}")
    except Exception as e:
        logger.error(f"获取文件内容时发生未知错误: {file_path}@{commit} - {e}")
    return None, ""