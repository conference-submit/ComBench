#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compilation error matching module

Used to compare actually extracted compilation errors with expected error information to determine if they match
"""

import re
from typing import List, Dict, Tuple, Optional

class ErrorMatcher:
    """Compilation error matcher"""
    
    def __init__(self):
        pass
        
    def are_error_lines_similar(self, line1: str, line2: str) -> bool:
        """
        Compare whether two error lines are similar, ignoring path prefix differences and character differences (such as quotes, question marks, etc.)
        
        Args:
            line1: First error line
            line2: Second error line
            
        Returns:
            bool: Whether similar
        """
        # If completely equal, return True directly
        if line1 == line2:
            return True
        
        # Parse error line format: path:line:column: error: message
        # Use regular expression matching
        pattern = r'^(.+?):(\d+):(\d+):\s*error:\s*(.+)$'
        
        match1 = re.match(pattern, line1)
        match2 = re.match(pattern, line2)
        
        if not match1 or not match2:
            # If format does not match, use original comparison
            return line1 == line2
        
        # Extract each part
        path1, line_num1, col1, message1 = match1.groups()
        path2, line_num2, col2, message2 = match2.groups()
        
        # Compare line and column numbers
        if line_num1 != line_num2 or col1 != col2:
            return False
        
        # Compare error messages, ignoring character differences like quotes and question marks
        normalized_message1 = self._normalize_error_message(message1)
        normalized_message2 = self._normalize_error_message(message2)

        print(f"normalized_message1: {normalized_message1}")
        print(f"normalized_message2: {normalized_message2}")
        
        if normalized_message1 != normalized_message2:
            return False
        
        # 比较路径后缀（忽略前缀差异）
        # 提取路径的最后部分进行比较
        path1_parts = path1.split('/')
        path2_parts = path2.split('/')
        
        # 从后往前比较，找到相同的后缀
        min_len = min(len(path1_parts), len(path2_parts))
        for i in range(1, min_len + 1):
            suffix1 = '/'.join(path1_parts[-i:])
            suffix2 = '/'.join(path2_parts[-i:])
            if suffix1 == suffix2:
                return True
        
        # 如果路径后缀也不匹配，返回False
        return False
    
    def are_error_lines_similar_with_tolerance(self, line1: str, line2: str, line_tolerance: int = 5) -> bool:
        """
        比较两个错误行是否相似，支持行号容差
        
        Args:
            line1: First error line
            line2: Second error line
            line_tolerance: 行号允许的差异范围
            
        Returns:
            bool: Whether similar
        """
        # If completely equal, return True directly
        if line1 == line2:
            return True
        
        # Parse error line format: path:line:column: error: message
        # Use regular expression matching
        pattern = r'^(.+?):(\d+):(\d+):\s*error:\s*(.+)$'
        
        match1 = re.match(pattern, line1)
        match2 = re.match(pattern, line2)
        
        if not match1 or not match2:
            # If format does not match, use original comparison
            return line1 == line2
        
        # Extract each part
        path1, line_num1, col1, message1 = match1.groups()
        path2, line_num2, col2, message2 = match2.groups()
        
        # 比较行号，允许容差
        line_diff = abs(int(line_num1) - int(line_num2))
        if line_diff > line_tolerance:
            return False
        
        # 比较列号（如果行号差异在容差范围内，列号可以不同）
        # 这里我们放宽列号的比较，因为修复后列号可能会变化
        
        # Compare error messages, ignoring character differences like quotes and question marks
        normalized_message1 = self._normalize_error_message(message1)
        normalized_message2 = self._normalize_error_message(message2)

        print(f"normalized_message1: {normalized_message1}")
        print(f"normalized_message2: {normalized_message2}")
        
        if normalized_message1 != normalized_message2:
            return False
        
        # 比较路径后缀（忽略前缀差异）
        # 提取路径的最后部分进行比较
        path1_parts = path1.split('/')
        path2_parts = path2.split('/')
        
        # 从后往前比较，找到相同的后缀
        min_len = min(len(path1_parts), len(path2_parts))
        for i in range(1, min_len + 1):
            suffix1 = '/'.join(path1_parts[-i:])
            suffix2 = '/'.join(path2_parts[-i:])
            if suffix1 == suffix2:
                return True
        
        # 如果路径后缀也不匹配，返回False
        return False
    
    def _normalize_error_message(self, message: str) -> str:
        """
        标准化错误消息，删除所有引号、问号等字符差异，并忽略分号后的提示信息
        
        Args:
            message: 原始错误消息
            
        Returns:
            str: 标准化后的错误消息
        """
        # 如果包含分号，只取分号前的部分（忽略提示信息）
        semicolon_index = message.find(';')
        if semicolon_index != -1:
            message = message[:semicolon_index].strip()
        
        # 删除所有问号
        normalized = message.replace('?', '')
        
        # 删除所有引号（包括各种类型的引号）
        normalized = normalized.replace('‘', '')
        normalized = normalized.replace('’', '')
        normalized = normalized.replace('"', '')
        normalized = normalized.replace(''', '')
        normalized = normalized.replace(''', '')
        normalized = normalized.replace("'", '')
        
        # 移除多余的空格
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def match_single_error(self, actual_error: str, expected_errors: List[Dict], line_tolerance: int = 0) -> Tuple[bool, Optional[Dict]]:
        """
        将单个实际错误与预期错误列表进行匹配
        
        Args:
            actual_error: 实际的错误文本
            expected_errors: 预期错误列表
            line_tolerance: 行号允许的差异范围
            
        Returns:
            Tuple[bool, Optional[Dict]]: (是否匹配, 匹配的预期错误)
        """
        if not expected_errors:
            return False, None
        
        for expected_error in expected_errors:
            # 从预期错误中提取错误文本
            expected_text = ""
            if isinstance(expected_error, dict):
                # 尝试不同的字段名
                for field in ['error_lines']:
                    if field in expected_error:
                        expected_value = expected_error[field]
                        if isinstance(expected_value, list):
                            expected_text = ' '.join(expected_value)
                        else:
                            expected_text = str(expected_value)
                        break
            else:
                expected_text = str(expected_error)
            
            if not expected_text:
                continue
            
            # 使用are_error_lines_similar_with_tolerance进行匹配
            if self.are_error_lines_similar_with_tolerance(actual_error, expected_text, line_tolerance):
                return True, expected_error
        
        return False, None
    
    def match_errors(self, actual_errors: List[str], expected_errors: List[Dict]) -> Dict[str, any]:
        """
        匹配实际错误和预期错误
        
        Args:
            actual_errors: 实际错误列表
            expected_errors: 预期错误列表
            
        Returns:
            匹配结果字典
        """
        if not actual_errors:
            return {
                'similarity_score': 0.0,
                'matched_errors': [],
                'unmatched_actual': [],
                'unmatched_expected': expected_errors,
                'reason': '未提取到实际错误'
            }
        
        if not expected_errors:
            return {
                'similarity_score': 0.0,
                'matched_errors': [],
                'unmatched_actual': actual_errors,
                'unmatched_expected': [],
                'reason': '未找到预期错误'
            }
        
        matched_pairs = []
        unmatched_actual = []
        used_expected_indices = set()
        
        # 为每个实际错误找到匹配
        for actual_error in actual_errors:
            is_matched, matched_error = self.match_single_error(actual_error, expected_errors)
            
            if is_matched:
                # 找到匹配的预期错误的索引
                best_match_index = None
                for i, expected_error in enumerate(expected_errors):
                    if expected_error == matched_error and i not in used_expected_indices:
                        best_match_index = i
                        break
                
                if best_match_index is not None:
                    matched_pairs.append({
                        'actual_error': actual_error,
                        'expected_error': matched_error,
                        'match_type': 'exact'
                    })
                    used_expected_indices.add(best_match_index)
                else:
                    unmatched_actual.append(actual_error)
            else:
                unmatched_actual.append(actual_error)
        
        # 找到未匹配的预期错误
        unmatched_expected = [
            expected_errors[i] for i in range(len(expected_errors))
            if i not in used_expected_indices
        ]
        
        # 计算总体相似度分数（简化为匹配率）
        if matched_pairs:
            overall_similarity = len(matched_pairs) / max(len(actual_errors), len(expected_errors))
        else:
            overall_similarity = 0.0
        
        # 生成匹配原因
        reason = ""
        if not matched_pairs:
            reason = "没有找到匹配的错误"
        elif unmatched_actual:
            reason = f"有 {len(unmatched_actual)} 个实际错误未匹配"
        elif unmatched_expected:
            reason = f"有 {len(unmatched_expected)} 个预期错误未匹配"
        else:
            reason = "所有错误都成功匹配"
        
        return {
            'similarity_score': overall_similarity,
            'matched_errors': matched_pairs,
            'unmatched_actual': unmatched_actual,
            'unmatched_expected': unmatched_expected,
            'reason': reason,
            'match_count': len(matched_pairs),
            'actual_count': len(actual_errors),
            'expected_count': len(expected_errors)
        } 