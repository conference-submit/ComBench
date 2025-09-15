#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compilation error extraction module

Directly reuse functions from extract_compiler_errors.py and integrate classification logic from error_classifier.py
"""

import sys
import os
from pathlib import Path
from typing import List, Tuple
import re

# Add compilation_error directory to Python path
compilation_error_path = Path(__file__).parent.parent.parent / "compilation_error"
if str(compilation_error_path) not in sys.path:
    sys.path.insert(0, str(compilation_error_path))

# Import functions from extract_compiler_errors.py
from extract_compiler_errors import (
    is_error_line,
    is_error_line_no_exclude,
    is_warning_line,
    is_build_progress_line,
    is_build_system_line,
    clean_log_line,
    collect_error_context,
    extract_error_lines_simple
)

# Import classification logic from error_classifier.py
from error_classifier import ErrorClassifier, ErrorType

class ErrorExtractor:
    """Compilation error extractor - directly reuses functions from extract_compiler_errors.py and integrates classification logic from error_classifier.py"""
    
    def __init__(self):
        # Initialize error classifier
        self.error_classifier = ErrorClassifier()
    
    def extract_errors(self, log_content: str) -> List[str]:
        """
        Extract compilation errors from log content, excluding warning_as_error types
        
        Args:
            log_content: Log content string
            
        Returns:
            List[str]: 错误行列表，已去重
        """
        if not log_content:
            return []
        
        # 去除ANSI转义序列
        log_content = self._strip_ansi_escape_sequences(log_content)
        
        # 直接使用extract_compiler_errors.py中的函数获取所有错误行
        error_lines, _ = extract_error_lines_simple(log_content)
        
        # 过滤掉warning_as_error类型的错误并去重
        filtered_error_lines = []
        seen_errors = set()
        
        for error_line in error_lines:
            main_type, detailed_type = self.error_classifier.identify_error_type(error_line)
            
            # 排除warning_as_error类型的错误
            if main_type != ErrorType.WERROR:
                # 标准化错误信息用于去重
                normalized_error = self._normalize_quotes(error_line.strip())
                if normalized_error and normalized_error not in seen_errors:
                    seen_errors.add(normalized_error)
                    filtered_error_lines.append(normalized_error)
        
        return filtered_error_lines
    
    def _strip_ansi_escape_sequences(self, text: str) -> str:
        """
        移除ANSI转义序列
        
        Args:
            text: 包含ANSI转义序列的文本
            
        Returns:
            去除ANSI转义序列后的文本
        """
        import re
        # ANSI转义序列的正则表达式
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    def _normalize_quotes(self, s: str) -> str:
        """统一所有引号为英文单引号"""
        return re.sub(r"[‘’“”\"]", "'", s)
    
    def extract_error_details(self, log_content: str) -> Tuple[List[str], List[str]]:
        """
        从日志内容中提取编译错误及其详细信息，排除warning_as_error类型
        
        Args:
            log_content: Log content string
            
        Returns:
            Tuple[List[str], List[str]]: (错误行列表, 错误详细信息列表，已去重)
        """
        if not log_content:
            return [], []
        
        # 去除ANSI转义序列
        log_content = self._strip_ansi_escape_sequences(log_content)
        
        # 直接使用extract_compiler_errors.py中的函数获取所有错误行
        error_lines, error_details = extract_error_lines_simple(log_content)
        
        # 过滤掉warning_as_error类型的错误并去重
        filtered_error_lines = []
        filtered_error_details = []
        seen_errors = set()
        
        for i, error_line in enumerate(error_lines):
            main_type, detailed_type = self.error_classifier.identify_error_type(error_line)
            
            # 排除warning_as_error类型的错误
            if main_type != ErrorType.WERROR and detailed_type != ErrorType.WERROR:
                # 标准化错误信息用于去重
                normalized_error = self._normalize_quotes(error_line.strip())
                if normalized_error and normalized_error not in seen_errors:
                    seen_errors.add(normalized_error)
                    filtered_error_lines.append(normalized_error)
                    if i < len(error_details):
                        filtered_error_details.append(error_details[i])
        
        return filtered_error_lines, filtered_error_details
    
    def classify_error_line(self, error_line: str) -> Tuple[ErrorType, ErrorType]:
        """
        对单个错误行进行分类
        
        Args:
            error_line: 错误行字符串
            
        Returns:
            Tuple[ErrorType, ErrorType]: (主要错误类型, 具体错误类型)
        """
        return self.error_classifier.identify_error_type(error_line)
    
    def classify_error_lines(self, error_lines: List[str]) -> dict:
        """
        对错误行列表进行分类
        
        Args:
            error_lines: 错误行列表
            
        Returns:
            dict: 错误类型统计字典
        """
        return self.error_classifier.classify_error_lines(error_lines)
    
    # 为了向后兼容，提供与extract_compiler_errors.py相同的函数接口
    def is_error_line(self, line: str) -> bool:
        """检查是否是错误行"""
        return is_error_line(line)
    
    def is_error_line_no_exclude(self, line: str) -> bool:
        """检查是否是错误行（不进行排除过滤）"""
        return is_error_line_no_exclude(line)
    
    def is_warning_line(self, line: str) -> bool:
        """检查是否是warning行"""
        return is_warning_line(line)
    
    def is_build_progress_line(self, line: str) -> bool:
        """检查是否是构建进度信息行"""
        return is_build_progress_line(line)
    
    def is_build_system_line(self, line: str) -> bool:
        """判断是否为构建系统相关的行"""
        return is_build_system_line(line)
    
    def clean_log_line(self, line: str) -> str:
        """清理日志行，移除时间戳和后面的第一个空格"""
        return clean_log_line(line)
    
    def collect_error_context(self, lines: List[str], error_line_index: int) -> List[str]:
        """收集错误行及其上下文信息"""
        return collect_error_context(lines, error_line_index) 