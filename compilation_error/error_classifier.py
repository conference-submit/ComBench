#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import argparse
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from enum import Enum

class ErrorType(Enum):
    """Compilation error type enumeration"""
    # Main error types (first level)
    SYNTAX = "syntax_error"      # Including basic syntax errors and format errors
    TYPE = "type_error"        # Including type conversion, comparison, etc.
    DECLARATION = "declaration_error"  # Including declaration, definition, redefinition, etc.
    FUNCTION = "function_error"    # Including function calls, constructors, Lambda, etc.
    MEMBER = "member_error"      # Including member access, virtual functions, override, templates, namespaces, etc.
    SEMANTIC = "semantic_error"    # Including operators, range loops, etc.
    TEMPLATE = "template_error"    # Including template instantiation, deduction, etc. errors
    WERROR = "warning_as_error"   # 警告被当作错误处理
    OTHER = "other_error"      # 未分类的错误
    
    # 具体错误类型（第二层）
    # 语法相关
    SYNTAX_MISSING = "missing_error"  # 包括缺少分号、括号等
    SYNTAX_UNEXPECTED = "unexpected_token"  # 包括意外的标记、标识符等
    SYNTAX_FORMAT = "format_error"  # 包括缩进、格式化等
    SYNTAX_INITIALIZATION = "initialization_error"  # 包括初始化列表、聚合初始化等
    SYNTAX_OPERATOR_PRECEDENCE = "operator_precedence_error"  # 包括运算符优先级和结合性错误
    SYNTAX_CONSTEXPR = "constexpr_error"  # 包括constexpr相关的语法错误
    SYNTAX_MACRO = "macro_error"  # 包括宏参数、宏定义相关的错误
    
    # 类型相关
    TYPE_MISMATCH = "type_mismatch"  # 包括不兼容类型、类型转换等
    TYPE_INCOMPLETE = "incomplete_type"  # 包括不完整类型定义等
    TYPE_CONVERSION = "type_conversion_error"  # 包括指针转换、整数转换等
    TYPE_COMPARISON = "type_comparison_error"  # 包括不同类型比较
    TYPE_REFERENCE = "reference_error"  # 包括成员引用、指针引用等
    TYPE_OVERLOAD = "overload_error"  # 包括操作符重载、函数重载等
    TYPE_ABSTRACT = "abstract_class_error"  # 包括抽象类实例化错误
    TYPE_FUNCTION_CALL = "function_call_type_error"  # 包括函数调用类型错误
    
    # 声明相关
    DECLARATION_UNDEFINED = "undefined_error"  # 包括未声明标识符等
    DECLARATION_CONFLICT = "conflict_error"  # 包括重定义、初始化顺序等
    DECLARATION_MISSING = "missing_declaration"  # 包括找不到文件等
    DECLARATION_SHADOWING = "shadowing_error" # 包括变量隐藏、遮蔽等
    
    # 函数相关
    FUNCTION_CALL = "call_error"  # 包括函数调用错误
    FUNCTION_LAMBDA = "lambda_error"  # 包括Lambda表达式错误
    FUNCTION_OVERLOAD = "function_overload_error"  # 包括函数重载错误
    FUNCTION_CONSTRUCTOR = "constructor_error"  # 包括构造函数错误
    
    # 成员相关
    MEMBER_ACCESS = "access_error"  # 包括私有成员访问等
    MEMBER_TEMPLATE = "template_error"  # 包括模板参数错误
    MEMBER_NAMESPACE = "namespace_error"  # 包括命名空间错误
    MEMBER_NOT_FOUND = "member_not_found"  # 包括找不到成员、虚函数、override等
    MEMBER_AMBIGUOUS = "ambiguous_member_error" # 成员访问不明确
    
    # 语义相关
    SEMANTIC_OPERATOR = "operator_error"  # 包括运算符使用错误
    SEMANTIC_RANGE = "range_error"  # 包括范围循环错误
    SEMANTIC_ADDRESS = "address_error" # 包括获取右值/构造/析构函数地址的错误
    SEMANTIC_CASE_DUPLICATE = "duplicate_case_error"  # 包括重复的case值错误
    SEMANTIC_CONSTEXPR = "semantic_constexpr_error"  # 包括constexpr语义错误
    SEMANTIC_AMBIGUOUS = "semantic_ambiguous_error"  # 包括语义上的不明确引用

    # 模板相关
    TEMPLATE_INSTANTIATION = "template_instantiation_error"  # 包括模板实例化错误
    TEMPLATE_ARGUMENT_DEDUCTION = "template_argument_deduction_error"  # 包括模板参数推导错误
    TEMPLATE_SPECIALIZATION = "template_specialization_error"  # 包括模板特化错误

    # 警告升级为错误相关
    WERROR_UNUSED = "unused_warning_error"  # 未使用变量/函数等警告升级为错误
    WERROR_FORMAT = "format_warning_error"  # 格式化字符串警告升级为错误
    WERROR_DEPRECATED = "deprecated_warning_error"  # 弃用功能警告升级为错误
    WERROR_CONVERSION = "conversion_warning_error"  # 类型转换警告升级为错误
    WERROR_SIGN_COMPARE = "sign_compare_warning_error"  # 符号比较警告升级为错误
    WERROR_RETURN = "return_warning_error"  # 返回值警告升级为错误
    WERROR_INFINITE_RECURSION = "infinite_recursion_warning_error"  # 无限递归警告升级为错误
    WERROR_UNUSED_RESULT = "unused_result_warning_error"  # 未使用结果警告升级为错误
    WERROR_INVALID_CONSTEXPR = "invalid_constexpr_warning_error"  # 无效constexpr警告升级为错误
    
    # 其他特殊错误类型
    OTHER_LINKER = "linker_error"  # 链接器错误
    OTHER_INTERNAL = "internal_error"  # 编译器内部错误

class ErrorClassifier:
    """错误分类器，基于新的ErrorType枚举"""
    
    # 主要错误类型规则（第一层）
    MAIN_ERROR_TYPE_RULES = [
        # 警告升级为错误 - 通用规则，避免硬编码特定警告类型
        (r'warning.*treated.*as.*error|-Werror|warnings.*as.*errors|warning.*becomes.*error|error.*\(from.*warning\)|warning:.*\\\[-Werror|\\\[-Werror.*\\\]|deprecated.*\\\[-Werror|\\\[-Werror,-W\w+\\\]|error:.*\[-W[a-zA-Z0-9_-]+\]|\[-W[a-zA-Z0-9_-]+\]', ErrorType.WERROR),
        
        # 声明相关错误 - 优化和扩展规则
        (r'implicit declaration|no include path|undeclared|not declared|cannot find|cannot open|no such file|#include|cannot find symbol|could not find|redefinition|redeclaration|multiple definition|shadows|conflict|has not been declared|std::\w+.*has not been declared|unknown type name|enumeration previously declared|ACLE intrinsics.*not enabled|does not match.*declaration|definition.*does not match|out-of-line.*definition.*does not match|label.*used.*but.*not.*defined|label.*not.*defined|no declaration matches|duplicate.*member|does not name a type|did you mean|was declared.*and later.*static|static.*declaration.*follows.*non-static|class member.*cannot.*redeclared|using.*namespace|namespace.*std', ErrorType.DECLARATION),
        
        # 类型相关错误 - 扩展规则以包含更多类型不匹配情况
        (r'cannot convert|cannot.*dynamic_cast|dynamic_cast.*cannot|incompatible types|type mismatch|incomplete type|comparison of integers|\'typeid\' of incomplete type|is incomplete|.*has incomplete type|cannot be used as|non-constant-expression cannot be narrowed|cannot initialize.*parameter|ISO C\+\+11 does not allow conversion|comparison.*of.*different.*signs|auto.*not allowed|placeholder.*auto|no viable conversion|viable conversion|allocating an object of abstract class|called object type.*is not a function|no matching conversion for functional-style cast|incompatible.*pointer.*type|from.*incompatible.*pointer.*type|ordered.*comparison.*of.*pointer.*with.*integer|cannot.*bind.*reference|lvalue.*required|rvalue.*reference|cannot initialize.*type.*with.*rvalue.*type|incompatible.*function.*pointer.*types|void.*is not a class type|no viable.*overloaded.*=|static data member.*can only be initialized|invalid use of undefined type|constexpr.*if.*condition.*evaluates.*cannot.*narrowed|constant.*expression.*evaluates.*cannot.*narrowed|invalid initialization of reference|cannot bind to.*unrelated type|does not refer to a value|incomplete definition of type|cannot initialize.*variable.*of.*type.*with.*lvalue.*of.*type', ErrorType.TYPE),
        
        # 语法相关错误 - 扩展规则以包含更多语法错误模式
        (r'syntax error|unexpected.*(?:token|symbol)|expected.*(?:;|\)|{|}|<|>|primary-expression|unqualified-id|expression|statement|declaration|type-specifier|identifier|function.*body|comma.*in.*macro.*parameter.*list|\(.*for.*function-style.*cast|:)|missing.*(?:semicolon|parenthesis|brace|bracket)|\'&&\' within \'\|\||initializer list|operator.*precedence|missing exception specification|exception specification|C\+\+.*style.*comments.*not.*allowed|comments.*not.*allowed.*in.*ISO.*C90|no.*newline.*at.*end.*of.*file|unterminated.*(?:conditional|directive|comment|string)|macro.*(?:passed.*arguments.*but.*takes|requires.*arguments.*but.*only.*given|takes.*arguments|expects.*arguments)|too.*(?:many|few).*arguments.*(?:provided.*to|to).*(?:function-like.*)?macro.*(?:invocation|FOO|[A-Z_]+)|(?:too.*(?:many|few).*arguments.*to.*macro)|macro.*(?:argument|parameter).*(?:error|mismatch)|non-friend.*class.*member.*cannot.*have.*qualified.*name|cannot.*have.*qualified.*name|function definition is not allowed here|extraneous.*before', ErrorType.SYNTAX),
        
        # 成员相关错误
        (r'is not a member|has no member|no member|no matching member function|member initializer|member function|this argument|non-static member|access.*specifier|private.*member|protected.*member|override|virtual', ErrorType.MEMBER),
        
        # 函数相关错误
        (r'function call|call to|lambda.*(?:capture|not found)|constructor|destructor|no matching function|function.*not.*found|(?:too many|too few|wrong).*arguments|arguments to function|ambiguous.*call to', ErrorType.FUNCTION),
        
        # 语义相关错误
        (r'operator.*(?:error|not defined)|range.*for|address.*of|cannot.*take.*address|invalid operands to binary expression|no match for.*operator|static_assert.*failed|duplicate.*case.*value|constexpr.*(?:variable.*must be initialized|function.*return type.*not.*literal|function.*never produces.*constant|function.*calls.*non-constexpr)|reference.*to.*is.*ambiguous', ErrorType.SEMANTIC),

        # 模板相关错误
        (r'implicit instantiation of undefined template|template.*argument.*deduction|template.*instantiation|no.*arguments.*depend.*on.*template.*parameter|class.*template|member.*template', ErrorType.TEMPLATE),
        
        # 其他错误
        (r'fatal error|internal error|error:.*$', ErrorType.OTHER)
    ]
    
    # 具体错误类型规则（第二层）
    DETAILED_ERROR_TYPE_RULES = [
        # 函数相关 (需要优先匹配，避免被其他规则误捕获)
        (r'lambda.*(?:capture|not found)|capture.*(?:lambda|not found)|closure', ErrorType.FUNCTION_LAMBDA),
        (r'(?:function call|call to.*function|invalid.*function.*call)|(?:wrong|too many|too few).*arguments', ErrorType.FUNCTION_CALL),
        (r'no matching function|function.*overload|ambiguous.*(?:call|function)|multiple.*functions.*match', ErrorType.FUNCTION_OVERLOAD),
        (r'constructor|no.*suitable.*constructor|cannot.*construct|constructor.*is.*private', ErrorType.FUNCTION_CONSTRUCTOR),
        
        # 语法相关 - 扩展规则
        (r'missing.*(?:semicolon|parenthesis|brace|bracket)|expected.*(?:;|\)|{|}|<|>|:)|type specifier.*required.*for.*all.*declarations|a type specifier is required', ErrorType.SYNTAX_MISSING),
        (r'unexpected.*(?:token|identifier|symbol)|expected.*(?:primary-expression|unqualified-id|expression|statement|declaration|type-specifier|identifier|function.*body|\(.*for.*function-style.*cast)|unterminated.*(?:conditional|directive|comment|string)|non-friend.*class.*member.*cannot.*have.*qualified.*name|cannot.*have.*qualified.*name|extraneous.*before', ErrorType.SYNTAX_UNEXPECTED),
        (r'format|indent|spacing|style|version control conflict marker|conflict marker in file|C\+\+.*style.*comments.*not.*allowed|comments.*not.*allowed.*in.*ISO.*C90|no.*newline.*at.*end.*of.*file', ErrorType.SYNTAX_FORMAT),
        (r'initializer.*list|aggregate.*initialization|brace.*initialization', ErrorType.SYNTAX_INITIALIZATION),
        (r'operator.*(?:precedence|binding)|precedence.*error|\'&&\' within \'\|\|', ErrorType.SYNTAX_OPERATOR_PRECEDENCE),
        (r'missing exception specification|exception specification', ErrorType.SYNTAX_CONSTEXPR),
        (r'macro.*(?:passed.*arguments.*but.*takes|requires.*arguments.*but.*only.*given|takes.*arguments|expects.*arguments)|expected.*comma.*in.*macro.*parameter.*list|too.*(?:many|few).*arguments.*(?:provided.*to|to).*(?:function-like.*)?macro.*(?:invocation|FOO|[A-Z_]+)|(?:too.*(?:many|few).*arguments.*to.*macro)|macro.*(?:argument|parameter).*(?:error|mismatch)', ErrorType.SYNTAX_MACRO),
        
        # 类型相关 - 扩展规则
        (r'cannot convert|incompatible types|type mismatch|cannot be used as|non-constant-expression cannot be narrowed|cannot initialize.*parameter|invalid.*conversion|auto.*not allowed|placeholder.*auto|no matching conversion for functional-style cast|incompatible.*pointer.*type|from.*incompatible.*pointer.*type|cannot initialize.*type.*with.*rvalue.*type|incompatible.*function.*pointer.*types|void.*is not a class type|no viable.*overloaded.*=|cannot initialize.*variable.*of.*type.*with.*lvalue.*of.*type', ErrorType.TYPE_MISMATCH),
        (r'\'typeid\' of incomplete type|incomplete.*(?:type|class)|.*has incomplete type|forward.*declaration', ErrorType.TYPE_INCOMPLETE),
        (r'ISO C\+\+11 does not allow conversion|.*conversion.*(?:from|to)|no viable conversion|viable conversion|constant.*expression.*evaluates.*cannot.*narrowed', ErrorType.TYPE_CONVERSION),
        (r'comparison.*(?:of integers of different signs|between.*different.*types)|ordered.*comparison.*of.*pointer.*with.*integer', ErrorType.TYPE_COMPARISON),
        (r'cannot.*bind.*reference|lvalue.*required|rvalue.*reference|reference.*to.*(?:non-const|const)|invalid.*reference', ErrorType.TYPE_REFERENCE),
        (r'operator.*overload|ambiguous.*overload|overload.*resolution|overloaded.*operator', ErrorType.TYPE_OVERLOAD),
        (r'allocating an object of abstract class|pure virtual.*function', ErrorType.TYPE_ABSTRACT),
        (r'called object type.*is not a function', ErrorType.TYPE_FUNCTION_CALL),
        
        # 声明相关 - 扩展规则以更好地匹配各种未声明错误
        (r'implicit declaration|(?:was )?not declared|undeclared.*(?:here|identifier)|use of undeclared identifier|use of undeclared label|not.*declared.*in.*scope|std::\w+.*(?:has )?not been declared|unknown type name|\'[\w_]+\'.*undeclared.*\(first.*use.*in.*this.*function\)|\'[\w_]+\'.*undeclared.*\(first.*use\)|label.*used.*but.*not.*defined|label.*not.*defined|no declaration matches|does not name a type|did you mean|undeclared.*not in.*function', ErrorType.DECLARATION_UNDEFINED),
        (r'redefinition|redeclaration|already.*defined|multiple.*definition|conflicting.*declaration|enumeration previously declared|does not match.*declaration|definition.*does not match|out-of-line.*definition.*does not match|duplicate.*member|was declared.*and later.*static|static.*declaration.*follows.*non-static|class member.*cannot.*redeclared', ErrorType.DECLARATION_CONFLICT),
        (r'no include path|cannot find.*(?:symbol|file)|not found|could not find|cannot open|no such file|#include.*not.*found|ACLE intrinsics.*not enabled', ErrorType.DECLARATION_MISSING),
        (r'shadows.*(?:parameter|previous|declaration|variable)', ErrorType.DECLARATION_SHADOWING),
        (r'using.*namespace|namespace.*std|no.*named.*in.*namespace|namespace.*does.*not.*contain', ErrorType.DECLARATION_MISSING),

        
        # 成员相关 (整合成员访问规则)
        (r'(?:access|private|protected)|cannot.*access|member.*is.*(?:private|protected)', ErrorType.MEMBER_ACCESS),
        (r'is not a member|has no member|no member|member.*not.*found|no.*such.*member|no matching member function|override|virtual.*function|pure.*virtual', ErrorType.MEMBER_NOT_FOUND),
        
        # 语义相关
        (r'operator.*(?:error|not.*defined)|invalid.*operator|invalid operands to binary expression|no match for.*operator|binary.*expression.*invalid', ErrorType.SEMANTIC_OPERATOR),
        (r'range.*(?:for|based.*loop|expression)', ErrorType.SEMANTIC_RANGE),
        (r'address.*of|cannot.*take.*address|address.*cannot.*be.*taken', ErrorType.SEMANTIC_ADDRESS),
        (r'duplicate.*case.*value', ErrorType.SEMANTIC_CASE_DUPLICATE),
        (r'constexpr.*(?:variable.*must be initialized|function.*return type.*not.*literal|function.*never produces.*constant|function.*calls.*non-constexpr)|constexpr.*if.*condition.*evaluates.*cannot.*narrowed', ErrorType.SEMANTIC_CONSTEXPR),
        (r'reference.*to.*is.*ambiguous', ErrorType.SEMANTIC_AMBIGUOUS),

        # 模板相关
        (r'implicit instantiation of undefined template', ErrorType.TEMPLATE_INSTANTIATION),
        (r'template.*argument.*deduction|no.*arguments.*depend.*on.*template.*parameter', ErrorType.TEMPLATE_ARGUMENT_DEDUCTION),
        (r'template.*instantiation', ErrorType.TEMPLATE_INSTANTIATION),
        (r'class.*template|member.*template|requires template arguments|no template named', ErrorType.TEMPLATE_SPECIALIZATION),
        
        # 警告升级为错误相关 (通用规则，避免硬编码特定警告类型)
        (r'unused.*(?:variable|function|parameter|label|value)|set.*but.*not.*used', ErrorType.WERROR_UNUSED),
        (r'format.*(?:string|specifier|security)|printf.*format', ErrorType.WERROR_FORMAT),
        (r'deprecated|is.*deprecated|use.*of.*deprecated', ErrorType.WERROR_DEPRECATED),
        (r'(?:implicit|narrowing).*conversion|conversion.*loses.*precision', ErrorType.WERROR_CONVERSION),
        (r'comparison.*between.*signed.*and.*unsigned|signed.*unsigned.*comparison', ErrorType.WERROR_SIGN_COMPARE),
        (r'non-void function does not return a value', ErrorType.WERROR_RETURN),
        (r'all paths through this function will call itself', ErrorType.WERROR_INFINITE_RECURSION),
        (r'ignoring return value.*nodiscard', ErrorType.WERROR_UNUSED_RESULT),
        (r'constexpr function never produces', ErrorType.WERROR_INVALID_CONSTEXPR),
        
        # 其他特殊错误类型 
        (r'linker error|undefined reference|unresolved external symbol|ld:|collect2:', ErrorType.OTHER_LINKER),
        (r'internal.*(?:error|compiler error)|ice:|fatal error.*internal', ErrorType.OTHER_INTERNAL)
    ]
    
    def __init__(self):
        """初始化错误分类器"""
        self.error_type_rules = self.MAIN_ERROR_TYPE_RULES + self.DETAILED_ERROR_TYPE_RULES
    
    def identify_error_type(self, error_message: str) -> Tuple[ErrorType, ErrorType]:
        """识别错误类型，返回主要错误类型和具体错误类型的组合
        
        Args:
            error_message: 错误消息
            
        Returns:
            Tuple[ErrorType, ErrorType]: (主要错误类型, 具体错误类型)
        """
        error_message = error_message.lower()
        
        # 首先识别主要错误类型
        main_type = ErrorType.OTHER
        for pattern, error_type in self.MAIN_ERROR_TYPE_RULES:
            if re.search(pattern, error_message, re.IGNORECASE):
                main_type = error_type
                break
        
        # 然后识别具体错误类型
        detailed_type = ErrorType.OTHER
        for pattern, error_type in self.DETAILED_ERROR_TYPE_RULES:
            if re.search(pattern, error_message, re.IGNORECASE):
                detailed_type = error_type
                break
        
        return (main_type, detailed_type)
    
    def classify_error_lines(self, error_lines: List[str]) -> Dict[str, int]:
        """分类错误行列表"""
        error_types = defaultdict(int)
        
        for error_line in error_lines:
            main_type, detailed_type = self.identify_error_type(error_line)
            
            error_types[main_type.value] += 1
        
        return dict(error_types)
    
    def generate_classification_report(self, compiler_errors: List[Dict]) -> Dict:
        """生成详细的错误分类报告"""
        report = {
            'summary': {},
            'by_category': {},
            'top_errors': {},
            'error_frequency': {},
            'commit_impact': {},
            'error_examples': {}
        }
        
        # 统计所有错误类型
        all_error_types = defaultdict(int)
        error_lines_by_type = defaultdict(list)
        commits_by_error_type = defaultdict(set)
        
        for error_record in compiler_errors:
            commit_sha = error_record.get('commit_sha', '') or error_record.get('failure_commit', '')
            error_lines = error_record.get('error_lines', [])
            
            # 重新分类错误行
            classified_types = self.classify_error_lines(error_lines)
            
            for error_type, count in classified_types.items():
                all_error_types[error_type] += count
                commits_by_error_type[error_type].add(commit_sha)
            
            # 收集错误行示例
            for error_line in error_lines[:2]:  # 只取前两个作为示例
                main_type, detailed_type = self.identify_error_type(error_line)
                error_type = detailed_type.value if detailed_type != ErrorType.OTHER else main_type.value
                
                # 截取错误行中的关键部分作为示例
                if 'error:' in error_line:
                    example = error_line.split('error:')[-1].strip()[:100]
                else:
                    example = error_line[:100]
                
                if example and example not in error_lines_by_type[error_type]:
                    error_lines_by_type[error_type].append(example)
        
        # 生成摘要
        report['summary'] = {
            'total_error_types': len(all_error_types),
            'total_error_instances': sum(all_error_types.values()),
            'most_common_error': max(all_error_types.items(), key=lambda x: x[1]) if all_error_types else None,
            'commits_affected': len(set().union(*commits_by_error_type.values())) if commits_by_error_type else 0
        }
        
        # 按类别分组（基于ErrorType枚举）
        syntax_errors = [ErrorType.SYNTAX.value, ErrorType.SYNTAX_MISSING.value, 
                        ErrorType.SYNTAX_UNEXPECTED.value, ErrorType.SYNTAX_FORMAT.value,
                        ErrorType.SYNTAX_INITIALIZATION.value, ErrorType.SYNTAX_OPERATOR_PRECEDENCE.value,
                        ErrorType.SYNTAX_CONSTEXPR.value, ErrorType.SYNTAX_MACRO.value]
        type_errors = [ErrorType.TYPE.value, ErrorType.TYPE_MISMATCH.value, 
                      ErrorType.TYPE_INCOMPLETE.value, ErrorType.TYPE_CONVERSION.value,
                      ErrorType.TYPE_COMPARISON.value, ErrorType.TYPE_REFERENCE.value,
                      ErrorType.TYPE_OVERLOAD.value, ErrorType.TYPE_ABSTRACT.value,
                      ErrorType.TYPE_FUNCTION_CALL.value]
        declaration_errors = [ErrorType.DECLARATION.value, ErrorType.DECLARATION_UNDEFINED.value,
                             ErrorType.DECLARATION_CONFLICT.value, ErrorType.DECLARATION_MISSING.value,
                             ErrorType.DECLARATION_SHADOWING.value]
        function_errors = [ErrorType.FUNCTION.value, ErrorType.FUNCTION_CALL.value,
                          ErrorType.FUNCTION_LAMBDA.value, ErrorType.FUNCTION_OVERLOAD.value,
                          ErrorType.FUNCTION_CONSTRUCTOR.value]
        member_errors = [ErrorType.MEMBER.value, ErrorType.MEMBER_ACCESS.value,
                        ErrorType.MEMBER_NOT_FOUND.value]
        semantic_errors = [ErrorType.SEMANTIC.value, ErrorType.SEMANTIC_OPERATOR.value,
                          ErrorType.SEMANTIC_RANGE.value, ErrorType.SEMANTIC_ADDRESS.value,
                          ErrorType.SEMANTIC_CASE_DUPLICATE.value, ErrorType.SEMANTIC_CONSTEXPR.value,
                          ErrorType.SEMANTIC_AMBIGUOUS.value]
        werror_errors = [ErrorType.WERROR.value, ErrorType.WERROR_UNUSED.value,
                        ErrorType.WERROR_FORMAT.value, ErrorType.WERROR_DEPRECATED.value,
                        ErrorType.WERROR_CONVERSION.value, ErrorType.WERROR_SIGN_COMPARE.value,
                        ErrorType.WERROR_RETURN.value, ErrorType.WERROR_INFINITE_RECURSION.value,
                        ErrorType.WERROR_UNUSED_RESULT.value, ErrorType.WERROR_INVALID_CONSTEXPR.value]
        template_errors = [ErrorType.TEMPLATE.value, ErrorType.TEMPLATE_INSTANTIATION.value,
                          ErrorType.TEMPLATE_ARGUMENT_DEDUCTION.value, ErrorType.TEMPLATE_SPECIALIZATION.value]
        
        report['by_category'] = {
            'syntax_errors': sum(all_error_types[et] for et in syntax_errors),
            'type_errors': sum(all_error_types[et] for et in type_errors),
            'declaration_errors': sum(all_error_types[et] for et in declaration_errors),
            'function_errors': sum(all_error_types[et] for et in function_errors),
            'member_errors': sum(all_error_types[et] for et in member_errors),
            'semantic_errors': sum(all_error_types[et] for et in semantic_errors),
            'template_errors': sum(all_error_types[et] for et in template_errors),
            'werror_errors': sum(all_error_types[et] for et in werror_errors),
            'other_errors': all_error_types[ErrorType.OTHER.value]
        }
        
        # 前十位错误类型
        report['top_errors'] = dict(sorted(all_error_types.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # 错误频率（影响的commit数）
        report['error_frequency'] = {
            error_type: len(commits) 
            for error_type, commits in commits_by_error_type.items()
        }
        
        # 错误示例
        report['error_examples'] = {
            error_type: examples[:3]  # 只保留前3个示例
            for error_type, examples in error_lines_by_type.items()
            if examples
        }
        
        return report

def load_compiler_errors_data(file_path: str) -> List[Dict]:
    """加载编译错误数据，支持两种格式"""
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return []
    
    compiler_errors = []
    
    # 判断文件格式
    if file_path.endswith('.jsonl'):
        # JSONL格式 (analyze_ci_fixes_unified.py的输出)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            compiler_errors.append(data)
                        except json.JSONDecodeError as e:
                            print(f"解析JSONL行时出错: {e}")
        except Exception as e:
            print(f"读取JSONL文件失败: {e}")
    else:
        # JSON格式 (extract_compiler_errors.py的输出)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                compiler_errors = data.get('compiler_errors', [])
        except Exception as e:
            print(f"读取JSON文件失败: {e}")
    
    print(f"成功加载 {len(compiler_errors)} 条编译错误记录")
    return compiler_errors

def save_classification_report(report: Dict, output_file: str):
    """保存分类报告"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"分类报告已保存到: {output_file}")
    except Exception as e:
        print(f"保存分类报告失败: {e}")

def print_classification_summary(report: Dict):
    """打印分类摘要"""
    print("\n" + "="*60)
    print("错误分类报告摘要")
    print("="*60)
    
    summary = report['summary']
    print(f"错误类型总数: {summary['total_error_types']}")
    print(f"错误实例总数: {summary['total_error_instances']}")
    print(f"影响的提交数: {summary['commits_affected']}")
    
    if summary['most_common_error']:
        error_type, count = summary['most_common_error']
        print(f"最常见错误: {error_type} ({count} 次)")
    
    print(f"\n=== 按类别统计 ===")
    for category, count in sorted(report['by_category'].items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"  {category.replace('_', ' ').title()}: {count} 次")
    
    print(f"\n=== 前10种具体错误类型 ===")
    for error_type, count in list(report['top_errors'].items())[:10]:
        print(f"  {error_type}: {count} 次")
    
    print(f"\n=== 错误频率（影响的提交数）===")
    sorted_frequency = sorted(report['error_frequency'].items(), key=lambda x: x[1], reverse=True)
    for error_type, commit_count in sorted_frequency[:10]:
        print(f"  {error_type}: {commit_count} 个提交")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='编译错误分类器 - 基于error_analyzer.py的分类逻辑')
    parser.add_argument('input_file', help='输入文件路径 (JSON或JSONL格式)')
    parser.add_argument('--output', '-o', help='输出文件路径 (可选)')
    
    args = parser.parse_args()
    
    print("编译错误分类器")
    print("="*50)
    
    # 加载数据
    compiler_errors = load_compiler_errors_data(args.input_file)
    if not compiler_errors:
        print("没有找到编译错误数据")
        return
    
    # 创建分类器
    classifier = ErrorClassifier()
    
    # 生成分类报告
    print("正在生成错误分类报告...")
    start_time = time.time()
    report = classifier.generate_classification_report(compiler_errors)
    elapsed_time = time.time() - start_time
    print(f"分类完成，耗时: {elapsed_time:.2f}秒")
    
    # 打印摘要
    print_classification_summary(report)
    
    # 保存报告
    if args.output:
        output_file = args.output
    else:
        # 根据输入文件名生成输出文件名
        base_name = os.path.splitext(args.input_file)[0]
        output_file = f"{base_name}_classification_report.json"
    
    save_classification_report(report, output_file)
    
    print(f"\n完成！详细分类报告已保存到: {output_file}")

if __name__ == "__main__":
    main() 