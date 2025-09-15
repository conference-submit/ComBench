#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified CI failure record collection tool
Supports CI failure record collection and analysis for multiple open source projects
"""

import os
import json
import requests
import argparse
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
from requests.exceptions import RequestException, ConnectionError, Timeout, HTTPError
import random
import chardet

# Configure logging
def setup_logging(repo_name: str):
    """Set logging configuration"""
    log_filename = f'collect_{repo_name}_ci_failures.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# GitHub API Configuration
# IMPORTANT: You must configure your GitHub Personal Access Token using environment variable
# Set GITHUB_TOKEN environment variable
# Example: export GITHUB_TOKEN="ghp_your_token_here"

def get_github_token():
    """Get GitHub token from environment variable"""
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        raise ValueError(
            "No GitHub token found! Please set environment variable:\n"
            "  export GITHUB_TOKEN='ghp_your_token_here'"
        )
    return token.strip()

GITHUB_TOKEN = get_github_token()
API_BASE_URL = "https://api.github.com"

# Error handling configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
REQUEST_TIMEOUT = 30  # seconds  
RATE_LIMIT_DELAY = 60  # seconds

def load_project_configs() -> Dict:
    """Load project configuration from config file"""
    config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'project_configs.json')
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            configs = json.load(f)
        return configs
    except Exception as e:
        logging.error(f"Failed to load configuration file: {e}")
        raise

# Load project configuration
REPO_CONFIGS = load_project_configs()

def get_github_token():
    """Get GitHub Token"""
    return GITHUB_TOKEN

def detect_encoding(content: bytes) -> str:
    """
    Automatically detect encoding of byte content
    
    Args:
        content: Byte content
        
    Returns:
        str: Detected encoding name
    """
    # Use chardet to detect encoding
    result = chardet.detect(content)
    encoding = result['encoding']
    confidence = result['confidence']
    
    # If confidence is too low, use default encoding
    if confidence < 0.7:
        return 'utf-8'
    
    # Handle some common encoding mappings
    encoding_map = {
        'ascii': 'utf-8',
        'iso-8859-1': 'latin1',
        'windows-1252': 'cp1252'
    }
    
    detected_encoding = encoding_map.get(encoding, encoding)
    
    # For pure ASCII content, prefer UTF-8
    if encoding == 'ascii':
        return 'utf-8'
    
    return detected_encoding

class CIFailureCollector:
    """CI失败记录收集器"""
    
    def __init__(self, repo_key: str):
        configs = load_project_configs()
        if repo_key not in configs:
            raise ValueError(f"不支持的仓库: {repo_key}. 支持的仓库: {list(configs.keys())}")
        
        self.repo_key = repo_key
        self.config = configs[repo_key]
        self.repo_owner = self.config['repo_owner']
        self.repo_name = self.config['repo_name']
        self.failures_file = f'{repo_key}_ci_failures.json'
        
        # 设置日志
        self.logger = setup_logging(repo_key)
        
        self.logger.info(f"初始化 {repo_key} 项目的CI失败收集器")
        self.logger.info(f"目标仓库: {self.repo_owner}/{self.repo_name}")
        
    def get_headers(self):
        """获取 GitHub API 请求头"""
        token = get_github_token()
        headers = {
            'Accept': 'application/vnd.github.v3+json',
            'Authorization': f'token {token}'
        }
        return headers

    def handle_rate_limit(self, response):
        """处理 GitHub API 限制"""
        if response.status_code == 403:
            reset_time = response.headers.get('x-ratelimit-reset')
            if reset_time:
                reset_timestamp = int(reset_time)
                current_timestamp = int(time.time())
                wait_time = max(reset_timestamp - current_timestamp, 60)
                self.logger.warning(f"遇到API限制，等待 {wait_time} 秒...")
                time.sleep(wait_time)
                return True
        return False

    def make_request_with_retry(self, url: str, params: dict = None, max_retries: int = MAX_RETRIES) -> Optional[requests.Response]:
        """带重试机制的请求函数"""
        session = self.create_session()
        
        for attempt in range(max_retries + 1):
            try:
                self.logger.debug(f"请求 URL: {url}, 尝试次数: {attempt + 1}")
                response = session.get(
                    url, 
                    headers=self.get_headers(), 
                    params=params, 
                    timeout=REQUEST_TIMEOUT
                )
                
                # 处理API限制
                if self.handle_rate_limit(response):
                    continue
                    
                # 检查HTTP状态码
                response.raise_for_status()
                return response
                
            except ConnectionError as e:
                self.logger.warning(f"连接错误 (尝试 {attempt + 1}/{max_retries + 1}): {str(e)}")
                if attempt < max_retries:
                    wait_time = RETRY_DELAY * (2 ** attempt)  # 指数退避
                    self.logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"连接失败，已达到最大重试次数")
                    return None
                    
            except Timeout as e:
                self.logger.warning(f"请求超时 (尝试 {attempt + 1}/{max_retries + 1}): {str(e)}")
                if attempt < max_retries:
                    time.sleep(RETRY_DELAY)
                else:
                    self.logger.error(f"请求超时，已达到最大重试次数")
                    return None
                    
            except HTTPError as e:
                if e.response.status_code == 403:
                    self.logger.warning("遇到API限制，等待后重试...")
                    time.sleep(RATE_LIMIT_DELAY)
                    continue
                elif e.response.status_code == 404:
                    self.logger.warning(f"资源未找到: {url}")
                    return None
                else:
                    self.logger.error(f"HTTP错误 {e.response.status_code}: {str(e)}")
                    return None
                    
            except RequestException as e:
                self.logger.warning(f"请求异常 (尝试 {attempt + 1}/{max_retries + 1}): {str(e)}")
                if attempt < max_retries:
                    time.sleep(RETRY_DELAY)
                else:
                    self.logger.error(f"请求失败，已达到最大重试次数")
                    return None
                    
            except Exception as e:
                self.logger.error(f"未知错误: {str(e)}")
                return None
        
        return None

    def get_workflows(self) -> List[Dict]:
        """获取仓库的所有 workflows"""
        url = f"{API_BASE_URL}/repos/{self.repo_owner}/{self.repo_name}/actions/workflows"
        
        try:
            response = self.make_request_with_retry(url)
            if response and response.status_code == 200:
                workflows = response.json()['workflows']
                self.logger.info("可用的 workflows:")
                for workflow in workflows:
                    self.logger.info(f"  Name: {workflow['name']}, ID: {workflow['id']}, Path: {workflow['path']}")
                return workflows
            else:
                self.logger.error("获取 workflows 失败")
                return []
        except json.JSONDecodeError as e:
            self.logger.error(f"解析 workflows JSON 失败: {str(e)}")
            return []
        except Exception as e:
            self.logger.error(f"获取 workflows 时发生未知错误: {str(e)}")
            return []

    def get_workflow_id_by_name(self, workflow_name: str) -> Optional[int]:
        """根据 workflow 名称获取其 ID。找不到则返回 None。"""
        try:
            workflows = self.get_workflows()
            for wf in workflows:
                if wf.get('name') == workflow_name:
                    return wf.get('id')
            self.logger.warning(f"未在仓库中找到名称为 '{workflow_name}' 的 workflow")
            return None
        except Exception as e:
            self.logger.error(f"查找 workflow '{workflow_name}' 的 ID 时出错: {e}")
            return None

    def get_workflow_runs_range(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """获取指定时间范围内的 workflow 运行记录，使用优化的API参数"""
        try:
            url = f"{API_BASE_URL}/repos/{self.repo_owner}/{self.repo_name}/actions/runs"
            
            all_runs = []
            page = 1
            max_pages = 100  # 限制最大页数，避免无限循环
            found_old_data = False
            current_end_time = end_time  # 动态调整结束时间
            
            self.logger.info(f"开始获取所有workflow的失败运行记录: {start_time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} ~ {end_time.strftime('%Y-%m-%dT%H:%M:%S+00:00')}")
            
            while page <= max_pages:
                # 构建API参数，使用status=failure和created参数优化查询
                params = {
                    'per_page': 100,  # 使用标准页面大小
                    'page': page,
                    'status': 'failure',  # 只获取失败的运行
                    'created': f'{start_time.isoformat()}..{current_end_time.isoformat()}'  # 获取指定时间范围内的失败记录
                }
                
                self.logger.debug(f"第 {page} 页请求，时间范围: <= {current_end_time.strftime('%Y-%m-%dT%H:%M:%S+00:00')}")
                
                response = self.make_request_with_retry(url, params)
                
                if not response or response.status_code != 200:
                    self.logger.error(f"获取 workflow runs 失败，页面: {page}")
                    break
                    
                try:
                    runs_data = response.json()
                    runs = runs_data.get('workflow_runs', [])
                    if not runs:
                        self.logger.info(f"第 {page} 页没有数据，调整时间范围继续获取")
                        # 调整时间范围，重设为已获得数据的最小时间值
                        if all_runs:
                            # 找到已获得数据的最小时间
                            min_time = min(datetime.fromisoformat(run['created_at'].replace('Z', '+00:00')) for run in all_runs)
                            # 确保新的结束时间严格小于最小时间，避免重复获取
                            if min_time < current_end_time:
                                # 将结束时间设置为比最小时间更早一点，确保不重复获取
                                current_end_time = min_time - timedelta(seconds=1)
                                self.logger.info(f"重设时间范围为已获得数据的最小时间减1秒: {current_end_time.strftime('%Y-%m-%dT%H:%M:%S+00:00')}")
                            else:
                                # 如果最小时间不比当前结束时间早，向前推进时间
                                current_end_time = current_end_time - timedelta(days=1)
                                self.logger.info(f"最小时间不比当前结束时间早，向前推进1天: {current_end_time.strftime('%Y-%m-%dT%H:%M:%S+00:00')}")
                        else:
                            # 如果还没有获得任何数据，向前推进时间
                            current_end_time = current_end_time - timedelta(days=7)
                            self.logger.info(f"尚未获得数据，向前推进时间: {current_end_time.strftime('%Y-%m-%dT%H:%M:%S+00:00')}")
                        
                        if current_end_time < start_time:
                            self.logger.info(f"时间范围已超出目标范围，停止获取")
                            break
                        page = 1  # 重置页码
                        continue
                    
                    page_runs_count = 0
                    old_data_count = 0
                    future_data_count = 0
                    non_compilation_runs = 0
                    earliest_time = None

                    # 处理当前页的数据
                    for run in runs:
                        # 只处理编译相关的workflow
                        if not self.is_compilation_workflow(run['name']):
                            self.logger.debug(f"跳过非编译相关workflow: {run['name']}")
                            non_compilation_runs += 1
                            continue
                        
                        # created_at' is in ISO 8601 format, e.g., '2021-08-25T14:30:00Z'
                        created_at = datetime.fromisoformat(run['created_at'].replace('Z', '+00:00'))
                        
                        # 检查是否在目标时间范围内
                        if start_time <= created_at < end_time:
                            all_runs.append(run)
                            page_runs_count += 1
                        elif created_at < start_time:
                            # 发现早于开始时间的数据
                            old_data_count += 1
                            found_old_data = True
                        elif created_at >= end_time:
                            # 发现晚于结束时间的数据
                            future_data_count += 1
                    
                    self.logger.info(f"第 {page} 页，找到 {len(runs)} 个失败记录，其中编译相关的 {page_runs_count} 个在时间范围内，{old_data_count} 个太旧，{future_data_count} 个太新，{non_compilation_runs} 个非编译相关")
                    
                    # 如果当前页没有找到任何在时间范围内的数据，且发现了旧数据，就停止
                    if page_runs_count == 0 and found_old_data:
                        self.logger.info(f"第 {page} 页没有在时间范围内的数据，且发现了旧数据，停止获取")
                        break
                    
                    page += 1
                    
                    # 添加延迟避免触发API限制
                    time.sleep(0.5)
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"解析 workflow runs JSON 失败: {str(e)}")
                    break
                    
            self.logger.info(f"总共获取到 {len(all_runs)} 个在时间范围内的编译相关失败 workflow runs")
            return all_runs
            
        except Exception as e:
            self.logger.error(f"获取 workflow runs 时发生未知错误: {str(e)}")
            return []

    def get_workflow_runs_range_for_workflow(self, workflow_id: int, start_time: datetime, end_time: datetime) -> List[Dict]:
        """按单个 workflow 拉取指定时间范围内的失败运行记录。"""
        try:
            url = f"{API_BASE_URL}/repos/{self.repo_owner}/{self.repo_name}/actions/workflows/{workflow_id}/runs"

            all_runs = []
            page = 1
            max_pages = 100
            found_old_data = False
            current_end_time = end_time

            self.logger.info(
                f"开始获取指定workflow(ID={workflow_id})的失败运行记录: {start_time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} ~ {end_time.strftime('%Y-%m-%dT%H:%M:%S+00:00')}"
            )

            while page <= max_pages:
                params = {
                    'per_page': 100,
                    'page': page,
                    'status': 'failure',
                    'created': f'{start_time.isoformat()}..{current_end_time.isoformat()}'
                }

                self.logger.debug(f"[workflow {workflow_id}] 第 {page} 页请求，时间范围: <= {current_end_time.strftime('%Y-%m-%dT%H:%M:%S+00:00')}")

                response = self.make_request_with_retry(url, params)
                if not response or response.status_code != 200:
                    self.logger.error(f"获取指定workflow的 runs 失败，页面: {page}")
                    break

                try:
                    runs_data = response.json()
                    runs = runs_data.get('workflow_runs', [])
                    if not runs:
                        self.logger.info(f"[workflow {workflow_id}] 第 {page} 页没有数据，调整时间范围继续获取")
                        if all_runs:
                            min_time = min(datetime.fromisoformat(run['created_at'].replace('Z', '+00:00')) for run in all_runs)
                            if min_time < current_end_time:
                                current_end_time = min_time - timedelta(seconds=1)
                                self.logger.info(f"[workflow {workflow_id}] 重设时间范围为已获得数据的最小时间减1秒: {current_end_time.strftime('%Y-%m-%dT%H:%M:%S+00:00')}")
                            else:
                                current_end_time = current_end_time - timedelta(days=1)
                                self.logger.info(f"[workflow {workflow_id}] 最小时间不比当前结束时间早，向前推进1天: {current_end_time.strftime('%Y-%m-%dT%H:%M:%S+00:00')}")
                        else:
                            current_end_time = current_end_time - timedelta(days=7)
                            self.logger.info(f"[workflow {workflow_id}] 尚未获得数据，向前推进时间: {current_end_time.strftime('%Y-%m-%dT%H:%M:%S+00:00')}")

                        if current_end_time < start_time:
                            self.logger.info(f"[workflow {workflow_id}] 时间范围已超出目标范围，停止获取")
                            break
                        page = 1
                        continue

                    page_runs_count = 0
                    old_data_count = 0
                    future_data_count = 0

                    for run in runs:
                        created_at = datetime.fromisoformat(run['created_at'].replace('Z', '+00:00'))
                        # 这里不再按名称过滤，因为已经限定到单个workflow
                        if start_time <= created_at < end_time:
                            all_runs.append(run)
                            page_runs_count += 1
                        elif created_at < start_time:
                            old_data_count += 1
                            found_old_data = True
                        elif created_at >= end_time:
                            future_data_count += 1

                    self.logger.info(
                        f"[workflow {workflow_id}] 第 {page} 页，找到 {len(runs)} 个失败记录，其中 {page_runs_count} 个在时间范围内，{old_data_count} 个太旧，{future_data_count} 个太新"
                    )

                    if page_runs_count == 0 and found_old_data:
                        self.logger.info(f"[workflow {workflow_id}] 第 {page} 页没有在时间范围内的数据，且发现了旧数据，停止获取")
                        break

                    page += 1
                    time.sleep(0.5)

                except json.JSONDecodeError as e:
                    self.logger.error(f"解析指定workflow的 runs JSON 失败: {str(e)}")
                    break

            self.logger.info(f"[workflow {workflow_id}] 总共获取到 {len(all_runs)} 个在时间范围内的失败 workflow runs")
            return all_runs

        except Exception as e:
            self.logger.error(f"获取指定workflow的 runs 时发生未知错误: {str(e)}")
            return []
    
    def get_workflow_jobs(self, run_id: int) -> List[Dict]:
        """获取特定 workflow run 的失败 jobs"""
        url = f"{API_BASE_URL}/repos/{self.repo_owner}/{self.repo_name}/actions/runs/{run_id}/jobs"
        
        try:
            response = self.make_request_with_retry(url)
            if not response or response.status_code != 200:
                self.logger.error(f"获取 run {run_id} 的 jobs 失败")
                return []
            
            try:
                all_jobs = response.json()['jobs']
                failed_jobs = [job for job in all_jobs if job.get('conclusion') == 'failure']
                self.logger.debug(f"Run {run_id}: 总 jobs {len(all_jobs)}，失败 jobs {len(failed_jobs)}")
                return failed_jobs
                
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.error(f"解析 jobs JSON 失败: {str(e)}")
                return []
                
        except Exception as e:
            self.logger.error(f"获取 run {run_id} 的 jobs 时发生未知错误: {str(e)}")
            return []

    def create_session(self):
        """创建一个带有重试机制的会话"""
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504, 429]  # 添加429（Too Many Requests）
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def get_job_logs(self, job_id: int) -> Optional[str]:
        """获取特定 job 的日志"""
        url = f"{API_BASE_URL}/repos/{self.repo_owner}/{self.repo_name}/actions/jobs/{job_id}/logs"
        
        try:
            response = self.make_request_with_retry(url)
            if not response or response.status_code != 200:
                self.logger.warning(f"获取 job {job_id} 的日志失败，状态码: {response.status_code if response else 'None'}")
                return None
            
            # 自动检测编码
            detected_encoding = detect_encoding(response.content)
            self.logger.info(f"Job {job_id} 日志检测到编码: {detected_encoding}")
            
            # 使用检测到的编码解码
            try:
                return response.content.decode(detected_encoding)
            except UnicodeDecodeError:
                # 如果检测的编码失败，尝试常见编码
                for encoding in ['utf-8', 'latin1', 'cp1252']:
                    try:
                        return response.content.decode(encoding)
                    except UnicodeDecodeError:
                        continue
                
                # 最后才使用replace，但记录警告
                self.logger.warning(f"Job {job_id} 日志包含无法解码的字符，使用replace模式")
                return response.content.decode('utf-8', errors='replace')
                
        except Exception as e:
            self.logger.error(f"获取 job {job_id} 的日志时发生未知错误: {str(e)}")
            return None

    def safe_write_to_file(self, failure_info: Dict) -> bool:
        """安全地写入失败信息到文件"""
        try:
            # 日志内容处理，使用自动编码检测
            if 'failure_logs' in failure_info and isinstance(failure_info['failure_logs'], str):
                logs = failure_info['failure_logs']
                # 将字符串转换为字节以检测编码
                logs_bytes = logs.encode('utf-8', errors='ignore')
                detected_encoding = detect_encoding(logs_bytes)
                
                try:
                    # 使用检测到的编码重新编码解码
                    failure_info['failure_logs'] = logs.encode(detected_encoding).decode(detected_encoding)
                except (UnicodeEncodeError, UnicodeDecodeError):
                    # 如果检测的编码失败，尝试常见编码
                    for encoding in ['utf-8', 'latin1', 'cp1252']:
                        try:
                            failure_info['failure_logs'] = logs.encode(encoding).decode(encoding)
                            break
                        except (UnicodeEncodeError, UnicodeDecodeError):
                            continue
                    else:
                        # 最后才使用replace，但记录警告
                        self.logger.warning(f"日志包含无法编码的字符，使用replace模式")
                        failure_info['failure_logs'] = logs.encode('utf-8', errors='replace').decode('utf-8')
            
            with open(self.failures_file, 'a', encoding='utf-8') as f:
                json.dump(failure_info, f, ensure_ascii=False)
                f.write('\n')
            return True
        except IOError as e:
            self.logger.error(f"写入文件失败: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"写入文件时发生未知错误: {str(e)}")
            return False

    def is_compilation_workflow(self, workflow_name: str) -> bool:
        """检查是否是编译相关的 workflow"""
        compilation_workflows = self.config.get('compilation_workflows', [])
        return workflow_name in compilation_workflows

    def collect_ci_failures(self, total_days=30):
        """收集 CI 失败记录"""
        self.logger.info(f"开始收集 {self.repo_key} 项目的 CI 失败记录，总天数：{total_days}")
        self.logger.info(f"注意：由于GitHub API限制，只能获取最近90天的数据")
        
        try:
            # 在开始时删除旧文件
            if os.path.exists(self.failures_file):
                os.remove(self.failures_file)
                self.logger.info(f"已删除旧文件: {self.failures_file}")
        except OSError as e:
            self.logger.error(f"删除旧文件失败: {str(e)}")
            return
        
        self.logger.info("收集编译相关的 workflow 失败记录")

        # 显示可用的 workflows
        self.get_workflows()
            
        now = datetime.now(timezone.utc)
        end_time = now
        start_time = now - timedelta(days=total_days)
        
        self.logger.info(f"采集时间段：{start_time.date()} ~ {end_time.date()}")
        
        # 收集所有编译相关的workflow runs
        self.logger.info("收集编译相关的workflow失败记录...")

        compilation_workflows = self.config.get('compilation_workflows', [])
        if isinstance(compilation_workflows, list) and len(compilation_workflows) == 1:
            only_wf_name = compilation_workflows[0]
            self.logger.info(f"检测到仅一个编译相关workflow: '{only_wf_name}'，将按该 workflow 限定获取 runs")
            wf_id = self.get_workflow_id_by_name(only_wf_name)
            if wf_id is not None:
                all_workflow_runs = self.get_workflow_runs_range_for_workflow(wf_id, start_time, end_time)
            else:
                self.logger.warning("未能解析到该 workflow 的ID，回退到仓库级 runs 拉取并按名称过滤")
                all_workflow_runs = self.get_workflow_runs_range(start_time, end_time)
        else:
            all_workflow_runs = self.get_workflow_runs_range(start_time, end_time)
        self.logger.info(f"总共找到 {len(all_workflow_runs)} 个失败的 workflow runs")
        
        total_failures_count = 0
        processed_job_ids = set()
        
        for run in all_workflow_runs:
            try:
                self.logger.info(f"处理 workflow run: {run['id']} - {run['name']}")
                jobs = self.get_workflow_jobs(run['id'])
                
                for job in jobs:
                    if job['id'] in processed_job_ids:
                        self.logger.info(f"跳过已处理的 job: {job['id']}")
                        continue
                    try:
                        self.logger.info(f"处理失败的 job: {job['id']}")
                        logs = self.get_job_logs(job['id'])
                        
                        if logs:
                            failure_info = {
                                'repo_name': self.repo_key,
                                'workflow_id': run['id'],
                                'workflow_name': run['name'],
                                'job_id': job['id'],
                                'job_name': job['name'],
                                'created_at': run['created_at'],
                                'failure_logs': logs,
                                'commit_sha': run['head_sha'],
                                'branch': run['head_branch']
                            }
                            
                            if self.safe_write_to_file(failure_info):
                                processed_job_ids.add(job['id'])
                                total_failures_count += 1
                                self.logger.info(f"已保存第 {total_failures_count} 条失败记录")
                            else:
                                self.logger.error(f"保存失败记录失败: job {job['id']}")
                        else:
                            self.logger.warning(f"无法获取 job {job['id']} 的日志")
                            
                    except Exception as e:
                        self.logger.error(f"处理 job {job['id']} 时发生错误: {str(e)}")
                        continue
                        
                    # 添加延迟避免API限制
                    time.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"处理 workflow run {run['id']} 时发生错误: {str(e)}")
                continue
                    
        self.logger.info(f"收集完成！")
        self.logger.info(f"  总失败记录: {total_failures_count}")
        self.logger.info(f"  文件保存位置: {self.failures_file}")

    def read_failures_from_file(self) -> List[Dict]:
        """从 JSON Lines 格式的文件中读取失败记录"""
        failures = []
        
        if not os.path.exists(self.failures_file):
            self.logger.warning(f"文件 {self.failures_file} 不存在")
            return failures
        
        try:
            # 首先读取文件内容以检测编码
            with open(self.failures_file, 'rb') as f:
                file_content = f.read()
            
            # 自动检测文件编码
            detected_encoding = detect_encoding(file_content)
            self.logger.info(f"文件 {self.failures_file} 检测到编码: {detected_encoding}")
            
            # 使用检测到的编码读取文件
            try:
                with open(self.failures_file, 'r', encoding=detected_encoding) as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:  # 跳过空行
                            try:
                                failure_info = json.loads(line)
                                failures.append(failure_info)
                            except json.JSONDecodeError as e:
                                self.logger.error(f"解析第 {line_num} 行时出错: {e}")
                                continue
            except UnicodeDecodeError:
                # 如果检测的编码失败，尝试常见编码
                self.logger.warning(f"文件 {self.failures_file} 使用检测编码失败，尝试其他编码")
                for encoding in ['utf-8', 'latin1', 'cp1252']:
                    try:
                        with open(self.failures_file, 'r', encoding=encoding) as f:
                            for line_num, line in enumerate(f, 1):
                                line = line.strip()
                                if line:  # 跳过空行
                                    try:
                                        failure_info = json.loads(line)
                                        failures.append(failure_info)
                                    except json.JSONDecodeError as e:
                                        self.logger.error(f"解析第 {line_num} 行时出错: {e}")
                                        continue
                        break  # 如果成功读取，跳出循环
                    except UnicodeDecodeError:
                        continue
                else:
                    # 最后才使用replace
                    self.logger.warning(f"文件 {self.failures_file} 包含无法解码的字符，使用replace模式")
                    with open(self.failures_file, 'r', encoding='utf-8', errors='replace') as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if line:  # 跳过空行
                                try:
                                    failure_info = json.loads(line)
                                    failures.append(failure_info)
                                except json.JSONDecodeError as e:
                                    self.logger.error(f"解析第 {line_num} 行时出错: {e}")
                                    continue
                            
            self.logger.info(f"成功读取 {len(failures)} 条失败记录")
            
        except IOError as e:
            self.logger.error(f"读取文件时出错: {e}")
        except Exception as e:
            self.logger.error(f"读取文件时发生未知错误: {e}")
        
        return failures

    def analyze_failure_patterns(self, failures: List[Dict]) -> Dict:
        """分析失败模式"""
        self.logger.info("分析失败模式...")
        
        try:
            analysis = {
                'repo_name': self.repo_key,
                'total_failures': len(failures),
                'workflow_stats': {},
                'job_stats': {},
                'branch_stats': {},
                'time_stats': {},
                'common_errors': []
            }
            
            # 按workflow统计
            for failure in failures:
                workflow_name = failure.get('workflow_name', 'Unknown')
                if workflow_name not in analysis['workflow_stats']:
                    analysis['workflow_stats'][workflow_name] = 0
                analysis['workflow_stats'][workflow_name] += 1
            
            # 按job统计
            for failure in failures:
                job_name = failure.get('job_name', 'Unknown')
                if job_name not in analysis['job_stats']:
                    analysis['job_stats'][job_name] = 0
                analysis['job_stats'][job_name] += 1
            
            # 按分支统计
            for failure in failures:
                branch = failure.get('branch', 'Unknown')
                if branch not in analysis['branch_stats']:
                    analysis['branch_stats'][branch] = 0
                analysis['branch_stats'][branch] += 1
            
            # 按时间统计
            for failure in failures:
                created_at = failure.get('created_at', '')
                if created_at:
                    date = created_at.split('T')[0]  # 提取日期部分
                    if date not in analysis['time_stats']:
                        analysis['time_stats'][date] = 0
                    analysis['time_stats'][date] += 1
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"分析失败模式时发生错误: {str(e)}")
            return {
                'repo_name': self.repo_key,
                'total_failures': 0,
                'workflow_stats': {},
                'job_stats': {},
                'branch_stats': {},
                'time_stats': {},
                'common_errors': []
            }

    def print_analysis_summary(self, analysis: Dict):
        """打印分析摘要"""
        try:
            self.logger.info("\n" + "="*60)
            self.logger.info(f"{self.repo_key.upper()} 项目 CI 失败分析摘要")
            self.logger.info("="*60)
            
            self.logger.info(f"\n总失败数: {analysis['total_failures']}")
            
            self.logger.info(f"\n按 Workflow 分类 (前5个):")
            sorted_workflows = sorted(analysis['workflow_stats'].items(), key=lambda x: x[1], reverse=True)
            for workflow, count in sorted_workflows[:5]:
                self.logger.info(f"  {workflow}: {count} 次失败")
            
            self.logger.info(f"\n按 Job 分类 (前5个):")
            sorted_jobs = sorted(analysis['job_stats'].items(), key=lambda x: x[1], reverse=True)
            for job, count in sorted_jobs[:5]:
                self.logger.info(f"  {job}: {count} 次失败")
            
            self.logger.info(f"\n按分支分类:")
            sorted_branches = sorted(analysis['branch_stats'].items(), key=lambda x: x[1], reverse=True)
            for branch, count in sorted_branches:
                self.logger.info(f"  {branch}: {count} 次失败")
                
            self.logger.info(f"\n按日期分类 (最近5天):")
            sorted_dates = sorted(analysis['time_stats'].items(), key=lambda x: x[0], reverse=True)
            for date, count in sorted_dates[:5]:
                self.logger.info(f"  {date}: {count} 次失败")
                
        except Exception as e:
            self.logger.error(f"打印分析摘要时发生错误: {str(e)}")

    def save_analysis(self, analysis: Dict):
        """保存分析结果"""
        analysis_file = f'{self.repo_key}_ci_analysis.json'
        try:
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)
            self.logger.info(f"分析结果已保存到: {analysis_file}")
        except IOError as e:
            self.logger.error(f"保存分析结果失败: {str(e)}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='统一的CI失败记录收集工具')
    parser.add_argument('repo', choices=list(REPO_CONFIGS.keys()), 
                        help='要收集的仓库名称')
    parser.add_argument('--days', type=int, default=90,
                        help='收集天数 (默认: 90)')
    parser.add_argument('--analyze-only', action='store_true',
                        help='仅分析已有数据，不收集新数据')
    
    args = parser.parse_args()
    
    try:
        if not GITHUB_TOKEN:
            print("请设置 GITHUB_TOKEN 环境变量")
            exit(1)
        
        # 创建收集器
        collector = CIFailureCollector(args.repo)
        
        if not args.analyze_only:
            # 收集失败记录
            collector.logger.info(f"开始收集 {args.repo} 项目的 CI 失败记录...")
            collector.collect_ci_failures(
                total_days=args.days
            )
        
        # 分析失败记录
        collector.logger.info(f"分析 {args.repo} 项目的失败记录...")
        failures = collector.read_failures_from_file()
        
        if failures:
            collector.logger.info(f"总共读取到 {len(failures)} 条失败记录")
            
            # 进行失败模式分析
            analysis = collector.analyze_failure_patterns(failures)
            collector.print_analysis_summary(analysis)
            collector.save_analysis(analysis)
        else:
            collector.logger.warning("没有找到失败记录")
            
    except KeyboardInterrupt:
        print("用户中断程序执行")
    except Exception as e:
        print(f"程序执行时发生未知错误: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 