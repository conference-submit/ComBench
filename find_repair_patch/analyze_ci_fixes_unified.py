#!/usr/bin/env python3
"""
Unified CI failure fix analyzer - supports compilation error fix analysis for multiple projects

This script supports analysis of different projects through configuration parameters:
- Git project
- LLVM project

Usage:
python analyze_ci_fixes_unified.py --project git
python analyze_ci_fixes_unified.py --project llvm
python analyze_ci_fixes_unified.py --project git --restart  # Restart analysis
python analyze_ci_fixes_unified.py --project git --test-mode  # Test mode
python analyze_ci_fixes_unified.py --project git --limit 50  # Only process first 50 records

Features:
- Default continue analysis: continue from last interrupted position without deleting existing results
- Support restart: use --restart parameter to delete existing results and restart
- Automatic backup: automatically backup existing result files before restart
- Progress display: real-time display of analysis progress and estimated remaining time
- File validation: verify integrity and format correctness of output files
- Resume from breakpoint: support continuing analysis after interruption, skip already processed records
- Extract file paths only: only extract compilation-related file paths, skip LLM analysis
"""

import json
import os
import re
import logging
import subprocess
import argparse
import shutil
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Set, Tuple
import requests
from dataclasses import dataclass
import multiprocessing
from functools import partial, lru_cache
import time
import random

# Create logger
logger = logging.getLogger(__name__)

# Global variables
global_repo_name = None

def set_global_repo_name(repo_name: str) -> None:
    """Set global repository name"""
    global global_repo_name
    global_repo_name = repo_name

def get_global_repo_name() -> str:
    """Get global repository name"""
    return global_repo_name

def setup_logging(project_name: str) -> None:
    """Set logging configuration
    
    Args:
        project_name: Project name, used to generate log file name
    """
    # Create log directory
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'log')
    os.makedirs(log_dir, exist_ok=True)
    
    # Use fixed log file name
    log_file = os.path.join(log_dir, f'{project_name}_repair_analysis.log')
    
    # If log file already exists, delete it first
    if os.path.exists(log_file):
        try:
            os.remove(log_file)
        except Exception as e:
            print(f"Warning: Unable to delete old log file: {e}")
    
    # Configure log format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            # File handler
            logging.FileHandler(log_file, encoding='utf-8'),
            # Console handler
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Log file: {log_file}")

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

def get_headers():
    """Create request headers"""
    return {'Authorization': f'token {GITHUB_TOKEN}', 'Accept': 'application/vnd.github.v3+json'}

def load_project_configs() -> Dict:
    """Load project configuration from config file"""
    config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'project_configs.json')
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            configs = json.load(f)
        logger.info(f"Successfully loaded project configuration file: {config_file}")
        return configs
    except Exception as e:
        logger.error(f"Failed to load configuration file: {e}")
        raise

class ProjectConfig:
    """Project configuration class"""
    def __init__(self, project_name: str):
        configs = load_project_configs()
        if project_name not in configs:
            raise ValueError(f"Unsupported project: {project_name}. Supported projects: {list(configs.keys())}")
        
        config = configs[project_name]
        self.project_name = project_name
        self.repo_owner = config['repo_owner']
        self.repo_name = config['repo_name']
        self.main_branch = config.get('main_branch', 'main')  # Default to 'main'
        self.compilation_workflows = config.get('compilation_workflows', [])
        self.include_paths = config.get('include_paths', ['-I.', '-Iinclude'])  # Read include paths from config file
        
        # File path processing - automatically generated based on project name
        current_dir = os.path.dirname(__file__)
        # Compilation error file path: ../compilation_error/{project_name}_compiler_errors_extracted.json
        self.failures_file = os.path.join(current_dir, '..', 'compilation_error', f'{project_name}_compiler_errors_extracted.json')
        # Fix analysis result file path: {project_name}_repair_analysis.jsonl
        self.output_file = os.path.join(current_dir, 'output', f'{project_name}_repair_analysis.jsonl')
        # Repository directory: {project_name}_repo
        self.repo_dir = os.path.join(current_dir, f'{project_name}_repo')
        
        self.api_base_url = "https://api.github.com"

@dataclass
class FailureRecord:
    """CI failure record"""
    commit_sha: str
    branch: str
    error_lines: List[str]
    error_types: Dict[str, float]
    error_count: int
    created_at: str
    workflow_name: str
    job_name: List[str]  # Changed to array type
    workflow_id: int
    job_id: List[str]  # Changed to array type
    error_details: Optional[List] = None

@dataclass
class RepairPair:
    """Error-fix pair"""
    failure_commit: str
    repair_commit: str
    error_lines: List[str]
    error_types: Dict[str, float]
    error_count: int
    workflow_name: str
    job_name: List[str]  # Changed to array type
    workflow_id: int
    job_id: List[str]  # Changed to array type
    diffs: List[str]  # 每个错误行对应的修复diff
    repair_source: str = "unknown"  # 可能的值: "same_branch", "dependency", "merge"
    error_details: Optional[List] = None
    compilation_related_paths: Optional[Dict[str, List[str]]] = None  # 编译相关文件路径
    compilation_related_paths_details: Optional[List[Dict[str, List[str]]]] = None  # 每个错误行对应的编译相关路径详情

def filter_commits_by_time(commits: List[Dict], failure_time: datetime) -> List[Dict]:
    """
    过滤提交列表，只保留时间晚于失败时间的提交
    
    Args:
        commits: 提交列表
        failure_time: 失败时间
        
    Returns:
        List[Dict]: 过滤后的提交列表
    """
    filtered_commits = []
    for commit in commits:
        try:
            commit_time_str = commit.get('commit', {}).get('author', {}).get('date') or \
                            commit.get('commit', {}).get('committer', {}).get('date')
            
            if not commit_time_str:
                logger.debug(f"跳过无时间信息的commit: {commit.get('sha', 'unknown')}")
                continue
                
            commit_time = datetime.fromisoformat(commit_time_str.replace('Z', '+00:00'))
            
            if commit_time > failure_time:
                filtered_commits.append(commit)
            else:
                logger.debug(f"过滤掉早于失败时间的commit: {commit.get('sha', 'unknown')} ({commit_time} <= {failure_time})")
                
        except Exception as e:
            logger.debug(f"解析commit时间失败: {commit.get('sha', 'unknown')} - {e}")
            continue
    
    logger.debug(f"时间过滤: {len(commits)} -> {len(filtered_commits)} 个commit")
    return filtered_commits

def is_compilation_workflow(workflow_name: str, config: ProjectConfig) -> bool:
    """检查是否是编译相关的 workflow"""
    return any(name in workflow_name for name in config.compilation_workflows)



def sanitize_commit_sha(commit_sha: Optional[str]) -> str:
    """规范化/清洗 commit sha，去除首尾与内部异常空白字符。

    说明：
    - 有日志显示候选 SHA 前带有换行符，导致 `git` 子进程参数为 "\n<sha>" 而找不到对象。
    - 这里使用严格清洗：先 strip 去除首尾空白，再移除所有内部空白字符（\r\n\t 等）。
    - 仅保留十六进制字符，防止混入不可见字符。
    """
    try:
        raw = (commit_sha or "")
        # 先做常规 strip，再去掉所有空白字符
        raw = raw.strip()
        raw = re.sub(r"\s+", "", raw)
        # 仅保留 0-9a-fA-F（git SHA）
        cleaned = re.sub(r"[^0-9a-fA-F]", "", raw)
        return cleaned
    except Exception:
        return (commit_sha or "").strip()

def check_commit_exists_local(repo_dir: str, commit_sha: str) -> bool:
    """检查commit是否在本地存在"""
    try:
        commit_sha = sanitize_commit_sha(commit_sha)
        if not commit_sha:
            return False
        result = subprocess.run(
            ['git', 'cat-file', '-e', commit_sha],
            cwd=repo_dir,
            capture_output=True
        )
        return result.returncode == 0
    except Exception:
        return False

def fetch_remote_commit(config: ProjectConfig, repo_dir: str, commit_sha: str) -> bool:
    """从远端尝试按 SHA 获取指定 commit 对象。

    策略：
    1) 依次尝试所有远端（origin 以及 fork_* 等）
    2) 优先使用浅深度抓取（--depth 1），失败再不带 depth
    成功任意一次即返回 True。
    """
    commit_sha = sanitize_commit_sha(commit_sha)
    if not commit_sha:
        return False
    try:
        # 获取所有远端名称
        try:
            remotes_out = subprocess.run(
                ['git', 'remote'], cwd=repo_dir, capture_output=True, text=True, check=True
            ).stdout.split()
            base_list = remotes_out or []
        except subprocess.CalledProcessError:
            base_list = []
        # 严格保证 origin 优先（不存在则只尝试其它远端；若无远端则回退仅 origin 名称）
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
                    logger.info(f"尝试从远端 {remote} 抓取对象 {commit_sha}")
                    r = subprocess.run(args, cwd=repo_dir, capture_output=True, text=True)
                    if r.returncode == 0:
                        return True
                    # 记录部分错误信息便于排查
                    err = (r.stderr or '').strip()
                    if err:
                        logger.debug(f"fetch 失败({remote}): {err[:200]}")
                except Exception as e:
                    logger.debug(f"fetch 异常({remote}): {e}")
                    continue
    except Exception as e:
        logger.error(f"获取远程commit失败: {e}")
        return False
    return False

 

def check_commit_dependency(config: ProjectConfig, repo_dir: str, failure_commit: str, candidate_commit: str) -> bool:
    """
    检查候选commit是否是失败commit的后续提交（在线性历史中是否是后继）
    
    Args:
        config: 项目配置
        repo_dir: 仓库目录
        failure_commit: 失败的commit SHA
        candidate_commit: 候选修复commit SHA
        
    Returns:
        True if failure_commit是candidate_commit的祖先提交
    """
    try:
        # 首先检查两个commit是否存在
        for commit in [(failure_commit or '').strip(), (candidate_commit or '').strip()]:
            result = subprocess.run(
                ['git', 'cat-file', '-e', commit],
                cwd=repo_dir,
                capture_output=True
            )
            if result.returncode != 0:
                logger.debug(f"提交 {commit} 不存在")
                return False
        
        # 使用git merge-base --is-ancestor检查线性历史中的祖先关系
        result = subprocess.run(
            ['git', 'merge-base', '--is-ancestor', (failure_commit or '').strip(), (candidate_commit or '').strip()],
            cwd=repo_dir,
            capture_output=True,
            text=True
        )
        # 返回值为0表示是祖先关系
        if result.returncode == 0:
            logger.debug(f"✓ {failure_commit} 是 {candidate_commit} 的祖先提交")
            return True
        else:
            logger.debug(f"✗ {failure_commit} 不是 {candidate_commit} 的祖先提交")
            return False
            
    except Exception as e:
        logger.error(f"检查提交祖先关系时出错: {e}")
        return False

def find_dependent_commits_all_branches(config: ProjectConfig, repo_dir: str, failure_commit: str, failure_date: str, author_email: str) -> List[Dict]:
    """
    查找失败提交的后续提交：
    1. 首先找到所有包含该提交的分支
    2. 在这些分支上查找一定时间范围内的后续提交（不限制作者）
    
    Args:
        config: 项目配置
        repo_dir: 仓库目录
        failure_commit: 失败的commit SHA
        failure_date: 失败时间
        author_email: 作者邮箱
        
    Returns:
        后继提交列表，按时间排序
    """
    # 不再限制作者；即使没有作者邮箱也继续查找
    
    try:
        failure_time = datetime.fromisoformat(failure_date.replace('Z', '+00:00'))
        end_time = failure_time + timedelta(days=30)
        
        # 直接用 git rev-list --all 查找所有分支的后续提交
        try:
            descendant_commits = []
            rev_list_cmd = [
                'git', 'rev-list', '--all',
                    '--ancestry-path', f'{failure_commit}..HEAD',
                    f'--since={failure_time.isoformat()}',
                    f'--until={end_time.isoformat()}'
                ]
            result = subprocess.run(
                rev_list_cmd,
                cwd=repo_dir,
                capture_output=True,
                text=True,
                check=True
            )
            descendant_shas = result.stdout.strip().split('\n')
            if descendant_shas and descendant_shas[0]:
                for sha in descendant_shas:
                    commit_data = subprocess.run(
                        ['git', 'show', '-s', '--format={"sha": "%H", "commit": {"message": "%s", "author": {"date": "%cI", "email": "%ae"}}}', sha],
                        cwd=repo_dir,
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    try:
                        commit_json = json.loads(commit_data.stdout)
                        if commit_json not in descendant_commits:  # 避免重复添加
                            descendant_commits.append(commit_json)
                            logger.debug(f"添加后继提交: {sha}")
                    except json.JSONDecodeError:
                        logger.debug(f"解析提交信息失败: {sha}")
                        continue
            
            return descendant_commits
            
        except subprocess.CalledProcessError as e:
            logger.error(f"执行git命令失败: {e}")
            return []
            
    except Exception as e:
        logger.error(f"查找后继提交失败: {e}")
        return []

def get_commits_after_failure(config: ProjectConfig, failure_date: str, branch: str = None, author_email: str = None, base_sha: str = None, days_window: int = 7) -> List[Dict]:
    """获取失败后的提交，在同分支查找时间更晚的commit。

    如果提供 base_sha，则在本地使用祖先关系过滤，仅保留 base_sha 的后继（ancestry-path）提交，避免混入默认分支提交。
    """
    failure_time = datetime.fromisoformat(failure_date.replace('Z', '+00:00'))
    end_time_window = failure_time + timedelta(days=days_window)
    
    # 获取提交列表
    url = f"{config.api_base_url}/repos/{config.repo_owner}/{config.repo_name}/commits"
    
    # 构建查询参数
    params = {
        'since': failure_time.isoformat(),
        'until': end_time_window.isoformat(),
        'per_page': 100
    }
    
    # 若传入的分支为主分支（考虑常见前缀形式），则跳过并返回空列表
    if branch:
        try:
            raw = branch.strip()
            candidates: List[str] = []
            if raw:
                candidates.append(raw)
            prefixes = [
                'refs/heads/',
                'heads/',
                'refs/remotes/origin/',
                'remotes/origin/',
                'origin/',
                'refs/',
            ]
            for p in prefixes:
                if raw.startswith(p):
                    candidates.append(raw[len(p):])
            # 去重保持顺序
            seen_local: Set[str] = set()
            ordered_candidates: List[str] = []
            for c in candidates:
                if c and c not in seen_local:
                    seen_local.add(c)
                    ordered_candidates.append(c)
            main_names: Set[str] = {config.main_branch, 'main', 'master'}
            if any(c in main_names for c in ordered_candidates):
                logger.info("分支为主分支，跳过同分支后续提交查找")
                return []
        except Exception:
            # 解析失败不影响后续逻辑
            pass
    
    try:
        commits = []
        if branch:
            # 对于该端点，应使用 sha 参数而非 ref；ref 会被忽略从而回落到默认分支
            def branch_candidates(raw: str):
                b = raw.strip()
                candidates = []
                # 原始
                if b:
                    candidates.append(b)
                # 常见前缀剥离
                prefixes = [
                    'refs/heads/',
                    'heads/',
                    'refs/remotes/origin/',
                    'remotes/origin/',
                    'origin/',
                    'refs/',
                ]
                for p in prefixes:
                    if b.startswith(p):
                        candidates.append(b[len(p):])
                # 去重保持顺序
                seen = set()
                ordered = []
                for c in candidates:
                    if c and c not in seen:
                        seen.add(c)
                        ordered.append(c)
                return ordered

            # 先尝试使用本地 git 分支/引用，若存在则直接用 git log 获取提交（避免 fork 分支导致的 404）
            def resolve_local_ref(cand: str) -> str:
                # 优先处理 PR 引用（pull/<num>/head 或 pr/<num>/head 映射到 refs/remotes/origin/pr/<num>/head）
                pr_match = None
                try:
                    pr_match = re.match(r'^(?:pull|pr)/(\d+)/(?:head|merge)$', cand)
                except Exception:
                    pr_match = None
                potential_refs = []
                if pr_match:
                    pr_num = pr_match.group(1)
                    potential_refs.append(f"refs/remotes/origin/pr/{pr_num}/head")
                potential_refs += [
                    f"refs/heads/{cand}",
                    f"refs/remotes/origin/{cand}",
                    f"refs/{cand}",
                    cand,
                ]
                for ref in potential_refs:
                    try:
                        result = subprocess.run(
                            ['git', 'rev-parse', '--verify', ref],
                            cwd=config.repo_dir,
                            capture_output=True,
                            text=True
                        )
                        if result.returncode == 0:
                            return ref
                    except Exception:
                        continue
                return ""

            def branch_exists_remote(owner: str, repo: str, cand: str) -> bool:
                try:
                    branch_url = f"{config.api_base_url}/repos/{owner}/{repo}/branches/{cand}"
                    resp = requests.get(branch_url, headers=get_headers())
                    if resp.status_code == 200:
                        return True
                    if resp.status_code == 404:
                        logger.info(f"远端不存在分支 {cand}（可能来自 fork），跳过远端查询")
                        return False
                    logger.warning(f"查询远端分支 {cand} 返回 {resp.status_code}")
                    return False
                except Exception as _:
                    return False

            # 预计算候选分支列表
            candidates_list = branch_candidates(branch)

            # 若分支以 users/ 开头，直接走 fork 远端逻辑，跳过本地解析
            skip_local = any(re.match(r'^users/[^/]+/.+$', c.strip() if isinstance(c, str) else '') for c in candidates_list)
            if skip_local:
                logger.info("分支以 users/ 开头：直接使用 fork 远端分支逻辑，跳过同分支本地解析引用")
            
            local_success = False
            if not skip_local:
                for cand in candidates_list:
                    resolved = resolve_local_ref(cand)
                    if not resolved:
                        continue
                    try:
                        # 构造 ancestry-path 命令，若提供 base_sha 则严格过滤其后继
                        if base_sha:
                            ancestry_cmd = [
                                'git', 'rev-list', '--ancestry-path', f'{base_sha}..{resolved}',
                                f'--since={failure_time.isoformat()}',
                                f'--until={end_time_window.isoformat()}'
                            ]
                            result_hashes = subprocess.run(
                                ancestry_cmd,
                                cwd=config.repo_dir,
                                capture_output=True,
                                text=True,
                                check=True
                            )
                            log_hashes = [h.strip() for h in result_hashes.stdout.splitlines() if h.strip()]
                            logger.info(f"同分支本地解析引用: {resolved}，祖先路径内候选: {len(log_hashes)}")
                        else:
                            log_cmd = [
                                'git', 'log', resolved,
                                '--since', failure_time.isoformat(),
                                '--until', end_time_window.isoformat(),
                                '--format=%H'
                            ]
                            result_hashes = subprocess.run(
                                log_cmd,
                                cwd=config.repo_dir,
                                capture_output=True,
                                text=True,
                                check=True
                            )
                            log_hashes = [h.strip() for h in result_hashes.stdout.splitlines() if h.strip()]
                        
                        if not log_hashes:
                            # 即使 ref 存在，但窗口内无提交，视为成功（返回空即可）
                            local_success = True
                            break
                        
                        # 仅收集 SHA，必要字段在返回前统一获取，避免逐提交多次 git 调用
                        for h in log_hashes:
                            if not h:
                                continue
                            commit_obj = {
                                'sha': h,
                                'source_branch_ref': resolved,
                            }
                            commits.append(commit_obj)
                        local_success = True
                        break
                    except subprocess.CalledProcessError:
                        continue

            if not local_success:
                # 本地没有该分支引用，改用 Git 远端回退：处理 PR 引用或 fork 用户仓库分支，无法解析时再浅拉取 origin 分支
                success = False
                for candidate in candidates_list:
                    try:
                        # 尝试通过失败提交关联的PR解析 fork 分支信息
                        def resolve_fork_branch_via_commit(commit_sha: str) -> Tuple[str, str]:
                            try:
                                api = f"{config.api_base_url}/repos/{config.repo_owner}/{config.repo_name}/commits/{commit_sha}/pulls"
                                # 该接口返回与commit相关的PR列表
                                resp = requests.get(api, headers=get_headers(), timeout=15)
                                if resp.status_code != 200:
                                    return "", ""
                                items = resp.json() or []
                                if not isinstance(items, list) or not items:
                                    return "", ""
                                head = items[0].get('head', {})
                                user_login = (head.get('user') or {}).get('login') or ""
                                ref_name = head.get('ref') or ""
                                return user_login.strip(), ref_name.strip()
                            except Exception:
                                return "", ""

                        # 1) 处理 PR 引用：pull/<num>/head 或 pr/<num>/head
                        pr_match = None
                        try:
                            pr_match = re.match(r'^(?:pull|pr)/(\d+)/(?:head|merge)$', candidate)
                        except Exception:
                            pr_match = None
                        if pr_match:
                            pr_num = pr_match.group(1)
                            # 确保有 PR 引用（幂等）
                            try:
                                subprocess.run(
                                    ['git', 'fetch', 'origin', '+refs/pull/*:refs/remotes/origin/pr/*'],
                                    cwd=config.repo_dir,
                                    capture_output=True,
                                    text=True,
                                    check=True
                                )
                            except subprocess.CalledProcessError:
                                pass
                            resolved_remote = f'refs/remotes/origin/pr/{pr_num}/head'
                            try:
                                if base_sha:
                                    ancestry_cmd = [
                                        'git', 'rev-list', '--ancestry-path', f'{base_sha}..{resolved_remote}',
                                        f'--since={failure_time.isoformat()}',
                                        f'--until={end_time_window.isoformat()}'
                                    ]
                                    result_hashes = subprocess.run(
                                        ancestry_cmd,
                                        cwd=config.repo_dir,
                                        capture_output=True,
                                        text=True,
                                        check=True
                                    )
                                    log_hashes = [h.strip() for h in result_hashes.stdout.splitlines() if h.strip()]
                                else:
                                    log_cmd = [
                                        'git', 'log', resolved_remote,
                                        '--since', failure_time.isoformat(),
                                        '--until', end_time_window.isoformat(),
                                        '--format=%H'
                                    ]
                                    result_hashes = subprocess.run(
                                        log_cmd,
                                        cwd=config.repo_dir,
                                        capture_output=True,
                                        text=True,
                                        check=True
                                    )
                                    log_hashes = [h.strip() for h in result_hashes.stdout.splitlines() if h.strip()]
                                for h in log_hashes:
                                    if not h:
                                        continue
                                    commits.append({'sha': h, 'source_branch_ref': resolved_remote})
                                success = True
                                break
                            except subprocess.CalledProcessError:
                                continue

                        # 2) 处理 users/<user>/<branch>：添加 fork 远端并拉取该分支
                        user_match = None
                        try:
                            user_match = re.match(r'^users/([^/]+)/(.+)$', candidate)
                        except Exception:
                            user_match = None
                        if user_match:
                            user_name = user_match.group(1)
                            user_branch = user_match.group(2)
                            # 在部分fork仓库中，分支名本身包含 users/<user>/ 前缀
                            full_users_branch = f'users/{user_name}/{user_branch}'
                            remote_name = f'fork_{user_name}'
                            # 检查/添加远端
                            try:
                                existing_remotes = subprocess.run(
                                    ['git', 'remote'],
                                    cwd=config.repo_dir,
                                    capture_output=True,
                                    text=True,
                                    check=True
                                ).stdout.split()
                            except subprocess.CalledProcessError:
                                existing_remotes = []
                            if remote_name not in existing_remotes:
                                try:
                                    fork_url = f'https://github.com/{user_name}/{config.repo_name}.git'
                                    subprocess.run(
                                        ['git', 'remote', 'add', remote_name, fork_url],
                                        cwd=config.repo_dir,
                                        capture_output=True,
                                        text=True,
                                        check=True
                                    )
                                except subprocess.CalledProcessError:
                                    # 添加失败则跳过该候选
                                    continue
                            # 检查远端分支是否存在（优先完整 users/<user>/<branch>）
                            try:
                                ls_full = subprocess.run(
                                    ['git', 'ls-remote', '--heads', remote_name, full_users_branch],
                                    cwd=config.repo_dir,
                                    capture_output=True,
                                    text=True,
                                    check=True
                                )
                                if ls_full.stdout.strip():
                                    fetch_target = full_users_branch
                                else:
                                    # 回退检查裸分支名
                                    ls_plain = subprocess.run(
                                        ['git', 'ls-remote', '--heads', remote_name, user_branch],
                                        cwd=config.repo_dir,
                                        capture_output=True,
                                        text=True,
                                        check=True
                                    )
                                    if ls_plain.stdout.strip():
                                        fetch_target = user_branch
                                    else:
                                        # 最后尝试原样 candidate
                                        ls_cand = subprocess.run(
                                            ['git', 'ls-remote', '--heads', remote_name, candidate],
                                            cwd=config.repo_dir,
                                            capture_output=True,
                                            text=True,
                                            check=True
                                        )
                                        if not ls_cand.stdout.strip():
                                            continue
                                        fetch_target = candidate
                            except subprocess.CalledProcessError:
                                continue

                            # 浅拉取该分支
                            try:
                                subprocess.run(
                                    ['git', 'fetch', '--depth', '200', '--shallow-since', failure_time.isoformat(), remote_name, fetch_target],
                                    cwd=config.repo_dir,
                                    capture_output=True,
                                    text=True,
                                    check=True
                                )
                            except subprocess.CalledProcessError:
                                try:
                                    subprocess.run(
                                        ['git', 'fetch', '--depth', '200', remote_name, fetch_target],
                                        cwd=config.repo_dir,
                                        capture_output=True,
                                        text=True,
                                        check=True
                                    )
                                except subprocess.CalledProcessError:
                                    continue

                            # 远端引用路径：保持与fetch_target一致（可能包含 users/<user>/ 前缀）
                            resolved_remote = f'refs/remotes/{remote_name}/{fetch_target}'
                            try:
                                if base_sha:
                                    ancestry_cmd = [
                                        'git', 'rev-list', '--ancestry-path', f'{base_sha}..{resolved_remote}',
                                        f'--since={failure_time.isoformat()}',
                                        f'--until={(failure_time + timedelta(days=7)).isoformat()}'
                                    ]
                                    result_hashes = subprocess.run(
                                        ancestry_cmd,
                                        cwd=config.repo_dir,
                                        capture_output=True,
                                        text=True,
                                        check=True
                                    )
                                    log_hashes = [h.strip() for h in result_hashes.stdout.splitlines() if h.strip()]
                                else:
                                    log_cmd = [
                                        'git', 'log', resolved_remote,
                                        '--since', failure_time.isoformat(),
                                        '--until', end_time_window.isoformat(),
                                        '--format=%H'
                                    ]
                                    result_hashes = subprocess.run(
                                        log_cmd,
                                        cwd=config.repo_dir,
                                        capture_output=True,
                                        text=True,
                                        check=True
                                    )
                                    log_hashes = [h.strip() for h in result_hashes.stdout.splitlines() if h.strip()]
                                # 确保对象存在于本地（fork 场景使用 fork 远端）
                                try:
                                    _ = [
                                        subprocess.run(
                                            ['git', 'fetch', '--depth', '1', remote_name, h],
                                            cwd=config.repo_dir,
                                            capture_output=True,
                                            text=True,
                                            check=False
                                        ) for h in log_hashes if h
                                    ]
                                except Exception:
                                    pass
                                for h in log_hashes:
                                    if not h:
                                        continue
                                    commits.append({'sha': h, 'source_branch_ref': resolved_remote})
                                success = True
                                break
                            except subprocess.CalledProcessError:
                                continue

                        # 3) 常规 origin/<candidate> 分支
                        # 检查远端是否存在该分支
                        ls_remote = subprocess.run(
                            ['git', 'ls-remote', '--heads', 'origin', candidate],
                            cwd=config.repo_dir,
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        if not ls_remote.stdout.strip():
                            logger.info(f"远端不存在分支 {candidate}")
                            # 4) 尝试通过失败commit关联的PR自动定位 fork/<user>/<ref>
                            if base_sha:
                                pr_user, pr_ref = resolve_fork_branch_via_commit(base_sha)
                                if pr_user and pr_ref:
                                    remote_name = f'fork_{pr_user}'
                                    try:
                                        existing_remotes = subprocess.run(
                                            ['git', 'remote'],
                                            cwd=config.repo_dir,
                                            capture_output=True,
                                            text=True,
                                            check=True
                                        ).stdout.split()
                                    except subprocess.CalledProcessError:
                                        existing_remotes = []
                                    if remote_name not in existing_remotes:
                                        try:
                                            fork_url = f'https://github.com/{pr_user}/{config.repo_name}.git'
                                            subprocess.run(
                                                ['git', 'remote', 'add', remote_name, fork_url],
                                                cwd=config.repo_dir,
                                                capture_output=True,
                                                text=True,
                                                check=True
                                            )
                                        except subprocess.CalledProcessError:
                                            pass
                                    # 拉取 PR head 引用
                                    try:
                                        subprocess.run(
                                            ['git', 'fetch', '--depth', '200', '--shallow-since', failure_time.isoformat(), remote_name, pr_ref],
                                            cwd=config.repo_dir,
                                            capture_output=True,
                                            text=True,
                                            check=True
                                        )
                                    except subprocess.CalledProcessError:
                                        try:
                                            subprocess.run(
                                                ['git', 'fetch', '--depth', '200', remote_name, pr_ref],
                                                cwd=config.repo_dir,
                                                capture_output=True,
                                                text=True,
                                                check=True
                                            )
                                        except subprocess.CalledProcessError:
                                            continue
                                    resolved_remote = f'refs/remotes/{remote_name}/{pr_ref}'
                                    try:
                                        if base_sha:
                                            ancestry_cmd = [
                                                'git', 'rev-list', '--ancestry-path', f'{base_sha}..{resolved_remote}',
                                                f'--since={failure_time.isoformat()}',
                                                f'--until={(failure_time + timedelta(days=7)).isoformat()}'
                                            ]
                                            result_hashes = subprocess.run(
                                                ancestry_cmd,
                                                cwd=config.repo_dir,
                                                capture_output=True,
                                                text=True,
                                                check=True
                                            )
                                            log_hashes = [h.strip() for h in result_hashes.stdout.splitlines() if h.strip()]
                                        else:
                                            log_cmd = [
                                                'git', 'log', resolved_remote,
                                                '--since', failure_time.isoformat(),
                                                '--until', (failure_time + timedelta(days=7)).isoformat(),
                                                '--format=%H'
                                            ]
                                            result_hashes = subprocess.run(
                                                log_cmd,
                                                cwd=config.repo_dir,
                                                capture_output=True,
                                                text=True,
                                                check=True
                                            )
                                            log_hashes = [h.strip() for h in result_hashes.stdout.splitlines() if h.strip()]
                                        for h in log_hashes:
                                            if not h:
                                                continue
                                            commits.append({'sha': h, 'source_branch_ref': resolved_remote})
                                        success = True
                                        break
                                    except subprocess.CalledProcessError:
                                        continue
                            # 未能通过 PR 解析，继续下一个候选
                            continue

                        # 尝试浅拉取，优先按时间窗口；失败则退化为按 depth
                        try:
                            subprocess.run(
                                ['git', 'fetch', '--depth', '200', '--shallow-since', failure_time.isoformat(), 'origin', candidate],
                                cwd=config.repo_dir,
                                capture_output=True,
                                text=True,
                                check=True
                            )
                        except subprocess.CalledProcessError:
                            subprocess.run(
                                ['git', 'fetch', '--depth', '200', 'origin', candidate],
                                cwd=config.repo_dir,
                                capture_output=True,
                                text=True,
                                check=True
                            )

                        # 使用更新后的远端引用进行同样的提交列表收集
                        resolved_remote = f'refs/remotes/origin/{candidate}'
                        try:
                            if base_sha:
                                ancestry_cmd = [
                                    'git', 'rev-list', '--ancestry-path', f'{base_sha}..{resolved_remote}',
                                    f'--since={failure_time.isoformat()}',
                                    f'--until={end_time_window.isoformat()}'
                                ]
                                result_hashes = subprocess.run(
                                    ancestry_cmd,
                                    cwd=config.repo_dir,
                                    capture_output=True,
                                    text=True,
                                    check=True
                                )
                                log_hashes = [h.strip() for h in result_hashes.stdout.splitlines() if h.strip()]
                            else:
                                log_cmd = [
                                    'git', 'log', resolved_remote,
                                    '--since', failure_time.isoformat(),
                                    '--until', end_time_window.isoformat(),
                                    '--format=%H'
                                ]
                                result_hashes = subprocess.run(
                                    log_cmd,
                                    cwd=config.repo_dir,
                                    capture_output=True,
                                    text=True,
                                    check=True
                                )
                                log_hashes = [h.strip() for h in result_hashes.stdout.splitlines() if h.strip()]

                            for h in log_hashes:
                                if not h:
                                    continue
                                commits.append({'sha': h, 'source_branch_ref': resolved_remote})
                            success = True
                            break
                        except subprocess.CalledProcessError:
                            continue
                    except Exception:
                        continue

                # 如果所有候选分支都失败了，记录日志并返回空列表
                if not success:
                    logger.warning(f"无法找到分支 '{branch}' 的后续提交，返回空列表")
        else:
            # 没有指定分支时，返回空列表
            logger.info("未指定分支，返回空列表")
        
        # 在返回前：统一使用本地git获取必要字段，并过滤掉主分支提交
        try:
            main_names: Set[str] = {config.main_branch, 'main', 'master'}
        except Exception:
            main_names: Set[str] = {'main', 'master'}

        # 批量 git show 获取必要字段
        def extract_branch_from_ref(ref_value: str) -> str:
            if not ref_value:
                return ''
            ref_value = ref_value.strip()
            # 取末尾段作为分支名（兼容 refs/heads/, refs/remotes/origin/ 等）
            return ref_value.split('/')[-1]

        sha_to_ref: Dict[str, str] = {}
        ordered_shas: List[str] = []
        for item in commits:
            sha_value = item.get('sha') if isinstance(item, dict) else None
            if not sha_value or sha_value in sha_to_ref:
                continue
            sha_to_ref[sha_value] = item.get('source_branch_ref', '') if isinstance(item, dict) else ''
            ordered_shas.append(sha_value)

        if not ordered_shas:
            return []

        rebuilt_commits: List[Dict] = []
        # 分批执行，避免命令过长
        chunk_size = 128
        fmt = '%H%x1f%ae%x1f%an%x1f%cI%x1f%B%x1e'
        for i in range(0, len(ordered_shas), chunk_size):
            batch = ordered_shas[i:i + chunk_size]
            try:
                result = subprocess.run(
                    ['git', 'show', '-s', f'--format={fmt}', *batch],
                    cwd=config.repo_dir,
                    capture_output=True,
                    text=True,
                    check=True
                )
            except subprocess.CalledProcessError:
                # 若批次失败，逐个回退尝试（最大化保留可解析的提交）
                for sha in batch:
                    try:
                        single = subprocess.run(
                            ['git', 'show', '-s', f'--format={fmt}', sha],
                            cwd=config.repo_dir,
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        output = single.stdout
                    except subprocess.CalledProcessError:
                        continue
                    records = [rec for rec in output.split('\x1e') if rec.strip()]
                    for rec in records:
                        parts = rec.split('\x1f')
                        if len(parts) < 5:
                            continue
                        out_sha, email, name, date_str, message = parts[0], parts[1], parts[2], parts[3], parts[4]
                        ref_val = sha_to_ref.get(out_sha, '')
                        branch_name = extract_branch_from_ref(ref_val)
                        if branch_name in main_names:
                            continue
                        commit_obj = {
                            'sha': out_sha,
                            'commit': {
                                'message': message.strip(),
                                'author': {
                                    'date': date_str.strip(),
                                    'email': email.strip(),
                                    'name': name.strip(),
                                }
                            }
                        }
                        if ref_val:
                            commit_obj['source_branch_ref'] = ref_val
                        rebuilt_commits.append(commit_obj)
                continue

            output = result.stdout
            records = [rec for rec in output.split('\x1e') if rec.strip()]
            for rec in records:
                parts = rec.split('\x1f')
                if len(parts) < 5:
                    continue
                out_sha, email, name, date_str, message = parts[0], parts[1], parts[2], parts[3], parts[4]
                ref_val = sha_to_ref.get(out_sha, '')
                branch_name = extract_branch_from_ref(ref_val)
                if branch_name in main_names:
                    continue
                commit_obj = {
                    'sha': out_sha,
                    'commit': {
                        'message': message.strip(),
                        'author': {
                            'date': date_str.strip(),
                            'email': email.strip(),
                            'name': name.strip(),
                        }
                    }
                }
                if ref_val:
                    commit_obj['source_branch_ref'] = ref_val
                rebuilt_commits.append(commit_obj)

        return rebuilt_commits
        
    except Exception as e:
        logger.error(f"获取提交失败: {e}")
        return []

def find_main_branch_commits(config: ProjectConfig, failure_date: str, author_email: str = None) -> List[Dict]:
    """
    在主分支上查找同作者的后续提交
    
    Args:
        config: 项目配置
        failure_date: 失败时间
        branch: 分支名称（可选）
        author_email: 作者邮箱（可选）
        
    Returns:
        符合条件的提交列表，每个提交包含:
        - sha: 提交哈希
        - commit: 提交信息（包含message和author信息）
    """
    if not author_email or not author_email.strip():
        logger.info("跳过主分支查找：没有作者邮箱信息")
        return []
        
    try:
        # 获取失败提交的时间戳
        failure_time = datetime.fromisoformat(failure_date.replace('Z', '+00:00'))
        end_time = failure_time + timedelta(days=7)  # 只向后查找7天
        
        # 在主分支上查找同作者的后续提交
        cmd = [
            'git', 'log', config.main_branch,
            '--author', author_email,
            '--since', failure_time.isoformat(),
            '--until', end_time.isoformat(),
            '--format=%H'  # 先只获取commit hash
        ]
        
        result = subprocess.run(cmd, cwd=config.repo_dir, capture_output=True, text=True, check=True)
        commit_hashes = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        
        if not commit_hashes:
            logger.info("主分支未找到同作者的后续提交")
            return []
            
        logger.info(f"在主分支找到 {len(commit_hashes)} 个同作者提交的hash")
        
        # 获取每个提交的详细信息
        main_commits = []
        for commit_hash in commit_hashes:
            try:
                # 分别获取各个字段，避免JSON格式问题
                message_cmd = ['git', 'show', '-s', '--format=%B', commit_hash]
                date_cmd = ['git', 'show', '-s', '--format=%cI', commit_hash]
                email_cmd = ['git', 'show', '-s', '--format=%ae', commit_hash]
                
                message = subprocess.run(message_cmd, cwd=config.repo_dir, capture_output=True, text=True, check=True).stdout.strip()
                date = subprocess.run(date_cmd, cwd=config.repo_dir, capture_output=True, text=True, check=True).stdout.strip()
                email = subprocess.run(email_cmd, cwd=config.repo_dir, capture_output=True, text=True, check=True).stdout.strip()
                
                commit_data = {
                    'sha': commit_hash,
                    'commit': {
                        'message': message,
                        'author': {
                            'date': date,
                            'email': email
                        }
                    }
                }
                main_commits.append(commit_data)
                
            except subprocess.CalledProcessError as e:
                logger.debug(f"获取提交 {commit_hash} 的信息失败: {e}")
                continue
        
        if not main_commits:
            logger.info("无法获取任何提交的详细信息")
            return []
            
        # 按时间排序（从早到晚）
        main_commits.sort(
            key=lambda x: datetime.fromisoformat(x['commit']['author']['date'].replace('Z', '+00:00'))
        )
        
        return main_commits
        
    except Exception as e:
        logger.error(f"主分支查找执行失败: {e}")
        return []

def check_commit_build_status(config: ProjectConfig, commit_sha: str, workflow_name: str = None) -> bool:
    """
    检查commit的编译状态
    
    Args:
        config: 项目配置
        commit_sha: commit的SHA
        workflow_name: 特定的workflow名称（可选），如果提供则只检查该workflow
        
    Returns:
        bool: 是否编译成功
    """
    try:
        # 获取workflow运行状态
        url = f"{config.api_base_url}/repos/{config.repo_owner}/{config.repo_name}/actions/runs"
        params = {'head_sha': commit_sha}
        response = requests.get(url, headers=get_headers(), params=params, timeout=15)
        response.raise_for_status()
        runs_data = response.json()
        
        all_success = True
        # 检查编译相关的workflow
        for run in runs_data.get('workflow_runs', []):
            # 如果指定了特定的workflow，只检查该workflow
            if workflow_name and run['name'] != workflow_name:
                continue
            # 否则检查所有编译相关的workflow
            elif not workflow_name and not is_compilation_workflow(run['name'], config):
                continue
                
            if run['conclusion'] == 'failure':
                all_success = False

        if len(runs_data.get('workflow_runs', [])) == 0:
            all_success = False
            logger.info(f"? Commit {commit_sha} 没有编译workflow记录")
        elif not all_success:
            logger.info(f"✗ Commit {commit_sha} 编译失败")
        elif all_success:
            logger.info(f"✓ Commit {commit_sha} 编译成功")
        return all_success
        
    except Exception as e:
        logger.error(f"检查编译状态失败: {e}")
        if isinstance(e, requests.exceptions.RequestException):
            resp = getattr(e, 'response', None)
            resp_text = getattr(resp, 'text', '无响应内容')
            logger.error(f"API响应: {resp_text}")
        return False

def get_commit_build_status_state(config: ProjectConfig, commit_sha: str, workflow_name: str = None) -> str:
    """
    获取commit的编译状态（区分无记录）。

    Returns:
        'success' | 'failure' | 'no_record'
    """
    try:
        url = f"{config.api_base_url}/repos/{config.repo_owner}/{config.repo_name}/actions/runs"
        params = {'head_sha': commit_sha}
        response = requests.get(url, headers=get_headers(), params=params, timeout=15)
        response.raise_for_status()
        runs_data = response.json()

        has_relevant_runs = False
        all_success = True
        for run in runs_data.get('workflow_runs', []):
            if workflow_name and run['name'] != workflow_name:
                continue
            elif not workflow_name and not is_compilation_workflow(run['name'], config):
                continue
            has_relevant_runs = True
            if run['conclusion'] == 'failure':
                all_success = False

        if not has_relevant_runs:
            return 'no_record'
        return 'success' if all_success else 'failure'
    except Exception as e:
        # 安全日志，并保守返回 no_record，避免因为网络异常误判为失败
        try:
            logger.error(f"获取编译状态(三态)失败: {e}")
            if isinstance(e, requests.exceptions.RequestException):
                resp = getattr(e, 'response', None)
                resp_text = getattr(resp, 'text', '无响应内容')
                logger.error(f"API响应: {resp_text}")
        except Exception:
            pass
        return 'no_record'

def get_file_content_from_git(repo_dir: str, commit_sha: str, file_path: str) -> Tuple[str, str]:
    """
    获取指定commit下的文件内容，优先直接路径，失败则用ls-tree查找。
    
    Args:
        repo_dir: 仓库目录
        commit_sha: 提交SHA
        file_path: 文件路径（会被转换为相对于项目根目录的路径）
        
    Returns:
        Tuple[str, str]: (文件内容, 找到的文件路径)
    """
    file_name = os.path.basename(file_path)
    
    commit_sha = sanitize_commit_sha(commit_sha)
    try:
        result = subprocess.run(
            ['git', 'show', f'{commit_sha}:{file_path}'],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8',
            errors='ignore'
        )
        logger.debug(f"直接成功: {commit_sha}:{file_path}")
        return result.stdout, file_path
    except subprocess.CalledProcessError:
        logger.debug(f"直接路径查找失败，尝试ls-tree: {commit_sha}:{file_path}")
        try:
            ls_result = subprocess.run(
                ['git', 'ls-tree', '-r', '--name-only', commit_sha],
                cwd=repo_dir,
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
                logger.warning(f"ls-tree中未找到文件: {file_path}@{commit_sha}")
                return "", ""

            result = subprocess.run(
                ['git', 'show', f'{commit_sha}:{best_match}'],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                check=True,
                encoding='utf-8',
                errors='ignore'
            )
            return result.stdout, best_match

        except subprocess.TimeoutExpired as e:
            logger.warning(f"获取文件内容超时: {file_path}@{commit_sha} - {e}")
        except Exception as e:
            logger.warning(f"ls-tree回退失败: {file_path}@{commit_sha} - {e}")
    except Exception as e:
        logger.error(f"获取文件内容时发生未知错误: {file_path}@{commit_sha} - {e}")
    return "", ""

def parse_includes(repo_dir: str, commit_sha: str, file_path: str, file_content: str = None) -> set:
    """
    通过git show获取指定commit下的file_path内容，解析其直接包含的头文件文件名集合（只返回有文件后缀的，如.h/.hpp等）
    支持传入file_content以避免重复获取。
    """
    includes = set()
    try:
        if file_content is None:
            file_content, _ = get_file_content_from_git(repo_dir, commit_sha, file_path)
        for line in (file_content or "").splitlines():
            match = re.match(r'\s*#include\s+[<\"]([^>\"]+)[>\"]', line)
            if match:
                header = match.group(1)
                # 只保留有文件后缀的include（如.h, .hpp, .c, .cpp等）
                if re.search(r'\.[a-zA-Z0-9]+$', header):
                    includes.add(header)
    except Exception as e:
        logger.warning(f"解析包含头文件失败: {file_path}@{commit_sha} - {e}")
    return includes







def find_relevant_diff_for_error_line(error_line: str, error_detail: Optional[str], diff_hunks: List[Dict], repo_dir: str, failure_commit: str, repair_commit: str, config: ProjectConfig = None, compilation_cache: Dict = None) -> Tuple[List[Dict], Dict[str, List[str]]]:
    """为单个错误行查找相关的修复diff，只提取编译相关文件路径"""
    # 初始化缓存字典
    if compilation_cache is None:
        compilation_cache = {}
    
    # 从错误行中提取文件名和路径
    matches = re.findall(r'([^:\s]+\.(?:c|h|cc|hh|cpp|hpp|cxx|hxx)):(\d+)(?::(\d+))?', error_line)
    if not matches:
        return [], {}
    file_path = matches[0][0]  # 可能带有相对路径
    file_name = os.path.basename(file_path)
    error_line_num = int(matches[0][1])  # 错误行号
    
    # 获取错误文件的内容
    full_content, real_file_path = get_file_content_from_git(repo_dir, failure_commit, file_path)
    
    # 使用缓存获取编译相关文件分析
    cache_key = f"{repair_commit}:{file_path}"
    if cache_key in compilation_cache:
        logger.info(f"使用缓存的编译相关文件分析: {file_path}")
        compilation_related = compilation_cache[cache_key]
    else:
        logger.info(f"分析编译相关文件: {file_path}")
        compilation_related = get_compilation_related_files(repo_dir, repair_commit, file_path)
        compilation_cache[cache_key] = compilation_related
    
    # 使用更智能的分类和排序逻辑
    same_file_hunks = []  # 同一文件的修改（最高优先级）
    direct_include_hunks = []  # 直接包含的头文件的修改（高优先级）
    indirect_include_hunks = []  # 间接包含的头文件的修改（中等优先级）
    other_header_hunks = []  # 其他头文件的修改（低优先级）
    source_hunks = []  # 其他源文件的修改（最低优先级）
    
    for hunk in diff_hunks:
        hunk_file = os.path.basename(hunk['file'])
        
        if hunk_file == file_name:
            same_file_hunks.append(hunk)
        elif hunk_file in compilation_related['direct_includes']:
            direct_include_hunks.append(hunk)
        elif hunk_file in compilation_related['indirect_includes']:
            indirect_include_hunks.append(hunk)
        elif hunk_file.endswith(('.h', '.hpp', '.hh', '.hxx')):
            other_header_hunks.append(hunk)
        elif is_source_file(hunk_file):
            source_hunks.append(hunk)
    
    logger.info(f"智能Hunk分类统计:")
    logger.info(f"  同一文件: {len(same_file_hunks)}")
    logger.info(f"  直接包含: {len(direct_include_hunks)}")
    logger.info(f"  间接包含: {len(indirect_include_hunks)}")
    logger.info(f"  其他头文件: {len(other_header_hunks)}")
    logger.info(f"  其他源文件: {len(source_hunks)}")
    
    # 收集compilation_related的文件路径信息（只保存有diff hunk的文件）
    compilation_related_paths = {}
    
    # 获取所有diff hunk中的文件路径
    diff_hunk_files = {hunk['file'] for hunk in diff_hunks}
    diff_hunk_basenames = {os.path.basename(hunk['file']) for hunk in diff_hunks}
    
    # 收集错误文件路径（如果存在same_file_hunks）
    if same_file_hunks:
        compilation_related_paths['error_file'] = [real_file_path]
    
    # 收集直接包含的头文件路径（只保存有diff hunk的）
    if compilation_related['direct_includes']:
        direct_include_paths = []
        for include_name in compilation_related['direct_includes']:
            # 检查是否有对应的diff hunk
            if include_name in diff_hunk_basenames:
                # 在diff_hunks中找到对应的完整路径
                for hunk in diff_hunks:
                    if os.path.basename(hunk['file']) == include_name:
                        direct_include_paths.append(hunk['file'])
                        break
        if direct_include_paths:
            compilation_related_paths['direct_includes'] = direct_include_paths
    
    # 收集间接包含的头文件路径（只保存有diff hunk的）
    if compilation_related['indirect_includes']:
        indirect_include_paths = []
        for include_name in compilation_related['indirect_includes']:
            # 检查是否有对应的diff hunk
            if include_name in diff_hunk_basenames:
                # 在diff_hunks中找到对应的完整路径
                for hunk in diff_hunks:
                    if os.path.basename(hunk['file']) == include_name:
                        indirect_include_paths.append(hunk['file'])
                        break
        if indirect_include_paths:
            compilation_related_paths['indirect_includes'] = indirect_include_paths
    
    logger.info("只提取文件路径模式，跳过LLM分析")
    return [], compilation_related_paths

def get_commit_diff(config: ProjectConfig, repo_dir: str, failure_commit: str, repair_commit: str, error_lines: List[str], error_details: Optional[List] = None, ignore_whitespace: bool = True) -> Tuple[List[Dict], Dict[str, List[str]], List[Dict[str, List[str]]]]:
    """获取修复提交的diff"""
    try:
        # 在日志中标记本次比较的提交范围
        logger.info(f"获取diff: {failure_commit} -> {repair_commit}")
        # 在打印diff信息之前，先打印错误信息
        try:
            if error_lines:
                logger.info(f"编译错误（共{len(error_lines)}条）：")
                for idx, el in enumerate(error_lines, 1):
                    logger.info(f"[{idx}] {el}")
        except Exception as _:
            # 打印错误信息不应阻塞主流程
            pass
        # 构建git diff命令参数
        cmd_args = ['git', 'diff', 
                   '--no-color',           # 禁用颜色输出
                   '--no-ext-diff',        # 禁用外部diff工具
                   '--unified=3',          # 设置上下文行数
                   '--minimal']            # 尽量生成最小的diff
        
        if ignore_whitespace:
            cmd_args.append('--ignore-space-change')  # 忽略空白字符变化
        
        cmd_args.extend([failure_commit, repair_commit])
        
        # 获取修复提交的diff - 添加更好的参数
        result = subprocess.run(
            cmd_args,
            cwd=repo_dir,
            capture_output=True,
            text=False,  # 改为False，手动处理编码
            check=True
        )
        
        # 尝试不同的编码方式解码
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        diff_text = None
        
        for encoding in encodings:
            try:
                diff_text = result.stdout.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
                
        if diff_text is None:
            # 如果所有编码都失败，使用latin1（它不会失败，但可能产生乱码）
            diff_text = result.stdout.decode('latin1')
            logger.warning(f"使用latin1编码解码diff，可能包含乱码")
        
        # 记录diff概要信息（--stat）并将完整diff写入日志目录的补丁文件
        try:
            stat_proc = subprocess.run(
                ['git', 'diff', '--stat', failure_commit, repair_commit],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                check=False
            )
            diff_stat = stat_proc.stdout.strip()

            # 将完整diff写入日志目录，便于离线查看
            log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'diffs')
            os.makedirs(log_dir, exist_ok=True)
            safe_fail = failure_commit[:12]
            safe_fix = repair_commit[:12]
            diff_path = os.path.join(log_dir, f"diff_{safe_fail}_to_{safe_fix}.patch")
            with open(diff_path, 'w', encoding='utf-8', newline='') as f:
                f.write(diff_text)
            line_count = diff_text.count('\n') + (0 if diff_text.endswith('\n') else 1)
            byte_count = len(diff_text.encode('utf-8'))
            logger.info(f"完整diff已保存: {diff_path} (行数: {line_count}, 字节: {byte_count})")
        except Exception as e:
            logger.warning(f"记录/保存diff信息时出错: {e}")

        # 解析diff块
        diff_hunks = parse_diff_hunks(diff_text)
        
        # 创建缓存字典，用于缓存get_compilation_related_files的结果
        compilation_cache = {}
        
        # 对每个错误行查找相关的修复diff（支持多个diff）
        repair_diffs = []
        all_compilation_related_paths = {}
        compilation_related_paths_details = []  # 新增：记录每个错误行的编译相关路径详情
        
        for i, error_line in enumerate(error_lines):
            # 获取对应的error_detail（如果存在）
            error_detail = None
            if error_details and i < len(error_details):
                error_detail = error_details[i]
            

            logger.info("=" * 80)
            logger.info("错误行详情:")
            logger.info(f"<{i+1}/{len(error_lines)}>: {error_line}")

            relevant_repairs, compilation_related_paths = find_relevant_diff_for_error_line(error_line, error_detail, diff_hunks, repo_dir, failure_commit, repair_commit, config, compilation_cache)
            
            if relevant_repairs:
                logger.info(f"找到 {len(relevant_repairs)} 个相关修复:")
                #for j, repair in enumerate(relevant_repairs, 1):
                    #logger.info(f"修复 {j}:")
                    #logger.info(f"  修改的文件: {repair['modified_file']}")
                    #logger.info(f"  修复diff: {repair['diff']}")
                    #if repair['explanation']:
                    #    logger.info(f"  修复解释: {repair['explanation']}")
                
                # 为每个错误行添加其相关修复数组
                repair_diffs.append({
                    'error_line': error_line,
                    'relevant_repairs': relevant_repairs
                })
            else:
                logger.info("未找到相关修复")
                # 即使没有找到修复也要添加条目
                repair_diffs.append({
                    'error_line': error_line,
                    'relevant_repairs': []
                })
            
            # 记录当前错误行的编译相关路径详情
            compilation_related_paths_details.append(compilation_related_paths)
            
            # 合并compilation_related_paths
            for key, paths in compilation_related_paths.items():
                if key not in all_compilation_related_paths:
                    all_compilation_related_paths[key] = []
                all_compilation_related_paths[key].extend(paths)
            
            logger.info("=" * 80)
        
        # 去重compilation_related_paths中的路径
        for key in all_compilation_related_paths:
            all_compilation_related_paths[key] = list(set(all_compilation_related_paths[key]))
            
        return repair_diffs, all_compilation_related_paths, compilation_related_paths_details
        
    except subprocess.CalledProcessError as e:
        logger.error(f"获取diff失败: {e}")
        if e.stderr:
            logger.error(f"错误输出: {e.stderr.decode('utf-8', errors='replace')}")
        return [], {}, []
    except subprocess.TimeoutExpired as e:
        logger.error(f"获取diff超时: {e}")
        return [], {}, []
    except Exception as e:
        logger.error(f"处理diff时发生错误: {e}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        return [], {}, []

def find_repair_commit(
    config: ProjectConfig,
    repo_dir: str,
    record: FailureRecord,
    author_email: str,
    failure_commit_committer_date: Optional[str] = None,
) -> Optional[RepairPair]:
    """查找修复提交

    Args:
        config: 项目配置
        repo_dir: 仓库目录
        record: 失败记录
        author_email: 作者邮箱
        failure_commit_committer_date: 失败提交的 committer date（ISO字符串，建议UTC，如 2025-07-25T11:47:51+00:00）
    """
    try:
        # 1. 获取失败时间：优先使用传入的 committer date，否则回退到记录创建时间
        failure_date = failure_commit_committer_date or record.created_at
        
        # 2. 获取所有可能的修复提交
        logger.info("正在查找可能的修复提交...")
        
        # 2.1 在同一分支上查找后续提交
        same_branch_commits = get_commits_after_failure(config, failure_date, record.branch, author_email, base_sha=record.commit_sha)
        logger.info(f"✓ 在同一分支上找到 {len(same_branch_commits)} 个后续提交")
        
        # 2.2 查找依赖提交
        dependent_commits = find_dependent_commits_all_branches(config, repo_dir, record.commit_sha, failure_date, author_email)
        logger.info(f"✓ 找到 {len(dependent_commits)} 个依赖提交")
        
        # 2.3 查找主分支提交
        main_branch_commits = find_main_branch_commits(config, failure_date, author_email)
        logger.info(f"✓ 找到 {len(main_branch_commits)} 个主分支提交")
        
        # 3. 合并所有提交，并在去重时按来源优先级保留优先级更高者
        all_commits = []
        for commit in same_branch_commits:
            commit['repair_source'] = 'same_branch'
            all_commits.append(commit)
        for commit in dependent_commits:
            commit['repair_source'] = 'dependency'
            all_commits.append(commit)
        for commit in main_branch_commits:
            commit['repair_source'] = 'main_branch'
            all_commits.append(commit)

        # 优先级映射：same_branch > dependency > main_branch
        source_priority = {
            'same_branch': 3,
            'dependency': 2,
            'main_branch': 1
        }

        # 先按sha去重，同时根据优先级保留最佳repair_source
        commit_by_sha = {}
        for commit in all_commits:
            sha = commit.get('sha')
            if not sha or sha == record.commit_sha:
                continue
            if sha not in commit_by_sha:
                commit_by_sha[sha] = commit
            else:
                existing = commit_by_sha[sha]
                existing_src = existing.get('repair_source', 'main_branch')
                new_src = commit.get('repair_source', 'main_branch')
                if source_priority.get(new_src, 0) > source_priority.get(existing_src, 0):
                    # 仅更新来源为优先级更高者
                    existing['repair_source'] = new_src
                    commit_by_sha[sha] = existing

        # 4. 去重后按优先级 + 时间 + 改动量排序（统一调用）
        filtered_commits = sort_commits_with_priority(
            commits=list(commit_by_sha.values()),
            source_priority=source_priority,
            failure_date=failure_date,
            config=config,
            base_sha=record.commit_sha,
        )

        logger.info(f"共找到 {len(filtered_commits)} 个候选修复提交")
        
        # 5. 预获取所有候选提交的编译状态（success/failure/no_record）
        def parse_time_safe(commit_item: Dict) -> datetime:
            try:
                return datetime.fromisoformat(commit_item['commit']['author']['date'].replace('Z', '+00:00'))
            except Exception:
                return datetime.min

        candidate_states: List[Tuple[Dict, str, datetime]] = []
        for c in filtered_commits:
            try:
                state = get_commit_build_status_state(config, c['sha'], record.workflow_name)
            except Exception as e:
                logger.warning(f"获取构建状态失败，暂按 no_record 处理: {c.get('sha', '')} - {e}")
                state = 'no_record'
            candidate_states.append((c, state, parse_time_safe(c)))

        any_success = any(state == 'success' for _, state, _ in candidate_states)
        if not any_success:
            logger.info("所有候选提交均无编译成功记录（仅失败或无记录），跳过该失败commit的分析")
            return None

        def exists_success_after(ts: datetime) -> bool:
            for _, st, t in candidate_states:
                if st == 'success' and t > ts:
                    return True
            return False

        # 6. 逐个检查提交（允许：自身success；或自身no_record且后续存在success）
        best_repair_pair = None
        best_fixed_count = 0

        repeated_count = 0
        
        for i, commit in enumerate(filtered_commits, 1):
            if best_fixed_count != 0 and repeated_count > 3:
                break
            commit_sha = sanitize_commit_sha(commit['sha'])
            commit_message = commit['commit']['message']
            repair_source = commit['repair_source']
            commit_time = datetime.fromisoformat(commit['commit']['author']['date'].replace('Z', '+00:00')) if commit.get('commit', {}).get('author', {}).get('date') else datetime.min
            
            # 查找此 commit 的预先状态
            cur_state = 'failure'
            for c, st, t in candidate_states:
                if c['sha'] == commit_sha:
                    cur_state = st
                    break
            
            # 打印 sha 以便观察是否含有异常空白
            logger.info(f"\n检查第 {i}/{len(filtered_commits)} 个候选提交: {repair_source} {commit_time} (build={cur_state}), sha='{commit_sha}'")
            # 若本地不存在该对象，尝试从所有远端按 SHA 抓取一次
            if not check_commit_exists_local(repo_dir, commit_sha):
                _fetched = fetch_remote_commit(config, repo_dir, commit_sha)
                if not _fetched:
                    logger.warning(f"修复commit {commit_sha} 缺失且抓取失败，跳过")
                    continue
            try:
                commit_info = get_commit_info_from_git(repo_dir, commit_sha)
            except Exception as e:
                logger.warning(f"获取提交信息失败{commit_sha}: {e}")


            # 5.1 编译状态放宽：允许 no_record 但有后续 success 的提交
            allow_analyze = (cur_state == 'success') or (cur_state == 'no_record' and exists_success_after(commit_time))
            if not allow_analyze:
                logger.info("跳过：编译未成功且无后续成功记录")
                continue
            # 5.2 检查错误所在文件是否被删除
            error_files_deleted = False
            for error_line in record.error_lines:
                # 从错误行中提取文件路径
                matches = re.findall(r'([^:\s]+\.(?:c|h|cc|hh|cpp|hpp|cxx|hxx)):(\d+)(?::(\d+))?', error_line)
                if matches:
                    file_path = matches[0][0]
                    try:
                        # 检查文件是否在repair commit中被删除
                        repair_file_content, _ = get_file_content_from_git(repo_dir, commit_sha, file_path)
                        if repair_file_content is None or repair_file_content.strip() == "":
                            logger.info(f"错误行：{error_line}")
                            logger.info(f"跳过：错误文件 {file_path} 在commit {commit_sha} 中被删除")
                            error_files_deleted = True
                            break
                    except Exception as e:
                        logger.info(f"跳过：无法在commit {commit_sha} 中获取错误文件 {file_path} 的内容，可能已被删除: {e}")
                        error_files_deleted = True
                        break
            
            if error_files_deleted:
                continue
                
            # 5.3 获取修复代码差异并提取编译相关文件路径
            try:
                diffs, compilation_related_paths, compilation_related_paths_details = get_commit_diff(
                    config, repo_dir, record.commit_sha, commit_sha, record.error_lines, record.error_details
                )
            except Exception as e:
                logger.warning(f"获取diff失败，跳过该提交 {commit_sha}: {e}")
                continue
            repeated_count+=1
            # 5.4 计算修复的错误数量（按照错误行统计，每个错误行最多计算一次）
            fixed_count = 0
            # 在只提取文件路径模式下，只要有编译相关路径就认为修复了该错误
            if compilation_related_paths:
                fixed_count = len(record.error_lines)  # 认为所有错误都被修复
                logger.info(f"只提取文件路径模式：发现编译相关路径，认为修复了所有 {fixed_count} 个错误")
            
            logger.info(f"{i}/{len(filtered_commits)} 修复的错误数量: {fixed_count}/{len(record.error_lines)}")
            # 5.5 如果这个提交修复了更多的错误，更新最佳修复
            if fixed_count > best_fixed_count:
                best_fixed_count = fixed_count
                best_repair_pair = RepairPair(
                    failure_commit=record.commit_sha,
                    repair_commit=commit_sha,
                    error_lines=record.error_lines,
                    error_details=record.error_details,
                    error_types=record.error_types,
                    error_count=record.error_count,
                    workflow_name=record.workflow_name,
                    job_name=record.job_name,
                    workflow_id=record.workflow_id,
                    job_id=record.job_id,
                    diffs=diffs,
                    repair_source=repair_source,
                    compilation_related_paths=compilation_related_paths,
                    compilation_related_paths_details=compilation_related_paths_details
                )
                
                # 如果所有错误都修复了，可以提前返回
                if fixed_count == len(record.error_lines):
                    logger.info(f"✓ 找到完整修复提交: {commit_sha}")
                    return best_repair_pair
        
        # 分析完所有提交后，返回修复最多错误的提交
        if best_repair_pair:
            logger.info(f"✓ 找到最佳修复提交: {best_repair_pair.repair_commit} (修复了 {best_fixed_count}/{len(record.error_lines)} 个错误)")
            return best_repair_pair
        
        logger.info("未找到修复提交")
        return None
        
    except Exception as e:
        logger.error(f"查找修复提交失败: {e}")
        return None

def load_failure_records(config: ProjectConfig) -> List[FailureRecord]:
    """加载失败记录"""
    if not os.path.exists(config.failures_file):
        logger.error(f"失败记录文件不存在: {config.failures_file}")
        return []
    
    with open(config.failures_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    records = []
    for item in data.get('compiler_errors', []):
        # 直接使用数组格式，无需兼容性处理
        job_name = item.get('job_name', [])
        job_id = item.get('job_id', [])
        
        record = FailureRecord(
            commit_sha=item['commit_sha'],
            branch=item.get('branch', ''),
            error_lines=item['error_lines'],
            error_types=item.get('error_types', {}),
            error_count=item.get('error_count', len(item['error_lines'])),
            created_at=item['created_at'],
            workflow_name=item.get('workflow_name', ''),
            workflow_id=item.get('workflow_id', 0),
            job_name=job_name,
            job_id=job_id,
            error_details=item.get('error_details', None)
        )
        records.append(record)
    
    logger.info(f"加载了 {len(records)} 个失败记录")
    return records

def get_last_processed_index(config: ProjectConfig, failure_records: List[FailureRecord]) -> int:
    """
    获取最后处理的记录索引
    
    Args:
        config: 项目配置
        failure_records: 所有失败记录列表（按处理顺序排序）
        
    Returns:
        int: 最后处理的记录索引，如果文件不存在或为空则返回-1
    """
    if not os.path.exists(config.output_file):
        logger.info(f"输出文件不存在: {config.output_file}")
        return -1
    
    try:
        # 读取输出文件中最后处理的记录信息
        last_processed_info = None
        with open(config.output_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    failure_commit = data.get('failure_commit')
                    job_id = data.get('job_id')
                    workflow_id = data.get('workflow_id')
                    if failure_commit and job_id is not None and workflow_id is not None:
                        last_processed_info = {
                            'failure_commit': failure_commit,
                            'job_id': job_id,
                            'workflow_id': workflow_id
                        }
                except json.JSONDecodeError as e:
                    logger.warning(f"解析第 {line_num} 行JSON失败: {e}")
                    continue
        
        if not last_processed_info:
            logger.info("输出文件中没有找到有效的处理记录")
            return -1
        
        # 在输入记录列表中查找最后处理的记录位置
        for i, record in enumerate(failure_records):
            # job_id现在总是数组格式
            if (record.commit_sha == last_processed_info['failure_commit'] and
                record.job_id == last_processed_info['job_id'] and
                record.workflow_id == last_processed_info['workflow_id']):
                logger.info(f"找到最后处理的记录: {last_processed_info['failure_commit']} "
                          f"(job_id: {last_processed_info['job_id']}, "
                          f"workflow_id: {last_processed_info['workflow_id']}, 索引: {i})")
                return i
        
        # 如果找不到匹配的记录，说明输入记录列表可能已经改变
        logger.warning(f"在输入记录列表中找不到最后处理的记录: {last_processed_info}")
        logger.warning("输入记录列表可能已经改变，建议使用 --restart 重新开始")
        return -1
        
    except Exception as e:
        logger.error(f"读取现有结果失败: {e}")
        return -1

def show_analysis_progress(config: ProjectConfig, total_records: int, processed_count: int) -> None:
    """
    显示分析进度和统计信息
    
    Args:
        config: 项目配置
        total_records: 总记录数
        processed_count: 已处理记录数
    """
    if total_records == 0:
        return
    
    progress_percent = (processed_count / total_records) * 100
    remaining_count = total_records - processed_count
    
    logger.info(f"分析进度: {processed_count}/{total_records} ({progress_percent:.1f}%)")
    logger.info(f"剩余记录: {remaining_count} 个")
    
    # 估算剩余时间（假设每个记录平均需要2分钟）
    if remaining_count > 0:
        estimated_minutes = remaining_count * 2
        estimated_hours = estimated_minutes / 60
        if estimated_hours >= 1:
            logger.info(f"预计剩余时间: {estimated_hours:.1f} 小时")
        else:
            logger.info(f"预计剩余时间: {estimated_minutes:.0f} 分钟")
    
    # 显示输出文件信息
    if os.path.exists(config.output_file):
        try:
            file_size = os.path.getsize(config.output_file)
            file_size_mb = file_size / (1024 * 1024)
            logger.info(f"输出文件大小: {file_size_mb:.1f} MB")
        except Exception:
            pass

def validate_output_file(config: ProjectConfig) -> Tuple[int, int]:
    """
    验证输出文件的完整性，返回有效行数和错误行数
    
    Args:
        config: 项目配置
        
    Returns:
        Tuple[int, int]: (有效行数, 错误行数)
    """
    valid_lines = 0
    error_lines = 0
    
    if not os.path.exists(config.output_file):
        return valid_lines, error_lines
    
    try:
        with open(config.output_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    # 检查必要字段
                    if data.get('failure_commit') and data.get('repair_commit'):
                        valid_lines += 1
                    else:
                        error_lines += 1
                        logger.warning(f"第 {line_num} 行缺少必要字段")
                except json.JSONDecodeError as e:
                    error_lines += 1
                    logger.warning(f"第 {line_num} 行JSON格式错误: {e}")
        
        logger.info(f"输出文件验证结果: {valid_lines} 个有效记录, {error_lines} 个错误记录")
        return valid_lines, error_lines
        
    except Exception as e:
        logger.error(f"验证输出文件失败: {e}")
        return valid_lines, error_lines

def backup_output_file(config: ProjectConfig) -> str:
    """
    备份输出文件
    
    Args:
        config: 项目配置
        
    Returns:
        str: 备份文件路径
    """
    if not os.path.exists(config.output_file):
        return ""
    
    try:
        # 生成备份文件名（包含时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{config.output_file}.backup_{timestamp}"
        
        # 复制文件
        shutil.copy2(config.output_file, backup_path)
        logger.info(f"已备份输出文件到: {backup_path}")
        
        return backup_path
        
    except Exception as e:
        logger.error(f"备份输出文件失败: {e}")
        return ""

def save_repair_result_append(output_file: str, repair_result: RepairPair):
    """以jsonl格式追加写入单条修复结果"""
    # 转为可序列化字典
    result_dict = {
        'failure_commit': repair_result.failure_commit,
        'repair_commit': repair_result.repair_commit,
        'error_lines': repair_result.error_lines,
        'workflow_id': repair_result.workflow_id,
        'job_id': repair_result.job_id,
        'workflow_name': repair_result.workflow_name,
        'job_name': repair_result.job_name,
        'diffs': [
            {
                'error_line': diff['error_line'],
                'relevant_repairs': diff['relevant_repairs']
            }
            for diff in repair_result.diffs
        ],
        'repair_source': repair_result.repair_source,
        'error_details': repair_result.error_details,
        'compilation_related_paths': repair_result.compilation_related_paths,
        'compilation_related_paths_details': repair_result.compilation_related_paths_details
    }
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
    logger.info(f"已追加写入修复结果到: {output_file}")

def process_single_record(config: ProjectConfig, record: FailureRecord, index: int, total: int) -> None:
    """处理单条记录"""
    logger.info(f"[{index}/{total}] 处理记录: {record.commit_sha}")
    
    # 检查错误行数量，如果超过30个则跳过处理
    error_line_count = len(record.error_lines)
    if error_line_count > 30:
        logger.info(f"⚠️  跳过处理：错误行数量 ({error_line_count}) 超过30个")
        return
    logger.info(f"✓ 分支: {record.branch}")
    # 获取作者信息
    author_email = None
    try:
        # 先检查commit是否在本地存在
        if not check_commit_exists_local(config.repo_dir, record.commit_sha):
            # 如果不存在，尝试从远程获取
            if not fetch_remote_commit(config, config.repo_dir, record.commit_sha):
                logger.warning(f"⚠️  无法获取commit {record.commit_sha}")
                return
        
        # 使用新的函数获取commit信息
        commit_info = get_commit_info_from_git(config.repo_dir, record.commit_sha)

        if commit_info:
            author_email = commit_info['author_email']
        else:
            logger.warning("⚠️  无法获取作者邮箱信息")
            
    except Exception as e:
        logger.warning(f"⚠️  获取作者信息失败: {e}")
    
    # 查找修复提交：优先传入 committer date（解析为UTC ISO）
    failure_commit_committer_date_iso_utc = None
    try:
        if commit_info and commit_info.get('commit_time'):
            # 示例格式: '2025-07-25 17:15:05 +0530'
            dt = datetime.strptime(commit_info['commit_time'], '%Y-%m-%d %H:%M:%S %z')
            dt_utc = dt.astimezone(timezone.utc)
            failure_commit_committer_date_iso_utc = dt_utc.isoformat()
    except Exception as _:
        failure_commit_committer_date_iso_utc = None

    repair_pair = find_repair_commit(
        config,
        config.repo_dir,
        record,
        author_email,
        failure_commit_committer_date=failure_commit_committer_date_iso_utc,
    )
    if repair_pair:
        save_repair_result_append(config.output_file, repair_pair)
        return
    
    logger.info("未找到修复提交")

def analyze_repairs(config: ProjectConfig, failure_records: List[FailureRecord], restart: bool = False) -> None:
    """
    分析修复提交
    
    Args:
        config: 项目配置
        failure_records: 失败记录列表（按处理顺序排序）
        restart: 是否重新开始分析（删除现有结果文件）
    """
    
    if restart:
        # 重新开始模式：先备份现有文件，然后删除
        if os.path.exists(config.output_file):
            backup_path = backup_output_file(config)
            if backup_path:
                logger.info(f"已备份现有结果到: {backup_path}")
            
            try:
                os.remove(config.output_file)
                logger.info(f"已删除现有的分析结果文件: {config.output_file}")
            except Exception as e:
                logger.error(f"删除分析结果文件失败: {e}")
                return
        logger.info("重新开始分析模式")
        start_index = 0
    else:
        # 继续分析模式（默认）：定位到断开的位置
        logger.info("继续分析模式：定位到断开的位置...")
        
        # 验证现有输出文件
        valid_lines, error_lines = validate_output_file(config)
        if error_lines > 0:
            logger.warning(f"发现 {error_lines} 个错误记录，建议使用 --restart 重新开始")
        
        # 获取最后处理的记录索引
        start_index = get_last_processed_index(config, failure_records)
        
        if start_index >= len(failure_records) - 1:
            logger.info("所有记录都已处理完成，无需继续分析")
            return
        elif start_index >= 0:
            logger.info(f"从第 {start_index + 2} 个记录开始继续分析（已处理 {start_index + 1} 个记录）")
        else:
            logger.info("未发现已处理的记录，将从头开始分析")
            start_index = -1
    
    # 计算需要处理的记录
    records_to_process = failure_records[start_index + 1:]
    total_to_process = len(records_to_process)
    total_records = len(failure_records)
    
    logger.info(f"开始处理 {total_to_process} 个记录（总共 {total_records} 个记录）...")
    
    # 显示初始进度
    if not restart:
        processed_count = start_index + 1
        show_analysis_progress(config, total_records, processed_count)
    
    # 逐条处理记录
    for i, record in enumerate(records_to_process, 1):
        global_index = start_index + i
        process_single_record(config, record, global_index + 1, total_records)
        
        # 每处理10个记录显示一次进度
        if i % 10 == 0 or i == total_to_process:
            if not restart:
                processed_count = start_index + i
                show_analysis_progress(config, total_records, processed_count)
    
    logger.info("分析完成")
    
    # 显示最终统计信息
    if os.path.exists(config.output_file):
        valid_lines, error_lines = validate_output_file(config)
        logger.info(f"最终统计: 成功分析 {valid_lines} 个记录")
        if error_lines > 0:
            logger.warning(f"发现 {error_lines} 个错误记录")

def ensure_full_history(repo_dir: str) -> None:
    """确保仓库不是浅克隆，尽量解浅到完整历史。

    在 bare/mirror 仓库下同样适用。
    """
    try:
        # 初始浅克隆检测
        result = subprocess.run(
            ['git', 'rev-parse', '--is-shallow-repository'],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        is_shallow = (result.stdout or '').strip().lower() == 'true'
    except Exception:
        is_shallow = False

    # 记录 shallow 文件信息（若存在）
    try:
        shallow_file = os.path.join(repo_dir, 'shallow')
        if os.path.isfile(shallow_file):
            try:
                with open(shallow_file, 'r', encoding='utf-8', errors='ignore') as f:
                    shallow_lines = sum(1 for _ in f)
                logger.info(f"shallow 文件存在: {shallow_file} (条目: {shallow_lines})")
            except Exception:
                logger.info(f"shallow 文件存在: {shallow_file}")
    except Exception:
        pass

    if not is_shallow:
        logger.info("仓库已为完整历史（非浅克隆）")
        return

    logger.info("检测到浅克隆，尝试解浅以获取完整历史...")

    # 逐步尝试多种解浅方案（按优先级降序）
    fetch_attempts: List[List[str]] = [
        ['git', 'fetch', 'origin', '--prune', '--tags', '--unshallow'],
        ['git', 'fetch', 'origin', '--prune', '--tags', '--deepen', '2147483647'],
        ['git', 'fetch', 'origin', '--prune', '--tags', '--depth', '2147483647'],
        ['git', 'fetch', '--all', '--tags', '--prune'],
    ]

    for cmd in fetch_attempts:
        try:
            logger.info(f"执行解浅命令: {' '.join(cmd)}")
            subprocess.run(cmd, cwd=repo_dir, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            err = (e.stderr or '').strip() if hasattr(e, 'stderr') else ''
            logger.warning(f"解浅命令失败: {' '.join(cmd)}; 错误: {err[:400]}")
        except Exception as e:
            logger.warning(f"解浅命令异常: {' '.join(cmd)}; {e}")

        # 每次尝试后复查是否仍为浅克隆
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--is-shallow-repository'],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                check=True,
            )
            if (result.stdout or '').strip().lower() == 'false':
                logger.info("✓ 解浅成功：仓库已为完整历史")
                break
        except Exception:
            # 如果检查失败，不中断，继续下一步
            pass

    # 结束时再次报告 shallow 文件状态
    try:
        shallow_file = os.path.join(repo_dir, 'shallow')
        if os.path.isfile(shallow_file):
            logger.warning("解浅完成后 shallow 文件仍存在，可能仍为浅历史或仅部分加深")
        else:
            logger.info("解浅完成后未检测到 shallow 文件")
    except Exception:
        pass

def detect_default_branch(repo_dir: str, fallback_main: str = 'main') -> str:
    """检测远端默认分支，若失败回退到提供的分支名或常见主分支名。"""
    # 优先使用回退分支（若本地存在该引用）
    try:
        r = subprocess.run(
            ['git', 'show-ref', '--verify', '--quiet', f'refs/heads/{fallback_main}'],
            cwd=repo_dir,
            check=False,
        )
        if r.returncode == 0:
            return fallback_main
    except Exception:
        pass

    # 读取远端 HEAD 指向
    try:
        proc = subprocess.run(
            ['git', 'remote', 'show', 'origin'],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        for line in (proc.stdout or '').splitlines():
            line = line.strip()
            if line.startswith('HEAD branch: '):
                return line.split(':', 1)[1].strip()
    except Exception:
        pass

    # 常见回退
    for name in ['main', 'master']:
        try:
            r = subprocess.run(
                ['git', 'show-ref', '--verify', '--quiet', f'refs/heads/{name}'],
                cwd=repo_dir,
                check=False,
            )
            if r.returncode == 0:
                return name
        except Exception:
            continue
    return fallback_main

def log_main_branch_time_range(repo_dir: str, branch_name: str) -> None:
    """记录主分支的提交数量与最早/最新提交时间范围。"""
    # 按优先级选择可用引用
    ref_candidates = [
        f'refs/heads/{branch_name}',
        f'refs/remotes/origin/{branch_name}',
        f'origin/{branch_name}',
    ]
    resolved_ref = ''
    for ref in ref_candidates:
        try:
            r = subprocess.run(['git', 'rev-parse', '--verify', ref], cwd=repo_dir, capture_output=True, text=True)
            if r.returncode == 0:
                resolved_ref = ref
                break
        except Exception:
            continue
    if not resolved_ref:
        logger.warning(f"无法定位主分支引用: {branch_name}")
        return

    try:
        cnt_proc = subprocess.run(['git', 'rev-list', '--count', resolved_ref], cwd=repo_dir, capture_output=True, text=True, check=True)
        count_val = (cnt_proc.stdout or '').strip()
    except Exception:
        count_val = '?'
    try:
        # 使用全局维度的“最早提交”获取方式，与你提供的命令一致
        first_proc = subprocess.run(
            'git log --reverse --format=%H\t%aI\t%cI | head -n 1 | cat',
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
            shell=True,
        )
        earliest = (first_proc.stdout or '').strip()
    except Exception:
        earliest = ''
    try:
        last_proc = subprocess.run(['git', 'log', '-n', '1', '--format=%H\t%aI\t%cI', resolved_ref], cwd=repo_dir, capture_output=True, text=True, check=True)
        latest = (last_proc.stdout or '').strip()
    except Exception:
        latest = ''

    logger.info(f"主分支 {branch_name} 引用: {resolved_ref}")
    logger.info(f"提交数: {count_val}")
    if earliest:
        logger.info(f"最早提交: {earliest}")
    if latest:
        logger.info(f"最新提交: {latest}")

def setup_git_repo(config: ProjectConfig, skip_update: bool = False):
    """设置或更新git仓库，并预先下载所有需要的数据

    Args:
        config: 项目配置
        skip_update: 若为True且仓库已存在，则跳过远端更新与PR引用获取
    """
    repo_url = f"https://github.com/{config.repo_owner}/{config.repo_name}.git"
    try:
        if not os.path.exists(config.repo_dir):
            logger.info(f"克隆 {config.project_name} 仓库...")
            
            # 使用--mirror克隆所有分支和引用
            subprocess.run([
                'git', 'clone', '--mirror',
                repo_url, config.repo_dir
            ], check=True, capture_output=True)
            
            logger.info("✓ 克隆完成")
            # mirror 克隆理论上是完整历史，但若为迁移遗留浅仓库，进行一次解浅确认
            ensure_full_history(config.repo_dir)
            # 记录主分支的时间范围，便于核对是否全量
            try:
                main_branch = detect_default_branch(config.repo_dir, getattr(config, 'main_branch', 'main'))
                log_main_branch_time_range(config.repo_dir, main_branch)
            except Exception:
                pass
        else:
            if skip_update:
                logger.info(f"仓库已存在，跳过更新 {config.project_name} 仓库 (--skip-update)")
            else:
                logger.info(f"仓库已存在，更新 {config.project_name} 仓库...")
                # 只更新origin的远程引用
                subprocess.run(
                    ['git', 'fetch', 'origin', '--prune'], 
                    cwd=config.repo_dir, 
                    check=True, 
                    capture_output=True
                )
                logger.info("✓ origin远程引用更新完成")
                # 检查是否已经有PR引用
                check_refs = subprocess.run(
                    ['git', 'ls-remote', '--refs', 'origin', 'refs/pull/*'],
                    cwd=config.repo_dir,
                    check=True,
                    capture_output=True,
                    text=True
                )
                if not check_refs.stdout.strip():
                    # 如果没有PR引用才获取
                    logger.info("获取所有PR引用...")
                    subprocess.run(
                        ['git', 'fetch', 'origin', '+refs/pull/*:refs/remotes/origin/pr/*'],
                        cwd=config.repo_dir,
                        check=True,
                        capture_output=True
                    )
                    logger.info("✓ PR引用获取完成")
                else:
                    logger.info("PR引用已存在,跳过获取")

                # 更新后进行解浅，确保完整历史
                ensure_full_history(config.repo_dir)
                # 打印主分支的时间范围
                try:
                    main_branch = detect_default_branch(config.repo_dir, getattr(config, 'main_branch', 'main'))
                    log_main_branch_time_range(config.repo_dir, main_branch)
                except Exception:
                    pass
        
        logger.info("✓ 仓库设置和数据预下载完成")
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Git命令执行失败: {e}")
        if e.stderr:
            logger.error(f"错误输出: {e.stderr.decode('utf-8', errors='replace')}")
        raise
    except Exception as e:
        logger.error(f"仓库设置失败: {e}")
        raise

def is_source_file(file_path: str) -> bool:
    """检查是否是源代码文件"""
    source_extensions = {'.c', '.h', '.cpp', '.hpp', '.cc', '.hh', '.cxx', '.hxx'}
    return any(file_path.lower().endswith(ext) for ext in source_extensions)

def parse_diff_hunks(diff_text: str) -> List[Dict]:
    """解析git diff输出的代码块，只保留对C++源代码的修改"""
    try:
        hunks = []
        current_file = None
        hunk_lines = []
        
        for line in diff_text.split('\n'):
            try:
                if line.startswith('diff --git'):
                    # 当遇到新的文件时，处理之前的文件（如果有的话）
                    if current_file and hunk_lines and is_source_file(current_file):
                        hunks.append({
                            'file': current_file,
                            'content': '\n'.join(hunk_lines)
                        })
                    # 获取新文件名
                    current_file = line.split(' b/')[-1]
                    # 只有当是源代码文件时才开始收集行
                    if is_source_file(current_file):
                        hunk_lines = [line]
                    else:
                        hunk_lines = []
                        current_file = None
                elif current_file and is_source_file(current_file):
                    # 只有当前文件是源代码文件时才继续收集行
                    hunk_lines.append(line)
            except Exception as e:
                logger.warning(f"处理diff行时出错: {e}")
                continue
                
        # 处理最后一个文件
        if current_file and hunk_lines and is_source_file(current_file):
            hunks.append({
                'file': current_file,
                'content': '\n'.join(hunk_lines)
            })
            
        # 记录日志
        if hunks:
            logger.info(f"找到 {len(hunks)} 个源代码文件的修改:")
        else:
            logger.info("没有找到源代码文件的修改")
            
        return hunks
        
    except Exception as e:
        logger.error(f"解析diff块时发生错误: {e}")
        return []


def sort_commits_with_priority(
    commits: List[Dict],
    source_priority: Dict[str, int],
    failure_date: str,
    config: ProjectConfig,
    base_sha: str
) -> List[Dict]:
    """按来源优先级、时间接近度和改动规模对提交排序。

    排序键：
    1) 来源优先级（降序：same_branch > dependency > main_branch）
    2) 与失败时间的时间差（升序，越近越好）
    3) 改动文件数（升序）
    4) 增删总量（升序）
    """
    failure_time = datetime.fromisoformat(failure_date.replace('Z', '+00:00'))

    def get_diff_stats(base_sha_local: str, head_sha_local: str) -> Tuple[int, int, int]:
        try:
            result = subprocess.run(
                ['git', 'diff', '--shortstat', base_sha_local, head_sha_local],
                cwd=config.repo_dir,
                capture_output=True,
                text=True,
                timeout=20
            )
            output = result.stdout.strip()
            files_changed = insertions = deletions = 0
            if output:
                files_match = re.search(r"(\d+) files? changed", output)
                insert_match = re.search(r"(\d+) insertions?\(\+\)", output)
                delete_match = re.search(r"(\d+) deletions?\(-\)", output)
                if files_match:
                    files_changed = int(files_match.group(1))
                if insert_match:
                    insertions = int(insert_match.group(1))
                if delete_match:
                    deletions = int(delete_match.group(1))
            return files_changed, insertions, deletions
        except Exception:
            return 10**9, 10**9, 10**9

    def sort_key(commit: Dict) -> Tuple[int, float, int, int]:
        src = commit.get('repair_source', 'main_branch')
        priority_score = -source_priority.get(src, 0)
        try:
            commit_time = datetime.fromisoformat(commit['commit']['author']['date'].replace('Z', '+00:00'))
            time_delta = (commit_time - failure_time).total_seconds()
            if time_delta < 0:
                time_delta = float('inf')
        except Exception:
            time_delta = float('inf')
        files_changed, insertions, deletions = get_diff_stats(base_sha, commit['sha'])
        total_changes = insertions + deletions
        return (priority_score, time_delta, files_changed, total_changes)

    return sorted(commits, key=sort_key)

def get_compilation_related_files(repo_dir: str, commit_sha: str, error_file_path: str) -> Dict[str, Set[str]]:
    """
    获取与错误文件编译相关的文件集合（修复死循环风险）
    
    Args:
        repo_dir: 仓库目录
        commit_sha: 提交SHA
        error_file_path: 错误文件路径
        
    Returns:
        Dict包含不同类型的相关文件集合:
        - direct_includes: 错误文件直接包含的头文件
        - indirect_includes: 错误文件间接包含的头文件（通过递归包含）
    """
    result = {
        'direct_includes': set(),
        'indirect_includes': set()
    }
    # import pdb; pdb.set_trace()
    try:
        error_file_name = os.path.basename(error_file_path)
        error_file_dir = os.path.dirname(error_file_path)
        
        # 1. 获取错误文件直接包含的头文件
        error_file_content, _ = get_file_content_from_git(repo_dir, commit_sha, error_file_path)
        if error_file_content:
            result['direct_includes'] = parse_includes(repo_dir, commit_sha, error_file_path, error_file_content)
        
        # 2. 修复递归获取间接包含的头文件（多重保护防止死循环）
        visited = set()
        recursion_count = 0  # 添加递归计数器
        max_recursion = 1000   # 最大递归次数限制
        
        def get_indirect_includes(file_path: str, depth: int = 0):
            nonlocal recursion_count
            recursion_count += 1
            
            # 多重保护机制
            if (depth >= 10 or                    # 深度限制
                file_path in visited or          # 访问记录
                recursion_count > max_recursion): # 总递归次数限制
                logger.debug(f"停止递归: depth={depth}, visited={file_path in visited}, count={recursion_count}")
                return
                
            visited.add(file_path)
            
            try:
                content, _ = get_file_content_from_git(repo_dir, commit_sha, file_path)
                if content:
                    includes = parse_includes(repo_dir, commit_sha, file_path, content)
                    for include in includes:
                        if include not in result['direct_includes']:
                            result['indirect_includes'].add(include)
                            get_indirect_includes(include, depth + 1)
            except Exception as e:
                logger.debug(f"获取间接包含失败: {file_path} - {e}")
        

        for direct_include in list(result['direct_includes']):  
            get_indirect_includes(direct_include, 0)  # 从深度0开始
        

        
        logger.info(f"编译相关文件分析结果:")
        logger.info(f"  直接包含: {len(result['direct_includes'])} 个文件")
        logger.info(f"  间接包含: {len(result['indirect_includes'])} 个文件")
        logger.info(f"  递归次数: {recursion_count}")
        
    except Exception as e:
        logger.error(f"获取编译相关文件失败: {e}")

    # 将result中的所有项都取basename
    result['direct_includes'] = set(os.path.basename(f) for f in result['direct_includes'])
    result['indirect_includes'] = set(os.path.basename(f) for f in result['indirect_includes'])
    logger.info(f"编译直接相关文件: {result['direct_includes']}")
    logger.info(f"编译间接相关文件: {result['indirect_includes']}")
    
    return result

def get_compilation_related_files_v2(repo_dir: str, commit_sha: str, error_file_path: str, config: ProjectConfig = None) -> Dict[str, Set[str]]:
    """
    使用g++ -MM命令获取与错误文件编译相关的文件集合
    
    Args:
        repo_dir: 仓库目录
        commit_sha: 提交SHA
        error_file_path: 错误文件路径
        config: 项目配置（可选，用于获取include路径）
        
    Returns:
        Dict包含不同类型的相关文件集合:
        - direct_includes: 错误文件直接包含的头文件
        - indirect_includes: 错误文件间接包含的头文件（通过递归包含）
    """
    result = {
        'direct_includes': set(),
        'indirect_includes': set()
    }
    
    try:
        # 检查文件是否存在
        if not os.path.exists(os.path.join(repo_dir, error_file_path)):
            logger.warning(f"错误文件不存在: {error_file_path}")
            return result
        
        # 获取文件扩展名
        file_ext = os.path.splitext(error_file_path)[1].lower()
        
        # 只处理C/C++源文件
        if file_ext not in ['.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.hh', '.hxx']:
            logger.info(f"跳过非C/C++文件: {error_file_path}")
            return result
        
        # 构建g++ -MM命令
        # 根据项目配置设置不同的include路径
        if config and hasattr(config, 'include_paths'):
            include_paths = config.include_paths
        else:
            # 如果没有提供config，使用默认路径
            include_paths = ['-I.', '-Iinclude']
        
        # 根据文件扩展名选择编译器
        file_ext = os.path.splitext(error_file_path)[1].lower()
        if file_ext in ['.c']:
            compiler = 'gcc'
        elif file_ext in ['.cpp', '.cc', '.cxx', '.h', '.hpp', '.hh', '.hxx']:
            compiler = 'g++'
        else:
            # 默认使用g++
            compiler = 'g++'
        
        # 构建完整的编译命令
        cmd = [compiler, '-MM'] + include_paths + [error_file_path]
        
        logger.info(f"执行命令: {' '.join(cmd)}")
        logger.info(f"工作目录: {repo_dir}")
        
        # 执行编译器 -MM命令
        result_process = subprocess.run(
            cmd,
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=30  # 30秒超时
        )
        
        if result_process.returncode != 0:
            logger.warning(f"{compiler} -MM命令执行失败: {result_process.stderr}")
            # 如果编译器 -MM失败，回退到原来的方法
            logger.info("回退到原来的解析方法")
            return get_compilation_related_files(repo_dir, commit_sha, error_file_path)
        
        # 解析编译器 -MM的输出
        output_lines = result_process.stdout.strip().split('\n')
        if not output_lines:
            logger.warning(f"{compiler} -MM没有输出")
            return result
        
        # 解析依赖文件
        dependencies = []
        
        # 处理所有行，处理多行依赖
        current_line = ""
        
        for line in output_lines:
            line = line.strip()
            if line.endswith('\\'):
                # 继续到下一行，去掉反斜杠并添加空格
                current_line += line[:-1] + " "
            else:
                # 完成当前依赖行
                current_line += line
                if ':' in current_line:
                    # 分割冒号后的部分，得到所有依赖文件
                    deps_part = current_line.split(':', 1)[1].strip()
                    line_deps = [dep.strip() for dep in deps_part.split() if dep.strip()]
                    dependencies.extend(line_deps)
                current_line = ""
        
        logger.info(f"{compiler} -MM输出解析结果:")
        logger.info(f"  原始输出: {result_process.stdout.strip()}")
        logger.info(f"  解析到的依赖文件: {dependencies}")
        
        # 分类依赖文件
        for dep_file in dependencies:
            if dep_file == error_file_path:
                continue  # 跳过源文件本身
            
            # 获取文件扩展名
            dep_ext = os.path.splitext(dep_file)[1].lower()
            
            if dep_ext in ['.h', '.hpp', '.hh', '.hxx']:
                # 头文件
                if os.path.basename(dep_file) not in result['direct_includes']:
                    result['direct_includes'].add(os.path.basename(dep_file))
            elif dep_ext in ['.c', '.cpp', '.cc', '.cxx']:
                # 源文件，通常不直接包含，但可能间接影响
                result['indirect_includes'].add(os.path.basename(dep_file))
        
        logger.info(f"编译相关文件分析结果:")
        logger.info(f"  直接包含: {len(result['direct_includes'])} 个文件")
        logger.info(f"  间接包含: {len(result['indirect_includes'])} 个文件")
        logger.info(f"  直接包含文件: {result['direct_includes']}")
        logger.info(f"  间接包含文件: {result['indirect_includes']}")
        
    except subprocess.TimeoutExpired:
        logger.warning(f"{compiler} -MM命令执行超时")
        # 回退到原来的方法
        return get_compilation_related_files(repo_dir, commit_sha, error_file_path)
    except Exception as e:
        logger.error(f"使用{compiler} -MM获取编译相关文件失败: {e}")
        # 回退到原来的方法
        return get_compilation_related_files(repo_dir, commit_sha, error_file_path)
    
    return result

def get_commit_info_from_git(repo_dir: str, commit_sha: str) -> Optional[Dict]:
    """
    从本地git仓库获取commit的详细信息
    
    Args:
        repo_dir: 仓库目录
        commit_sha: commit的SHA
        
    Returns:
        Optional[Dict]: 包含commit信息的字典，格式如下：
        {
            'author_email': str,
            'author_name': str,
            'commit_sha': str,
            'commit_time': str,
            'branch': str,
            'message': str
        }
        如果获取失败则返回None
    """
    try:
        # 检查commit是否存在
        commit_sha = sanitize_commit_sha(commit_sha)
        if not check_commit_exists_local(repo_dir, commit_sha):
            logger.warning(f"Commit 不存在(本地): '{commit_sha}'")
            return None
        
        # 获取commit的基本信息
        t0 = time.time()
        result = subprocess.run(
            ['git', 'show', '-s', '--format=%ae%n%an%n%H%n%ci%n%s', commit_sha],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )
        logger.debug(f"git show 元数据耗时: {(time.time()-t0)*1000:.1f} ms @ {commit_sha}")
        
        commit_info_lines = result.stdout.strip().split('\n')
        logger.debug(f"获取到的提交信息行数: {len(commit_info_lines)}")
        logger.debug(f"提交信息内容: {commit_info_lines}")
        
        # 解析提交信息
        if len(commit_info_lines) >= 5:  # 需要至少5个字段
            author_email = commit_info_lines[0] if len(commit_info_lines) > 0 else ""
            author_name = commit_info_lines[1] if len(commit_info_lines) > 1 else ""
            commit_sha_actual = commit_info_lines[2] if len(commit_info_lines) > 2 else ""
            commit_time = commit_info_lines[3] if len(commit_info_lines) > 3 else ""
            message = commit_info_lines[4] if len(commit_info_lines) > 4 else ""
            
            # 单独获取分支信息（默认跳过以避免耗时）
            branch = get_commit_branch(repo_dir, commit_sha)
            
            # 构建返回结果
            commit_info = {
                'author_email': author_email.strip(),
                'author_name': author_name.strip(),
                'commit_sha': commit_sha_actual.strip(),
                'commit_time': commit_time.strip(),
                'branch': branch,
                'message': message.strip()
            }
            
            logger.info(f"成功获取commit信息:")
            logger.info(f"  作者: {commit_info['author_name']} <{commit_info['author_email']}>")
            logger.info(f"  时间: {commit_info['commit_time']}")
            if commit_info['branch']:
                logger.info(f"  分支: {commit_info['branch']}")
            logger.info(f"  消息: {commit_info['message'][:100]}{'...' if len(commit_info['message']) > 100 else ''}")
            
            return commit_info
        else:
            logger.warning(f"提交信息格式不正确，期望至少5个字段，实际获得 {len(commit_info_lines)} 个字段")
            logger.debug(f"提交信息内容: {commit_info_lines}")
            return None
            
    except subprocess.CalledProcessError as e:
        logger.error(f"执行git命令失败: {e}")
        if e.stderr:
            logger.error(f"错误输出: {e.stderr.decode('utf-8', errors='replace')}")
        return None
    except Exception as e:
        logger.error(f"获取commit信息失败: {e}")
        return None

def get_commit_branch(repo_dir: str, commit_sha: str) -> str:
    """
    获取commit所在的分支信息
    
    Args:
        repo_dir: 仓库目录
        commit_sha: commit的SHA
        
    Returns:
        str: 分支名称，如果无法获取则返回空字符串
    """
    try:
        # 方法1: 使用git branch --contains查找包含该commit的分支
        result = subprocess.run(
            ['git', 'branch', '--contains', commit_sha, '--format=%(refname:short)'],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True
        )
        
        branches = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        
        if branches:
            # 优先返回main或master分支
            for branch in branches:
                if branch in ['main', 'master']:
                    logger.debug(f"找到主分支: {branch}")
                    return branch
            
            # 如果没有主分支，返回第一个分支
            logger.debug(f"返回第一个分支: {branches[0]}")
            return branches[0]
        
        # 方法2: 如果方法1失败，尝试使用git name-rev
        result = subprocess.run(
            ['git', 'name-rev', '--name-only', commit_sha],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True
        )
        
        branch_name = result.stdout.strip()
        if branch_name and branch_name != 'undefined':
            logger.debug(f"使用name-rev找到分支: {branch_name}")
            return branch_name
        
        # 方法3: 检查是否是HEAD
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.stdout.strip() == commit_sha:
            # 获取当前分支名
            result = subprocess.run(
                ['git', 'branch', '--show-current'],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                check=True
            )
            
            current_branch = result.stdout.strip()
            if current_branch:
                logger.debug(f"当前HEAD分支: {current_branch}")
                return current_branch
        
        logger.debug(f"无法确定commit {commit_sha} 的分支信息")
        return ""
        
    except subprocess.CalledProcessError as e:
        logger.debug(f"获取分支信息失败: {e}")
        return ""
    except Exception as e:
        logger.debug(f"获取分支信息时发生错误: {e}")
        return ""

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="统一CI失败修复分析器")
    parser.add_argument('project', help='项目名称 (例如: git, electron, llvm)')
    parser.add_argument('--test-mode', action='store_true', 
                       help='测试模式：只处理少量记录')
    parser.add_argument('--restart', action='store_true',
                       help='重新开始：删除现有结果文件，从头开始分析（默认是继续分析）')
    parser.add_argument('--limit', type=int, default=None,
                       help='限制处理的记录数量 (例如: --limit 50 只处理前50个记录)')
    parser.add_argument('--skip-update', action='store_true',
                        help='仓库已存在时跳过远端更新与PR引用抓取，降低启动耗时')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.project)
    
    logger.info(f"开始分析 {args.project.upper()} 项目的编译错误修复...")
    logger.info("运行模式：只提取编译相关文件路径（跳过LLM分析）")
    
    try:
        # 创建项目配置
        config = ProjectConfig(args.project)
        
        # 设置全局仓库名称
        set_global_repo_name(config.repo_name)
        
        # 设置git仓库
        setup_git_repo(config, skip_update=args.skip_update)
        
        # 加载失败记录
        failure_records = load_failure_records(config)
        if not failure_records:
            logger.error("没有找到失败记录")
            return
        
        # 按时间排序，从最新开始分析
        failure_records.sort(key=lambda x: x.created_at, reverse=True)
        
        # 测试模式：只处理前10个记录
        if args.test_mode:
            failure_records = failure_records[:10]
            logger.info(f"测试模式：只处理前 {len(failure_records)} 个记录")
        
        # 限制处理记录数量
        if args.limit is not None:
            original_count = len(failure_records)
            failure_records = failure_records[:args.limit]
            logger.info(f"限制模式：只处理前 {len(failure_records)} 个记录（总共 {original_count} 个记录）")
        
        # 分析修复提交（默认继续分析，只有加--restart才重新开始）
        analyze_repairs(config, failure_records, restart=args.restart)
        
        logger.info(f"{args.project.upper()} 项目分析完成")
        
    except Exception as e:
        logger.error(f"分析过程中出错: {e}")
        return

if __name__ == "__main__":
    main() 