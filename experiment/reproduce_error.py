#!/usr/bin/env python3
"""
Compilation error reproduction script

Used to reproduce compilation errors in projects by running GitHub Actions workflows locally using the act tool.

Usage:
    python reproduce_error.py <project_name> <failure_commit> <job_name> <workflow_name> [options]

Parameters:
    project_name: Project name (e.g., llvm)
    failure_commit: Failed commit SHA
    job_name: Job name (e.g., "premerge-checks-linux")
    workflow_name: Workflow name (e.g., "CI Checks")

Options:
    --dry-run: Only show operations to be performed, do not actually run
    --force-rebuild: Force rebuild Docker image even if it already exists

Examples:
    python reproduce_error.py llvm 4f8acb6898ca282321d688b960ca02f8e80bad26 "premerge-checks-linux" "CI Checks"
    python reproduce_error.py llvm 4f8acb6898ca282321d688b960ca02f8e80bad26 "premerge-checks-linux" "CI Checks" --force-rebuild
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import yaml
import time
import re
from itertools import product
from difflib import SequenceMatcher

def string_similarity(a, b):
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

# ====== Configuration Section ======
# ACT tool path configuration
script_dir = Path(__file__).parent.absolute()
possible_act_paths = [
    "act",  # act in PATH (prefer global installation)
    Path("/usr/local/bin/act"),  # System installation path
    Path("/usr/bin/act"),  # System installation path
    script_dir / "bin/act",  # bin/act in experiment directory (backup)
]

ACT_PATH = None
for act_path in possible_act_paths:
    if isinstance(act_path, str):
        # Check if act is in PATH
        try:
            result = subprocess.run(["which", "act"], capture_output=True, text=True)
            if result.returncode == 0:
                ACT_PATH = "act"
                break
        except:
            continue
    elif act_path.exists():
        ACT_PATH = str(act_path)
        break

if not ACT_PATH:
    print("Warning: act tool not found, please ensure act is installed")
    print("Installation command: curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash")
    ACT_PATH = "act"  # Use default value to make runtime error clearer

print(f"Using ACT path: {ACT_PATH}")

# Non-Ubuntu runtime environments to skip
NON_UBUNTU_RUNNERS = {
    # Windows runtime environment
    "windows-latest",
    "windows-2022",
    "windows-2019",
    "windows-2016",
    "windows-11",
    "windows-10",
    
    # macOS runtime environment
    "macos-latest",
    "macos-14",
    "macos-13",
    "macos-12",
    "macos-11",
    "macos-10.15",

}

# ====== Script Main Body ======

def load_project_config(config_file="act_project_configs.json"):
    """Load project configuration"""
    # Try to load configuration file from project root directory
    config_path = Path(__file__).parent / config_file
    
    try:
        print(f"Loading configuration file: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Convert relative paths to absolute paths
        for project_name, project_config in config.items():
            if project_name != "default" and isinstance(project_config, dict) and 'repo_path' in project_config:
                repo_path = project_config['repo_path']
                if not os.path.isabs(repo_path):
                    # Convert relative path to absolute path
                    absolute_repo_path = (Path(__file__).parent / repo_path).resolve()
                    project_config['repo_path'] = str(absolute_repo_path)
                    print(f"Updated {project_name} repo_path to: {absolute_repo_path}")
        
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file does not exist: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Configuration file format error: {config_path}: {e}")
        sys.exit(1)


def find_workflow_file(repo_path, workflow_name):
    """Find workflow file"""
    print(f"Finding workflow file: {workflow_name}")
    workflow_dir = repo_path / '.github' / 'workflows'
    
    if not workflow_dir.exists():
        print(f"Error: Workflow directory does not exist: {workflow_dir}")
        return None
    
    # Traverse workflow directory to find matching files
    for file in workflow_dir.glob('*.y*ml'):
        try:
            with open(file) as f:
                workflow = yaml.safe_load(f)
                if workflow.get('name') == workflow_name:
                    print(f"✅ Found workflow file: {file}")
                    return file
        except Exception as e:
            print(f"Warning: Error parsing {file}: {e}")
            continue
    
    print(f"Error: Workflow file named {workflow_name} not found")
    return None


def find_job_id(repo_path, workflow_file, display_name):
    """Use act -l to find real job ID and event based on display name"""
    print(f"\n🔍 Finding job ID: {display_name} using 'act -l'")
    
    act_cmd = [ACT_PATH, "-l", "-W", str(workflow_file)]
    
    try:
        process = subprocess.run(
            act_cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(repo_path),
            env=os.environ.copy()
        )
        
        output = process.stdout
        print("--- `act -l` output ---")
        print(output)
        print("-----------------------")

        lines = output.strip().split('\n')
        job_lines = [line for line in lines if line.strip() and not line.strip().startswith("Stage")]

        if not job_lines:
            print(f"❌ 'act -l' did not return any jobs for workflow {workflow_file}.")
            return None, None
        
        parsed_jobs = {}
        for line in job_lines:
            parts = re.split(r'\s{2,}', line.strip())
            if len(parts) < 3:
                continue
            
            job_id, job_name = parts[1], parts[2]
            
            if job_id not in parsed_jobs:
                parsed_jobs[job_id] = {'id': job_id, 'name': job_name, 'event': None}
            else:
                # Prefer longer names, as short ones might be truncated
                if len(job_name) > len(parsed_jobs[job_id]['name']):
                    parsed_jobs[job_id]['name'] = job_name

            if len(parts) >= 6 and parts[5]:
                # A simple check to see if it looks like an event name
                event_candidate = parts[-1]
                if event_candidate and not any(c in event_candidate for c in ' /\\.'):
                    parsed_jobs[job_id]['event'] = event_candidate

        found_jobs = list(parsed_jobs.values())
        
        variable_jobs = []
        best_match = None
        best_similarity = 0
        similarity_threshold = 0.6  # Similarity threshold

        # Preprocess target name - remove bracket content for better matching
        base_display_name = display_name
        display_name_no_brackets = re.sub(r'\s*\([^)]*\)', '', display_name)

        # If job name contains " / ", use the first part for matching
        # This also becomes our main search target
        search_target = display_name_no_brackets

        # Special handling for when in subworkflow context (test_php.yml)
        current_workflow = str(workflow_file)
        if " / " in display_name_no_brackets:
            search_target = display_name_no_brackets.split(" / ")[0].strip()
            print(f"ℹ️ Job name contains ' / ', using base name '{search_target}' for matching")

        for job_info in found_jobs:
            job_id, job_name = job_info['id'], job_info['name']
            job_name_no_brackets = re.sub(r'\s*\([^)]*\)', '', job_name)

            # Check exact match: compare ID and Name
            if job_id == search_target or job_name_no_brackets == search_target:
                print(f"✅ Found exact match by ID or Name: {job_id}")
                return job_id, job_info['event']
            
            # Also check original display_name (in case "a / b" is the real name)
            if job_id == display_name or job_name == display_name:
                 print(f"✅ Found exact match by original display name or ID: {job_id}")
                 return job_id, job_info['event']

            # If job name contains variable expressions
            if '${{' in job_name:
                variable_jobs.append(job_info)
                continue

            # Calculate similarity - use search_target as target
            name_similarity = string_similarity(job_name_no_brackets, search_target)
            id_similarity = string_similarity(job_id, search_target)
            max_similarity = max(name_similarity, id_similarity)
            
            if max_similarity > best_similarity:
                best_similarity = max_similarity
                best_match = job_info

        # If there are jobs with variables, prioritize parsing matrix variables
        if variable_jobs:
            print(f"⚠️ Found jobs with variables. Trying to expand matrix for '{display_name}'...")
            try:
                with open(workflow_file, 'r') as f:
                    workflow_content = yaml.safe_load(f)
                
                jobs_def = workflow_content.get('jobs', {})
                
                for var_job in variable_jobs:
                    job_def = jobs_def.get(var_job['id'])
                    if not job_def or 'strategy' not in job_def or 'matrix' not in job_def['strategy']:
                        continue

                    name_template = job_def.get('name', var_job['name'])
                    matrix_def = job_def['strategy']['matrix']

                    matrix_combinations = []
                    base_matrix_vars = {k: v for k, v in matrix_def.items() if k not in ['include', 'exclude']}
                    
                    if base_matrix_vars:
                        keys = list(base_matrix_vars.keys())
                        values = [v if isinstance(v, list) else [v] for v in base_matrix_vars.values()]
                        for instance in product(*values):
                            matrix_combinations.append(dict(zip(keys, instance)))
                    
                    if 'include' in matrix_def:
                        includes = matrix_def['include']
                        if not matrix_combinations:
                            matrix_combinations.extend(includes)
                        else:
                            # This simplified logic just adds includes as new combos
                            matrix_combinations.extend(includes)

                    def get_value_from_path(path, context):
                        parts = path.split('.')
                        val = context
                        for part in parts:
                            if isinstance(val, dict) and part in val:
                                val = val[part]
                            else:
                                return None
                        return val

                    best_matrix_match = None
                    best_matrix_similarity = 0

                    for combo in matrix_combinations:
                        context = {'matrix': combo}
                        generated_name = name_template
                        
                        # Replace all variable expressions
                        for var_match in re.finditer(r'\$\{\{([^}]+)\}\}', name_template):
                            var_expr = var_match.group(1).strip()
                            value = None
                            
                            # Handle conditional expressions
                            if '&&' in var_expr and '||' in var_expr:
                                try:
                                    condition_var, true_val, false_val = re.split(r'\s*&&\s*|\s*\|\|\s*', var_expr)
                                    if get_value_from_path(condition_var, context):
                                        value = true_val.strip("'\"")
                                    else:
                                        value = false_val.strip("'\"")
                                except Exception:
                                    pass
                            
                            if value is None:
                                value = get_value_from_path(var_expr, context)

                            if value is not None:
                                generated_name = generated_name.replace(var_match.group(0), str(value))

                        # Improved matrix matching algorithm
                        generated_name_no_brackets = re.sub(r'\s*\([^)]*\)', '', generated_name)
                        
                        # 1. Exact match check
                        if generated_name == display_name or generated_name_no_brackets == display_name_no_brackets:
                            print(f"✅ Found exact matrix match: {generated_name}")
                            return var_job['id'], var_job['event']
                        
                        # 2. Keyword matching - especially for cases like "Linux CMake C++20"
                        display_keywords = set(display_name.lower().split())
                        generated_keywords = set(generated_name.lower().split())
                        
                        # Calculate keyword matching score
                        common_keywords = display_keywords & generated_keywords
                        keyword_similarity = len(common_keywords) / max(len(display_keywords), len(generated_keywords)) if max(len(display_keywords), len(generated_keywords)) > 0 else 0
                        
                        # 3. String similarity
                        string_sim = max(
                            string_similarity(generated_name, display_name),
                            string_similarity(generated_name_no_brackets, display_name_no_brackets)
                        )
                        
                        # 4. Priority check - if generated name contains target keywords, give extra points
                        priority_bonus = 0
                        target_keywords = ['c++20', 'ninja', 'shared', 'package', 'fetch']
                        
                        # Check if target keywords are included
                        for target in target_keywords:
                            if target in generated_name.lower():
                                priority_bonus = 0.3
                                break
                        
                        # Additional check: if target name contains "Linux" but generated name does not, apply penalty
                        platform_penalty = 0
                        if 'linux' in display_name.lower() and 'linux' not in generated_name.lower():
                            platform_penalty = -0.5
                        elif 'windows' in display_name.lower() and 'windows' not in generated_name.lower():
                            platform_penalty = -0.5
                        elif 'macos' in display_name.lower() and 'macos' not in generated_name.lower():
                            platform_penalty = -0.5
                        
                        # 5. Comprehensive scoring - keyword matching has higher weight, plus priority bonus and platform penalty
                        matrix_similarity = (keyword_similarity * 0.7) + (string_sim * 0.3) + priority_bonus + platform_penalty
                        
                        print(f"  Comparing matrix combination: '{generated_name}' vs '{display_name}'")
                        print(f"    Keyword matching: {keyword_similarity:.2f} (common keywords: {common_keywords})")
                        print(f"    String similarity: {string_sim:.2f}")
                        print(f"    Priority bonus: {priority_bonus:.2f}")
                        print(f"    Platform penalty: {platform_penalty:.2f}")
                        print(f"    Comprehensive score: {matrix_similarity:.2f}")
                        
                        if matrix_similarity > best_matrix_similarity:
                            best_matrix_similarity = matrix_similarity
                            best_matrix_match = {'name': generated_name, 'id': var_job['id']}

                    # If found sufficiently similar matrix job
                    if best_matrix_match and best_matrix_similarity >= similarity_threshold:
                        print(f"✅ Found similar matrix job (similarity: {best_matrix_similarity:.2f})")
                        print(f"  - Original name: {display_name}")
                        print(f"  - Matched job: {best_matrix_match['name']} (ID: {best_matrix_match['id']})")
                        return best_matrix_match['id'], var_job['event']

            except Exception as e:
                print(f"❌ Error while trying to expand matrix variables: {e}")
            
            # If there is only one variable job and no match found, fall back to that job
            if len(variable_jobs) == 1:
                job_id = variable_jobs[0]['id']
                event = variable_jobs[0]['event']
                print(f"⚠️ Could not definitively match matrix job, but only one candidate exists. Falling back to job ID: {job_id}")
                return job_id, event

        # If found sufficiently similar match (only considered when there are no variable jobs)
        if best_match and best_similarity >= similarity_threshold:
            print(f"✅ Found similar job (similarity: {best_similarity:.2f})")
            print(f"  - Original name: {display_name}")
            print(f"  - Matched job: {best_match['name']} (ID: {best_match['id']})")
            return best_match['id'], best_match['event']

        print(f"❌ Could not find a job with display name or ID '{display_name}' in workflow '{workflow_file}'")
        print("Available jobs in this workflow:")
        for job in found_jobs:
            print(f"  - ID: {job['id']}, Name: {job['name']}")

        return None, None

    except subprocess.CalledProcessError as e:
        print(f"Error: 'act -l' execution failed: {e}")
        print(f"Stderr: {e.stderr}")
        return None, None
    except Exception as e:
        print(f"Error: Exception occurred while parsing 'act -l' output: {e}")
        return None, None


def check_docker_image_exists(image_name):
    """Check if Docker image exists"""
    try:
        check_image_cmd = ["docker", "images", image_name, "--format", "{{.Repository}}:{{.Tag}}"]
        image_check_result = subprocess.run(check_image_cmd, capture_output=True, text=True)
        
        if image_check_result.returncode == 0 and image_check_result.stdout.strip():
            return True
        return False
    except Exception as e:
        print(f"Warning: Error checking image: {e}")
        return False


def build_project_image(project_name, pre_build_command, base_image="catthehacker/ubuntu:act-latest", force_rebuild=False):
    """Build project-specific Docker image"""
    image_name = f"{project_name}-build-env:latest"
    
    # If force rebuild or image already exists, delete old image first
    if force_rebuild and check_docker_image_exists(image_name):
        print(f"🗑️ Deleting old {project_name} build image: {image_name}")
        try:
            delete_result = subprocess.run(
                ["docker", "rmi", "-f", image_name],
                capture_output=True,
                text=True
            )
            if delete_result.returncode == 0:
                print(f"✅ Successfully deleted old image: {image_name}")
            else:
                print(f"⚠️ Failed to delete old image: {delete_result.stderr}")
        except Exception as e:
            print(f"⚠️ Exception occurred while deleting old image: {e}")
    
    # Check if image already exists (check again after deletion)
    if not force_rebuild and check_docker_image_exists(image_name):
        print(f"✅ {project_name} build image already exists, skipping build: {image_name}")
        return True
    
    print(f"🔧 Starting to build {project_name} build image...")
    print(f"Executing pre-build command: {pre_build_command}")
    
    try:
        # First start a temporary container to execute pre-build command
        temp_container_cmd = [
            "docker", "run", "-d", "--rm",
            "-e", "DEBIAN_FRONTEND=noninteractive",
            base_image,
            "tail", "-f", "/dev/null"
        ]
        
        print(f"Starting temporary container to execute pre-build command...")
        container_result = subprocess.run(temp_container_cmd, capture_output=True, text=True)
        if container_result.returncode == 0:
            container_id = container_result.stdout.strip()
            print(f"Temporary container ID: {container_id}")
            
            # Execute pre-build command in container
            exec_result = subprocess.run([
                "docker", "exec",  "-u", "0", container_id,
                "bash", "-c", pre_build_command
            ], capture_output=True, text=True)
            
            if exec_result.returncode == 0:
                print("✅ Pre-build command executed successfully in container")
                # Commit container as new image
                commit_cmd = ["docker", "commit", container_id, image_name]
                commit_result = subprocess.run(commit_cmd, capture_output=True, text=True)
                if commit_result.returncode == 0:
                    print(f"✅ Successfully created {project_name} build image: {image_name}")
                    return True
                else:
                    print(f"⚠️ Failed to commit image: {commit_result.stderr}")
            else:
                print(f"⚠️ Pre-build command execution failed in container: {exec_result.stderr}")
                if exec_result.stdout:
                    print(f"Command output: {exec_result.stdout}")
            
            # Clean up temporary container
            subprocess.run(["docker", "stop", container_id], capture_output=True)
        else:
            print(f"⚠️ Failed to start temporary container: {container_result.stderr}")
        
        return False
            
    except Exception as e:
        print(f"⚠️ Exception occurred while building image: {e}")
        return False


def has_job_dependencies(workflow_file, job_id):
    """Check if job has needs dependencies"""
    try:
        with open(workflow_file, 'r') as f:
            workflow_content = yaml.safe_load(f)
            
        jobs_def = workflow_content.get('jobs', {})
        job_def = jobs_def.get(job_id, {})
        
        # Check if there are needs dependencies
        if 'needs' in job_def:
            needs = job_def['needs']
            if isinstance(needs, list) and needs:
                print(f"⚠️ Job '{job_id}' has the following dependencies: {needs}")
                return True
            elif isinstance(needs, str) and needs:
                print(f"⚠️ Job '{job_id}' has dependencies: {needs}")
                return True
        
        return False
    except Exception as e:
        print(f"Warning: Error checking job dependencies: {e}")
        return False


def parse_compiler_from_job_name(job_name):
    """Parse compiler information from job name"""
    print(f"🔍 Parsing compiler information from job name: {job_name}")
    
    # Common compiler patterns
    compiler_patterns = [
        # systemd format: "build (gcc, 13, mold)"
        r'build\s*\(\s*(gcc|clang)\s*,\s*(\d+)\s*,\s*(\w+)\s*\)',
        # Other format: "build (gcc-13)"
        r'build\s*\(\s*(gcc|clang)-(\d+)\s*\)',
        # Simple format: "gcc-13"
        r'(gcc|clang)-(\d+)',
        # With linker format: "gcc-13-mold"
        r'(gcc|clang)-(\d+)-(\w+)',
    ]
    
    for pattern in compiler_patterns:
        match = re.search(pattern, job_name, re.IGNORECASE)
        if match:
            if len(match.groups()) == 3:
                # Format: "build (gcc, 13, mold)" or "gcc-13-mold"
                compiler = match.group(1).lower()
                version = match.group(2)
                linker = match.group(3).lower()
                print(f"✅ Parsed compiler: {compiler} {version}, linker: {linker}")
                return {
                    'COMPILER': compiler,
                    'COMPILER_VERSION': version,
                    'LINKER': linker
                }
            elif len(match.groups()) == 2:
                # Format: "build (gcc-13)" or "gcc-13"
                compiler = match.group(1).lower()
                version = match.group(2)
                # Default linker
                linker = 'bfd' if compiler == 'gcc' else 'lld'
                print(f"✅ Parsed compiler: {compiler} {version}, default linker: {linker}")
                return {
                    'COMPILER': compiler,
                    'COMPILER_VERSION': version,
                    'LINKER': linker
                }
    
    print(f"⚠️ Unable to parse compiler information from job name: {job_name}")
    return {}


def extract_best_matrix_combo(workflow_file, job_id, display_name):
    """Extract the best matching matrix combination from workflow file"""
    try:
        with open(workflow_file, 'r') as f:
            workflow_content = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Error reading workflow file: {e}")
        return None

    jobs_def = workflow_content.get('jobs', {})
    job_def = jobs_def.get(job_id)

    if not job_def:
        return None

    # Check if there is a matrix strategy
    if 'strategy' not in job_def or 'matrix' not in job_def['strategy']:
        return None

    matrix_def = job_def['strategy']['matrix']
    
    # Build all matrix combinations
    matrix_combinations = []
    
    # Process standard matrix variables
    base_matrix_vars = {k: v for k, v in matrix_def.items() if k not in ['include', 'exclude']}
    
    if base_matrix_vars:
        keys = list(base_matrix_vars.keys())
        values = [v if isinstance(v, list) else [v] for v in base_matrix_vars.values()]
        for instance in product(*values):
            matrix_combinations.append(dict(zip(keys, instance)))
    
    # Process combinations in include
    if 'include' in matrix_def:
        includes = matrix_def['include']
        if not matrix_combinations:
            matrix_combinations.extend(includes)
        else:
            matrix_combinations.extend(includes)

    print(f"🔍 Found {len(matrix_combinations)} matrix combinations:")
    for i, combo in enumerate(matrix_combinations):
        print(f"  {i+1}. {combo}")

    # Try to match the correct matrix combination based on display_name
    best_combo = None
    best_similarity = 0
    
    # Extract keywords from display_name
    display_keywords = display_name.lower().split()
    print(f"🔍 Extracted keywords from display_name: {display_keywords}")
    
    for combo in matrix_combinations:
        # Handle nested matrix structure, such as {'build': {'name': 'libressl heimdal valgrind', ...}}
        combo_name = None
        if isinstance(combo, dict):
            # Find sub-dictionary containing name field
            for key, value in combo.items():
                if isinstance(value, dict) and 'name' in value:
                    combo_name = value['name'].lower()
                    break
            # If no nested name found, check current level
            if not combo_name and 'name' in combo:
                combo_name = combo['name'].lower()
        
        if combo_name:
            combo_keywords = combo_name.split()
            
            # Improved matching algorithm
            # 1. Exact match check
            if combo_name == display_name.lower():
                print(f"✅ Found exact match: '{combo_name}'")
                return combo
            
            # 2. Keyword matching
            common_keywords = set(display_keywords) & set(combo_keywords)
            keyword_similarity = len(common_keywords) / max(len(display_keywords), len(combo_keywords)) if max(len(display_keywords), len(combo_keywords)) > 0 else 0
            
            # 3. String similarity
            string_similarity_score = string_similarity(combo_name, display_name.lower())
            
            # 4. Comprehensive scoring - keyword matching has higher weight
            similarity = (keyword_similarity * 0.7) + (string_similarity_score * 0.3)
            
            print(f"  Comparing '{combo_name}' with '{display_name.lower()}'")
            print(f"    Keyword matching: {keyword_similarity:.2f} (common keywords: {common_keywords})")
            print(f"    String similarity: {string_similarity_score:.2f}")
            print(f"    Comprehensive score: {similarity:.2f}")
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_combo = combo
        else:
            
            # Generic matrix combination inference mechanism
            inferred_name = infer_matrix_combo_name(combo, display_name)
            if inferred_name:
                print(f"    🔍 Inferred name from matrix: {inferred_name}")
                
                # Calculate similarity with display_name
                combo_keywords = inferred_name.lower().split()
                common_keywords = set(display_keywords) & set(combo_keywords)
                keyword_similarity = len(common_keywords) / max(len(display_keywords), len(combo_keywords)) if max(len(display_keywords), len(combo_keywords)) > 0 else 0
                string_similarity_score = string_similarity(inferred_name.lower(), display_name.lower())
                similarity = (keyword_similarity * 0.7) + (string_similarity_score * 0.3)
                
                print(f"    Keyword matching: {keyword_similarity:.2f} (common keywords: {common_keywords})")
                print(f"    String similarity: {string_similarity_score:.2f}")
                print(f"    Comprehensive score: {similarity:.2f}")
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_combo = combo
                    # Add inferred name for this combination
                    best_combo['inferred_name'] = inferred_name
                continue
            
            # Try to infer name from flags field (keep original logic)
            if 'flags' in combo:
                flags = combo['flags']
                # Check if flags contains specific keywords
                flags_lower = flags.lower()
                inferred_name = None
                
                if 'c++20' in flags_lower or 'cxx_standard=20' in flags_lower:
                    inferred_name = 'c++20'
                elif 'ninja' in flags_lower:
                    inferred_name = 'ninja'
                elif 'shared_libs=on' in flags_lower:
                    inferred_name = 'shared'
                elif 'local_dependencies_only=on' in flags_lower:
                    inferred_name = 'package'
                elif 'force_fetch_dependencies=on' in flags_lower:
                    inferred_name = 'fetch'
                
                if inferred_name:
                    print(f"    🔍 Inferred name from flags: {inferred_name}")
                    combo_keywords = [inferred_name]
                    common_keywords = set(display_keywords) & set(combo_keywords)
                    keyword_similarity = len(common_keywords) / max(len(display_keywords), len(combo_keywords)) if max(len(display_keywords), len(combo_keywords)) > 0 else 0
                    string_similarity_score = string_similarity(inferred_name, display_name.lower())
                    similarity = (keyword_similarity * 0.7) + (string_similarity_score * 0.3)
                    
                    print(f"    Keyword matching: {keyword_similarity:.2f} (common keywords: {common_keywords})")
                    print(f"    String similarity: {string_similarity_score:.2f}")
                    print(f"    Comprehensive score: {similarity:.2f}")
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_combo = combo
                        # Add inferred name for this combination
                        best_combo['inferred_name'] = inferred_name
    
    if best_combo and best_similarity > 0.5:
        # Get the name of the matrix combination
        combo_name = None
        if isinstance(best_combo, dict):
            if 'name' in best_combo:
                combo_name = best_combo['name']
            elif 'inferred_name' in best_combo:
                combo_name = best_combo['inferred_name']
            else:
                # Find nested name
                for key, value in best_combo.items():
                    if isinstance(value, dict) and 'name' in value:
                        combo_name = value['name']
                        break
        
        print(f"🎯 Found best matching matrix combination: {combo_name} (similarity: {best_similarity:.2f})")
        return best_combo
    elif not best_combo:
        print(f"⚠️ No matching matrix combination found, using first combination")
        return matrix_combinations[0] if matrix_combinations else None
    else:
        print(f"⚠️ Best match similarity too low ({best_similarity:.2f}), using first combination")
        return matrix_combinations[0] if matrix_combinations else None


def build_act_matrix_args(matrix_combo):
    """Convert matrix combination to act --matrix parameter format"""
    matrix_args = []
    
    if not matrix_combo:
        return matrix_args
    # Flatten matrix combination, build key-value pairs
    flattened = flatten_matrix_combo_for_act(matrix_combo)
    
    for key, value in flattened.items():
        # Special handling for relabel field
        if key == "relabel":
            if value in [True, "True", "true", 1]:
                value = "yes"
            elif value in [False, "False", "false", 0]:
                value = "no"
            # Other cases convert directly to string
            else:
                value = str(value)
        matrix_args.extend(['--matrix', f'{key}:{value}'])
    
    return matrix_args


def flatten_matrix_combo_for_act(combo, prefix=''):
    """Flatten matrix combination for act command"""
    result = {}
    
    for key, value in combo.items():
        # Skip internally used fields, such as inferred_name
        if key == "inferred_name":
            continue
            
        full_key = f"{prefix}.{key}" if prefix else key
        # Special handling for relabel field
        if key == "relabel":
            if value in [True, "True", "true", 1]:
                result[full_key] = "yes"
            elif value in [False, "False", "false", 0]:
                result[full_key] = "no"
            else:
                result[full_key] = str(value)
        elif isinstance(value, dict):
            # Recursively handle nested dictionaries
            result.update(flatten_matrix_combo_for_act(value, full_key))
        elif isinstance(value, (str, int, float, bool)):
            # Keep simple type values
            result[full_key] = str(value)
        elif isinstance(value, list):
            # Convert list to JSON string
            import json
            result[full_key] = json.dumps(value, separators=(',', ':'))
        else:
            # For other complex types, convert to string
            result[full_key] = str(value)
    
    return result


def extract_simple_matrix_env_vars(matrix_combo):
    """Extract simple environment variables from matrix combination"""
    env_vars = {}
    
    if not matrix_combo:
        return env_vars
    
    # Flatten and extract simple environment variables
    flattened = flatten_matrix_combo(matrix_combo)
    for key, value in flattened.items():
        if isinstance(value, (str, int, float, bool)):
            env_key = key.upper().replace('-', '_').replace('.', '_')
            # Only keep key environment variables to avoid too much noise
            if any(keyword in env_key.lower() for keyword in ['name', 'build', 'install', 'configure', 'generate']):
                env_vars[env_key] = str(value)
    
    return env_vars


def extract_matrix_env_vars(workflow_file, job_id, display_name):
    """Extract matrix environment variables from workflow file"""
    try:
        with open(workflow_file, 'r') as f:
            workflow_content = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Error reading workflow file: {e}")
        return {}

    jobs_def = workflow_content.get('jobs', {})
    job_def = jobs_def.get(job_id)

    if not job_def:
        return {}

    # Check if there is a matrix strategy
    if 'strategy' not in job_def or 'matrix' not in job_def['strategy']:
        return {}

    matrix_def = job_def['strategy']['matrix']
    
    # Build all matrix combinations
    matrix_combinations = []
    
    # Process standard matrix variables
    base_matrix_vars = {k: v for k, v in matrix_def.items() if k not in ['include', 'exclude']}
    
    if base_matrix_vars:
        keys = list(base_matrix_vars.keys())
        values = [v if isinstance(v, list) else [v] for v in base_matrix_vars.values()]
        for instance in product(*values):
            matrix_combinations.append(dict(zip(keys, instance)))
    
    # Process combinations in include
    if 'include' in matrix_def:
        includes = matrix_def['include']
        if not matrix_combinations:
            matrix_combinations.extend(includes)
        else:
            matrix_combinations.extend(includes)

    # Try to match the correct matrix combination based on display_name
    best_combo = None
    best_similarity = 0
    
    # Extract keywords from display_name
    display_keywords = display_name.lower().split()
    print(f"🔍 Extracted keywords from display_name: {display_keywords}")
    
    for combo in matrix_combinations:
        # Handle nested matrix structure, such as {'build': {'name': 'libressl heimdal valgrind', ...}}
        combo_name = None
        if isinstance(combo, dict):
            # Find sub-dictionary containing name field
            for key, value in combo.items():
                if isinstance(value, dict) and 'name' in value:
                    combo_name = value['name'].lower()
                    break
            # If no nested name found, check current level
            if not combo_name and 'name' in combo:
                combo_name = combo['name'].lower()
        
        if combo_name:
            combo_keywords = combo_name.split()
            
            # Calculate similarity
            common_keywords = set(display_keywords) & set(combo_keywords)
            similarity = len(common_keywords) / max(len(display_keywords), len(combo_keywords))
            
            print(f"  Comparing '{combo_name}' with '{display_name.lower()}' - similarity: {similarity:.2f}")
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_combo = combo
    
    if best_combo and best_similarity > 0.5:
        # Get the name of the matrix combination
        combo_name = None
        if isinstance(best_combo, dict):
            for key, value in best_combo.items():
                if isinstance(value, dict) and 'name' in value:
                    combo_name = value['name']
                    break
            if not combo_name and 'name' in best_combo:
                combo_name = best_combo['name']
        
        print(f"🎯 Found best matching matrix combination: {combo_name} (similarity: {best_similarity:.2f})")
    elif not best_combo:
        print(f"⚠️ No matching matrix combination found, using first combination")
        best_combo = matrix_combinations[0] if matrix_combinations else None

    # If matching combination found, extract environment variables
    if best_combo:
        print(f"🎯 Using matrix combination: {best_combo}")
        
        # Extract environment variables, prioritize job-level env configuration
        env_vars = {}
        context = {'matrix': best_combo}
        
        # Handle job-level env configuration
        if 'env' in job_def:
            env_config = job_def['env']
            if isinstance(env_config, dict):
                for env_key, env_value in env_config.items():
                    resolved_value = resolve_template_expression(env_value, context)
                    env_vars[env_key] = resolved_value
        
        # Handle environment variables in matrix combination
        # Flatten nested structure, extract useful environment variables
        flattened_vars = flatten_matrix_combo(best_combo)
        for key, value in flattened_vars.items():
            # Skip complex objects, only handle simple key-value pairs
            if isinstance(value, (str, int, float, bool)):
                env_key = key.upper().replace('-', '_').replace('.', '_')
                env_vars[env_key] = str(value)
        
        print(f"📋 Extracted environment variables: {env_vars}")
        return env_vars
    else:
        print(f"⚠️ Unable to find matching matrix combination")
        return {}


def flatten_matrix_combo(combo, prefix=''):
    """Flatten matrix combination, convert nested structure to flat key-value pairs"""
    result = {}
    
    for key, value in combo.items():
        # Skip internally used fields, such as inferred_name
        if key == "inferred_name":
            continue
            
        full_key = f"{prefix}_{key}" if prefix else key
        
        if isinstance(value, dict):
            # Recursively handle nested dictionaries
            result.update(flatten_matrix_combo(value, full_key))
        elif isinstance(value, (str, int, float, bool)):
            # Only keep simple type values
            result[full_key] = str(value)
        # Skip lists and other complex types
    
    return result


def infer_matrix_combo_name(combo, target_name):
    """
    Generic matrix combination name inference mechanism
    Intelligently infer the most matching name based on target name patterns and matrix combination fields
    
    Args:
        combo: Matrix combination dictionary
        target_name: Target job name
        
    Returns:
        str: Inferred name, returns None if unable to infer
    """
    if not combo or not target_name:
        return None
    
    # Analyze target name pattern
    target_pattern = analyze_name_pattern(target_name)
    
    # Infer name based on pattern
    if target_pattern['type'] == 'parentheses_format':
        return infer_parentheses_format_name(combo, target_pattern)
    elif target_pattern['type'] == 'simple_format':
        return infer_simple_format_name(combo, target_pattern)
    else:
        return infer_generic_format_name(combo, target_name)


def analyze_name_pattern(name):
    """
    Analyze name pattern characteristics
    
    Args:
        name: Name to analyze
        
    Returns:
        dict: Dictionary containing pattern information
    """
    pattern = {
        'type': 'unknown',
        'prefix': '',
        'suffix': '',
        'separator': '',
        'has_parentheses': False,
        'parentheses_content': '',
        'keywords': []
    }
    
    # Check if contains parentheses
    if '(' in name and ')' in name:
        pattern['has_parentheses'] = True
        start = name.find('(')
        end = name.rfind(')')
        pattern['prefix'] = name[:start].strip()
        pattern['parentheses_content'] = name[start+1:end].strip()
        pattern['suffix'] = name[end+1:].strip()
        pattern['type'] = 'parentheses_format'
        
        # Analyze content inside parentheses
        if ',' in pattern['parentheses_content']:
            pattern['separator'] = ','
            pattern['keywords'] = [item.strip() for item in pattern['parentheses_content'].split(',')]
        else:
            pattern['keywords'] = [pattern['parentheses_content']]
    else:
        pattern['type'] = 'simple_format'
        pattern['keywords'] = name.split()
    
    return pattern


def infer_parentheses_format_name(combo, pattern):
    """
    Infer parentheses format name
    
    Args:
        combo: Matrix combination
        pattern: Name pattern
        
    Returns:
        str: Inferred name
    """
    prefix = pattern['prefix']
    keywords = pattern['keywords']
    
    # Infer format based on prefix and keyword count
    if prefix.lower() == 'build':
        return infer_build_format(combo, keywords)
    elif prefix.lower() == 'ci':
        return infer_ci_format(combo, keywords)
    else:
        return infer_generic_parentheses_format(combo, pattern)


def infer_build_format(combo, keywords):
    """Infer build format name"""
    # Check if there is env field (systemd format)
    if 'env' in combo and isinstance(combo['env'], dict):
        env_vars = combo['env']
        if 'COMPILER' in env_vars and 'COMPILER_VERSION' in env_vars and 'LINKER' in env_vars:
            return f"build ({env_vars['COMPILER']}, {env_vars['COMPILER_VERSION']}, {env_vars['LINKER']})"
    
    # Check if there are direct compiler fields
    if 'compiler' in combo and 'version' in combo:
        return f"build ({combo['compiler']}, {combo['version']})"
    
    return None


def infer_ci_format(combo, keywords):
    """Infer ci format name"""
    # Check if there are distro and release fields (mkosi format)
    if 'distro' in combo and 'release' in combo:
        distro = combo.get('distro', '')
        release = combo.get('release', '')
        llvm = combo.get('llvm', 0)
        cflags = combo.get('cflags', '')
        relabel = combo.get('relabel', False)
        vm = combo.get('vm', 0)
        
        # Handle relabel field - may be boolean or string
        if isinstance(relabel, bool):
            relabel_str = 'yes' if relabel else 'no'
        else:
            # If it's a string, use directly
            relabel_str = str(relabel)
        
        return f"ci ({distro}, {release}, {llvm}, {cflags}, {relabel_str}, {vm})"
    
    return None


def infer_generic_parentheses_format(combo, pattern):
    """Infer generic parentheses format name"""
    prefix = pattern['prefix']
    keywords = pattern['keywords']
    
    # Try to extract values matching keywords from matrix combination
    inferred_parts = []
    
    for keyword in keywords:
        # Clean keywords (remove special characters)
        clean_keyword = re.sub(r'[^\w\s]', '', keyword).strip()
        
        # Find matching fields in matrix combination
        best_match = None
        best_similarity = 0
        
        for key, value in combo.items():
            if isinstance(value, (str, int, float, bool)):
                # Convert value to string for comparison
                value_str = str(value).lower()
                similarity = string_similarity(clean_keyword.lower(), value_str)
                
                if similarity > best_similarity and similarity > 0.3:  # Set minimum similarity threshold
                    best_similarity = similarity
                    best_match = str(value)
        
        if best_match:
            inferred_parts.append(best_match)
        else:
            # If no match found, use original keyword
            inferred_parts.append(keyword)
    
    if inferred_parts:
        return f"{prefix} ({', '.join(inferred_parts)})"
    
    return None


def infer_simple_format_name(combo, pattern):
    """Infer simple format name"""
    keywords = pattern['keywords']
    
    # Try to extract matching values from matrix combination
    inferred_parts = []
    
    for keyword in keywords:
        best_match = None
        best_similarity = 0
        
        for key, value in combo.items():
            if isinstance(value, (str, int, float, bool)):
                value_str = str(value).lower()
                similarity = string_similarity(keyword.lower(), value_str)
                
                if similarity > best_similarity and similarity > 0.3:
                    best_similarity = similarity
                    best_match = str(value)
        
        if best_match:
            inferred_parts.append(best_match)
        else:
            inferred_parts.append(keyword)
    
    if inferred_parts:
        return ' '.join(inferred_parts)
    
    return None


def infer_generic_format_name(combo, target_name):
    """Infer generic format name"""
    # Flatten matrix combination
    flattened = flatten_matrix_combo(combo)
    
    # Try to build name from flattened fields
    name_parts = []
    
    # Sort by field name alphabetically to ensure consistency
    sorted_keys = sorted(flattened.keys())
    
    for key in sorted_keys:
        value = flattened[key]
        if isinstance(value, (str, int, float, bool)) and value:
            name_parts.append(str(value))
    
    if name_parts:
        return ' '.join(name_parts)
    
    return None


def resolve_template_expression(template, context):
    """Resolve template expressions, support conditional expressions"""
    if not isinstance(template, str) or '${{' not in template:
        return str(template)
    
    resolved_template = str(template)
    
    # Handle template expressions
    for var_match in re.finditer(r'\$\{\{([^}]+)\}\}', str(template)):
        var_expr = var_match.group(1).strip()
        
        # Handle conditional expressions, such as "matrix.build.generate && 'cmake' || 'autotools'"
        if '&&' in var_expr and '||' in var_expr:
            try:
                # Simple conditional expression parsing
                parts = var_expr.split('&&')
                if len(parts) == 2:
                    condition_part = parts[0].strip()
                    value_part = parts[1].strip()
                    
                    # Extract condition variable value
                    condition_value = get_value_from_path(condition_part, context)
                    
                    if '||' in value_part:
                        true_false_parts = value_part.split('||')
                        if len(true_false_parts) == 2:
                            true_val = true_false_parts[0].strip().strip("'\"")
                            false_val = true_false_parts[1].strip().strip("'\"")
                            
                            result = true_val if condition_value else false_val
                            resolved_template = resolved_template.replace(var_match.group(0), result)
                            continue
            except Exception as e:
                print(f"⚠️ Failed to parse conditional expression: {var_expr}, error: {e}")
        
        # Handle simple variable references
        value = get_value_from_path(var_expr, context)
        if value is not None:
            resolved_template = resolved_template.replace(var_match.group(0), str(value))
    
    return resolved_template


def get_value_from_path(path, context):
    """Get value corresponding to path from context"""
    parts = path.split('.')
    val = context
    for part in parts:
        if isinstance(val, dict) and part in val:
            val = val[part]
        else:
            return None
    return val


def resolve_template(template, context):
    """Resolve template variables"""
    resolved_template = str(template)
    for var_match in re.finditer(r'\$\{\{([^}]+)\}\}', str(template)):
        var_expr = var_match.group(1).strip()
        parts = var_expr.split('.')
        val = context
        for part in parts:
            if isinstance(val, dict) and part in val:
                val = val[part]
            else:
                val = None
                break
        if val is not None:
            resolved_template = resolved_template.replace(var_match.group(0), str(val))
    return resolved_template


def run_with_act(repo_path, job_name, workflow_file, failure_commit, dry_run=False, project_config=None, event=None, display_name=None, main_branch=None, force_rebuild=False, line_number=None):
    """Run workflow job using act"""
    print("\n🚀 Running workflow job using act...")
    
    # Pre-parse workflow file to get uses mapping (for subsequent logic)
    job_uses_mapping = parse_job_uses_mapping(workflow_file)
    
    # Initialize project_name variable
    project_name = project_config.get('project_name', 'default') if project_config else 'default'
    
    # Execute pre-build command (obtained from environment variables)
    if not dry_run and project_config:
        pre_build_command = None
        project_specific_config = project_config.get(project_name, {})
        additional_args = project_specific_config.get('additional_args', [])
        
        # Find PRE_BUILD_COMMAND environment variable
        for i, arg in enumerate(additional_args):
            if arg == '--env' and i + 1 < len(additional_args):
                next_arg = additional_args[i + 1]
                if next_arg.startswith('PRE_BUILD_COMMAND='):
                    pre_build_command = next_arg.split('=', 1)[1]
                    break
        
        if pre_build_command:
            # Use new build function, will automatically check if image exists
            base_image = "catthehacker/ubuntu:act-latest"
        
            if project_name == "iree":
                base_image = "ghcr.io/iree-org/cpubuilder_ubuntu_jammy_ghr_x86_64:main"

            if project_name == "electron":
                base_image = "ghcr.io/electron/build:latest"

            if project_name == "llvm":
                base_image = "ghcr.io/llvm/ci-ubuntu-24.04:latest"
            
            build_success = build_project_image(project_name, pre_build_command, base_image, force_rebuild)
            if not build_success:
                print("Will continue using original image...")
    
    # Redis project's pre-build command is now handled through PRE_BUILD_COMMAND in config file
    
    # Check if job has dependencies or uses subworkflow
    has_deps = has_job_dependencies(workflow_file, job_name)
    has_uses = job_name in job_uses_mapping

    if project_name != "iree":
        has_uses = False

    # Initialize input_args variable
    input_args = []
    
    if has_uses:
        print("Detected job uses subworkflow, trying to discover and run subworkflow...")
        
        if job_name in job_uses_mapping:
            uses_info = job_uses_mapping[job_name]
            uses_path = uses_info['uses']
            with_params = uses_info.get('with', {})
            
            # Convert with parameters to act --input parameters
            print(f"📋 Processing subworkflow input parameters: {with_params}")
            for param_key, param_value in with_params.items():
                resolved_value = str(param_value)
                
                # Add --input parameter
                input_args.extend(["--input", f"{param_key}={resolved_value}"])
            
            # Convert relative path to absolute path
            if uses_path.startswith('./'):
                sub_workflow_path = repo_path / uses_path[2:]  # Remove './'
            else:
                sub_workflow_path = repo_path / uses_path
                
            if sub_workflow_path.exists():
                print(f"✅ Found subworkflow: {sub_workflow_path}")
                
                # Switch to subworkflow and re-find job ID
                workflow_file = sub_workflow_path
                print(f"🔄 Switching to subworkflow: {workflow_file}")
                
                # If there is a display name, re-find job ID in subworkflow
                if display_name:
                    print(f"🔄 Re-finding job ID in subworkflow: {display_name}")
                    # In subworkflow, use the latter part for matching
                    search_target = display_name.split(" / ")[1].strip()
                    print(f"ℹ️ In subworkflow, job name contains ' / ', using latter part '{search_target}' for matching")
                    new_job_id, new_event = find_job_id(repo_path, workflow_file, search_target)
                    if new_job_id:
                        job_name = new_job_id
                        event = new_event
                        # Update has_uses variable since job_name has changed
                        has_uses = job_name in job_uses_mapping
                        print(f"✅ Found job ID in subworkflow: {job_name}")
                    else:
                        print(f"⚠️ Job ID not found in subworkflow, using original job name: {job_name}")
            else:
                print(f"⚠️ Subworkflow file does not exist: {sub_workflow_path}")
        else:
            print(f"⚠️ Uses statement not found for job '{job_name}' in workflow")
            
    
    # Create log directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create log file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if line_number:
        log_file_path = f"logs/act_{project_name}_line{line_number}.log"
    else:
        log_file_path = f"logs/act_{project_name}_{timestamp}.log"
    
    print(f"📝 Log file: {log_file_path}")
    
    # Add safe directory configuration
    try:
        subprocess.run(["git", "config", "--global", "--add", "safe.directory", str(repo_path)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to add safe directory configuration: {e}")
    

    
    # Set environment variables
    env = os.environ.copy()
    
    # Get project name and configuration from config
    project_name = project_config.get('project_name', 'default') if project_config else 'default'
    project_specific_config = project_config.get(project_name, {}) if project_config else {}
    default_config = project_config.get('default', {}) if project_config else {}

    # Get repository information
    try:
        # Get remote repository URL
        remote_url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=str(repo_path),
            text=True
        ).strip()
        
        # Extract owner and repo from URL
        owner = "owner"
        repo = "repo"
        
        if remote_url.endswith('.git'):
            remote_url = remote_url[:-4]
        
        if 'github.com' in remote_url:
            if remote_url.startswith('git@github.com:'):
                owner_repo = remote_url.split('git@github.com:')[1]
            else:
                owner_repo = remote_url.split('github.com/')[1]
            owner, repo = owner_repo.split('/')
        else:
            # Try to get repository information from directory name
            repo_dir = os.path.basename(repo_path)
            if repo_dir.endswith('_repo'):
                repo = repo_dir[:-5]  # Remove '_repo' suffix
                owner = project_name  # Use project name as organization name
            else:
                print(f"Warning: Unable to parse repository information from URL or directory name: {repo_path}")
    except Exception as e:
        print(f"Warning: Failed to get repository information: {e}")
        owner = "owner"
        repo = "repo"

    # Create custom event payload file for workflow_dispatch event
    event_file_path = None
    if event and 'workflow_dispatch' in str(event):
        # Create custom event payload file
        event_payload = {
            "repository": {
                "name": repo,
                "full_name": f"{owner}/{repo}",
                "default_branch": main_branch or "main"
            },
            "pull_request": {
                "head": {
                    "sha": failure_commit
                },
                "number": 1  # Add PR number to avoid undefined error
            }
        }
        
        # Create event payload file
        event_file_path = repo_path / "event.json"
        try:
            import json
            with open(event_file_path, 'w') as f:
                json.dump(event_payload, f, indent=2)
            print(f"📋 Created custom event payload file: {event_file_path}")
            print(f"📋 Event payload content: {json.dumps(event_payload, indent=2)}")
        except Exception as e:
            print(f"⚠️ Failed to create event payload file: {e}")
            event_file_path = None

    # Get token from token pool
    token_pool = default_config.get('token_pool', [])
    if token_pool:
        # Use the first available token
        github_token = token_pool[0]
    else:
        github_token = "dummy-token"
        print("Warning: No available GitHub token found")

    # Get GitHub environment variable configuration
    github_env = default_config.get('github_env', {})
    env_vars = {}
    
    # Replace placeholders in environment variables
    for key, value in github_env.items():
        try:
            env_vars[key] = value.format(
                repo_path=str(repo_path),
                commit=failure_commit,
                owner=owner,
                repo=repo,
                token=github_token,
                main_branch=main_branch
            )
        except KeyError as e:
            # If a placeholder is missing, use original value
            print(f"Warning: Unrecognized placeholder {e} in environment variable {key}")
            env_vars[key] = value

    # Extract matrix parameters for act --matrix option
    original_display_name = display_name or job_name
    matrix_combo = extract_best_matrix_combo(workflow_file, job_name, original_display_name)
    matrix_args = []
    

    
    if matrix_combo:
        print(f"🎯 Using act --matrix parameters: {matrix_combo}")
        # Convert matrix combination to act --matrix parameter format
        matrix_args = build_act_matrix_args(matrix_combo)
        if matrix_args:
            print(f"📋 Built matrix parameters: {matrix_args}")
        
        # Also extract some key environment variables for backward compatibility
        #matrix_env_vars = extract_simple_matrix_env_vars(matrix_combo)
        #if matrix_env_vars:
            #print(f"🔧 Supplementary environment variables: {matrix_env_vars}")
            #env_vars.update(matrix_env_vars)
    
    # Parse compiler information from job name
    compiler_env_vars = parse_compiler_from_job_name(original_display_name)
    if compiler_env_vars:
        print(f"🎯 Adding compiler environment variables: {compiler_env_vars}")
        env_vars.update(compiler_env_vars)
    
    # Update environment variables
    env.update(env_vars)
    env["XDG_CACHE_HOME"] = "/data/cache"
    


    # Build act command
    act_cmd = [
        ACT_PATH,
        "-W", str(workflow_file),
        "--env", "PIP_BREAK_SYSTEM_PACKAGES=1",
    ]

    # If custom event payload file is created, add --eventpath parameter
    if event_file_path and event_file_path.exists():
        act_cmd.extend(["--eventpath", str(event_file_path)])
        print(f"🎯 Adding custom event payload file: {event_file_path}")

    act_cmd.extend(["-j", job_name])
    is_subworkflow = (job_name in job_uses_mapping and str(workflow_file).endswith(job_uses_mapping[job_name]['uses'].split('/')[-1])) or has_uses or str(workflow_file).endswith('test_php.yml') or str(workflow_file).endswith('test_yaml.yml') or str(workflow_file).endswith('test_bazel.yml') or str(workflow_file).endswith('test_cpp.yml') or str(workflow_file).endswith('test_java.yml') or str(workflow_file).endswith('test_python.yml') or str(workflow_file).endswith('test_ruby.yml') or str(workflow_file).endswith('test_php_ext.yml') or str(workflow_file).endswith('test_csharp.yml') or str(workflow_file).endswith('test_objectivec.yml') or str(workflow_file).endswith('test_rust.yml') or str(workflow_file).endswith('test_upb.yml') or str(workflow_file).endswith('staleness_check.yml')
    
    if is_subworkflow:
        print("ℹ️ Running subworkflow, forcing use of workflow_call event...")
        # Subworkflow forces use of workflow_call event
        act_cmd.insert(1, "workflow_call")
        print(f"ℹ️ Subworkflow forcing use of event: workflow_call")
        # If there are matrix environment variables, they have been handled earlier
    elif event:
        # If event contains multiple events (separated by commas), select the first one
        if ',' in str(event):
            selected_event = str(event).split(',')[0].strip()
            print(f"ℹ️  Multiple events detected ({event}), using first event: '{selected_event}'")
        else:
            selected_event = str(event).strip()
            print(f"ℹ️  Running entire workflow with event '{selected_event}'")
        
        if "push" in str(event).split(','):
            selected_event = "push"
        
        act_cmd.insert(1, selected_event)
    else:
        # Try to get default event from workflow file
        default_event = get_default_event_for_workflow(workflow_file)
        print(f"⚠️  Job has dependencies, but no event type could be determined. Using default event '{default_event}'.")
        act_cmd.insert(1, default_event)

    # Add matrix parameters to act command
    if matrix_args:
        act_cmd.extend(matrix_args)
        print(f"🎯 Adding matrix parameters to act command: {' '.join(matrix_args)}")
    
    # Add input parameters to act command
    if input_args:
        act_cmd.extend(input_args)
        print(f"🎯 Adding input parameters to act command: {' '.join(input_args)}")
    
    # Add environment variables to act command
    for key, value in env_vars.items():
        act_cmd.extend(["--env", f"{key}={value}"])
    
    # Add token
    act_cmd.extend(["-s", f"GITHUB_TOKEN={github_token}"])

    # Add platform mappings from project configuration
    platform_mappings = {}
    if project_config:
        # First get mappings from default configuration
        if 'platform_mappings' in default_config:
            platform_mappings.update(default_config['platform_mappings'])
        
        # Get project-specific configuration mappings, new ones will be added, old ones will be overwritten
        if 'platform_mappings' in project_specific_config:
            platform_mappings.update(project_specific_config['platform_mappings'])
    
    # Add platform mappings to act command
    for runner, image in platform_mappings.items():
        act_cmd.extend(["-P", f"{runner}={image}"])

        # Add other configuration parameters
    if project_config:
        # Merge additional command line arguments, and prioritize project-level configuration when --network is included
        merged_additional_args = []
        if 'additional_args' in project_specific_config:
            merged_additional_args.extend(project_specific_config['additional_args'])
        if 'additional_args' in default_config:
            merged_additional_args.extend(default_config['additional_args'])

        if merged_additional_args:
            deduped_args = []
            network_seen = False
            for arg in merged_additional_args:
                # Only keep the first occurrence of --network=, since project-level parameters are added first, project-level takes priority
                if isinstance(arg, str) and arg.startswith("--network="):
                    if network_seen:
                        continue
                    network_seen = True
                deduped_args.append(arg)

            act_cmd.extend(deduped_args)
        

    if dry_run:
        tmpdir = env.get('TMPDIR', '/data/tmp')
        print(f"Simulated run command: TMPDIR={tmpdir} {' '.join(act_cmd)}")
        return True
    
    # Debug info: print complete act command
    print(f"🔧 Built act command: {' '.join(act_cmd)}")
        
    try:
        # Run act command and redirect output to log file
        with open(log_file_path, 'w') as log_file:
            process = subprocess.Popen(
                act_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=str(repo_path),
                env=env
            )
            
            # Real-time log output with timeout control
            start_time = time.time()
            timeout_seconds = 43200  # 12 hour timeout
            
            while process.poll() is None:
                # Check for timeout
                if time.time() - start_time > timeout_seconds:
                    print(f"⚠️  Timeout ({timeout_seconds} seconds), terminating process...")
                    process.terminate()
                    try:
                        process.wait(timeout=30)  # Give 30 seconds for graceful exit
                    except subprocess.TimeoutExpired:
                        process.kill()  # Force kill
                    return False
                
                # Read output
                line = process.stdout.readline()
                if line:
                    print(line, end='')
                    log_file.write(line)
                    log_file.flush()
                else:
                    time.sleep(0.1)  # Avoid high CPU usage
        
        if process.returncode != 0:
            print(f"Error: act run failed, exit code: {process.returncode}")
            return False
        
        print("✅ act run successful")
        
        # Clean up temporary event payload file
        if event_file_path and event_file_path.exists():
            try:
                event_file_path.unlink()
                print(f"🧹 Cleaned up temporary event payload file: {event_file_path}")
            except Exception as e:
                print(f"⚠️ Failed to clean up temporary file: {e}")
        
        return True
        
    except Exception as e:
        print(f"Error: Exception occurred while running act: {e}")
        
        # Clean up temporary event payload file
        if event_file_path and event_file_path.exists():
            try:
                event_file_path.unlink()
                print(f"🧹 Cleaned up temporary event payload file: {event_file_path}")
            except Exception as cleanup_error:
                print(f"⚠️ Failed to clean up temporary file: {cleanup_error}")
        
        return False


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('project_name',
                       help='Project name (e.g., llvm)')
    parser.add_argument('failure_commit',
                       help='Failed commit SHA')
    parser.add_argument('job_name',
                       help='Job name (e.g., "premerge-checks-linux")')
    parser.add_argument('workflow_name',
                       help='Workflow name (e.g., "CI Checks")')
    parser.add_argument('--repo-path', type=str,
                       help='Custom repository path (optional, defaults to path in config file)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Only show operations to be performed, do not actually run')
    parser.add_argument('--force-rebuild', action='store_true',
                       help='Force rebuild Docker image even if it already exists')
    parser.add_argument('--no-switch', action='store_true',
                       help='Do not switch to specified commit, keep current state')
    parser.add_argument('--line-number', type=int,
                       help='Line number in JSONL file, used for log file naming')
    
    return parser.parse_args()


def switch_to_commit(repo_path, failure_commit, main_branch):
    """Switch to target commit"""
    print(f"\n🔄 Switching to commit: {failure_commit}")
    try:
        # Modify permissions
        user = os.getlogin()
        subprocess.run(
            ["sudo", "chown", "-R", f"{user}:{user}", str(repo_path)],
            check=True
        )

        # Check remote repository configuration
        print("Checking remote repository configuration...")
        try:
            remote_result = subprocess.run(
                ["git", "-C", str(repo_path), "remote", "-v"],
                capture_output=True,
                text=True,
                check=True
            )
            remotes = remote_result.stdout.strip()
            if not remotes:
                print("⚠️ No remote repository configured, trying to switch to local commit directly")
                # Try to switch to local commit directly
                subprocess.run(
                    ["git", "-C", str(repo_path), "checkout", "-f", failure_commit],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print("✅ Switch successful")
                return True
            else:
                print(f"Remote repository configuration: {remotes}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Failed to check remote repository: {e.stderr}")
            # If checking remote repository fails, try direct switch
            try:
                subprocess.run(
                    ["git", "-C", str(repo_path), "checkout", "-f", failure_commit],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print("✅ Switch successful")
                return True
            except subprocess.CalledProcessError:
                print("❌ Direct switch also failed")
                return False

        # First try to fetch specific commit
        print(f"Trying to fetch commit {failure_commit}...")
        try:
            subprocess.run(
                ["git", "-C", str(repo_path), "fetch", "origin", failure_commit],
                check=True,
                capture_output=True,
                text=True
            )
            print("✅ Successfully fetched target commit")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Failed to fetch commit directly: {e.stderr}")
            # If fetching specific commit fails, try to fetch all updates
            try:
                subprocess.run(
                    ["git", "-C", str(repo_path), "fetch", "--all"],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print("✅ Successfully fetched all updates")
            except subprocess.CalledProcessError as fetch_error:
                print(f"⚠️ Failed to fetch all updates: {fetch_error.stderr}")
                # If all fetch attempts fail, try direct switch (commit might already be local)
                print("Trying to switch to local commit directly...")

        # Switch to target commit
        print(f"Switching to commit {failure_commit}...")
        try:
            subprocess.run(
                ["git", "-C", str(repo_path), "checkout", "-f", failure_commit],
                check=True,
                capture_output=True,
                text=True
            )
            print("✅ Switch successful")
            return True
        except subprocess.CalledProcessError as checkout_error:
            print(f"❌ Failed to switch to commit {failure_commit}: {checkout_error.stderr}")
            # Try using full commit hash
            try:
                # Get full commit hash
                result = subprocess.run(
                    ["git", "-C", str(repo_path), "rev-parse", failure_commit],
                    capture_output=True,
                    text=True,
                    check=True
                )
                full_commit = result.stdout.strip()
                print(f"Trying to use full commit hash: {full_commit}")
                subprocess.run(
                    ["git", "-C", str(repo_path), "checkout", "-f", full_commit],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print("✅ Successfully switched using full commit hash")
                return True
            except subprocess.CalledProcessError as rev_parse_error:
                print(f"❌ Failed to get full commit hash: {rev_parse_error.stderr}")
                return False
                
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to switch to commit {failure_commit}: {e.stderr}")
        return False


def get_job_runs_on(workflow_file, job_id, display_name):
    """
    Get the 'runs-on' value for a given job, resolving matrix variables if necessary.
    """
    try:
        with open(workflow_file, 'r') as f:
            workflow_content = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Error while reading workflow file: {e}")
        return None

    jobs_def = workflow_content.get('jobs', {})
    job_def = jobs_def.get(job_id)

    if not job_def:
        return None

    runs_on_template = job_def.get('runs-on')
    if not runs_on_template:
        return "unknown" # No runs-on defined

    if '${{' not in str(runs_on_template):
        return runs_on_template if isinstance(runs_on_template, str) else ' '.join(runs_on_template)

    if 'strategy' not in job_def or 'matrix' not in job_def['strategy']:
        return str(runs_on_template)

    name_template = job_def.get('name', job_id)
    matrix_def = job_def['strategy']['matrix']

    matrix_combinations = []
    base_matrix_vars = {k: v for k, v in matrix_def.items() if k not in ['include', 'exclude']}
    
    if base_matrix_vars:
        keys = list(base_matrix_vars.keys())
        values = [v if isinstance(v, list) else [v] for v in base_matrix_vars.values()]
        for instance in product(*values):
            matrix_combinations.append(dict(zip(keys, instance)))
    
    if 'include' in matrix_def:
        includes = matrix_def['include']
        if not matrix_combinations:
            matrix_combinations.extend(includes)
        else:
            matrix_combinations.extend(includes)

    def get_value_from_path(path, context):
        parts = path.split('.')
        val = context
        for part in parts:
            if isinstance(val, dict) and part in val:
                val = val[part]
            else:
                return None
        return val

    def resolve_template(template, context):
        resolved_template = str(template)
        for var_match in re.finditer(r'\$\{\{([^}]+)\}\}', str(template)):
            var_expr = var_match.group(1).strip()
            value = get_value_from_path(var_expr, context)
            if value is not None:
                resolved_template = resolved_template.replace(var_match.group(0), str(value))
        return resolved_template

    best_similarity = -1
    best_combo = None

    for combo in matrix_combinations:
        context = {'matrix': combo}
        generated_name = resolve_template(name_template, context)
        
        generated_name_no_brackets = re.sub(r'\s*\([^)]*\)', '', generated_name)
        display_name_no_brackets = re.sub(r'\s*\([^)]*\)', '', display_name)

        similarity = max(
            string_similarity(generated_name, display_name),
            string_similarity(generated_name_no_brackets, display_name_no_brackets)
        )
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_combo = combo

    if best_combo and best_similarity > 0.9:
        return resolve_template(runs_on_template, {'matrix': best_combo})

    return str(runs_on_template)


def get_workflow_events(workflow_file):
    """
    Read workflow file and get supported event types
    Returns: List of event types
    """
    import yaml
    
    try:
        with open(workflow_file, 'r', encoding='utf-8') as f:
            workflow_content = yaml.safe_load(f)
        
        if 'on' not in workflow_content:
            return []
        
        events = workflow_content['on']
        if isinstance(events, str):
            return [events]
        elif isinstance(events, list):
            return events
        elif isinstance(events, dict):
            return list(events.keys())
        else:
            return []
            
    except Exception as e:
        print(f"⚠️  Error reading workflow events: {e}")
        return []


def get_default_event_for_workflow(workflow_file):
    """
    Get default event type for workflow
    Priority: push > pull_request > workflow_dispatch > others
    """
    events = get_workflow_events(workflow_file)
    if not events:
        return "push"
    
    # Prefer commonly used events
    priority_events = ["push", "pull_request", "workflow_dispatch", "workflow_call"]
    for priority_event in priority_events:
        if priority_event in events:
            return priority_event
    
    # If no common events, return the first one
    return events[0]


def parse_job_uses_mapping(workflow_file):
    """
    Parse workflow file to find each job and its corresponding subworkflow
    Returns: {job_id: {'uses': 'path/to/workflow.yml', 'with': {...}}}
    """
    import yaml
    
    job_uses_mapping = {}
    
    try:
        with open(workflow_file, 'r', encoding='utf-8') as f:
            workflow_content = yaml.safe_load(f)
        
        if 'jobs' not in workflow_content:
            return job_uses_mapping
            
        for job_id, job_config in workflow_content['jobs'].items():
            if 'uses' in job_config:
                uses_path = job_config['uses']
                # Extract with parameters
                with_params = job_config.get('with', {})
                
                job_uses_mapping[job_id] = {
                    'uses': uses_path,
                    'with': with_params,
                    'name': job_config.get('name', job_id)
                }
                
    except Exception as e:
        print(f"⚠️  Error parsing workflow file: {e}")
    
    return job_uses_mapping


def main():
    """Main function"""
    args = parse_arguments()
    
    # Load act configuration
    project_config = load_project_config()
    if not project_config:
        return False
        
    # Load project general configuration
    general_configs_path = Path(__file__).parent.parent / "project_configs.json"
    if not general_configs_path.exists():
        print(f"Error: Configuration file does not exist: {general_configs_path}")
        return False
    with open(general_configs_path, 'r') as f:
        general_configs = json.load(f)

    # Add project name to configuration
    if args.project_name in project_config:
        project_config['project_name'] = args.project_name
    else:
        print(f"Error: Project {args.project_name} is not defined in act_project_configs.json")
        return False
    
    # Get project-specific configuration
    project_specific_config = project_config.get(args.project_name, {})
    
    # Get main branch name
    main_branch = general_configs.get(args.project_name, {}).get("main_branch")
    if not main_branch:
        print(f"Error: Project {args.project_name} configuration is missing main_branch")
        return False

    # Check necessary configuration
    if args.repo_path:
        # Use command line specified repository path
        repo_path = Path(args.repo_path)
        print(f"Using custom repository path: {repo_path}")
    else:
        # Use repository path from configuration file
        if 'repo_path' not in project_specific_config:
            print(f"Error: Project {args.project_name} configuration is missing repo_path")
            return False
        
        repo_path = Path(project_specific_config['repo_path'])
        print(f"Using repository path from configuration file: {repo_path}")
    
    if not repo_path.exists():
        print(f"Error: Repository path does not exist: {repo_path}")
        return False
    
    # Switch to target commit (unless --no-switch is specified)
    if not args.no_switch:
        if not switch_to_commit(repo_path, args.failure_commit, main_branch):
            return False
    else:
        print(f"⚠️  Skipping switch to commit {args.failure_commit}, keeping current state")

    # Find workflow file
    workflow_file = find_workflow_file(repo_path, args.workflow_name)
    if not workflow_file:
        return False
    
    # Find real job ID
    job_id, event = find_job_id(repo_path, workflow_file, args.job_name)
    if not job_id:
        return False
    
    # Check job running operating system
    runs_on = get_job_runs_on(workflow_file, job_id, args.job_name)
    if not runs_on:
        print(f"⚠️  Unable to determine running environment for job '{args.job_name}', skipping system check.")
    elif runs_on.lower() in NON_UBUNTU_RUNNERS:
        print(f"✅ Job '{args.job_name}' runs on '{runs_on}', not Ubuntu, skipped.")
        return True # Return success, as it's skipped as expected

    # Run act
    return run_with_act(
        repo_path=repo_path,
        job_name=job_id,
        workflow_file=workflow_file,
        failure_commit=args.failure_commit,
        dry_run=args.dry_run,
        project_config=project_config,
        event=event,
        display_name=args.job_name,
        main_branch=main_branch,
        force_rebuild=args.force_rebuild,
        line_number=args.line_number
    )


if __name__ == "__main__":
    main() 