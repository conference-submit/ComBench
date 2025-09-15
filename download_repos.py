#!/usr/bin/env python3
"""
Download all project repositories for ComBench
This script downloads all project repositories to the experiment/repos directory
"""

import os
import json
import subprocess
import sys
from pathlib import Path

def load_project_configs():
    """Load project configurations from project_configs.json"""
    config_path = Path(__file__).parent / 'project_configs.json'
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        sys.exit(1)

def create_repos_directory():
    """Create the repos directory if it doesn't exist"""
    repos_dir = Path(__file__).parent / 'experiment' / 'repos'
    repos_dir.mkdir(parents=True, exist_ok=True)
    return repos_dir

def clone_repository(repo_url, target_dir, project_name, main_branch='main'):
    """Clone a repository to the target directory"""
    target_path = Path(target_dir)
    
    if target_path.exists():
        print(f"Repository {project_name} already exists at {target_path}")
        return True
    
    print(f"Cloning {project_name} from {repo_url}...")
    try:
        # Clone with shallow history to save space and time
        cmd = [
            'git', 'clone',
            '--depth', '1',
            '--branch', main_branch,
            repo_url,
            str(target_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Successfully cloned {project_name}")
            return True
        else:
            print(f"✗ Failed to clone {project_name}: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Error cloning {project_name}: {e}")
        return False

def download_all_repos():
    """Download specified project repositories"""
    print("=== ComBench Repository Downloader ===")
    print()
    
    # Load project configurations
    all_projects = load_project_configs()
    
    # Only download specified projects
    target_projects = ['openssl', 'llvm', 'rocksdb', 'bitcoin']
    projects = {name: config for name, config in all_projects.items() if name in target_projects}
    
    # Create repos directory
    repos_dir = create_repos_directory()
    print(f"Repository directory: {repos_dir}")
    print()
    
    # Download each repository
    success_count = 0
    total_count = len(projects)
    
    for project_name, config in projects.items():
        repo_owner = config.get('repo_owner')
        repo_name = config.get('repo_name')
        main_branch = config.get('main_branch', 'main')
        
        if not repo_owner or not repo_name:
            print(f"✗ Skipping {project_name}: missing repo_owner or repo_name")
            continue
        
        repo_url = f"https://github.com/{repo_owner}/{repo_name}.git"
        target_dir = repos_dir / f"{project_name}_repo"
        
        if clone_repository(repo_url, target_dir, project_name, main_branch):
            success_count += 1
        
        print()
    
    # Summary
    print("=== Download Summary ===")
    print(f"Total projects: {total_count}")
    print(f"Successfully downloaded: {success_count}")
    print(f"Failed: {total_count - success_count}")
    
    if success_count == total_count:
        print("✓ All repositories downloaded successfully!")
    else:
        print("⚠ Some repositories failed to download. Check the error messages above.")
    
    return success_count == total_count

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("Usage: python download_repos.py")
        print()
        print("This script downloads specified project repositories to experiment/repos/")
        print("Downloads: openssl, llvm, rocksdb, bitcoin")
        print("Each repository will be cloned to a directory named {project_name}_repo")
        print()
        print("Requirements:")
        print("- Git must be installed and available in PATH")
        print("- Internet connection for downloading repositories")
        print("- Sufficient disk space (repositories can be large)")
        return
    
    try:
        success = download_all_repos()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
