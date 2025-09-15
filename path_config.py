
#!/usr/bin/env python3
"""
Path configuration file
Unified management of all path settings in the project, avoiding hardcoded absolute paths
"""

import os
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Basic path configuration
BASE_PATHS = {
    # Project root directory
    'project_root': PROJECT_ROOT,
    
    # Experiment-related paths
    'experiment': PROJECT_ROOT / 'experiment',
    'compilation_error': PROJECT_ROOT / 'compilation_error',
    'find_repair_patch': PROJECT_ROOT / 'find_repair_patch',
    
    # Data file paths (relative to project root directory)
    'compilation_error_data': PROJECT_ROOT / 'compilation_error',
    'repair_patch_data': PROJECT_ROOT / 'find_repair_patch',
    
    # Configuration file paths
    'project_configs': PROJECT_ROOT / 'project_configs.json',
}

# External dependency paths (can be overridden by environment variables)
EXTERNAL_PATHS = {
    # OpenHands path - can be overridden by environment variable OPENHANDS_PATH
    'openhands': os.environ.get('OPENHANDS_PATH', 
                               PROJECT_ROOT.parent / 'compiler-error' / 'collect-compiler-fix' / 'experiment' / 'agent' / 'OpenHands'),
}


def get_path(key: str) -> Path:
    """
    Get specified path
    
    Args:
        key: Path key name
        
    Returns:
        Path object
    """
    if key in BASE_PATHS:
        return BASE_PATHS[key]
    elif key in EXTERNAL_PATHS:
        path_value = EXTERNAL_PATHS[key]
        return Path(path_value) if isinstance(path_value, str) else path_value
    else:
        raise KeyError(f"Unknown path key: {key}")

def get_compilation_error_file(project_name: str) -> Path:
    """
    Get compilation error data file path
    
    Args:
        project_name: Project name
        
    Returns:
        Compilation error data file path
    """
    return get_path('compilation_error_data') / f"{project_name}_compiler_errors_extracted.json"

def get_repair_patch_file(project_name: str) -> Path:
    """
    Get fix patch data file path
    
    Args:
        project_name: Project name
        
    Returns:
        Fix patch data file path
    """
    return get_path('repair_patch_data') / f"{project_name}_repair_analysis.jsonl"


def setup_sys_path():
    """
    Set sys.path, add necessary module paths
    """
    import sys
    compilation_error_path = str(get_path('compilation_error_data'))
    if compilation_error_path not in sys.path:
        sys.path.append(compilation_error_path)

# Verify if paths exist
def validate_paths():
    """
    Verify if key paths exist
    
    Returns:
        dict: Path verification results
    """
    results = {}
    
    # Verify basic paths
    for key, path in BASE_PATHS.items():
        results[key] = {
            'path': str(path),
            'exists': path.exists(),
            'type': 'directory' if path.is_dir() else 'file' if path.is_file() else 'unknown',
            'source': 'project'
        }
    
    # Verify external paths
    for key, path in EXTERNAL_PATHS.items():
        path_obj = Path(path)
        results[key] = {
            'path': str(path_obj),
            'exists': path_obj.exists(),
            'type': 'directory' if path_obj.is_dir() else 'file' if path_obj.is_file() else 'unknown',
            'source': 'environment' if key in os.environ else 'default'
        }
    
    
    return results

if __name__ == "__main__":
    # Print path configuration information
    print("=== ComBench Path Configuration ===")
    print(f"Project root directory: {PROJECT_ROOT}")
    print()
    
    print("Basic paths:")
    for key, path in BASE_PATHS.items():
        print(f"  {key}: {path}")
    print()
    
    print("External paths:")
    for key, path in EXTERNAL_PATHS.items():
        print(f"  {key}: {path}")
    print()
    
    print("Path verification results:")
    validation_results = validate_paths()
    for key, result in validation_results.items():
        status = "✓" if result['exists'] else "✗"
        source_info = f" ({result['source']})" if 'source' in result else ""
        print(f"  {status} {key}: {result['path']}{source_info}")
