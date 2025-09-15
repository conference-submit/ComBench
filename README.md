# ComBench - Compilation Error Repair Benchmark Platform

ComBench is a benchmark platform for evaluating the performance of automated compilation error repair (ACER) tools. The platform provides a complete workflow for compilation error collection, analysis, and evaluation.

## 📁 Project Structure

```
ComBench/
├── act/                      # Act tool for local GitHub Actions execution
│   ├── main.go              # Act tool main source code
│   ├── install.sh           # Act installation script
│   ├── Makefile             # Build configuration
│   └── go.mod               # Go module dependencies
├── ci_failures/              # CI failure record collection
│   └── collect_ci_failures.py
├── compilation_error/          # Compilation error data collection
│   ├── extract_compiler_errors.py # Extract compilation errors from CI logs
│   ├── error_classifier.py        # Compilation error classifier
│   ├── process_output_files.py    # Compilation error data processing
│   ├── batch_extract_errors.py    # Batch extract compilation errors
│   └── output/                    # Extracted compilation error data
├── find_repair_patch/         # Repair patch analysis
│   ├── analyze_ci_fixes_unified.py # Unified repair patch analysis tool
│   └── *_repair_analysis.jsonl    # Repair patch analysis results for each project
├── experiment/                # Experiments and evaluation
│   ├── Setting/              # Evaluation settings
│   │   ├── Agent/            # Agent-based tool evaluation
│   │   │   ├── agent_evaluator.py    # agent-based tool evaluator
│   │   │   ├── run_evaluation.py     # Run evaluation script
│   │   │   ├── config_*.json         # Evaluation configuration for each project
│   │   │   ├── reproduced_data/      # Reproduced datasets (as input)
│   │   │   └── final_results/        # Final evaluation results
│   │   ├── PFL/              # PFL method evaluation
│   │   │   ├── pfl_evaluator.py      # PFL evaluator
│   │   │   ├── run_evaluation.py     # Run evaluation script
│   │   │   ├── config_*.json         # Evaluation configuration for each project
│   │   │   └── final_results/        # Final evaluation results
│   │   ├── run_comparison.py # Run comparison analysis
│   │   └── compare_with_ground_truth.py # Compare with ground truth repairs
│   ├── reproduce/            # Error reproduction
│   │   ├── error_extractor.py        # Error extractor
│   │   ├── error_matcher.py          # Error matcher
│   │   ├── metadata_loader.py        # Metadata loader
│   │   └── reproduce_compiler_errors.py # Reproduce compilation errors
│   ├── extract_and_reproduce.py # Extract and reproduce errors
│   ├── batch_reproduce_project.py # Batch reproduce projects
│   ├── reproduce_error.py    # Direct error reproduction using act tool
│   └── act_project_configs.json # Configuration for act tool experiments
├── project_configs.json      # Project configuration file
├── path_config.py           # Path configuration management
├── download_repos.py        # Script to download all project repositories
└── requirements.txt         # Python dependency package list
```

## 🚀 Quick Start

### Environment Requirements

- Python 3.8+
- Git
- Docker (for reproducing compilation environments)
- Go 1.20+ (for building act tool)
- Related compilation toolchains (GCC, Clang, etc.)

### Install Dependencies

```bash
# Clone repository
git clone <repository-url>
cd ComBench

# Install Python dependencies
pip install -r requirements.txt

# Build and install act tool (required for error reproduction)
cd act
make install
cd ..
```

### Configure Paths

The project uses `path_config.py` to centrally manage path configurations. External dependency paths can be set through environment variables:

```bash
# Set OpenHands path
export OPENHANDS_PATH="/path/to/OpenHands"

# View current path configuration
python path_config.py
```

### Configure GitHub API Token

Before using ComBench tools, you need to configure your GitHub Personal Access Token using environment variable:

```bash
# Set your GitHub Personal Access Token
export GITHUB_TOKEN="ghp_your_token_here"
```

**Note**: Create GitHub Personal Access Token with `repo` and `actions:read` permissions. Never commit tokens to version control!

## 📊 Main Features

### 1. CI Failure Record Collection

Collect CI failure records from GitHub Actions:

```bash
# Collect CI failure records for a specific project
python ci_failures/collect_ci_failures.py llvm

# Collect data from the last 30 days
python ci_failures/collect_ci_failures.py llvm --days 30

# Analyze existing data only
python ci_failures/collect_ci_failures.py llvm --analyze-only
```

### 2. Compilation Error Data Collection

Extract compilation errors from GitHub Actions CI logs:

```bash
# Extract compilation errors for a specific project
python compilation_error/extract_compiler_errors.py --project llvm

# Batch extract all projects
python compilation_error/batch_extract_errors.py

# Process compilation error files in output folder
python compilation_error/process_output_files.py
```

### 3. Repair Patch Analysis

Find repair patches:

```bash
# Analyze repair patches
python find_repair_patch/analyze_ci_fixes_unified.py --project llvm
```

### 4. Error Reproduction

First download all project repositories:

```bash
# Download all project repositories
python download_repos.py
```

Then reproduce compilation errors:

```bash
# Reproduce specific error in dir find_repair_patch
python experiment/extract_and_reproduce.py --project llvm --line-number 1

# Direct error reproduction using reproduce_error.py
python experiment/reproduce_error.py <project_name> <failure_commit> <job_name> <workflow_name>
```

### 5. ACER Tool Evaluation

Evaluate the compilation error repair capability of ACER tools (Compilation Success):

```bash
# Run Agent-based tool evaluation (Require Openhands)
cd experiment/Setting/Agent
python run_evaluation.py --config config_llvm.json

# Run PFL evaluation (Require Implementing LLM API calls in experiment/Setting/PFL/model/)
cd experiment/Setting/PFL
python run_evaluation.py --config config_llvm.json
```

### 6. Consistency between Tool-generated patches and Developer's patches

Compare performance of different methods (Exactly Match):

```bash
# Run comparison analysis
cd experiment/Setting
python run_comparison.py PFL/final_results/results-bitcoin
```

## ⚙️ Configuration

### Project Configuration (`project_configs.json`)

Each project's configuration includes:

```json
{
  "llvm": {
    "repo_owner": "llvm",
    "repo_name": "llvm-project",
    "main_branch": "main",
    "main_language": "C++",
    "compilation_workflows": ["CI Checks"],
    "reproducible_jobs": ["Build and Test Linux"],
    "include_paths": ["-I.", "-Iinclude", "-Illvm/include"]
  }
}
```

### Evaluation Configuration

Agent-based tool and PFL evaluations use JSON configuration files:

```json
{
  "project_name": "llvm",
  "model_name": "claude-sonnet-4",
  "data_path": "reproduced_data/llvm.jsonl",
  "repo_path": "/path/to/llvm-project",
  "output_dir": "final_results/results-llvm"
}
```


## 📝 Data Formats

### CI Failure Record Format

```json
{
  "repo_name": "llvm",
  "workflow_id": 12345678,
  "workflow_name": "CI Checks",
  "job_id": 87654321,
  "job_name": "Build and Test Linux",
  "created_at": "2024-01-01T12:00:00Z",
  "failure_logs": "完整的CI日志内容...",
  "commit_sha": "abc123def456",
  "branch": "main"
}
```

### Compilation Error Data Format

```json
{
  "commit_sha": "76295efe08f864e9760b65f9d3aa03735db68354",
  "branch": "p2p-fix-nscore-overflow-24049",
  "workflow_name": "CI",
  "job_name": [
    "macOS 14 native, arm64, no depends, sqlite only, gui",
    "macOS 14 native, arm64, fuzz"
  ],
  "workflow_id": 16558331993,
  "job_id": [
    46841335938,
    46841335949
  ],
  "created_at": "2025-07-28T02:07:04Z",
  "error_lines": [
    "/Users/runner/work/bitcoin/bitcoin/src/net.cpp:181:30: error: non-constant-expression cannot be narrowed from type 'int64_t' (aka 'long long') to 'int' in initializer list [-Wc++11-narrowing]"
  ],
  "error_details": [
    "/Users/runner/work/bitcoin/bitcoin/src/net.cpp:181:30: error: non-constant-expression cannot be narrowed from type 'int64_t' (aka 'long long') to 'int' in initializer list [-Wc++11-narrowing]\n            const int nScore{local_service_info.nScore};\n                             ^~~~~~~~~~~~~~~~~~~~~~~~~"
  ],
  "error_count": 1,
  "error_types": {
    "type_error": 1
  },
  "error_line_types": [
    "type_error"
  ]
}
```

### Evaluation Result Format

#### Agent-based Tool Results (使用 diff_text)

```json
{
  "instance_index": 1,
  "error_index": 1,
  "error_result": {
    "is_successful": true,
    "is_exact_match": false,
    "is_valid": true,
    "error_line": "file.cpp:10:5: error: expected ';' before '}'",
    "diff_text": "diff --git a/file.cpp b/file.cpp\nindex 1234567..abcdefg 100644\n--- a/file.cpp\n+++ b/file.cpp\n@@ -7,7 +7,7 @@\n   int x = 1\n-  int y = 2\n+  int y = 2;\n }\n",
    "errors_before": ["file.cpp:10:5: error: expected ';' before '}'"],
    "errors_after": [],
    "error_detail": "file.cpp:10:5: error: expected ';' before '}'\n  10 | }\n      | ^",
    "time": 15.5,
    "evaluation_mode": "full"
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

#### PFL Method Results (使用 patches)

```json
{
  "instance_index": 1,
  "error_index": 1,
  "error_result": {
    "is_successful": true,
    "is_exact_match": false,
    "is_valid": false,
    "error_line": "file.cpp:10:5: error: expected ';' before '}'",
    "patches": [
      {
        "error_line": "file.cpp:10:5: error: expected ';' before '}'",
        "file_path": "file.cpp",
        "start_line": 10,
        "end_line": 10,
        "original_code": "  int y = 2",
        "fixed_code": "  int y = 2;",
        "confidence": 0.8,
        "explanation": "添加缺失的分号"
      }
    ],
    "errors_before": ["file.cpp:10:5: error: expected ';' before '}'"],
    "errors_after": [],
    "error_detail": "file.cpp:10:5: error: expected ';' before '}'\n  10 | }\n      | ^",
    "time": 15.5,
    "evaluation_mode": "full"
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Path Errors**: Check path configuration in `path_config.py`
2. **Missing Dependencies**: Ensure all external dependency paths are correctly set
3. **Permission Issues**: Ensure sufficient file system permissions
4. **Docker Issues**: Ensure Docker service is running normally

### Debug Mode

```bash
# Enable detailed logging
export LOG_LEVEL=DEBUG
```