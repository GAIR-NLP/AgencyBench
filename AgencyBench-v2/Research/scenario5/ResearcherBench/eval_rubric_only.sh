#!/bin/bash

# ===================================================
# DARS Benchmark - ResearcherBench Evaluation Script
# ===================================================
# This script evaluates AI model responses using rubric evaluation
# Usage: 
#   1. Set MODEL environment variable
#   2. Run: ./eval_rubric_only.sh
# ===================================================

set -e  # Exit on any error

# Print output functions
print_status() {
    echo "[INFO] $1"
}

print_success() {
    echo "[SUCCESS] $1"
}

print_warning() {
    echo "[WARNING] $1"
}

print_error() {
    echo "[ERROR] $1"
}

# Print banner
echo "================================================="
echo "  🔍 ResearcherBench Rubric Evaluation Script     "
echo "================================================="

# Environment variables configuration

# # Set up environment variables
# export MODEL=<your_model_name>  
# export OPENAI_API_KEY="your-openai-api-key"
# export OPENAI_BASE_URL="https://api.openai.com/v1" 

export JUDGE_MODEL="o3-mini"  # for rubric evaluation
export MAX_WORKERS=10  # 并行线程数，可以根据API限制和机器性能调整

# Check if MODEL is set
if [ -z "$MODEL" ]; then
    print_error "MODEL environment variable is not set!"
    echo ""
    echo "Please set MODEL before running this script:"
    echo "  export MODEL=OpenAI"
    echo "  ./eval_rubric_only.sh"
    echo ""
    echo "Available models in data/user_data/:"
    if [ -d "data/user_data" ]; then
        for file in data/user_data/*.json; do
            if [ -f "$file" ]; then
                echo "  - $(basename "$file" .json)"
            fi
        done
    fi
    exit 1
fi

print_status "Target Model: $MODEL"

# Check if model data file exists
MODEL_DATA_FILE="data/user_data/${MODEL}.json"
if [ ! -f "$MODEL_DATA_FILE" ]; then
    print_error "Model data file not found: $MODEL_DATA_FILE"
    echo ""
    echo "Available models in data/user_data/:"
    if [ -d "data/user_data" ]; then
        for file in data/user_data/*.json; do
            if [ -f "$file" ]; then
                echo "  - $(basename "$file" .json)"
            fi
        done
    fi
    exit 1
fi

print_success "Found model data file: $MODEL_DATA_FILE"

# Check required environment variables
print_status "Checking environment variables..."

if [ -z "$OPENAI_API_KEY" ]; then
    print_warning "No OpenAI API key found!"
    echo "Please set:"
    echo "  export OPENAI_API_KEY=your_openai_api_key"
    echo ""
    echo "Optional environment variables:"
    echo "  export JUDGE_MODEL=o3-mini             # Judge model for evaluation (default: o3-mini)"
    echo "  export MAX_WORKERS=10                  # Concurrent workers (default: 17)"
    echo "  export MAX_RETRIES=3                   # API retry attempts (default: 3)"
    exit 1
fi

if [ -n "$OPENAI_API_KEY" ]; then
    print_success "OpenAI API key found"
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p results
mkdir -p logs
mkdir -p claims
print_success "Directories created"

# 记录开始时间
start_time=$(date +%s)

print_status "Starting evaluation for model: $MODEL"
print_status "Using judge model: ${JUDGE_MODEL:-o3-mini}"
print_status "Using ${MAX_WORKERS:-10} parallel workers"
echo "=========================================="

# Rubric Evaluation
print_status "Starting rubric evaluation..."
echo "================================"

python code/rubric_eval/main.py \
    --model_file "data/user_data/$MODEL.json" \
    --rubrics_file "data/eval_data/rubric.json" \
    --result_dir "results" \
    --judge_model "${JUDGE_MODEL:-o3-mini}" \
    --max_workers "${MAX_WORKERS:-10}"

# Check if rubric evaluation completed successfully
if [ $? -eq 0 ]; then
    print_success "Rubric evaluation completed successfully!"
else
    print_error "Rubric evaluation failed!"
    exit 1
fi

# echo ""
# print_status "Phase 2: Starting factual evaluation..."
# echo "=========================================="

# # Run the factual evaluation
# python3 -m code.faithfulness_eval.faithfulness_script \
#     --mode both \
#     --model "$MODEL" \
#     --judge_model "${FACTUAL_JUDGE_MODEL:-gpt-4.1}" \
#     --data_dir "./data" \
#     --output_dir "./results" \
#     --max_workers "${MAX_WORKERS:-10}" \
#     --max_retries "${MAX_RETRIES:-3}"

# # Check if evaluation completed successfully
# if [ $? -eq 0 ]; then
#     print_success "Factual evaluation completed successfully!"
# else
#     print_error "Factual evaluation failed!"
#     exit 1
# fi

# 记录结束时间
end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "=========================================="
print_success "All evaluations completed successfully!"
echo "Total duration: ${duration} seconds"
echo ""
echo "Results saved to:"
echo "  - Rubric evaluation: results/$MODEL/rubric_eval/"
echo ""
