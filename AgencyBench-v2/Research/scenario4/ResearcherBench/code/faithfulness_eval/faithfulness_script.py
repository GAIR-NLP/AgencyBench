from code.faithfulness_eval.faithful_evaluator import FaithfulnessEvaluator
import json
import os
import asyncio
import argparse
import sys
from collections import defaultdict
sys.path.append('..')
from .config import Config

# ===== EVALUATION FUNCTIONS =====

async def attempt_model(model_data, model_name, config):
    """
    Evaluate all questions for a single model.
    
    Args:
        model_data: List of {"id": int, "question": str, "response": str}
        args: Command line arguments
        model_name: Name of the model
        config: Configuration object
    
    Returns:
        List of evaluation results
    """
    
    async def bound_func(evaluator, item):
        question_id = item["id"]
        question = item["question"]
        response = item["response"]

        async with semaphore:
            result = await evaluator.evaluate_response_faithfulness(question, response, question_id, model_name)
            return result
            
    semaphore = asyncio.Semaphore(config.max_workers)
    evaluator = FaithfulnessEvaluator(config=config)

    tasks = [bound_func(evaluator, item) for item in model_data]
    results = await asyncio.gather(*tasks)
    return results


def load_model_data(model_name, args):
    """
    Load model data from data/user_data/<model_name>.json
    
    Args:
        model_name: Name of the model
        args: Command line arguments
        
    Returns:
        List of {"id": int, "question": str, "response": str} or None if file not found
    """
    data_path = os.path.join(args.data_dir, "user_data", f"{model_name}.json")
    
    if not os.path.exists(data_path):
        print(f"❌ Model data file not found: {data_path}")
        return None
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ Loaded {len(data)} questions for model {model_name}")
        return data
        
    except Exception as e:
        print(f"❌ Error loading model data from {data_path}: {e}")
        return None


def save_model_results(results, model_name, args):
    """
    Save evaluation results for a single model.
    
    Args:
        results: List of evaluation results
        model_name: Name of the model
        args: Command line arguments
    """
    # Create model-specific result directory
    model_output_dir = os.path.join(args.output_dir, "factual_eval", model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Save overall results for the model
    combined_result_path = os.path.join(model_output_dir, "factual_results.json")
    
    print(f"Saving combined results to {combined_result_path}")
    with open(combined_result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"✅ Saved {len(results)} results for model {model_name}")


def load_response(answer_path):
    """Load response from file."""
    if not os.path.exists(answer_path):
        return None
    with open(answer_path, 'r', encoding='utf-8') as f:
        return f.read()

def run_evaluation(args):
    """Run faithfulness evaluation."""
    # Create configuration and update with command line arguments
    config = Config().update_from_args(args)
    
    # Show configuration for debugging
    print("🔧 Configuration:")
    config.show_config()
    
    if not config.validate():
        print("❌ Configuration validation failed. Please check your API keys.")
        return
    
    # If specific model is provided, evaluate only that model
    if args.model:
        model_name = args.model
    else:
        print("❌ No specific model provided.")
        return
    
    print(f"🚀 Starting evaluation for {model_name}...")
    
    # Load model data
    model_data = load_model_data(model_name, args)
    print(f"🔄 Processing {len(model_data)} questions for {model_name}...")
    
    # Run evaluation for this model
    try:
        results = asyncio.run(attempt_model(model_data, model_name, config))
        save_model_results(results, model_name, args)
        print(f"✅ Completed evaluation for {model_name}: {len(results)} results")
        
    except Exception as e:
        print(f"❌ Error evaluating {model_name}: {e}")

# ===== ANALYSIS FUNCTIONS =====

def process_entries_for_analysis(args, data):
    """Process JSON entries and return model metrics for analysis."""
    model_metrics = defaultdict(lambda: defaultdict(list))
    
    for entry in data:
        try:
            model_name = entry["model"]
            faith_score = entry["faithfulness_score"]
            ground_score = entry['groundedness_score']
        except KeyError:
            print(f"Missing faithfulness score for model {model_name}. Skipping...")
            continue

        # Add score to appropriate list
        model_metrics[model_name]["faithfulness_score"].append(faith_score)
        model_metrics[model_name]["groundedness_score"].append(ground_score)
    
    return model_metrics


def calculate_averages(model_metrics):
    """Calculate average scores for each model and metric."""
    average_scores = {}
    
    for model, metrics in model_metrics.items():
        average_scores[model] = {}
        
        for metric, scores in metrics.items():
            avg_score = sum(scores) / len(scores) if scores else 0
            average_scores[model][metric] = round(avg_score, 2)
    
    return average_scores


def print_average_table(average_scores):
    """Print a formatted table of average scores."""
    print("\nAverage Factual Scores by Model:")
    print("-" * 80)
    print(f"{'Model':<15} | {'Faithfulness':<15} | {'Groundedness':<15} ")
    print("-" * 80)
    
    # 按照 faithfulness_score 的顺序排序模型
    sorted_models = sorted(average_scores.keys(), key=lambda m: average_scores[m].get('faithfulness_score', 0), reverse=True)
    
    for model in sorted_models:
        faith_score  = average_scores[model].get('faithfulness_score', {})
        ground_score = average_scores[model].get('groundedness_score', {})
        
        faith_display = f"{faith_score:.2f}" if faith_score != 'N/A' else 'N/A'
        ground_display = f"{ground_score:.2f}" if ground_score != 'N/A' else 'N/A'
        
        print(f"{model:<15} | {faith_display:<15} | {ground_display:<15} ")
    
    return sorted_models


def save_analysis_results(average_scores, output_dir, model_name):
    """Save the organized analysis results to a JSON file."""
    
    try:
        result_path = os.path.join(output_dir, "factual_eval", model_name)
        analysis_file = os.path.join(result_path, "factual_analysis.json")
        os.makedirs(result_path, exist_ok=True)
        
        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump(average_scores, f, indent=4, ensure_ascii=False)
        
        print(f"\nDetailed analysis results saved to '{analysis_file}'")
    except Exception as e:
        print(f"❌ Error saving analysis results: {e}")

def print_analysis_summary(args, average_scores, sorted_models):
    """Print a summary of just the average scores."""
    summary = {}
    for model in sorted_models:
        summary[model] = {
            "faithfulness": average_scores[model].get("faithfulness_score", {}),
            "groundedness": average_scores[model].get("groundedness_score", {}),
        }
    
    print("-" * 70)
    print(f"{'Model':<25} | {'Faithfulness':<15} | {'Groundedness':<15}")
    print("-" * 70)

    
    for model, scores in summary.items():
        faith = scores["faithfulness"]
        ground = scores["groundedness"]
        faith_display = faith if faith == 'N/A' else f'{faith:.2f}'
        ground_display = ground if ground == 'N/A' else f'{ground:.2f}'
        print(f"{model:<25} | {faith_display:<15} | {ground_display:<15}")


def analyze_results(args):
    """Analyze existing factual evaluation results."""
    # If specific model is provided, analyze only that model
    if args.model:
        models_to_analyze = [args.model]
    else:
        # Find all models in the result directory
        models_to_analyze = []
        result_path = os.path.join(args.output_dir, "factual_eval")
        if os.path.exists(result_path):
            for item in os.listdir(result_path):
                model_path = os.path.join(result_path, item)
                if os.path.isdir(model_path):
                    factual_eval_path = os.path.join(model_path, "factual_results.json")
                    if os.path.exists(factual_eval_path):
                        models_to_analyze.append(item)
        
        if not models_to_analyze:
            print("❌ No evaluation results found in the result directory.")
            print("   Please run evaluation first before analysis.")
            return
    
    print(f"📊 Analyzing results for {models_to_analyze} model...")
    
    # Collect all data for combined analysis
    all_data = []
    result_path = os.path.join(args.output_dir, "factual_eval")
    
    for model_name in models_to_analyze:
        # Path to the model's result JSON file
        result_file = os.path.join(result_path, model_name, "factual_results.json")
        
        if not os.path.exists(result_file):
            print(f"⚠️  Result file not found for {model_name}: {result_file}")
            continue
        
        print(f"📂 Loading results from: {result_file}")
        
        # Load and add to combined data
        try:
            with open(result_file, 'r', encoding="UTF-8") as f:
                model_data = json.load(f)
            all_data.extend(model_data)
            print(f"✅ Loaded {len(model_data)} results for {model_name}")
        except Exception as e:
            print(f"❌ Error loading results for {model_name}: {e}")
            continue
    
    if not all_data:
        print("❌ No valid data found for analysis.")
        return
    
    print(f"📊 Analyzing combined data: {len(all_data)} total results")
    
    # Process data for analysis
    model_metrics = process_entries_for_analysis(args, all_data)
    
    if not model_metrics:
        print("❌ No data found matching the specified criteria.")
        return
    
    # Calculate averages and print results
    average_scores = calculate_averages(model_metrics)
    sorted_models = print_average_table(average_scores)
    
    # Save results and print summary
    save_analysis_results(average_scores, args.output_dir, model_name)
    print_analysis_summary(args, average_scores, sorted_models)


def main(args):
    """Main function that handles both evaluation and analysis modes."""
    if args.mode == "evaluate":
        run_evaluation(args)
    elif args.mode == "analyze":
        analyze_results(args)
    elif args.mode == "both":
        print("🔄 Running evaluation followed by analysis...")
        run_evaluation(args)
        analyze_results(args)
    else:
        print(f"❌ Unknown mode: {args.mode}. Use 'evaluate', 'analyze', or 'both'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate and analyze factual of AI responses')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='evaluate', 
                       choices=['evaluate', 'analyze', 'both'],
                       help='Mode: evaluate (run evaluation), analyze (analyze results), or both')
    
    # Basic evaluation parameters
    parser.add_argument('--data_dir', type=str, default='./data', 
                       help='Base data directory')
    parser.add_argument('--output_dir', type=str, default="./results", 
                       help='Directory to save results')
    
    # Model and API configuration
    parser.add_argument('--model', type=str, default='OpenAI', 
                       help='Target model to evaluate (e.g., OpenAI, Grok3, Google)')
    parser.add_argument('--judge_model', type=str, default='gpt-4.1', 
                       help='Judge Model for evaluation (e.g., gpt-4.1, o3-mini)')
    parser.add_argument('--openai_api_key', type=str, default=None,
                       help='API key for OpenAI')
    parser.add_argument('--jina_api_key', type=str, default=None,
                       help='Jina API key for web content extraction')
    # parser.add_argument('--azure_endpoint', type=str, default=None,
    #                    help='Azure OpenAI endpoint (overrides environment variable)')
    
    # Performance parameters
    parser.add_argument('--max_retries', type=int, default=3, 
                       help='Maximum number of retries for API calls')
    parser.add_argument('--max_workers', type=int, default=5, 
                       help='Maximum number of concurrent workers')

    args = parser.parse_args()
    main(args)
