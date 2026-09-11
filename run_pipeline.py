"""
Run Pipeline — Single Entry Point for Full Reproducibility
============================================================
Runs the entire Hiver AI Support Agent pipeline end-to-end:

1. Preprocess data (filter AppleSupport, build threads)
2. Discover intents (clustering + taxonomy)
3. Build golden evaluation set
4. Build retrieval index
5. Run agent on golden set
6. Run baselines
7. Run LLM-as-judge evaluation
8. Generate results and reports

Usage:
    python run_pipeline.py              # Run everything
    python run_pipeline.py --skip-llm   # Skip LLM calls (use cached results)
    python run_pipeline.py --step 3     # Run from step 3 onwards
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


FORCE_RUN = False

def check_prerequisites():
    """Check that all required files and dependencies are available."""
    print("Checking prerequisites...")
    
    # Check API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("  ⚠ OPENROUTER_API_KEY not set. LLM calls will fail.")
        print("    Set it: export OPENROUTER_API_KEY=your_key_here")
        print("    Or create a .env file with OPENROUTER_API_KEY=your_key_here")
        return False
    else:
        print("  ✓ OPENROUTER_API_KEY found")
    
    # Check raw data
    raw_csv = Path("data/raw/twcs.csv")
    if not raw_csv.exists():
        print(f"  ✗ Raw dataset not found at {raw_csv}")
        print("    Download from: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter")
        print("    Place twcs.csv in data/raw/")
        return False
    else:
        print(f"  ✓ Raw dataset found ({raw_csv.stat().st_size / 1e6:.0f} MB)")
    
    # Check Python dependencies
    required = ['pandas', 'numpy', 'sklearn', 'sentence_transformers', 'openai']
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"  ✗ Missing packages: {', '.join(missing)}")
        print(f"    Run: pip install -r requirements.txt")
        return False
    else:
        print("  ✓ All Python dependencies installed")
    
    return True


def step_1_preprocess():
    """Step 1: Data preprocessing."""
    output_file = Path("data/processed/apple_threads.jsonl")
    if not FORCE_RUN and output_file.exists():
        n_lines = sum(1 for _ in open(output_file))
        print(f"\n  → Preprocessed data already exists ({n_lines} threads). Skipping.")
        return True
    
    from data.preprocess import main as preprocess_main
    preprocess_main()
    return True


def step_2_discover_intents():
    """Step 2: Intent discovery."""
    taxonomy_file = Path("src/intent_taxonomy.json")
    if not FORCE_RUN and taxonomy_file.exists():
        with open(taxonomy_file) as f:
            data = json.load(f)
        n_intents = len(data.get("intents", {}))
        print(f"\n  → Intent taxonomy already exists ({n_intents} intents). Skipping.")
        return True
    
    from src.intent_discovery import main as discover_main
    discover_main()
    return True


def step_3_build_golden_set():
    """Step 3: Build golden evaluation set."""
    golden_file = Path("evaluation/golden_set.jsonl")
    if not FORCE_RUN and golden_file.exists():
        n_lines = sum(1 for _ in open(golden_file))
        print(f"\n  → Golden set already exists ({n_lines} examples). Skipping.")
        return True
    
    from evaluation.build_golden_set import build_golden_set
    build_golden_set()
    return True


def step_4_run_agent():
    """Step 4: Run agent on golden set."""
    results_file = Path("results/agent_results.jsonl")
    if not FORCE_RUN and results_file.exists():
        n_lines = sum(1 for _ in open(results_file))
        print(f"\n  → Agent results already exist ({n_lines} predictions). Skipping.")
        return True
    
    from evaluation.eval_harness import run_agent_on_golden_set
    run_agent_on_golden_set()
    return True


def step_5_run_baselines():
    """Step 5: Run baselines."""
    trivial_file = Path("results/trivial_baseline_results.jsonl")
    simple_file = Path("results/simple_baseline_results.jsonl")
    
    if not FORCE_RUN and trivial_file.exists() and simple_file.exists():
        print(f"\n  → Baseline results already exist. Skipping.")
        return True
    
    from evaluation.baselines import run_baselines
    run_baselines()
    return True


def step_6_run_judge():
    """Step 6: Run LLM-as-judge."""
    judge_file = Path("results/judge_results.jsonl")
    if not FORCE_RUN and judge_file.exists():
        n_lines = sum(1 for _ in open(judge_file))
        print(f"\n  → Judge results already exist ({n_lines} evaluations). Skipping.")
        return True
    
    from evaluation.llm_judge import run_judge_on_golden_set
    run_judge_on_golden_set()
    return True


def step_7_generate_reports():
    """Step 7: Generate evaluation reports."""
    from evaluation.eval_harness import (
        _load_jsonl,
        compute_intent_metrics,
        compute_escalation_metrics,
        compute_judge_summary,
        generate_comparison_report,
        generate_visualizations,
    )
    
    golden = _load_jsonl(Path("evaluation/golden_set.jsonl"))
    agent_results = _load_jsonl(Path("results/agent_results.jsonl"))
    trivial_results = _load_jsonl(Path("results/trivial_baseline_results.jsonl"))
    simple_results = _load_jsonl(Path("results/simple_baseline_results.jsonl"))
    
    judge_results = []
    judge_file = Path("results/judge_results.jsonl")
    if judge_file.exists():
        judge_results = _load_jsonl(judge_file)
    
    # Generate report
    report = generate_comparison_report(
        golden, agent_results, trivial_results, simple_results, judge_results
    )
    with open(Path("results/evaluation_summary.md"), 'w') as f:
        f.write(report)
    
    # Generate visualizations
    generate_visualizations(golden, agent_results, simple_results)
    
    # Print headline results
    agent_intent = compute_intent_metrics(golden, agent_results, "LLM Agent")
    simple_intent = compute_intent_metrics(golden, trivial_results, "Trivial")
    
    print(f"\n{'='*60}")
    print("HEADLINE RESULTS")
    print(f"{'='*60}")
    print(f"  Intent Accuracy: {agent_intent['accuracy']:.1%}")
    print(f"  Intent Macro F1: {agent_intent['macro_f1']:.3f}")
    
    if judge_results:
        judge_summary = compute_judge_summary(judge_results)
        if judge_summary.get("overall"):
            print(f"  Reply Quality:   {judge_summary['overall']['mean']}/5.0")
    
    print(f"\n  Reports saved to results/")
    print(f"{'='*60}")
    
    return True


def main():
    global FORCE_RUN
    
    parser = argparse.ArgumentParser(
        description="Run the Hiver AI Support Agent pipeline"
    )
    parser.add_argument(
        "--step", type=int, default=1,
        help="Start from this step (1-7). Default: 1"
    )
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Skip steps that require LLM API calls"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force run all steps from scratch (ignores cache)"
    )
    args = parser.parse_args()
    
    FORCE_RUN = args.force
    
    print("=" * 60)
    print("Hiver AI Support Agent — Full Pipeline")
    print("=" * 60)
    
    start_time = time.time()
    
    # Check prerequisites
    if not check_prerequisites():
        if not args.skip_llm:
            print("\n⚠ Prerequisites check failed. Fix issues above and retry.")
            print("  Or run with --skip-llm to skip LLM-dependent steps.")
            sys.exit(1)
    
    steps = [
        (1, "Data Preprocessing", step_1_preprocess),
        (2, "Intent Discovery", step_2_discover_intents),
        (3, "Golden Evaluation Set", step_3_build_golden_set),
        (4, "Run Agent on Golden Set", step_4_run_agent),
        (5, "Run Baselines", step_5_run_baselines),
        (6, "LLM-as-Judge Evaluation", step_6_run_judge),
        (7, "Generate Reports", step_7_generate_reports),
    ]
    
    llm_steps = {2, 4, 6}  # Steps that require LLM calls
    
    for step_num, step_name, step_func in steps:
        if step_num < args.step:
            continue
        
        if args.skip_llm and step_num in llm_steps:
            print(f"\n[{step_num}/7] {step_name} — SKIPPED (--skip-llm)")
            continue
        
        print(f"\n{'─'*60}")
        print(f"[{step_num}/7] {step_name}")
        print(f"{'─'*60}")
        
        try:
            success = step_func()
            if not success:
                print(f"\n✗ Step {step_num} failed. Stopping.")
                sys.exit(1)
        except Exception as e:
            print(f"\n✗ Step {step_num} failed with error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Pipeline complete! Total time: {elapsed/60:.1f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
