"""
Evaluation Harness — Automated Metrics & Comparison
=====================================================
Runs the full evaluation pipeline:
1. Run agent on golden set
2. Compute intent classification metrics (accuracy, F1, confusion matrix)
3. Run LLM-as-judge on reply quality
4. Compute escalation decision metrics
5. Compare against baselines
6. Generate results tables and visualizations

Usage:
    python -m evaluation.eval_harness
"""

import json
import numpy as np
import os
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    cohen_kappa_score,
)

# ─── Config ────────────────────────────────────────────────────────────────────

GOLDEN_SET_FILE = Path("evaluation/golden_set.jsonl")
RESULTS_DIR = Path("results")

AGENT_RESULTS_FILE = RESULTS_DIR / "agent_results.jsonl"
TRIVIAL_RESULTS_FILE = RESULTS_DIR / "trivial_baseline_results.jsonl"
SIMPLE_RESULTS_FILE = RESULTS_DIR / "simple_baseline_results.jsonl"
JUDGE_RESULTS_FILE = RESULTS_DIR / "judge_results.jsonl"
EVAL_REPORT_FILE = RESULTS_DIR / "evaluation_report.json"
EVAL_SUMMARY_FILE = RESULTS_DIR / "evaluation_summary.md"


# ─── Run Agent on Golden Set ────────────────────────────────────────────────────

def run_agent_on_golden_set():
    """Run the main agent on all golden set examples."""
    from src.agent import SupportAgent
    
    print("=" * 60)
    print("Running Agent on Golden Set")
    print("=" * 60)
    
    # Load golden set
    golden = []
    with open(GOLDEN_SET_FILE) as f:
        for line in f:
            golden.append(json.loads(line))
    print(f"Loaded {len(golden)} golden set examples")
    
    # Initialize agent
    agent = SupportAgent()
    agent.ensure_index()
    
    # Run predictions
    results = []
    for i, entry in enumerate(golden):
        print(f"  Processing {i+1}/{len(golden)}: {entry['id']}", end='\r')
        
        try:
            prediction = agent.process(entry['customer_message'])
            prediction['id'] = entry['id']
            results.append(prediction)
        except Exception as e:
            results.append({
                "id": entry['id'],
                "input_message": entry['customer_message'],
                "intent": {"intent": "error", "confidence": 0.0, "reasoning": str(e)},
                "reply": {"reply": "", "grounding_sources": [], "strategy": "error"},
                "escalation": {"decision": "escalate", "reason": f"Error: {e}", "confidence": 0.0, "trigger": "error"},
                "similar_threads": [],
            })
    
    print()
    
    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(AGENT_RESULTS_FILE, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    
    print(f"Agent results saved to {AGENT_RESULTS_FILE}")
    print(f"LLM usage: {agent.llm.get_usage_stats()}")
    
    return results


# ─── Metrics Computation ───────────────────────────────────────────────────────

def compute_intent_metrics(
    golden: list[dict],
    predictions: list[dict],
    system_name: str = "Agent",
) -> dict:
    """Compute intent classification metrics."""
    
    gold_intents = [g['gold_intent'] for g in golden]
    pred_intents = [p.get('intent', {}).get('intent', 'unknown') 
                    if isinstance(p.get('intent'), dict) 
                    else p.get('predicted_intent', 'unknown') 
                    for p in predictions]
    
    # Accuracy
    accuracy = accuracy_score(gold_intents, pred_intents)
    
    # Per-class metrics
    labels = sorted(set(gold_intents + pred_intents))
    precision, recall, f1, support = precision_recall_fscore_support(
        gold_intents, pred_intents, labels=labels, average=None, zero_division=0
    )
    
    # Macro F1
    macro_f1 = float(np.mean(f1))
    
    # Confusion matrix
    cm = confusion_matrix(gold_intents, pred_intents, labels=labels)
    
    per_class = {}
    for i, label in enumerate(labels):
        per_class[label] = {
            "precision": round(float(precision[i]), 3),
            "recall": round(float(recall[i]), 3),
            "f1": round(float(f1[i]), 3),
            "support": int(support[i]),
        }
    
    return {
        "system": system_name,
        "accuracy": round(accuracy, 3),
        "macro_f1": round(macro_f1, 3),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }


def compute_escalation_metrics(
    golden: list[dict],
    predictions: list[dict],
    system_name: str = "Agent",
) -> dict:
    """Compute escalation decision metrics."""
    
    gold_esc = [g.get('gold_escalation', 'auto_handle') for g in golden]
    pred_esc = [p.get('escalation', {}).get('decision', 'auto_handle')
                if isinstance(p.get('escalation'), dict)
                else p.get('predicted_escalation', 'auto_handle')
                for p in predictions]
    
    accuracy = accuracy_score(gold_esc, pred_esc)
    
    labels = ['auto_handle', 'escalate']
    precision, recall, f1, support = precision_recall_fscore_support(
        gold_esc, pred_esc, labels=labels, average=None, zero_division=0
    )
    
    # We care most about escalation recall (missed escalations are bad)
    escalate_idx = labels.index('escalate') if 'escalate' in labels else -1
    
    return {
        "system": system_name,
        "accuracy": round(accuracy, 3),
        "escalation_precision": round(float(precision[escalate_idx]), 3) if escalate_idx >= 0 else 0.0,
        "escalation_recall": round(float(recall[escalate_idx]), 3) if escalate_idx >= 0 else 0.0,
        "escalation_f1": round(float(f1[escalate_idx]), 3) if escalate_idx >= 0 else 0.0,
        "confusion_matrix": confusion_matrix(gold_esc, pred_esc, labels=labels).tolist(),
    }


def compute_judge_summary(judge_results: list[dict]) -> dict:
    """Summarize LLM judge scores."""
    from evaluation.llm_judge import RUBRIC_DIMENSIONS
    
    summary = {}
    for dim in RUBRIC_DIMENSIONS:
        scores = []
        for r in judge_results:
            dim_data = r.get(dim, {})
            if isinstance(dim_data, dict) and "score" in dim_data:
                scores.append(dim_data["score"])
        
        if scores:
            summary[dim] = {
                "mean": round(float(np.mean(scores)), 2),
                "std": round(float(np.std(scores)), 2),
                "min": int(min(scores)),
                "max": int(max(scores)),
                "median": round(float(np.median(scores)), 2),
            }
    
    overall = [r.get("overall_score", 0) for r in judge_results if r.get("overall_score")]
    if overall:
        summary["overall"] = {
            "mean": round(float(np.mean(overall)), 2),
            "std": round(float(np.std(overall)), 2),
        }
    
    # Breakdown by difficulty
    for difficulty in ["easy", "medium", "hard"]:
        diff_scores = [
            r.get("overall_score", 0) for r in judge_results
            if r.get("difficulty") == difficulty and r.get("overall_score")
        ]
        if diff_scores:
            summary[f"overall_by_{difficulty}"] = {
                "mean": round(float(np.mean(diff_scores)), 2),
                "n": len(diff_scores),
            }
    
    return summary


# ─── Generate Comparison Table ──────────────────────────────────────────────────

def generate_comparison_report(
    golden: list[dict],
    agent_results: list[dict],
    trivial_results: list[dict],
    simple_results: list[dict],
    judge_results: list[dict],
) -> str:
    """Generate a markdown comparison report."""
    
    # Compute metrics for all systems
    agent_intent = compute_intent_metrics(golden, agent_results, "LLM Agent")
    trivial_intent = compute_intent_metrics(golden, trivial_results, "Trivial")
    simple_intent = compute_intent_metrics(golden, simple_results, "Simple (TF-IDF)")
    
    agent_esc = compute_escalation_metrics(golden, agent_results, "LLM Agent")
    trivial_esc = compute_escalation_metrics(golden, trivial_results, "Trivial")
    simple_esc = compute_escalation_metrics(golden, simple_results, "Simple")
    
    judge_summary = compute_judge_summary(judge_results) if judge_results else {}
    
    report = f"""# Evaluation Results

_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_

## Intent Classification

| System | Accuracy | Macro F1 |
|--------|----------|----------|
| **LLM Agent (ours)** | **{agent_intent['accuracy']:.1%}** | **{agent_intent['macro_f1']:.3f}** |
| Simple (TF-IDF + LogReg) | {simple_intent['accuracy']:.1%} | {simple_intent['macro_f1']:.3f} |
| Trivial (majority class) | {trivial_intent['accuracy']:.1%} | {trivial_intent['macro_f1']:.3f} |

### Per-Intent Breakdown (LLM Agent)

| Intent | Precision | Recall | F1 | Support |
|--------|-----------|--------|----|---------|\n"""
    
    for label, metrics in sorted(agent_intent['per_class'].items()):
        report += f"| {label} | {metrics['precision']:.2f} | {metrics['recall']:.2f} | {metrics['f1']:.2f} | {metrics['support']} |\n"
    
    report += f"""
## Escalation Decision

| System | Accuracy | Escalation Recall | Escalation Precision |
|--------|----------|-------------------|---------------------|
| **LLM Agent (ours)** | **{agent_esc['accuracy']:.1%}** | **{agent_esc['escalation_recall']:.2f}** | **{agent_esc['escalation_precision']:.2f}** |
| Simple (keyword) | {simple_esc['accuracy']:.1%} | {simple_esc['escalation_recall']:.2f} | {simple_esc['escalation_precision']:.2f} |
| Trivial (never) | {trivial_esc['accuracy']:.1%} | {trivial_esc['escalation_recall']:.2f} | {trivial_esc['escalation_precision']:.2f} |

> **Note**: Escalation recall is the most important metric — missed escalations (false negatives) 
> are far worse than unnecessary escalations (false positives).

## Reply Quality (LLM-as-Judge)

_Evaluated using Claude Sonnet as judge (different model from pipeline to avoid self-evaluation bias)_

"""
    
    if judge_summary:
        report += "| Dimension | Mean Score | Std Dev |\n"
        report += "|-----------|-----------|--------|\n"
        for dim in ["relevance", "tone", "actionability", "grounding", "safety"]:
            if dim in judge_summary:
                report += f"| {dim.capitalize()} | {judge_summary[dim]['mean']}/5.0 | ±{judge_summary[dim]['std']} |\n"
        
        if "overall" in judge_summary:
            report += f"| **Overall** | **{judge_summary['overall']['mean']}/5.0** | **±{judge_summary['overall']['std']}** |\n"
        
        report += "\n### Quality by Difficulty\n\n"
        for diff in ["easy", "medium", "hard"]:
            key = f"overall_by_{diff}"
            if key in judge_summary:
                report += f"- **{diff.capitalize()}**: {judge_summary[key]['mean']}/5.0 (n={judge_summary[key]['n']})\n"
    
    return report


# ─── Visualization ──────────────────────────────────────────────────────────────

def generate_visualizations(golden, agent_results, simple_results):
    """Generate confusion matrix and other plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("  matplotlib/seaborn not installed, skipping visualizations")
        return
    
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # 1. Intent confusion matrix
    gold_intents = [g['gold_intent'] for g in golden]
    pred_intents = [
        p.get('intent', {}).get('intent', 'unknown')
        if isinstance(p.get('intent'), dict)
        else p.get('predicted_intent', 'unknown')
        for p in agent_results
    ]
    
    labels = sorted(set(gold_intents + pred_intents))
    cm = confusion_matrix(gold_intents, pred_intents, labels=labels)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted Intent')
    ax.set_ylabel('True Intent')
    ax.set_title('Intent Classification — Confusion Matrix (LLM Agent)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'confusion_matrix_agent.png', dpi=150)
    plt.close()
    
    print(f"  Saved confusion matrix to {RESULTS_DIR / 'confusion_matrix_agent.png'}")
    
    # 2. Comparison bar chart
    systems = ['Trivial', 'Simple\n(TF-IDF)', 'LLM Agent\n(Ours)']
    
    trivial_metrics = compute_intent_metrics(golden, 
        _load_jsonl(TRIVIAL_RESULTS_FILE), "Trivial")
    simple_metrics = compute_intent_metrics(golden,
        _load_jsonl(SIMPLE_RESULTS_FILE), "Simple")
    agent_metrics = compute_intent_metrics(golden, agent_results, "Agent")
    
    accuracies = [
        trivial_metrics['accuracy'],
        simple_metrics['accuracy'],
        agent_metrics['accuracy'],
    ]
    f1s = [
        trivial_metrics['macro_f1'],
        simple_metrics['macro_f1'],
        agent_metrics['macro_f1'],
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['#ff6b6b', '#ffd93d', '#6bcb77']
    
    ax1.bar(systems, accuracies, color=colors)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Intent Classification Accuracy')
    ax1.set_ylim(0, 1)
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')
    
    ax2.bar(systems, f1s, color=colors)
    ax2.set_ylabel('Macro F1')
    ax2.set_title('Intent Classification Macro F1')
    ax2.set_ylim(0, 1)
    for i, v in enumerate(f1s):
        ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'baseline_comparison.png', dpi=150)
    plt.close()
    
    print(f"  Saved baseline comparison to {RESULTS_DIR / 'baseline_comparison.png'}")


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file."""
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


# ─── Main Evaluation Pipeline ──────────────────────────────────────────────────

def run_full_evaluation():
    """Run the complete evaluation pipeline."""
    print("=" * 60)
    print("FULL EVALUATION PIPELINE")
    print("=" * 60)
    
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Load golden set
    print("\n[1/6] Loading golden set...")
    golden = _load_jsonl(GOLDEN_SET_FILE)
    print(f"  {len(golden)} examples")
    
    # Run agent
    print("\n[2/6] Running agent on golden set...")
    if AGENT_RESULTS_FILE.exists():
        print("  Agent results already exist, loading...")
        agent_results = _load_jsonl(AGENT_RESULTS_FILE)
    else:
        agent_results = run_agent_on_golden_set()
    
    # Run baselines
    print("\n[3/6] Running baselines...")
    if TRIVIAL_RESULTS_FILE.exists() and SIMPLE_RESULTS_FILE.exists():
        print("  Baseline results already exist, loading...")
        trivial_results = _load_jsonl(TRIVIAL_RESULTS_FILE)
        simple_results = _load_jsonl(SIMPLE_RESULTS_FILE)
    else:
        from evaluation.baselines import run_baselines
        trivial_results, simple_results = run_baselines()
    
    # Run LLM judge
    print("\n[4/6] Running LLM-as-judge...")
    if JUDGE_RESULTS_FILE.exists():
        print("  Judge results already exist, loading...")
        judge_results = _load_jsonl(JUDGE_RESULTS_FILE)
    else:
        from evaluation.llm_judge import LLMJudge
        judge = LLMJudge()
        judge_results = judge.evaluate_batch(golden, agent_results)
        with open(JUDGE_RESULTS_FILE, 'w') as f:
            for r in judge_results:
                f.write(json.dumps(r) + '\n')
    
    # Compute all metrics
    print("\n[5/6] Computing metrics...")
    agent_intent = compute_intent_metrics(golden, agent_results, "LLM Agent")
    trivial_intent = compute_intent_metrics(golden, trivial_results, "Trivial")
    simple_intent = compute_intent_metrics(golden, simple_results, "Simple (TF-IDF)")
    
    agent_esc = compute_escalation_metrics(golden, agent_results, "LLM Agent")
    judge_summary = compute_judge_summary(judge_results)
    
    # Print headline results
    print(f"\n{'='*60}")
    print("HEADLINE RESULTS")
    print(f"{'='*60}")
    print(f"\n  Intent Classification Accuracy:")
    print(f"    LLM Agent: {agent_intent['accuracy']:.1%}")
    print(f"    Simple:    {simple_intent['accuracy']:.1%}")
    print(f"    Trivial:   {trivial_intent['accuracy']:.1%}")
    print(f"\n  Intent Macro F1:")
    print(f"    LLM Agent: {agent_intent['macro_f1']:.3f}")
    print(f"    Simple:    {simple_intent['macro_f1']:.3f}")
    print(f"    Trivial:   {trivial_intent['macro_f1']:.3f}")
    
    if judge_summary.get("overall"):
        print(f"\n  Reply Quality (LLM Judge):")
        print(f"    Overall: {judge_summary['overall']['mean']}/5.0 ± {judge_summary['overall']['std']}")
    
    print(f"\n  Escalation Recall: {agent_esc['escalation_recall']:.2f}")
    
    # Generate report
    print("\n[6/6] Generating reports and visualizations...")
    report = generate_comparison_report(
        golden, agent_results, trivial_results, simple_results, judge_results
    )
    with open(EVAL_SUMMARY_FILE, 'w') as f:
        f.write(report)
    
    # Save full metrics
    full_report = {
        "timestamp": datetime.now().isoformat(),
        "golden_set_size": len(golden),
        "intent_metrics": {
            "agent": agent_intent,
            "simple": simple_intent,
            "trivial": trivial_intent,
        },
        "escalation_metrics": {
            "agent": agent_esc,
        },
        "judge_summary": judge_summary,
    }
    with open(EVAL_REPORT_FILE, 'w') as f:
        json.dump(full_report, f, indent=2)
    
    # Visualizations
    generate_visualizations(golden, agent_results, simple_results)
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"  Summary: {EVAL_SUMMARY_FILE}")
    print(f"  Full report: {EVAL_REPORT_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_full_evaluation()
