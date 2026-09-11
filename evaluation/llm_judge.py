"""
LLM-as-Judge — Reply Quality Evaluation
==========================================
Uses a different, stronger LLM (Claude Sonnet via OpenRouter) to judge
the quality of agent-generated replies.

Rubric dimensions (1-5 scale):
    1. Relevance — Does the reply address the customer's actual issue?
    2. Tone — Is it professional, empathetic, on-brand for AppleSupport?
    3. Actionability — Does it provide concrete next steps?
    4. Grounding — Is it consistent with how Apple actually responds?
    5. Safety — Does it avoid hallucinating policies or making promises?

Includes judge calibration: measures agreement between LLM judge and human labels.

Usage:
    from evaluation.llm_judge import LLMJudge
    judge = LLMJudge()
    scores = judge.evaluate(customer_msg, agent_reply, actual_reply)
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional

from src.llm_client import LLMClient

# ─── Config ────────────────────────────────────────────────────────────────────

RUBRIC_DIMENSIONS = [
    "relevance",
    "tone",
    "actionability",
    "grounding",
    "safety",
]

JUDGE_RESULTS_FILE = Path("results/judge_results.jsonl")
CALIBRATION_FILE = Path("results/judge_calibration.json")


class LLMJudge:
    """
    LLM-as-Judge for evaluating reply quality.
    
    Uses Claude Sonnet (via OpenRouter) as the judge — deliberately different
    from the pipeline model (Gemini Flash) to avoid self-evaluation bias.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
    
    def evaluate(
        self,
        customer_message: str,
        generated_reply: str,
        actual_reply: str = "",
        intent: str = "",
    ) -> dict:
        """
        Evaluate a generated reply using the LLM judge.
        
        Args:
            customer_message: The original customer message
            generated_reply: The agent's generated reply
            actual_reply: The actual historical reply (if available)
            intent: The classified intent
            
        Returns:
            Dict with scores for each rubric dimension (1-5) + overall + reasoning
        """
        
        system_prompt = """You are an expert evaluator assessing the quality of AI-generated customer 
support replies for @AppleSupport on Twitter.

You must evaluate each reply on 5 dimensions using a 1-5 scale:
1 = Very poor, 2 = Below average, 3 = Acceptable, 4 = Good, 5 = Excellent

Be strict and calibrated:
- A score of 5 should be rare — only for replies that are truly excellent
- A score of 3 means "adequate but not great"
- A score of 1 means the reply would actively harm the customer experience

Always provide specific reasoning for each score."""

        actual_context = ""
        if actual_reply:
            actual_context = f"""
For reference, here is how Apple actually responded to this customer:
Actual Apple response: "{actual_reply}"
Use this as a reference point for what a good response looks like, but note that
the actual response is not necessarily perfect either."""

        prompt = f"""Evaluate this AI-generated customer support reply.

Customer message: "{customer_message}"
Detected intent: {intent}
AI-generated reply: "{generated_reply}"
{actual_context}

Rate the reply on each dimension (1-5 scale):

1. **Relevance**: Does the reply address the customer's actual issue?
   - 5: Directly addresses the specific problem with accurate information
   - 3: Generally relevant but misses key details
   - 1: Completely off-topic or addresses wrong issue

2. **Tone**: Is it professional, empathetic, and on-brand for @AppleSupport?
   - 5: Perfect Apple voice — warm, professional, empathetic
   - 3: Adequate tone but could be more empathetic or feels generic
   - 1: Rude, dismissive, or completely off-brand

3. **Actionability**: Does it provide concrete, useful next steps?
   - 5: Clear, specific steps the customer can take right now
   - 3: Some guidance but vague or incomplete
   - 1: No actionable information at all

4. **Grounding**: Is it consistent with how Apple actually handles similar issues?
   - 5: Matches Apple's actual support practices perfectly
   - 3: Generally plausible but some details may be off
   - 1: Contains made-up policies or contradicts Apple's practices

5. **Safety**: Does it avoid hallucinating policies, prices, or making promises?
   - 5: No hallucinations, appropriate disclaimers
   - 3: Minor assumptions but nothing harmful
   - 1: Makes up policies, promises outcomes, or gives dangerous advice

Respond in JSON:
{{
    "relevance": {{"score": <1-5>, "reasoning": "<specific reasoning>"}},
    "tone": {{"score": <1-5>, "reasoning": "<specific reasoning>"}},
    "actionability": {{"score": <1-5>, "reasoning": "<specific reasoning>"}},
    "grounding": {{"score": <1-5>, "reasoning": "<specific reasoning>"}},
    "safety": {{"score": <1-5>, "reasoning": "<specific reasoning>"}},
    "overall_score": <float, weighted average>,
    "overall_assessment": "<2-3 sentence summary>"
}}"""

        try:
            result = self.llm.call_json(
                prompt,
                model_type="judge",  # Uses Claude Sonnet
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=1024,
            )
            
            # Calculate overall score if not provided
            if "overall_score" not in result or not isinstance(result.get("overall_score"), (int, float)):
                scores = []
                for dim in RUBRIC_DIMENSIONS:
                    dim_data = result.get(dim, {})
                    if isinstance(dim_data, dict):
                        scores.append(dim_data.get("score", 3))
                    elif isinstance(dim_data, (int, float)):
                        scores.append(dim_data)
                result["overall_score"] = round(np.mean(scores), 2) if scores else 3.0
            
            return result
            
        except Exception as e:
            return {
                dim: {"score": 0, "reasoning": f"Judge failed: {str(e)}"}
                for dim in RUBRIC_DIMENSIONS
            } | {
                "overall_score": 0.0,
                "overall_assessment": f"Evaluation failed: {str(e)}",
            }
    
    def evaluate_batch(
        self,
        examples: list[dict],
        predictions: list[dict],
    ) -> list[dict]:
        """
        Evaluate a batch of predictions.
        
        Args:
            examples: Golden set entries
            predictions: Agent prediction results
            
        Returns:
            List of judge evaluation results
        """
        results = []
        
        for i, (example, pred) in enumerate(zip(examples, predictions)):
            print(f"  Judging example {i+1}/{len(examples)}...", end='\r')
            
            result = self.evaluate(
                customer_message=example['customer_message'],
                generated_reply=pred.get('reply', {}).get('reply', ''),
                actual_reply=example.get('agent_actual_response', ''),
                intent=pred.get('intent', {}).get('intent', ''),
            )
            
            result['id'] = example['id']
            result['difficulty'] = example.get('difficulty', 'unknown')
            results.append(result)
        
        print()
        return results
    
    def calibrate(
        self,
        human_scores: list[dict],
        judge_scores: list[dict],
    ) -> dict:
        """
        Measure agreement between human labels and LLM judge.
        
        Args:
            human_scores: List of {id, dimension, score} from human labeling
            judge_scores: List of {id, dimension, score} from LLM judge
            
        Returns:
            Calibration metrics: Pearson r, Cohen's kappa, MAE per dimension
        """
        from scipy import stats
        
        results = {}
        
        for dim in RUBRIC_DIMENSIONS:
            human_vals = []
            judge_vals = []
            
            for h, j in zip(human_scores, judge_scores):
                h_score = h.get(dim, {}).get("score") if isinstance(h.get(dim), dict) else h.get(dim)
                j_score = j.get(dim, {}).get("score") if isinstance(j.get(dim), dict) else j.get(dim)
                
                if h_score is not None and j_score is not None:
                    human_vals.append(float(h_score))
                    judge_vals.append(float(j_score))
            
            if len(human_vals) >= 5:
                pearson_r, p_value = stats.pearsonr(human_vals, judge_vals)
                mae = np.mean(np.abs(np.array(human_vals) - np.array(judge_vals)))
                
                # Cohen's Kappa (after binning to categorical: low/mid/high)
                def bin_score(s):
                    if s <= 2:
                        return "low"
                    elif s <= 3:
                        return "mid"
                    else:
                        return "high"
                
                from sklearn.metrics import cohen_kappa_score
                human_bins = [bin_score(s) for s in human_vals]
                judge_bins = [bin_score(s) for s in judge_vals]
                kappa = cohen_kappa_score(human_bins, judge_bins)
                
                results[dim] = {
                    "pearson_r": round(pearson_r, 3),
                    "p_value": round(p_value, 4),
                    "mae": round(mae, 3),
                    "cohens_kappa": round(kappa, 3),
                    "n_samples": len(human_vals),
                }
            else:
                results[dim] = {"error": f"Too few samples ({len(human_vals)})"}
        
        return results


def run_judge_on_golden_set():
    """Run the LLM judge on agent predictions for the golden set."""
    print("=" * 60)
    print("Running LLM-as-Judge Evaluation")
    print("=" * 60)
    
    # Load golden set
    golden = []
    with open(Path("evaluation/golden_set.jsonl")) as f:
        for line in f:
            golden.append(json.loads(line))
    
    # Load agent predictions
    agent_results_file = Path("results/agent_results.jsonl")
    if not agent_results_file.exists():
        print("Error: Run the agent on the golden set first (results/agent_results.jsonl)")
        return
    
    predictions = []
    with open(agent_results_file) as f:
        for line in f:
            predictions.append(json.loads(line))
    
    # Run judge
    judge = LLMJudge()
    print(f"\nEvaluating {len(golden)} examples with LLM judge (Claude Sonnet)...")
    results = judge.evaluate_batch(golden, predictions)
    
    # Save results
    Path("results").mkdir(exist_ok=True)
    with open(JUDGE_RESULTS_FILE, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    
    # Print summary
    print(f"\n--- Judge Results Summary ---")
    for dim in RUBRIC_DIMENSIONS:
        scores = []
        for r in results:
            dim_data = r.get(dim, {})
            if isinstance(dim_data, dict) and "score" in dim_data:
                scores.append(dim_data["score"])
        if scores:
            print(f"  {dim}: mean={np.mean(scores):.2f}, std={np.std(scores):.2f}")
    
    overall = [r.get("overall_score", 0) for r in results if r.get("overall_score")]
    if overall:
        print(f"  OVERALL: mean={np.mean(overall):.2f}, std={np.std(overall):.2f}")
    
    print(f"\nResults saved to {JUDGE_RESULTS_FILE}")
    return results


if __name__ == "__main__":
    run_judge_on_golden_set()
