"""
Baselines — Trivial and Simple Baselines for Comparison
=========================================================
Implements two baseline systems to compare against the main agent:

1. Trivial Baseline:
   - Intent: always predict majority class
   - Reply: fixed template reply
   - Escalation: never escalate

2. Simple Baseline:
   - Intent: TF-IDF + Logistic Regression classifier
   - Reply: BM25 retrieval → return closest historical response verbatim
   - Escalation: keyword-based rules only

Usage:
    python -m evaluation.baselines
"""

import json
import re
import math
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# ─── Config ────────────────────────────────────────────────────────────────────

THREADS_FILE = Path("data/processed/apple_threads.jsonl")
MESSAGES_FILE = Path("data/processed/customer_messages.json")
TAXONOMY_FILE = Path("src/intent_taxonomy.json")
GOLDEN_SET_FILE = Path("evaluation/golden_set.jsonl")


# ═══════════════════════════════════════════════════════════════════════════════
# TRIVIAL BASELINE
# ═══════════════════════════════════════════════════════════════════════════════

class TrivialBaseline:
    """
    The simplest possible baseline:
    - Always predicts the most common intent
    - Returns a fixed template reply
    - Never escalates
    
    Purpose: Lower bound. If our agent can't beat this, something is very wrong.
    """
    
    def __init__(self):
        self.majority_intent = None
        self.template_reply = (
            "Hi there! We'd like to help. Can you DM us the details "
            "of your issue so we can look into it? We're here to help!"
        )
    
    def fit(self, messages: list[dict]):
        """Find the majority intent from training data."""
        intents = [m.get('cluster_id', 0) for m in messages]
        
        # Load taxonomy mapping
        taxonomy = {}
        if TAXONOMY_FILE.exists():
            with open(TAXONOMY_FILE) as f:
                data = json.load(f)
            taxonomy = data.get("intents", {})
        
        # Count intents
        cluster_counts = Counter(intents)
        majority_cluster = cluster_counts.most_common(1)[0][0]
        
        self.majority_intent = taxonomy.get(
            str(majority_cluster), {}
        ).get("intent_label", f"cluster_{majority_cluster}")
        
        print(f"  Trivial baseline: majority intent = '{self.majority_intent}'")
    
    def predict(self, message: str) -> dict:
        """Predict using trivial strategy."""
        return {
            "intent": {
                "intent": self.majority_intent,
                "confidence": 1.0,
                "reasoning": "Majority class prediction",
            },
            "reply": {
                "reply": self.template_reply,
                "grounding_sources": [],
                "strategy": "Fixed template reply",
            },
            "escalation": {
                "decision": "auto_handle",
                "reason": "Never escalates (trivial baseline)",
                "confidence": 1.0,
                "trigger": "trivial",
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SIMPLE BASELINE
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleBaseline:
    """
    A simple but non-trivial baseline:
    - Intent: TF-IDF vectorization + Logistic Regression
    - Reply: BM25-style retrieval → return closest historical response
    - Escalation: Keyword-based rules
    
    Purpose: Shows the value the LLM adds beyond classical ML.
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
        )
        self.classifier = LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class='multinomial',
        )
        self.corpus_messages = []
        self.corpus_responses = []
        self.corpus_tfidf = None
        self.intent_map = {}
    
    def fit(self, messages: list[dict]):
        """Train the TF-IDF + LogReg classifier and build BM25 corpus."""
        # Load taxonomy
        taxonomy = {}
        if TAXONOMY_FILE.exists():
            with open(TAXONOMY_FILE) as f:
                data = json.load(f)
            taxonomy = data.get("intents", {})
        
        # Prepare training data
        texts = [m['text'] for m in messages]
        cluster_ids = [m.get('cluster_id', 0) for m in messages]
        
        # Map cluster_id -> intent_label
        self.intent_map = {}
        for cid, intent_data in taxonomy.items():
            self.intent_map[int(cid)] = intent_data.get("intent_label", f"cluster_{cid}")
        
        labels = [self.intent_map.get(cid, f"cluster_{cid}") for cid in cluster_ids]
        
        # Train TF-IDF + LogReg
        print("  Training TF-IDF + LogReg classifier...")
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.classifier, X, labels, cv=5, scoring='accuracy')
        print(f"  CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Build retrieval corpus
        self.corpus_messages = texts
        self.corpus_responses = [m.get('agent_response', '') for m in messages]
        self.corpus_tfidf = X
        
        print(f"  Corpus: {len(texts)} messages for retrieval")
    
    def predict(self, message: str) -> dict:
        """Predict using simple ML baseline."""
        # Intent classification
        X = self.vectorizer.transform([message])
        predicted_intent = self.classifier.predict(X)[0]
        confidence = float(max(self.classifier.predict_proba(X)[0]))
        
        # BM25-style retrieval (using TF-IDF cosine similarity)
        similarities = (self.corpus_tfidf @ X.T).toarray().flatten()
        best_idx = int(np.argmax(similarities))
        best_response = self.corpus_responses[best_idx] if self.corpus_responses[best_idx] else \
            "Please DM us your details so we can assist you further."
        
        # Keyword-based escalation
        escalation = self._keyword_escalation(message)
        
        return {
            "intent": {
                "intent": predicted_intent,
                "confidence": confidence,
                "reasoning": "TF-IDF + Logistic Regression classification",
            },
            "reply": {
                "reply": best_response,
                "grounding_sources": [],
                "strategy": "BM25 retrieval — closest historical response",
            },
            "escalation": escalation,
        }
    
    def _keyword_escalation(self, message: str) -> dict:
        """Simple keyword-based escalation."""
        msg_lower = message.lower()
        
        escalation_words = [
            "lawsuit", "legal", "attorney", "lawyer", "sue",
            "scam", "fraud", "stolen", "hacked", "urgent",
            "dangerous", "safety", "fire", "burning",
        ]
        
        for word in escalation_words:
            if word in msg_lower:
                return {
                    "decision": "escalate",
                    "reason": f"Keyword trigger: '{word}'",
                    "confidence": 0.8,
                    "trigger": "keyword",
                }
        
        return {
            "decision": "auto_handle",
            "reason": "No escalation keywords detected",
            "confidence": 0.6,
            "trigger": "keyword",
        }


# ─── Run Baselines on Golden Set ───────────────────────────────────────────────

def run_baselines():
    """Run both baselines on the golden set and save results."""
    print("=" * 60)
    print("Running Baselines")
    print("=" * 60)
    
    # Load training data (messages with cluster assignments)
    print("\nLoading training data...")
    with open(MESSAGES_FILE) as f:
        messages = json.load(f)
    print(f"  Training messages: {len(messages)}")
    
    # Load golden set
    print("Loading golden set...")
    golden = []
    with open(GOLDEN_SET_FILE) as f:
        for line in f:
            golden.append(json.loads(line))
    print(f"  Golden set: {len(golden)} examples")
    
    # Initialize and train baselines
    print("\n--- Trivial Baseline ---")
    trivial = TrivialBaseline()
    trivial.fit(messages)
    
    print("\n--- Simple Baseline ---")
    simple = SimpleBaseline()
    simple.fit(messages)
    
    # Run predictions
    print("\nRunning predictions on golden set...")
    trivial_results = []
    simple_results = []
    
    for entry in golden:
        msg = entry['customer_message']
        
        trivial_pred = trivial.predict(msg)
        simple_pred = simple.predict(msg)
        
        trivial_results.append({
            "id": entry['id'],
            "gold_intent": entry['gold_intent'],
            "predicted_intent": trivial_pred['intent']['intent'],
            "predicted_reply": trivial_pred['reply']['reply'],
            "predicted_escalation": trivial_pred['escalation']['decision'],
        })
        
        simple_results.append({
            "id": entry['id'],
            "gold_intent": entry['gold_intent'],
            "predicted_intent": simple_pred['intent']['intent'],
            "predicted_reply": simple_pred['reply']['reply'],
            "predicted_escalation": simple_pred['escalation']['decision'],
        })
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / "trivial_baseline_results.jsonl", 'w') as f:
        for r in trivial_results:
            f.write(json.dumps(r) + '\n')
    
    with open(results_dir / "simple_baseline_results.jsonl", 'w') as f:
        for r in simple_results:
            f.write(json.dumps(r) + '\n')
    
    # Quick accuracy check
    trivial_correct = sum(1 for r in trivial_results if r['predicted_intent'] == r['gold_intent'])
    simple_correct = sum(1 for r in simple_results if r['predicted_intent'] == r['gold_intent'])
    
    print(f"\n--- Quick Results ---")
    print(f"  Trivial intent accuracy: {trivial_correct}/{len(golden)} ({trivial_correct/len(golden)*100:.1f}%)")
    print(f"  Simple intent accuracy: {simple_correct}/{len(golden)} ({simple_correct/len(golden)*100:.1f}%)")
    
    print(f"\n{'='*60}")
    print("Baseline results saved to results/")
    print(f"{'='*60}")
    
    return trivial_results, simple_results


if __name__ == "__main__":
    run_baselines()
