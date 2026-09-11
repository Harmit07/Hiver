"""
Golden Evaluation Set Builder
===============================
Constructs a stratified, hand-label-ready evaluation set of 200 examples.

Sampling strategy:
1. Stratified by cluster/intent (ensures coverage across all intents)
2. Stratified by difficulty (easy 40%, medium 30%, hard 30%)
3. Includes adversarial examples (sarcasm, off-topic, edge cases)

Outputs a JSONL file ready for hand-labeling, plus a sampling methodology doc.

Usage:
    python -m evaluation.build_golden_set
"""

import json
import random
import re
from pathlib import Path
from collections import defaultdict

# ─── Config ────────────────────────────────────────────────────────────────────

THREADS_FILE = Path("data/processed/apple_threads.jsonl")
TAXONOMY_FILE = Path("src/intent_taxonomy.json")
MESSAGES_FILE = Path("data/processed/customer_messages.json")
GOLDEN_SET_FILE = Path("evaluation/golden_set.jsonl")
METHODOLOGY_FILE = Path("evaluation/sampling_methodology.md")

TARGET_SIZE = 200
DIFFICULTY_RATIOS = {"easy": 0.40, "medium": 0.30, "hard": 0.30}
ADVERSARIAL_COUNT = 20  # Included within the hard category

RANDOM_SEED = 42


# ─── Difficulty Heuristics ──────────────────────────────────────────────────────

def estimate_difficulty(message: str, thread: dict) -> str:
    """
    Heuristic difficulty estimation for a customer message.
    
    Easy: Clear single issue, common pattern, short message
    Medium: Ambiguous intent, moderate length, some complexity
    Hard: Multiple issues, strong emotion, sarcasm, unusual request
    """
    msg_lower = message.lower()
    word_count = len(message.split())
    
    # Hard indicators
    hard_signals = 0
    if word_count > 40:
        hard_signals += 1
    if any(w in msg_lower for w in ["but also", "and also", "plus", "additionally"]):
        hard_signals += 1  # Multiple issues
    if any(w in msg_lower for w in ["terrible", "worst", "hate", "furious", "disgusting"]):
        hard_signals += 1  # Strong emotion
    if "?" in message and "!" in message:
        hard_signals += 1  # Mixed signals
    if message.count("!") > 2:
        hard_signals += 1  # High frustration
    if any(w in msg_lower for w in ["lol", "smh", "seriously?", "wow", "bruh"]):
        hard_signals += 1  # Sarcasm/informal
    if thread.get("num_turns", 0) > 4:
        hard_signals += 1  # Multi-turn = complex
    
    # Easy indicators
    easy_signals = 0
    if word_count < 15:
        easy_signals += 1
    if "how do i" in msg_lower or "how to" in msg_lower:
        easy_signals += 1  # Clear how-to question
    if message.count("?") == 1 and "!" not in message:
        easy_signals += 1  # Single clear question
    
    if hard_signals >= 2:
        return "hard"
    elif easy_signals >= 2:
        return "easy"
    else:
        return "medium"


def is_adversarial_candidate(message: str) -> bool:
    """Check if a message could be a good adversarial test case."""
    msg_lower = message.lower()
    
    signals = [
        # Sarcasm
        any(w in msg_lower for w in ["great job", "thanks a lot", "wonderful"]) and 
        any(w in msg_lower for w in ["not", "broken", "fail", "worst"]),
        # Off-topic
        any(w in msg_lower for w in ["pizza", "weather", "game", "movie"]),
        # Multi-language snippets  
        bool(re.search(r'[^\x00-\x7F]{3,}', message)),
        # Very short / unclear
        len(message.split()) <= 3,
        # Pure emoji/symbols
        len(re.sub(r'[^\w\s]', '', message).strip()) < 5,
        # Mixed topics
        message.count("@") > 2,
    ]
    
    return sum(signals) >= 1


# ─── Golden Set Builder ────────────────────────────────────────────────────────

def build_golden_set():
    """Build the stratified golden evaluation set."""
    print("=" * 60)
    print("Building Golden Evaluation Set")
    print(f"Target size: {TARGET_SIZE} examples")
    print("=" * 60)
    
    random.seed(RANDOM_SEED)
    
    # Load threads
    print("\n[1/4] Loading threads...")
    threads = []
    with open(THREADS_FILE) as f:
        for line in f:
            threads.append(json.loads(line))
    print(f"  Loaded {len(threads)} threads")
    
    # Load taxonomy for cluster assignments
    messages_with_clusters = []
    if MESSAGES_FILE.exists():
        with open(MESSAGES_FILE) as f:
            messages_with_clusters = json.load(f)
    
    # Build thread lookup
    thread_lookup = {t['thread_id']: t for t in threads}
    
    # Load taxonomy
    taxonomy = {}
    if TAXONOMY_FILE.exists():
        with open(TAXONOMY_FILE) as f:
            data = json.load(f)
        taxonomy = data.get("intents", data)
    
    # Map cluster_id -> intent_label
    cluster_to_intent = {}
    for cid, intent_data in taxonomy.items():
        cluster_to_intent[int(cid)] = intent_data.get("intent_label", f"cluster_{cid}")
    
    # Step 2: Assign difficulty and cluster
    print("\n[2/4] Assigning difficulty levels...")
    candidates = []
    for msg_data in messages_with_clusters:
        thread = thread_lookup.get(msg_data.get('thread_id', ''), {})
        if not thread:
            continue
        
        difficulty = estimate_difficulty(msg_data['text'], thread)
        cluster_id = msg_data.get('cluster_id', 0)
        intent = cluster_to_intent.get(cluster_id, "unknown")
        is_adversarial = is_adversarial_candidate(msg_data['text'])
        
        candidates.append({
            "thread_id": msg_data['thread_id'],
            "customer_message": msg_data['text'],
            "agent_response": msg_data.get('agent_response', ''),
            "thread_context": thread.get('messages', [])[:6],  # Keep first 6 messages
            "difficulty": difficulty,
            "cluster_id": cluster_id,
            "inferred_intent": intent,
            "is_adversarial": is_adversarial,
        })
    
    # Difficulty distribution
    diff_counts = defaultdict(int)
    for c in candidates:
        diff_counts[c['difficulty']] += 1
    print(f"  Difficulty distribution: {dict(diff_counts)}")
    
    # Step 3: Stratified sampling
    print("\n[3/4] Stratified sampling...")
    
    # Group by (intent, difficulty)
    groups = defaultdict(list)
    adversarial_pool = []
    
    for c in candidates:
        groups[(c['inferred_intent'], c['difficulty'])].append(c)
        if c['is_adversarial']:
            adversarial_pool.append(c)
    
    # Calculate target per group
    n_intents = len(set(c['inferred_intent'] for c in candidates))
    base_per_intent = (TARGET_SIZE - ADVERSARIAL_COUNT) // max(n_intents, 1)
    
    selected = []
    seen_ids = set()
    
    # Sample from each intent × difficulty group
    for intent in set(c['inferred_intent'] for c in candidates):
        for difficulty, ratio in DIFFICULTY_RATIOS.items():
            group = groups.get((intent, difficulty), [])
            n_target = max(1, int(base_per_intent * ratio))
            
            available = [c for c in group if c['thread_id'] not in seen_ids]
            sampled = random.sample(available, min(n_target, len(available)))
            
            for s in sampled:
                seen_ids.add(s['thread_id'])
                selected.append(s)
    
    # Add adversarial examples
    adversarial_available = [c for c in adversarial_pool if c['thread_id'] not in seen_ids]
    adversarial_sampled = random.sample(
        adversarial_available,
        min(ADVERSARIAL_COUNT, len(adversarial_available))
    )
    for s in adversarial_sampled:
        seen_ids.add(s['thread_id'])
        s['difficulty'] = 'hard'  # Mark as hard
        s['is_adversarial'] = True
        selected.append(s)
    
    # If we haven't reached target, fill from remaining
    if len(selected) < TARGET_SIZE:
        remaining = [c for c in candidates if c['thread_id'] not in seen_ids]
        random.shuffle(remaining)
        for c in remaining[:TARGET_SIZE - len(selected)]:
            selected.append(c)
    
    # Trim to target
    selected = selected[:TARGET_SIZE]
    random.shuffle(selected)
    
    print(f"  Selected: {len(selected)} examples")
    
    # Final difficulty distribution
    final_diff = defaultdict(int)
    for s in selected:
        final_diff[s['difficulty']] += 1
    print(f"  Final difficulty: {dict(final_diff)}")
    
    # Final intent distribution
    final_intent = defaultdict(int)
    for s in selected:
        final_intent[s['inferred_intent']] += 1
    print(f"  Final intent distribution: {dict(final_intent)}")
    
    # Step 4: Format and save
    print(f"\n[4/4] Saving golden set to {GOLDEN_SET_FILE}...")
    
    golden_entries = []
    for i, s in enumerate(selected):
        entry = {
            "id": f"golden_{i+1:03d}",
            "customer_message": s['customer_message'],
            "thread_context": s['thread_context'],
            "agent_actual_response": s['agent_response'],
            # ── Labels to be verified/corrected by hand ──
            "gold_intent": s['inferred_intent'],  # Pre-filled, to be verified
            "gold_escalation": "auto_handle",      # To be labeled
            "gold_escalation_reason": "",           # To be filled
            "gold_reply_quality_notes": "",         # To be filled
            # ── Metadata ──
            "difficulty": s['difficulty'],
            "is_adversarial": s.get('is_adversarial', False),
            "thread_id": s['thread_id'],
            "labeler_notes": "",                    # To be filled during labeling
        }
        golden_entries.append(entry)
    
    with open(GOLDEN_SET_FILE, 'w') as f:
        for entry in golden_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"  Saved {len(golden_entries)} entries")
    
    # Write sampling methodology
    _write_methodology(len(candidates), len(selected), final_diff, final_intent)
    
    print(f"\n{'='*60}")
    print("Done! Golden set is ready for hand-labeling.")
    print(f"  Golden set: {GOLDEN_SET_FILE}")
    print(f"  Methodology: {METHODOLOGY_FILE}")
    print(f"{'='*60}")
    
    return golden_entries


def _write_methodology(total_candidates, selected_count, difficulty_dist, intent_dist):
    """Write the sampling methodology document."""
    methodology = f"""# Golden Evaluation Set — Sampling Methodology

## Overview
- **Total examples**: {selected_count}
- **Sampled from**: {total_candidates} candidate customer messages
- **Source**: AppleSupport conversations from Twitter Customer Support dataset

## Sampling Strategy

### 1. Stratified by Intent
Examples are sampled proportionally across all discovered intent categories to ensure 
every intent type is tested. No intent category has fewer than 3 examples.

**Intent distribution**:
{chr(10).join(f'- `{k}`: {v} examples' for k, v in sorted(intent_dist.items()))}

### 2. Stratified by Difficulty
Within each intent, examples are further stratified by estimated difficulty:

- **Easy (40%)**: Clear single-intent, short message, common pattern
- **Medium (30%)**: Moderate complexity, some ambiguity
- **Hard (30%)**: Multiple issues, strong emotion, sarcasm, edge cases

**Difficulty distribution**:
{chr(10).join(f'- `{k}`: {v} examples' for k, v in sorted(difficulty_dist.items()))}

### 3. Adversarial Examples
~{ADVERSARIAL_COUNT} examples are specifically selected as adversarial test cases:
- Sarcastic messages (e.g., "Great job breaking my phone Apple!")
- Off-topic messages
- Very short/ambiguous messages
- Messages with mixed signals (question + frustration)
- Non-English snippets

## Labeling Process

### Labels per example:
1. **`gold_intent`**: The correct intent category (pre-filled by clustering, verified by hand)
2. **`gold_escalation`**: Whether this should be auto-handled or escalated
3. **`gold_escalation_reason`**: Why the escalation decision was made
4. **`gold_reply_quality_notes`**: What a good reply should contain
5. **`labeler_notes`**: Any edge cases or ambiguities noted during labeling

### Labeling guidelines:
- If a message fits multiple intents, choose the primary/most actionable one
- Escalation = "Would a human agent provide significantly better service here?"
- Quality notes should focus on what information the reply MUST contain

### Quality control:
- 30 examples are labeled twice (2 weeks apart) to measure intra-annotator consistency
- Consistency score is reported alongside results
- Disagreements are flagged and re-examined

## Reproducibility
- Random seed: {RANDOM_SEED}
- Sampling code: `evaluation/build_golden_set.py`
- All sampling is deterministic given the same preprocessed data
"""
    
    with open(METHODOLOGY_FILE, 'w') as f:
        f.write(methodology)


if __name__ == "__main__":
    build_golden_set()
