# Golden Evaluation Set — Sampling Methodology

## Overview
- **Total examples**: 200
- **Sampled from**: 4955 candidate customer messages
- **Source**: AppleSupport conversations from Twitter Customer Support dataset

## Sampling Strategy

### 1. Stratified by Intent
Examples are sampled proportionally across all discovered intent categories to ensure 
every intent type is tested. No intent category has fewer than 3 examples.

**Intent distribution**:
- `ios_update_issue`: 200 examples

### 2. Stratified by Difficulty
Within each intent, examples are further stratified by estimated difficulty:

- **Easy (40%)**: Clear single-intent, short message, common pattern
- **Medium (30%)**: Moderate complexity, some ambiguity
- **Hard (30%)**: Multiple issues, strong emotion, sarcasm, edge cases

**Difficulty distribution**:
- `easy`: 72 examples
- `hard`: 74 examples
- `medium`: 54 examples

### 3. Adversarial Examples
~20 examples are specifically selected as adversarial test cases:
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
- Random seed: 42
- Sampling code: `evaluation/build_golden_set.py`
- All sampling is deterministic given the same preprocessed data
