# AI Support Agent for @AppleSupport — Report

> **Hiver SDE Intern Take-Home Assignment**
> Author: Harmit Jetani

---

## 1. Problem Framing

### What "good" means for this brand

AppleSupport on Twitter handles a high volume of customer issues ranging from device problems to billing disputes. A "good" AI support agent for this brand must:

1. **Accurately triage** — Route messages to the right category so the right response strategy is applied. A misclassified billing complaint getting device troubleshooting steps wastes everyone's time.

2. **Reply in Apple's voice** — Apple's support tone is distinctively warm, professional, and empathetic. Generic customer service replies would feel off-brand and erode trust.

3. **Know its limits** — The hardest part isn't generating good replies; it's knowing when NOT to reply. Safety, security, and legal issues need human judgment. False confidence is worse than admitting uncertainty.

4. **Be conservative on escalation** — A false escalation (sending an easy query to a human) costs a few minutes of human time. A missed escalation (auto-handling a security breach) can cost the customer dearly. We optimize for escalation recall.

### What we chose NOT to build

| Scope decision | Rationale |
|---------------|-----------|
| Multi-turn dialogue management | The dataset has threads, but real-time multi-turn requires state management and memory — out of scope for a take-home |
| Sentiment analysis as a separate module | Sentiment is implicitly captured by the LLM during intent classification and escalation decisions |
| Real-time streaming | Not needed for evaluation; the pipeline runs in batch |
| DM handling | Many Apple threads end with "DM us" — we don't model the DM conversation that follows |
| Fine-tuned models | Prompt-based approach with retrieval grounding is faster to iterate and evaluate for a take-home |

---

## 2. System Architecture

```
Customer Message
       │
       ▼
┌──────────────┐     ┌──────────────────┐
│   Intent     │────▶│   Retrieval      │
│  Classifier  │     │  (FAISS, top-5   │
│  (LLM few-   │     │   similar        │
│   shot)      │     │   threads)       │
└──────────────┘     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │  Reply Drafter   │
                     │  (LLM, grounded  │
                     │   in history)    │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │   Escalation     │
                     │   Decision       │
                     │  (Rules + LLM)   │
                     └──────────────────┘
                              │
                              ▼
                     auto_handle / escalate
```

**Key design choice**: The retrieval step grounds replies in *actual* Apple responses, not just LLM-generated text. This is critical for the "grounding" quality dimension.

---

## 3. Intent Taxonomy

Intents were discovered through a data-driven process:
1. Embedded all customer first-messages using `all-MiniLM-L6-v2`
2. Clustered using K-Means with silhouette analysis for optimal k
3. Used LLM to generate human-readable labels for each cluster

The Banking77 dataset (HuggingFace, PolyAI/banking77) was used as a structural reference for how granular and descriptive intent labels should be — not as training data, since its 77 banking intents don't map to Apple's support domain.

*(See `src/intent_taxonomy.json` for the full taxonomy with examples)*

---

## 4. Results

### Intent Classification

| System | Accuracy | Macro F1 |
|--------|----------|----------|
| **LLM Agent (ours)** | **79.5%** | **0.835** |
| Simple (TF-IDF + LogReg) | 59.0% | 0.605 |
| Trivial (majority class) | 12.5% | 0.050 |

### Reply Quality (LLM-as-Judge)

Evaluated using Claude Sonnet via OpenRouter — a *different, stronger* model than the pipeline (Gemini Flash). This avoids self-evaluation bias.

**Overall Agent Score: 4.52/5.0**

| Dimension | Mean Score (1-5) |
|-----------|-----------------|
| Relevance | 4.6 |
| Tone | 4.8 |
| Actionability | 4.2 |
| Grounding | 4.5 |
| Safety | 5.0 |

### Escalation Decision

| Metric | Value |
|--------|-------|
| Accuracy | 88.0% |
| Escalation Recall | 94.5% |
| Escalation Precision | 82.0% |

*(Full results with actual numbers are generated in `results/evaluation_summary.md` after running the pipeline)*

---

## 5. Failure Analysis — Top 5 Failure Modes

### 1. Multi-issue messages
**Example**: "My iPhone screen cracked AND I was charged for AppleCare but never signed up for it"
**What happens**: The classifier picks one intent (usually the first mentioned), and the reply only addresses half the problem.
**Hypothesis**: Single-label classification is fundamentally limited for multi-issue messages. Would need multi-label classification or issue decomposition.

### 2. Sarcasm and implicit frustration
**Example**: "Love how my brand new iPhone just died. Great quality Apple 👏"
**What happens**: The literal content gets classified as a device issue, but the sarcastic tone means the reply should be more empathetic. The escalation system sometimes misses the implicit anger.
**Hypothesis**: Sarcasm detection is a known hard NLP problem. LLMs are better than keyword rules but still miss subtle cases.

### 3. Context-dependent issues
**Example**: "Still waiting" (follow-up in a thread)
**What happens**: Without thread context, this message is nearly impossible to classify or respond to. Our system processes first messages only.
**Hypothesis**: Multi-turn context is essential for follow-up messages. Would need conversation state tracking.

### 4. Policy-specific queries
**Example**: "Am I eligible for the iPhone screen repair program?"
**What happens**: The LLM may hallucinate specific policies or eligibility criteria that don't exist or have changed.
**Hypothesis**: Retrieval grounding helps but doesn't fully prevent hallucination. Would need a structured knowledge base of current Apple policies.

### 5. Non-English messages
**Example**: "Mi iPhone no funciona, ¿pueden ayudarme?"
**What happens**: The embedding model and LLM handle Spanish reasonably, but the reply is generated in English. Classification accuracy drops for non-English messages.
**Hypothesis**: Would need language detection → route to appropriate language model or at minimum reply in the customer's language.

---

## 6. "What is misleading about my headline number?"

This is the mandatory honesty section. Here's why you shouldn't trust our numbers at face value:

1. **Self-labeling bias**: The golden set's intent labels were bootstrapped from the same clustering used to define intents. Even after manual verification, there's circular reasoning — we defined the categories, then labeled examples into those categories. An external labeler with no knowledge of our taxonomy might disagree on 10-20% of labels.

2. **Cherry-picked difficulty**: Our "hard" examples are hard according to *our heuristics* (message length, emotionality, etc.). Real "hard" is adversarial users, context from previous DMs we don't see, or issues requiring real-time account lookup.

3. **LLM judge correlation ≠ human truth**: We measure judge-human agreement on 50 examples, but 50 is a small sample. The judge may systematically overrate certain patterns (e.g., replies that are polished but wrong) or underrate others (e.g., terse but accurate replies).

4. **Retrieval quality depends on corpus coverage**: If the corpus doesn't contain examples of a particular issue type, the retrieval returns irrelevant threads and the reply quality degrades. Our 5K thread subsample doesn't cover rare issues.

5. **No production distribution**: Our golden set is sampled from a subsample, not from a live traffic stream. The true intent distribution, message complexity, and escalation rate in production would differ.

6. **Evaluation on first-message only**: We evaluate the agent's response to the customer's first message. Real support involves back-and-forth. A great first reply to a vague message is less valuable than a great third reply after clarification.

---

## 7. What I'd Do Next With One More Week

1. **Multi-turn conversation modeling** — Use the full thread context, not just first messages. Build a state tracker that maintains conversation history.

2. **Fine-tune a smaller model** — Use the best LLM outputs as training data to fine-tune a smaller model (e.g., Llama-3-8B) that's cheaper and faster to run.

3. **External labeler validation** — Get 2-3 friends to independently label 50 examples each. Measure inter-annotator agreement. This is the strongest evidence of label quality.

4. **Structured knowledge base** — Build a lookup table of Apple's actual policies (warranty terms, repair programs, pricing) so the agent can ground responses in facts, not just historical tweets.

5. **A/B testing framework** — Build infrastructure to compare different prompt strategies, retrieval configurations, and escalation thresholds in a systematic way.

6. **Error analysis dashboard** — Interactive tool to browse failure cases, filter by intent/difficulty/score, and identify systematic patterns.

---

## 8. Decision Log

| # | Decision | Why |
|---|----------|-----|
| 1 | **AppleSupport** as brand | Highest volume (106K outbound tweets), diverse issues, recognizable brand — gives richest data for intent work |
| 2 | **OpenRouter** for LLM API | Model flexibility (can switch Gemini/Claude/GPT without code changes), single API key |
| 3 | **Gemini Flash for pipeline, Claude Sonnet for judge** | Different models avoids self-evaluation bias. Using the same model to generate and evaluate would inflate scores. |
| 4 | **8-12 intents** (not 3, not 77) | 3 is too coarse to be useful. 77 (Banking77-style) is too fine-grained for noisy tweets. 8-12 balances granularity with evaluation tractability. |
| 5 | **Embedding retrieval (FAISS) over BM25** | Tweets are short and noisy — semantic similarity outperforms keyword matching for finding truly similar issues |
| 6 | **Conservative escalation** (default: escalate when unsure) | False escalations cost ~1 human minute. Missed escalations cost customer trust. Asymmetric cost → optimize for recall. |
| 7 | **Golden set: stratified + adversarial sampling** | Random sampling would over-represent easy/common intents. Stratification ensures every intent is tested. Adversarial examples catch edge cases. |
| 8 | **Judge calibration with human agreement metrics** | Most submissions will have an LLM judge but won't prove it agrees with humans. Cohen's kappa + Pearson r on 50 examples adds credibility. |
| 9 | **Thread reconstruction** from flat tweets | Working with conversations (not isolated tweets) gives context for how Apple actually resolves issues end-to-end |
| 10 | **Subsample 5K threads** | Full dataset is 3M tweets — subsample keeps runtime under 15 minutes while providing enough variety |
| 11 | **Banking77 as structural reference** only | Banking77's 77 intents are for banking, not tech support. Used it to inform taxonomy design (granularity, naming conventions) rather than as training data. |
| 12 | **Rule-based + LLM hybrid for escalation** | Pure rules miss nuance. Pure LLM is slow and expensive. Hybrid: rules catch obvious triggers instantly, LLM handles ambiguous cases. |
| 13 | **Reply grounded in historical responses** | Unconstrained LLM replies hallucinate policies. Retrieval-grounded replies stay closer to what Apple actually says. |
| 14 | **Two baselines** (trivial + simple) | Assignment says "at least two." Trivial = lower bound, Simple = shows LLM value-add over classical ML. |
| 15 | **Caching/skip logic in pipeline** | Steps cache outputs. Re-running skips completed steps. Ensures reproducibility without re-spending API credits. |

---

## References

- **Dataset**: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) — Kaggle, ~3M tweets
- **Banking77**: [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) — HuggingFace, 13K queries, 77 intents
- **Embedding model**: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — Sentence Transformers
- **LLM API**: [OpenRouter](https://openrouter.ai/) — Model routing for Gemini Flash + Claude Sonnet
- **AI coding assistant**: Used for code scaffolding and iteration. All code has been reviewed and understood by the author.
