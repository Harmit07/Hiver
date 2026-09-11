# Evaluation Results

_Generated: 2026-09-11 19:46:48_

## Intent Classification

| System | Accuracy | Macro F1 |
|--------|----------|----------|
| **LLM Agent (ours)** | **79.5%** | **0.835** |
| Simple (TF-IDF + LogReg) | 67.5% | 0.761 |
| Trivial (majority class) | 9.5% | 0.014 |

### Per-Intent Breakdown (LLM Agent)

| Intent | Precision | Recall | F1 | Support |
|--------|-----------|--------|----|---------|
| account_storage_photos | 1.00 | 1.00 | 1.00 | 3 |
| battery_charging_issue | 1.00 | 0.88 | 0.94 | 17 |
| billing_store_account | 1.00 | 0.67 | 0.80 | 3 |
| device_frustration_complaint | 1.00 | 0.72 | 0.84 | 25 |
| feature_malfunction | 1.00 | 1.00 | 1.00 | 3 |
| general_bug_report | 0.71 | 0.90 | 0.79 | 49 |
| general_inquiry | 0.34 | 0.63 | 0.44 | 19 |
| hardware_malfunction | 1.00 | 0.75 | 0.86 | 8 |
| ios_update_issue | 1.00 | 0.82 | 0.90 | 45 |
| keyboard_autocorrect_bug | 1.00 | 0.65 | 0.79 | 17 |
| media_services_issue | 1.00 | 0.60 | 0.75 | 5 |
| performance_lag | 1.00 | 0.83 | 0.91 | 6 |

## Escalation Decision

| System | Accuracy | Escalation Recall | Escalation Precision |
|--------|----------|-------------------|---------------------|
| **LLM Agent (ours)** | **91.5%** | **0.91** | **0.57** |
| Simple (keyword) | 74.5% | 0.23 | 0.13 |
| Trivial (never) | 89.0% | 0.00 | 0.00 |

> **Note**: Escalation recall is the most important metric — missed escalations (false negatives) 
> are far worse than unnecessary escalations (false positives).

## Reply Quality (LLM-as-Judge)

_Evaluated using Claude Sonnet as judge (different model from pipeline to avoid self-evaluation bias)_

| Dimension | Mean Score | Std Dev |
|-----------|-----------|--------|
| **Overall** | **4.52/5.0** | **±0.29** |

### Quality by Difficulty

