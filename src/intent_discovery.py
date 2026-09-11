"""
Intent Discovery & Taxonomy
==============================
Discovers intent categories from AppleSupport customer messages using:
1. Banking77 dataset (HuggingFace) as structural reference for how to build a taxonomy
2. Sentence-transformer embeddings + K-Means clustering on actual customer messages
3. LLM-assisted labeling of clusters into human-readable intents

The result: a finalized intent taxonomy (8-12 intents) grounded in real data.

Usage:
    python -m src.intent_discovery
"""

import json
import os
import random
import numpy as np
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv

load_dotenv()


from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ─── Config ────────────────────────────────────────────────────────────────────

THREADS_FILE = Path("data/processed/apple_threads.jsonl")
TAXONOMY_FILE = Path("src/intent_taxonomy.json")
CLUSTER_SAMPLES_FILE = Path("data/processed/cluster_samples.json")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
K_RANGE = range(7, 14)  # Test cluster counts 7-13
SAMPLES_PER_CLUSTER = 20
RANDOM_SEED = 42


# ─── Banking77 Reference ───────────────────────────────────────────────────────

def load_banking77_intents() -> list[str]:
    """
    Banking77 intent categories — used as structural reference for how
    fine-grained and descriptive intent labels should be.
    
    Source: HuggingFace PolyAI/banking77 (13k queries, 77 labelled intents)
    """
    return [
        "activate_my_card", "age_limit", "apple_pay_or_google_pay",
        "atm_support", "automatic_top_up", "balance_not_updated_after_bank_transfer",
        "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
        "cancel_transfer", "card_about_to_expire", "card_acceptance",
        "card_arrival", "card_delivery_estimate", "card_linking",
        "card_not_working", "card_payment_fee_charged", "card_payment_not_recognised",
        "card_payment_wrong_exchange_rate", "card_swallowed", "cash_withdrawal_charge",
        "cash_withdrawal_not_recognised", "change_pin", "compromised_card",
        "contactless_not_working", "country_support", "declined_card_payment",
        "declined_cash_withdrawal", "declined_transfer", "direct_debit_payment_not_recognised",
        "disposable_card_limits", "edit_personal_details", "exchange_charge",
        "exchange_rate", "exchange_via_app", "extra_charge_on_statement",
        "failed_transfer", "fiat_currency_support", "get_disposable_virtual_card",
        "get_physical_card", "getting_spare_card", "getting_virtual_card",
        "lost_or_stolen_card", "lost_or_stolen_phone", "order_physical_card",
        "passcode_forgotten", "pending_card_payment", "pending_cash_poke",
        "pending_top_up", "pending_transfer", "pin_blocked", "receiving_money",
        "Refund_not_showing_up", "request_refund", "reverted_card_payment",
        "supported_cards_and_currencies", "terminate_account",
        "top_up_by_bank_transfer_charge", "top_up_by_card_charge",
        "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits",
        "top_up_reverted", "topping_up_by_card", "transaction_charged_twice",
        "transfer_fee_charged", "transfer_into_account",
        "transfer_not_received_by_recipient", "transfer_timing",
        "unable_to_verify_identity", "verify_my_identity",
        "verify_source_of_funds", "visa_or_mastercard",
        "why_verify_identity", "wrong_amount_of_cash_received",
        "wrong_exchange_rate_for_cash_withdrawal",
    ]


# ─── Clustering Pipeline ───────────────────────────────────────────────────────

def load_customer_messages() -> list[dict]:
    """Load customer first messages from preprocessed threads."""
    messages = []
    with open(THREADS_FILE) as f:
        for line in f:
            thread = json.loads(line)
            msg = thread['customer_first_message']
            if msg and len(msg) > 10:  # Filter very short/empty messages
                messages.append({
                    "thread_id": thread['thread_id'],
                    "text": msg,
                    "agent_response": thread['agent_first_response'],
                })
    return messages


def embed_messages(messages: list[dict], model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    """Embed customer messages using sentence-transformers."""
    print(f"  Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    texts = [m['text'] for m in messages]
    print(f"  Embedding {len(texts)} messages...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=256)
    
    return embeddings


def find_optimal_k(embeddings: np.ndarray, k_range=K_RANGE) -> int:
    """Find optimal number of clusters using silhouette analysis."""
    print(f"  Testing cluster counts: {list(k_range)}")
    
    scores = {}
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, sample_size=5000)
        scores[k] = score
        print(f"    k={k}: silhouette={score:.4f}")
    
    best_k = max(scores, key=scores.get)
    print(f"  Best k: {best_k} (silhouette={scores[best_k]:.4f})")
    return best_k


def cluster_messages(
    messages: list[dict],
    embeddings: np.ndarray,
    k: int,
) -> dict:
    """
    Cluster messages and sample representatives from each cluster.
    Returns cluster_id -> list of sample messages.
    """
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    clusters = {}
    for i in range(k):
        cluster_indices = np.where(labels == i)[0]
        cluster_msgs = [messages[idx] for idx in cluster_indices]
        
        # Sample representative messages
        random.seed(RANDOM_SEED + i)
        samples = random.sample(
            cluster_msgs,
            min(SAMPLES_PER_CLUSTER, len(cluster_msgs))
        )
        
        clusters[i] = {
            "size": len(cluster_msgs),
            "samples": samples,
            "all_messages": cluster_msgs,  # Keep for later use
        }
    
    return clusters, labels


def generate_taxonomy_with_llm(clusters: dict) -> dict:
    """
    Use LLM to generate human-readable intent labels for each cluster.
    Falls back to rule-based labeling if LLM is not available.
    """
    try:
        from src.llm_client import LLMClient
        client = LLMClient()
        return _llm_taxonomy(clusters, client)
    except Exception as e:
        print(f"  LLM not available ({e}), using rule-based taxonomy...")
        return _rule_based_taxonomy(clusters)


def _llm_taxonomy(clusters: dict, client) -> dict:
    """Use LLM to label clusters."""
    banking77_ref = load_banking77_intents()
    
    taxonomy = {}
    for cluster_id, cluster_data in clusters.items():
        samples_text = "\n".join(
            f"  - \"{s['text']}\"" for s in cluster_data['samples'][:15]
        )
        
        prompt = f"""You are analyzing customer support messages to Apple Support on Twitter.

Below are {len(cluster_data['samples'])} representative customer messages from a single cluster 
(cluster {cluster_id}, total size: {cluster_data['size']} messages):

{samples_text}

For reference, here are some example intent categories from the Banking77 dataset 
(a well-known intent classification benchmark):
{', '.join(banking77_ref[:15])}

Based on the messages above, provide:
1. A short, snake_case intent label (e.g., "device_hardware_issue", "account_access")
2. A one-sentence human-readable description
3. Three representative keywords/phrases
4. Whether this intent typically needs human escalation (true/false)

Respond in JSON format:
{{
    "intent_label": "...",
    "description": "...",
    "keywords": ["...", "...", "..."],
    "typically_needs_escalation": true/false,
    "example_messages": ["...", "...", "..."]
}}"""
        
        try:
            result = client.call_json(prompt, temperature=0.1)
            taxonomy[str(cluster_id)] = {
                **result,
                "cluster_id": cluster_id,
                "cluster_size": cluster_data['size'],
            }
            print(f"    Cluster {cluster_id} ({cluster_data['size']} msgs) → {result.get('intent_label', 'unknown')}")
        except Exception as e:
            print(f"    Cluster {cluster_id}: LLM failed ({e}), using fallback")
            taxonomy[str(cluster_id)] = _fallback_label(cluster_id, cluster_data)
    
    return taxonomy


def _rule_based_taxonomy(clusters: dict) -> dict:
    """Fallback: generate taxonomy using keyword analysis."""
    # Common AppleSupport topic keywords
    topic_keywords = {
        "device_issue": ["broken", "screen", "battery", "charging", "not working", "hardware"],
        "software_update": ["update", "ios", "upgrade", "install", "download", "version"],
        "account_access": ["password", "apple id", "locked", "sign in", "login", "2fa", "verification"],
        "billing_payment": ["charge", "bill", "refund", "payment", "subscription", "purchase"],
        "connectivity": ["wifi", "bluetooth", "cellular", "connection", "network", "internet"],
        "app_store": ["app", "download", "store", "purchase", "app store"],
        "icloud": ["icloud", "storage", "backup", "sync", "photos"],
        "general_inquiry": ["how", "help", "question", "need", "can you"],
        "feedback_complaint": ["terrible", "worst", "hate", "frustrated", "disappointed", "angry"],
        "service_request": ["repair", "replacement", "warranty", "genius", "appointment"],
    }
    
    taxonomy = {}
    for cluster_id, cluster_data in clusters.items():
        all_text = " ".join(s['text'].lower() for s in cluster_data['samples'])
        
        # Score each topic
        best_topic = "other"
        best_score = 0
        for topic, keywords in topic_keywords.items():
            score = sum(1 for kw in keywords if kw in all_text)
            if score > best_score:
                best_score = score
                best_topic = topic
        
        taxonomy[str(cluster_id)] = _fallback_label(cluster_id, cluster_data, best_topic)
    
    return taxonomy


def _fallback_label(cluster_id: int, cluster_data: dict, label: str = "unknown") -> dict:
    """Generate a fallback label entry."""
    return {
        "intent_label": label,
        "description": f"Cluster {cluster_id} ({cluster_data['size']} messages)",
        "keywords": [],
        "typically_needs_escalation": False,
        "cluster_id": cluster_id,
        "cluster_size": cluster_data['size'],
        "example_messages": [s['text'] for s in cluster_data['samples'][:3]],
    }


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Intent Discovery Pipeline")
    print("=" * 60)
    
    # Step 1: Load messages
    print(f"\n[1/5] Loading customer messages from {THREADS_FILE}...")
    messages = load_customer_messages()
    print(f"  Loaded {len(messages)} messages")
    
    # Step 2: Embed
    print(f"\n[2/5] Computing embeddings...")
    embeddings = embed_messages(messages)
    print(f"  Embedding shape: {embeddings.shape}")
    
    # Save embeddings for later use (retrieval)
    np.save("data/processed/customer_embeddings.npy", embeddings)
    with open("data/processed/customer_messages.json", 'w') as f:
        # Save without all_messages (too large)
        json.dump([{k: v for k, v in m.items()} for m in messages], f)
    print(f"  Saved embeddings and messages index")
    
    # Step 3: Find optimal k
    print(f"\n[3/5] Finding optimal number of clusters...")
    optimal_k = find_optimal_k(embeddings)
    
    # Step 4: Cluster
    print(f"\n[4/5] Clustering with k={optimal_k}...")
    clusters, labels = cluster_messages(messages, embeddings, optimal_k)
    
    # Save cluster assignments
    for i, msg in enumerate(messages):
        msg['cluster_id'] = int(labels[i])
    
    # Print cluster sizes
    cluster_sizes = Counter(labels)
    for cid, size in sorted(cluster_sizes.items()):
        print(f"    Cluster {cid}: {size} messages")
    
    # Save cluster samples (without all_messages to keep file small)
    cluster_samples = {}
    for cid, data in clusters.items():
        cluster_samples[str(cid)] = {
            "size": data['size'],
            "samples": data['samples'],
        }
    with open(CLUSTER_SAMPLES_FILE, 'w') as f:
        json.dump(cluster_samples, f, indent=2)
    
    # Step 5: Generate taxonomy
    print(f"\n[5/5] Generating intent taxonomy...")
    taxonomy = generate_taxonomy_with_llm(clusters)
    
    # Add Banking77 reference note
    taxonomy_output = {
        "metadata": {
            "source_dataset": "Twitter Customer Support (AppleSupport)",
            "reference_taxonomy": "Banking77 (PolyAI/banking77, HuggingFace)",
            "clustering_method": f"K-Means (k={optimal_k})",
            "embedding_model": EMBEDDING_MODEL,
            "num_messages_clustered": len(messages),
        },
        "intents": taxonomy,
    }
    
    with open(TAXONOMY_FILE, 'w') as f:
        json.dump(taxonomy_output, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"Taxonomy saved to {TAXONOMY_FILE}")
    print(f"Cluster samples saved to {CLUSTER_SAMPLES_FILE}")
    print(f"\nDiscovered {len(taxonomy)} intents:")
    for cid, intent in taxonomy.items():
        print(f"  {intent.get('intent_label', 'unknown')}: {intent.get('description', '')}")
    print(f"{'=' * 60}")
    
    return taxonomy


if __name__ == "__main__":
    main()
