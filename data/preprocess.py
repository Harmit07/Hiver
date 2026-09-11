"""
Data Preprocessing Pipeline
============================
Filters AppleSupport conversations from the Twitter Customer Support dataset,
reconstructs multi-turn conversation threads, cleans text, and outputs 
structured JSONL for downstream use.

Usage:
    python -m data.preprocess
"""

import pandas as pd
import json
import re
import os
from pathlib import Path
from collections import defaultdict

# ─── Config ────────────────────────────────────────────────────────────────────
BRAND = "AppleSupport"
RAW_CSV = Path("data/raw/twcs.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "apple_threads.jsonl"
STATS_FILE = OUTPUT_DIR / "preprocessing_stats.json"
MAX_THREADS = 5000  # Subsample for working corpus

# ─── Text Cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Clean tweet text while preserving meaningful content."""
    if not isinstance(text, str):
        return ""
    
    # Remove @mentions (anonymized handles like @115712)
    text = re.sub(r'@\w+', '', text)
    
    # Normalize masked PII tokens
    text = text.replace('__email__', '[EMAIL]')
    text = text.replace('__url__', '[URL]')
    text = text.replace('__phone__', '[PHONE]')
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_raw_mentions(text: str) -> list:
    """Extract @mentions from raw text before cleaning."""
    if not isinstance(text, str):
        return []
    return re.findall(r'@(\w+)', text)


# ─── Thread Reconstruction ─────────────────────────────────────────────────────

def _normalize_id(val) -> int:
    """Convert a tweet ID to int, handling float NaN and string formats."""
    try:
        if pd.isna(val):
            return -1
        return int(float(val))
    except (ValueError, TypeError):
        return -1


def build_threads(df: pd.DataFrame) -> list[dict]:
    """
    Reconstruct conversation threads from flat tweet data.
    
    Uses response_tweet_id and in_response_to_tweet_id to link tweets
    into multi-turn conversations. Each thread starts with a customer 
    inbound message and includes all subsequent exchanges.
    """
    # Normalize IDs to int for consistent lookup
    df = df.copy()
    df['tweet_id_int'] = df['tweet_id'].apply(_normalize_id)
    df['in_response_to_int'] = df['in_response_to_tweet_id'].apply(_normalize_id)
    
    # Index tweets by int ID for fast lookup
    tweet_lookup = {}
    for _, row in df.iterrows():
        tid = row['tweet_id_int']
        if tid >= 0:
            tweet_lookup[tid] = row
    
    # Build adjacency: parent_id -> list of child tweet_ids
    children = defaultdict(list)
    for _, row in df.iterrows():
        parent_id = row['in_response_to_int']
        if parent_id >= 0:
            children[parent_id].append(row['tweet_id_int'])
    
    # Find thread roots: inbound tweets that are NOT responses to another tweet in our dataset
    roots = []
    for _, row in df.iterrows():
        if row['inbound'] == True:
            parent_id = row['in_response_to_int']
            if parent_id < 0 or parent_id not in tweet_lookup:
                roots.append(row['tweet_id_int'])
    
    print(f"    Found {len(roots)} potential thread roots")
    
    # BFS to build each thread
    threads = []
    seen = set()
    
    for root_id in roots:
        if root_id in seen:
            continue
        
        thread_messages = []
        queue = [root_id]
        
        while queue:
            current_id = queue.pop(0)
            if current_id in seen or current_id not in tweet_lookup:
                continue
            seen.add(current_id)
            
            tweet = tweet_lookup[current_id]
            role = "customer" if tweet['inbound'] == True else "agent"
            
            thread_messages.append({
                "role": role,
                "text": clean_text(tweet['text']),
                "raw_text": tweet['text'] if isinstance(tweet['text'], str) else "",
                "tweet_id": str(tweet['tweet_id']),
                "author_id": str(tweet['author_id']),
                "created_at": str(tweet.get('created_at', '')),
            })
            
            # Add children (responses to this tweet)
            for child_id in children.get(current_id, []):
                queue.append(child_id)
            
            # Also follow response_tweet_id links
            resp_ids = str(tweet.get('response_tweet_id', ''))
            if resp_ids and resp_ids != 'nan':
                for rid in resp_ids.split(','):
                    rid = rid.strip()
                    if rid:
                        try:
                            queue.append(int(float(rid)))
                        except (ValueError, TypeError):
                            pass
        
        # Sort messages chronologically within thread
        thread_messages.sort(key=lambda m: m.get('created_at', ''))
        
        if len(thread_messages) >= 2:
            has_customer = any(m['role'] == 'customer' for m in thread_messages)
            has_agent = any(m['role'] == 'agent' for m in thread_messages)
            
            if has_customer and has_agent:
                customer_msgs = [m for m in thread_messages if m['role'] == 'customer']
                agent_msgs = [m for m in thread_messages if m['role'] == 'agent']
                
                threads.append({
                    "thread_id": str(root_id),
                    "messages": thread_messages,
                    "customer_first_message": customer_msgs[0]['text'],
                    "agent_first_response": agent_msgs[0]['text'],
                    "num_turns": len(thread_messages),
                    "num_customer_msgs": len(customer_msgs),
                    "num_agent_msgs": len(agent_msgs),
                })
    
    return threads


# ─── Main Pipeline ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Data Preprocessing Pipeline")
    print(f"Brand: {BRAND}")
    print("=" * 60)
    
    # Step 1: Load raw data
    print(f"\n[1/5] Loading raw dataset from {RAW_CSV}...")
    df = pd.read_csv(RAW_CSV)
    print(f"  Total tweets: {len(df):,}")
    
    # Normalize ID columns to int for consistent matching
    df['tweet_id_int'] = df['tweet_id'].apply(_normalize_id)
    df['in_response_to_int'] = df['in_response_to_tweet_id'].apply(_normalize_id)
    
    # Step 2: Filter to AppleSupport conversations
    print(f"\n[2/5] Filtering to {BRAND} conversations...")
    
    # Get all tweets where AppleSupport is the author (outbound)
    apple_outbound = df[df['author_id'] == BRAND]
    print(f"  AppleSupport outbound tweets: {len(apple_outbound):,}")
    
    # Get all tweet IDs that AppleSupport responded to (parent tweets)
    apple_responded_to = set(
        apple_outbound['in_response_to_int'][apple_outbound['in_response_to_int'] >= 0]
    )
    print(f"  Apple responded to {len(apple_responded_to):,} unique parent tweets")
    
    # Get all AppleSupport tweet IDs (for finding replies TO Apple)
    apple_tweet_ids = set(
        apple_outbound['tweet_id_int'][apple_outbound['tweet_id_int'] >= 0]
    )
    
    # Get inbound tweets: those Apple responded to OR those responding to Apple
    apple_inbound_mask = (
        (df['tweet_id_int'].isin(apple_responded_to)) |
        (df['in_response_to_int'].isin(apple_tweet_ids))
    )
    apple_inbound = df[apple_inbound_mask & (df['inbound'] == True)]
    print(f"  Related inbound tweets: {len(apple_inbound):,}")
    
    # Combine
    apple_df = pd.concat([apple_outbound, apple_inbound]).drop_duplicates(subset='tweet_id')
    apple_df = apple_df.sort_values('created_at')
    print(f"  Total AppleSupport-related tweets: {len(apple_df):,}")
    
    # Step 3: Reconstruct threads
    print(f"\n[3/5] Reconstructing conversation threads...")
    threads = build_threads(apple_df)
    print(f"  Complete threads found: {len(threads):,}")
    
    if len(threads) == 0:
        print("  ERROR: No threads found! Check data filtering logic.")
        return []
    
    # Thread length distribution
    lengths = [t['num_turns'] for t in threads]
    print(f"  Avg thread length: {sum(lengths)/len(lengths):.1f} messages")
    print(f"  Min/Max: {min(lengths)}/{max(lengths)} messages")
    
    # Step 4: Subsample
    print(f"\n[4/5] Subsampling to {MAX_THREADS} threads...")
    
    import random
    random.seed(42)
    
    short = [t for t in threads if t['num_turns'] <= 3]
    medium = [t for t in threads if 4 <= t['num_turns'] <= 6]
    long = [t for t in threads if t['num_turns'] > 6]
    
    # Proportional sampling
    total = len(threads)
    n_short = min(len(short), int(MAX_THREADS * len(short) / total))
    n_medium = min(len(medium), int(MAX_THREADS * len(medium) / total))
    n_long = min(len(long), MAX_THREADS - n_short - n_medium)
    
    sampled = (
        random.sample(short, min(n_short, len(short))) +
        random.sample(medium, min(n_medium, len(medium))) +
        random.sample(long, min(n_long, len(long)))
    )
    random.shuffle(sampled)
    
    print(f"  Sampled: {len(sampled)} threads")
    print(f"    Short (≤3 msgs): {min(n_short, len(short))}")
    print(f"    Medium (4-6 msgs): {min(n_medium, len(medium))}")
    print(f"    Long (>6 msgs): {min(n_long, len(long))}")
    
    # Step 5: Save
    print(f"\n[5/5] Saving to {OUTPUT_FILE}...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, 'w') as f:
        for thread in sampled:
            f.write(json.dumps(thread) + '\n')
    
    # Save stats
    stats = {
        "brand": BRAND,
        "raw_tweets_total": int(len(df)),
        "apple_tweets_total": int(len(apple_df)),
        "threads_total": len(threads),
        "threads_sampled": len(sampled),
        "thread_length_distribution": {
            "short_lte_3": min(n_short, len(short)),
            "medium_4_6": min(n_medium, len(medium)),
            "long_gt_6": min(n_long, len(long)),
        },
        "avg_thread_length": round(sum(lengths) / len(lengths), 2),
    }
    
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"Done! Output: {OUTPUT_FILE}")
    print(f"Stats: {STATS_FILE}")
    print(f"{'=' * 60}")
    
    return sampled


if __name__ == "__main__":
    main()
