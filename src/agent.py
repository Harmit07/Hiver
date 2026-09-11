"""
AI Support Agent for AppleSupport
===================================
The core agent that performs three tasks:
1. Classify incoming customer messages into intents
2. Draft replies grounded in historical AppleSupport responses
3. Decide whether to auto-handle or escalate to a human

Usage:
    from src.agent import SupportAgent
    agent = SupportAgent()
    result = agent.process("My iPhone screen is cracked")
"""

import json
import re
from pathlib import Path
from typing import Optional

from src.llm_client import LLMClient
from src.retrieval import ThreadRetriever

# ─── Config ────────────────────────────────────────────────────────────────────

TAXONOMY_FILE = Path("src/intent_taxonomy.json")


# ─── Escalation Rules ──────────────────────────────────────────────────────────

# Keywords that trigger automatic escalation
ESCALATION_KEYWORDS = {
    "high_urgency": [
        "lawsuit", "legal", "attorney", "lawyer", "sue",
        "bbb", "better business bureau", "consumer protection",
        "scam", "fraud", "stolen identity", "identity theft",
        "safety", "dangerous", "fire", "burning", "explod",
    ],
    "strong_negative_emotion": [
        "worst company", "never buying", "boycott",
        "class action", "report you", "going viral",
    ],
    "privacy_security": [
        "hacked", "data breach", "compromised", "unauthorized access",
        "someone logged in", "account compromised",
    ],
}


class SupportAgent:
    """
    AI Customer Support Agent for @AppleSupport.
    
    Pipeline:
        message → classify_intent → retrieve_similar → draft_reply → decide_escalation
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
        self.retriever = ThreadRetriever()
        self.taxonomy = self._load_taxonomy()
        self._index_built = False
    
    def _load_taxonomy(self) -> dict:
        """Load the intent taxonomy."""
        if TAXONOMY_FILE.exists():
            with open(TAXONOMY_FILE) as f:
                data = json.load(f)
            return data.get("intents", data)
        return {}
    
    def ensure_index(self):
        """Build retrieval index if not yet built."""
        if not self._index_built:
            self.retriever.build_index()
            self._index_built = True
    
    # ── 1. Intent Classification ────────────────────────────────────────────────
    
    def classify_intent(self, message: str) -> dict:
        """
        Classify a customer message into one of the defined intents.
        
        Uses few-shot LLM classification with the taxonomy + example messages.
        
        Returns:
            {
                "intent": str,          # The intent label
                "confidence": float,    # 0.0 - 1.0
                "reasoning": str,       # Why this intent was chosen
            }
        """
        # Build intent descriptions for the prompt
        intent_list = []
        for cid, intent_data in self.taxonomy.items():
            label = intent_data.get("intent_label", f"cluster_{cid}")
            desc = intent_data.get("description", "")
            keywords = intent_data.get("keywords", [])
            examples = intent_data.get("example_messages", [])
            
            intent_str = f"- **{label}**: {desc}"
            if keywords:
                intent_str += f"\n  Keywords: {', '.join(keywords[:5])}"
            if examples:
                intent_str += f"\n  Examples: " + " | ".join(f'"{e[:60]}"' for e in examples[:3])
            intent_list.append(intent_str)
        
        intents_text = "\n".join(intent_list)
        intent_labels = [
            self.taxonomy[cid].get("intent_label", f"cluster_{cid}")
            for cid in self.taxonomy
        ]
        
        system_prompt = """You are an intent classifier for Apple customer support.
Your job is to classify incoming customer messages into exactly one intent category.
Be precise and consider the full context of the message.
Always respond in valid JSON format."""

        prompt = f"""Classify the following customer message into one of these intent categories:

{intents_text}

Customer message: "{message}"

Respond in JSON:
{{
    "intent": "<one of: {', '.join(intent_labels)}>",
    "confidence": <float between 0.0 and 1.0>,
    "reasoning": "<one sentence explaining why>"
}}"""

        try:
            result = self.llm.call_json(prompt, system_prompt=system_prompt, temperature=0.1)
            # Validate intent label
            if result.get("intent") not in intent_labels:
                # Find closest match
                result["intent"] = self._closest_intent(result.get("intent", ""), intent_labels)
            return result
        except Exception as e:
            return {
                "intent": "other",
                "confidence": 0.0,
                "reasoning": f"Classification failed: {str(e)}",
            }
    
    def _closest_intent(self, predicted: str, valid_labels: list) -> str:
        """Find the closest valid intent label (fuzzy match)."""
        predicted_lower = predicted.lower().replace(" ", "_")
        for label in valid_labels:
            if label.lower() in predicted_lower or predicted_lower in label.lower():
                return label
        return valid_labels[0] if valid_labels else "other"
    
    # ── 2. Reply Drafting ───────────────────────────────────────────────────────
    
    def draft_reply(
        self,
        message: str,
        intent: str,
        similar_threads: Optional[list] = None,
    ) -> dict:
        """
        Draft a reply grounded in historical AppleSupport responses.
        
        Uses retrieved similar threads to understand how Apple typically
        responds to similar issues, then generates a reply in Apple's voice.
        
        Returns:
            {
                "reply": str,               # The drafted reply
                "grounding_sources": list,  # Thread IDs used as reference
                "strategy": str,            # Brief description of reply strategy
            }
        """
        self.ensure_index()
        
        # Retrieve similar threads if not provided
        if similar_threads is None:
            similar_threads = self.retriever.search(message, k=5)
        
        # Build grounding context from similar threads
        grounding_context = []
        source_ids = []
        for i, thread in enumerate(similar_threads[:5]):
            grounding_context.append(
                f"Example {i+1} (similarity: {thread['similarity_score']:.2f}):\n"
                f"  Customer: \"{thread['customer_message']}\"\n"
                f"  Apple Support: \"{thread['agent_response']}\""
            )
            source_ids.append(thread['thread_id'])
        
        grounding_text = "\n\n".join(grounding_context)
        
        system_prompt = """You are drafting a customer support reply on behalf of @AppleSupport on Twitter.

Rules:
1. Match AppleSupport's real voice: professional, empathetic, concise (Twitter character limits)
2. Provide concrete, actionable next steps when possible
3. Reference specific Apple support channels (DM, apple.com/support, Genius Bar) when appropriate
4. Never make up policies, prices, or technical specs — if unsure, direct to official support
5. Keep it under 280 characters when possible (it's Twitter)
6. Always acknowledge the customer's frustration before providing solutions
7. Never promise specific outcomes (refunds, replacements) — only direct to appropriate channels"""

        prompt = f"""Draft a reply for this customer message.

Intent category: {intent}

Customer message: "{message}"

Here are real examples of how @AppleSupport has historically handled similar issues:

{grounding_text}

Generate a reply that follows AppleSupport's actual communication style shown above.

Respond in JSON:
{{
    "reply": "<the drafted reply text>",
    "strategy": "<brief description of what approach this reply takes>"
}}"""

        try:
            result = self.llm.call_json(prompt, system_prompt=system_prompt, temperature=0.4)
            result["grounding_sources"] = source_ids
            return result
        except Exception as e:
            return {
                "reply": "We'd like to help! Please DM us your details so we can look into this further.",
                "grounding_sources": source_ids,
                "strategy": f"Fallback generic reply (error: {str(e)})",
            }
    
    # ── 3. Escalation Decision ──────────────────────────────────────────────────
    
    def decide_escalation(
        self,
        message: str,
        intent: str,
        draft_reply: str,
    ) -> dict:
        """
        Decide whether a message should be auto-handled or escalated.
        
        Uses a hybrid approach:
        - Rule-based checks for clear escalation triggers (safety, legal, security)
        - LLM reasoning for ambiguous cases
        
        Returns:
            {
                "decision": "auto_handle" | "escalate",
                "reason": str,
                "confidence": float,
                "trigger": str,  # "rule" or "llm"
            }
        """
        # Phase 1: Rule-based escalation check
        rule_result = self._check_escalation_rules(message)
        if rule_result:
            return rule_result
        
        # Phase 2: Check taxonomy for intent-level escalation
        for cid, intent_data in self.taxonomy.items():
            if intent_data.get("intent_label") == intent:
                if intent_data.get("typically_needs_escalation", False):
                    return {
                        "decision": "escalate",
                        "reason": f"Intent '{intent}' is flagged as typically needing human review",
                        "confidence": 0.7,
                        "trigger": "taxonomy",
                    }
        
        # Phase 3: LLM-based decision for ambiguous cases
        return self._llm_escalation_decision(message, intent, draft_reply)
    
    def _check_escalation_rules(self, message: str) -> Optional[dict]:
        """Check rule-based escalation triggers."""
        msg_lower = message.lower()
        
        for category, keywords in ESCALATION_KEYWORDS.items():
            for kw in keywords:
                if kw in msg_lower:
                    return {
                        "decision": "escalate",
                        "reason": f"Triggered by {category} keyword: '{kw}'",
                        "confidence": 0.95,
                        "trigger": "rule",
                    }
        return None
    
    def _llm_escalation_decision(
        self, message: str, intent: str, draft_reply: str
    ) -> dict:
        """Use LLM to decide escalation for ambiguous cases."""
        
        system_prompt = """You are a support triage system deciding whether a customer message can be 
auto-handled by an AI bot or needs to be escalated to a human agent.

Escalate when:
- The issue requires access to account-specific data (order lookup, billing details)
- The customer is very upset and needs human empathy
- The issue is complex and multi-faceted
- There's a safety or security concern
- The AI's draft reply seems inadequate

Auto-handle when:
- The issue can be resolved with general troubleshooting steps
- The customer is asking a common FAQ-style question
- The draft reply adequately addresses the concern
- The issue is simple and well-understood"""

        prompt = f"""Customer message: "{message}"
Detected intent: {intent}
AI draft reply: "{draft_reply}"

Should this be auto-handled or escalated to a human?

Respond in JSON:
{{
    "decision": "auto_handle" or "escalate",
    "reason": "<specific reason for the decision>",
    "confidence": <float 0.0 to 1.0>
}}"""

        try:
            result = self.llm.call_json(prompt, system_prompt=system_prompt, temperature=0.1)
            result["trigger"] = "llm"
            return result
        except Exception:
            # Default: escalate if unsure (conservative)
            return {
                "decision": "escalate",
                "reason": "Escalation decision failed — defaulting to human review (conservative)",
                "confidence": 0.3,
                "trigger": "fallback",
            }
    
    # ── Full Pipeline ───────────────────────────────────────────────────────────
    
    def process(self, message: str) -> dict:
        """
        Run the full agent pipeline on a customer message.
        
        Returns:
            {
                "input_message": str,
                "intent": {...},
                "similar_threads": [...],
                "reply": {...},
                "escalation": {...},
            }
        """
        self.ensure_index()
        
        # Step 1: Classify intent
        intent_result = self.classify_intent(message)
        
        # Step 2: Retrieve similar threads
        similar_threads = self.retriever.search(message, k=5)
        
        # Step 3: Draft reply
        reply_result = self.draft_reply(message, intent_result["intent"], similar_threads)
        
        # Step 4: Decide escalation
        escalation_result = self.decide_escalation(
            message, intent_result["intent"], reply_result["reply"]
        )
        
        return {
            "input_message": message,
            "intent": intent_result,
            "similar_threads": [
                {
                    "thread_id": t["thread_id"],
                    "customer_message": t["customer_message"],
                    "agent_response": t["agent_response"],
                    "similarity_score": t["similarity_score"],
                }
                for t in similar_threads[:3]  # Keep top 3 for output
            ],
            "reply": reply_result,
            "escalation": escalation_result,
        }


# ─── Quick Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = SupportAgent()
    
    test_messages = [
        "My iPhone 14 screen is cracked after a small drop. Is this covered under warranty?",
        "I can't sign into my Apple ID and I've been locked out for 2 days",
        "How do I update my iPhone to iOS 17?",
        "I was charged $9.99 for an app I never downloaded!!! This is fraud!",
        "Your products are terrible and I'm going to sue Apple",
    ]
    
    for msg in test_messages:
        print(f"\n{'='*60}")
        print(f"Customer: {msg}")
        result = agent.process(msg)
        print(f"Intent: {result['intent']['intent']} (conf={result['intent']['confidence']:.2f})")
        print(f"Reply: {result['reply']['reply']}")
        print(f"Escalation: {result['escalation']['decision']} — {result['escalation']['reason']}")
