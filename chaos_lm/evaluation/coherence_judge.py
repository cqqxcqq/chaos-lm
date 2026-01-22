# evaluation/coherence_judge.py
"""
CHAOS-LM Coherence Judge
Uses external LLM to evaluate coherence of generated text.
"""

import os
from typing import Optional, Dict, Any, List
import json
from dataclasses import dataclass

from config.config import MetricsConfig


@dataclass
class CoherenceScore:
    """Container for coherence evaluation results"""
    score: float  # 1-10
    reasoning: str
    aspects: Dict[str, float]  # Individual aspect scores
    raw_response: str


class CoherenceJudge:
    """
    Uses an external LLM (via Groq API) to evaluate text coherence.
    
    The judge evaluates:
    - Grammatical correctness
    - Logical flow
    - Semantic coherence
    - (Ignores factual accuracy - this is expected to be wrong)
    """
    
    SYSTEM_PROMPT = """You are a text coherence evaluator. Your job is to rate the coherence of text on a scale of 1-10.

IMPORTANT: You should IGNORE factual accuracy. The text may contain intentionally false or nonsensical facts - this is expected.

Focus ONLY on:
1. Grammar: Is the text grammatically structured? (Even alien grammar counts if consistent)
2. Logic Flow: Do sentences connect to each other logically?
3. Internal Consistency: Does the text contradict itself?
4. Readability: Can a human parse and understand the structure?

Rating Scale:
1-2: Complete gibberish, random tokens
3-4: Some word patterns but mostly incoherent
5-6: Partially coherent, has structure but breaks down
7-8: Mostly coherent with minor issues
9-10: Fully coherent (grammatically, not factually)

Respond in JSON format:
{
    "score": <1-10>,
    "grammar_score": <1-10>,
    "logic_score": <1-10>,
    "consistency_score": <1-10>,
    "readability_score": <1-10>,
    "reasoning": "<brief explanation>"
}"""

    EVAL_PROMPT_TEMPLATE = """Please evaluate the coherence of the following text:

---
{text}
---

Remember: Ignore factual accuracy. Focus only on grammatical structure and logical flow."""

    def __init__(
        self,
        config: MetricsConfig,
        api_key: Optional[str] = None
    ):
        self.config = config
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = config.judge_model
        self.client = None
        
        if self.api_key:
            self._init_client()
    
    def _init_client(self):
        """Initialize Groq client"""
        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
        except ImportError:
            print("Warning: groq package not installed. Coherence judging unavailable.")
            self.client = None
        except Exception as e:
            print(f"Warning: Failed to initialize Groq client: {e}")
            self.client = None
    
    def evaluate(self, text: str) -> Optional[CoherenceScore]:
        """
        Evaluate coherence of a single text.
        
        Args:
            text: Text to evaluate
            
        Returns:
            CoherenceScore object or None if evaluation fails
        """
        if not self.client:
            return self._fallback_evaluation(text)
        
        if not text or len(text.strip()) < 10:
            return CoherenceScore(
                score=1.0,
                reasoning="Text too short to evaluate",
                aspects={},
                raw_response=""
            )
        
        try:
            prompt = self.EVAL_PROMPT_TEMPLATE.format(text=text[:2000])  # Limit length
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            raw_response = response.choices[0].message.content
            
            # Parse JSON response
            try:
                # Find JSON in response
                json_start = raw_response.find('{')
                json_end = raw_response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = raw_response[json_start:json_end]
                    result = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
                
                return CoherenceScore(
                    score=float(result.get('score', 5)),
                    reasoning=result.get('reasoning', ''),
                    aspects={
                        'grammar': float(result.get('grammar_score', 5)),
                        'logic': float(result.get('logic_score', 5)),
                        'consistency': float(result.get('consistency_score', 5)),
                        'readability': float(result.get('readability_score', 5))
                    },
                    raw_response=raw_response
                )
            except (json.JSONDecodeError, ValueError) as e:
                # Try to extract score from text
                import re
                score_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*10', raw_response)
                if score_match:
                    return CoherenceScore(
                        score=float(score_match.group(1)),
                        reasoning=raw_response,
                        aspects={},
                        raw_response=raw_response
                    )
                return self._fallback_evaluation(text)
                
        except Exception as e:
            print(f"Coherence evaluation error: {e}")
            return self._fallback_evaluation(text)
    
    def _fallback_evaluation(self, text: str) -> CoherenceScore:
        """
        Fallback evaluation using heuristics when API is unavailable.
        """
        if not text:
            return CoherenceScore(
                score=1.0,
                reasoning="Empty text",
                aspects={},
                raw_response=""
            )
        
        # Simple heuristics
        words = text.split()
        
        # Check for basic structure
        has_punctuation = any(p in text for p in '.!?')
        has_capital_start = text[0].isupper() if text else False
        avg_word_length = sum(len(w) for w in words) / max(len(words), 1)
        
        # Check for repetition
        unique_ratio = len(set(words)) / max(len(words), 1)
        
        # Check for common patterns
        has_articles = any(w.lower() in ['a', 'an', 'the'] for w in words)
        has_verbs = any(w.lower().endswith(('ing', 'ed', 'es', 's')) for w in words)
        
        # Compute score
        score = 5.0
        
        if has_punctuation:
            score += 1
        if has_capital_start:
            score += 0.5
        if 3 < avg_word_length < 8:
            score += 1
        if unique_ratio > 0.5:
            score += 1
        if has_articles:
            score += 0.5
        if has_verbs:
            score += 0.5
        
        # Penalty for very short or very repetitive text
        if len(words) < 5:
            score -= 2
        if unique_ratio < 0.3:
            score -= 2
        
        score = max(1.0, min(10.0, score))
        
        return CoherenceScore(
            score=score,
            reasoning="Heuristic evaluation (API unavailable)",
            aspects={
                'grammar': score,
                'logic': 