"""LLM-as-Judge evaluator for memory quality scoring.

Uses an LLM to evaluate the quality of retrieved memories against
ground-truth answers. Falls back to keyword matching when no judge is available.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass
class EvalScore:
    """Score for a single query evaluation."""

    system_name: str
    query_id: str
    accuracy: int  # 0-5: did it recall the right information?
    staleness_avoidance: int  # 0-5: did it avoid stale/outdated info?
    precision: int  # 0-5: was the answer concise and on-topic?
    total: int  # sum of above (max 15)

    reasoning: str = ""


class LLMJudge:
    """Evaluates memory recall quality using an LLM."""

    SCORING_PROMPT = """You are evaluating a memory system's recall quality.

SYSTEM: {system_name}
QUERY: {query}
GOLDEN ANSWER (correct): {golden_answer}
STALE SIGNALS (should NOT appear): {stale_signals}
CURRENT SIGNALS (SHOULD appear): {current_signals}
RECALLED ANSWER: {recalled_answer}

Score each dimension from 0 (worst) to 5 (best):

1. ACCURACY (0-5): Does the recalled answer contain the correct information
   from the golden answer? Rate 5 if fully correct, 0 if completely wrong.

2. STALENESS AVOIDANCE (0-5): Does the recalled answer AVOID stale/outdated
   information? Rate 5 if no stale info appears, 0 if it's full of old data.

3. PRECISION (0-5): Is the recalled answer concise and on-topic?
   Rate 5 if it's directly relevant with no fluff, 0 if mostly irrelevant.

Reply with ONLY this JSON format:
{{"accuracy": <0-5>, "staleness_avoidance": <0-5>, "precision": <0-5>, "reasoning": "<brief>"}}"""

    def __init__(self, model: str = "", api_key: str = ""):
        self.model = model or "gpt-4o-mini"
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def score(
        self,
        system_name: str,
        query_id: str,
        query: str,
        golden_answer: str,
        stale_signals: list[str],
        current_signals: list[str],
        recalled_answer: str,
    ) -> EvalScore:
        """Score a single query recall. Falls back to keyword scoring if no LLM."""

        if not self.api_key:
            return self._keyword_score(
                system_name, query_id, query, golden_answer,
                stale_signals, current_signals, recalled_answer,
            )

        try:
            prompt = self.SCORING_PROMPT.format(
                system_name=system_name,
                query=query,
                golden_answer=golden_answer,
                stale_signals=", ".join(stale_signals) if stale_signals else "none",
                current_signals=", ".join(current_signals) if current_signals else "none",
                recalled_answer=recalled_answer,
            )

            import urllib.request

            data = json.dumps({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a precise memory evaluation judge. Reply only with JSON."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 200,
            }).encode()

            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=data,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
                content = result["choices"][0]["message"]["content"]
                # Extract JSON from response
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    parsed = json.loads(content[start:end])
                else:
                    raise ValueError("No JSON found in response")

                return EvalScore(
                    system_name=system_name,
                    query_id=query_id,
                    accuracy=min(5, max(0, parsed.get("accuracy", 0))),
                    staleness_avoidance=min(5, max(0, parsed.get("staleness_avoidance", 0))),
                    precision=min(5, max(0, parsed.get("precision", 0))),
                    total=0,
                    reasoning=parsed.get("reasoning", ""),
                )
        except Exception as e:
            print(f"    judge LLM error: {e}, falling back to keyword scoring")
            return self._keyword_score(
                system_name, query_id, query, golden_answer,
                stale_signals, current_signals, recalled_answer,
            )

    @staticmethod
    def _keyword_score(
        system_name: str,
        query_id: str,
        query: str,
        golden_answer: str,
        stale_signals: list[str],
        current_signals: list[str],
        recalled_answer: str,
    ) -> EvalScore:
        """Simple keyword-based fallback scoring."""
        recalled_lower = recalled_answer.lower()
        golden_lower = golden_answer.lower()

        # Accuracy: how many golden answer words are in the recall?
        golden_words = set(golden_lower.split())
        if golden_words:
            overlap = sum(1 for w in golden_words if w in recalled_lower)
            accuracy = min(5, int((overlap / len(golden_words)) * 5))
        else:
            accuracy = 0

        # Staleness avoidance: penalize if stale signals appear
        staleness = 5
        for stale in stale_signals:
            if stale.lower() in recalled_lower:
                staleness = max(0, staleness - 3)

        # Precision: based on how much of the answer is relevant
        # Simple heuristic: shorter, more focused answers get higher precision
        recalled_words = len(recalled_answer.split())
        if recalled_words == 0:
            precision = 0
        elif recalled_words <= 20:
            precision = 5
        elif recalled_words <= 50:
            precision = 4
        elif recalled_words <= 100:
            precision = 3
        else:
            precision = 2

        total = accuracy + staleness + precision
        return EvalScore(
            system_name=system_name,
            query_id=query_id,
            accuracy=accuracy,
            staleness_avoidance=staleness,
            precision=precision,
            total=total,
            reasoning="keyword-based fallback scoring",
        )
