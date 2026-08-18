#!/usr/bin/env python3
"""
recall_keywords.py — Extract search terms from a prompt for memory recall.

Called by memory_recall.sh. Reads prompt from stdin.
Outputs two lines:
  line 1: search_terms (for grep/keyword search)
  line 2: retro_query (for semantic search; may be empty or same as line 1)

This is a simple heuristic version. Replace with your own logic if needed.
The main job is to:
  1. Decide if the prompt is worth searching at all (output nothing to skip)
  2. Extract the most searchable terms for keyword matching
  3. Optionally reformulate for semantic search
"""

import sys
import re

SKIP_PATTERNS = [
    r'^\s*$',                          # empty
    r'^(ok|okay|yes|no|thanks|sure)\s*\.?\s*$',  # very short acknowledgments
    r'^\d+$',                          # pure numbers
]

MIN_LENGTH = 6  # prompts shorter than this are skipped


def extract_terms(prompt: str) -> tuple[str, str]:
    """
    Returns (search_terms, retro_query).
    search_terms: for grep (keywords)
    retro_query: for semantic search (may be empty = use original prompt)
    """
    # Strip common filler words for keyword search
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                 'could', 'should', 'may', 'might', 'shall', 'can', 'i', 'you',
                 'we', 'they', 'it', 'this', 'that', 'what', 'how', 'why',
                 'when', 'where', 'who', 'which', 'and', 'or', 'but', 'so',
                 'if', 'then', 'just', 'me', 'my', 'your', 'our'}

    words = re.findall(r'\b\w{3,}\b', prompt.lower())
    keywords = [w for w in words if w not in stopwords]

    if not keywords:
        return "", ""

    # Use top 3 most distinctive words for grep
    search_terms = "|".join(keywords[:3])

    # For semantic search, use the original prompt (or a reformulation)
    # Return empty to signal "use original prompt"
    retro_query = ""

    return search_terms, retro_query


def main():
    prompt = sys.stdin.read().strip()

    if len(prompt) < MIN_LENGTH:
        sys.exit(0)

    for pattern in SKIP_PATTERNS:
        if re.match(pattern, prompt, re.IGNORECASE):
            sys.exit(0)

    search_terms, retro_query = extract_terms(prompt)

    if not search_terms:
        sys.exit(0)

    print(search_terms)
    print(retro_query)


if __name__ == "__main__":
    main()
