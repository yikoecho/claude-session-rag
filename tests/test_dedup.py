"""
Standalone verification for rag.hybrid._dedup_by_vector.

Verifies the three fixes without hitting the real embedding API or LanceDB:
  Case 1: two semantically-duplicate items, different dates -> newer kept
  Case 2: dup pair where the OLDER one is a keyword hit (_exempt) -> exempt kept (not newer)
  Case 3: vector reuse -> items carrying a vector do NOT trigger re-embedding

We inject vectors two ways:
  - via vec_map (simulates vectors reused from the retrieval layer)
  - via a fake embedding client (simulates the fallback re-embed path)
The fake client records which texts it was asked to embed, so we can assert
that reused-vector items never reach it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import types
# Stub heavy modules pulled in at import time (index.vector -> lancedb, openai).
# We only exercise _dedup_by_vector, which needs neither.
class _StubOpenAI:
    def __init__(self, *a, **k): pass
openai_stub = types.ModuleType("openai"); openai_stub.OpenAI = _StubOpenAI
sys.modules.setdefault("openai", openai_stub)
lancedb_stub = types.ModuleType("lancedb"); lancedb_stub.__path__ = []
lancedb_stub.connect = lambda *a, **k: None
lancedb_index = types.ModuleType("lancedb.index"); lancedb_index.FTS = object
sys.modules.setdefault("lancedb", lancedb_stub)
sys.modules.setdefault("lancedb.index", lancedb_index)

import numpy as np
import rag.hybrid as H

# ---- fake embedding client -------------------------------------------------
class _Emb:
    def __init__(self, vecs): self.data = [type("D", (), {"embedding": v})() for v in vecs]

class _FakeEmbeddings:
    def __init__(self, parent): self.parent = parent
    def create(self, model, input):
        # record every text the fallback path had to embed
        self.parent.embedded.extend(input)
        # hand back a deterministic vector per text (from the registry)
        return _Emb([self.parent.registry[t] for t in input])

class FakeClient:
    def __init__(self, registry):
        self.registry = registry      # text -> raw vector
        self.embedded = []            # texts that got re-embedded (should exclude reused ones)
        self.embeddings = _FakeEmbeddings(self)

def run_case(name, results, vec_map, registry, threshold=0.9):
    fake = FakeClient(registry)
    H.get_client = lambda: fake                      # bypass real API
    kept = H._dedup_by_vector([dict(r) for r in results], threshold, dict(vec_map))
    return kept, fake.embedded

def texts(items): return [r["text"] for r in items]

# Two nearly-identical unit vectors (cos ~0.9999 > 0.9) => "duplicates".
# A third orthogonal vector => unrelated, must survive.
DUP_A = [1.0, 0.0, 0.0]
DUP_B = [0.999, 0.0447, 0.0]     # almost parallel to DUP_A
ORTHO = [0.0, 1.0, 0.0]

fails = 0
def check(cond, msg):
    global fails
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond: fails += 1

# ---- Case 1: newer date wins among plain duplicates ------------------------
print("Case 1: duplicate pair, no exempt -> newer date kept")
results = [
    {"text": "2026-07-14: 台风天只睡1小时那次的记录", "score": 0.5},
    {"text": "2026-07-29: 熬夜到2:40的记录（语义重复）", "score": 0.5},
    {"text": "2026-06-15: 完全无关的docker部署", "score": 0.5},
]
vec_map = {
    results[0]["text"][:80]: DUP_A,
    results[1]["text"][:80]: DUP_B,
    results[2]["text"][:80]: ORTHO,
}
kept, embedded = run_case("c1", results, vec_map, registry={})
t = texts(kept)
check(any("2026-07-29" in x for x in t), "newer (07-29) survived")
check(not any("2026-07-14" in x for x in t), "older (07-14) removed")
check(any("2026-06-15" in x for x in t), "unrelated item survived")
check(embedded == [], "no re-embedding (all vectors reused from vec_map)")

# ---- Case 2: exempt keyword hit beats a newer non-exempt duplicate ---------
print("Case 2: older item is _exempt keyword hit -> exempt kept over newer")
results = [
    {"text": "2026-07-14: 关键词命中的旧记录", "score": 0.9, "_exempt": True},
    {"text": "2026-07-29: 语义重复但较新的RRF结果", "score": 0.5},
]
vec_map = {
    results[0]["text"][:80]: DUP_A,
    results[1]["text"][:80]: DUP_B,
}
kept, embedded = run_case("c2", results, vec_map, registry={})
t = texts(kept)
check(any("关键词命中" in x for x in t), "exempt (older) kept")
check(not any("较新的RRF" in x for x in t), "newer non-exempt dropped (exempt wins over date)")
check(all("_exempt" not in r for r in kept), "_exempt marker stripped from output")

# ---- Case 3: vector reuse vs fallback re-embed -----------------------------
print("Case 3: only vector-less items hit the embed fallback")
results = [
    {"text": "2026-08-01: 向量层来的候选（带vector）", "score": 0.5},
    {"text": "2026-08-02: BM25来的候选（无vector，需补算）", "score": 0.5},
]
vec_map = { results[0]["text"][:80]: ORTHO }          # only item 0 carries a vector
registry = { results[1]["text"]: [0.0, 0.0, 1.0] }    # item 1 must be re-embedded
kept, embedded = run_case("c3", results, vec_map, registry=registry)
check(results[0]["text"] not in embedded, "vector-carrying item NOT re-embedded")
check(results[1]["text"] in embedded, "vector-less item WAS re-embedded")
check(len(embedded) == 1, "exactly one re-embed call happened")

print()
print("RESULT:", "ALL PASS" if fails == 0 else f"{fails} CHECK(S) FAILED")
sys.exit(1 if fails else 0)
