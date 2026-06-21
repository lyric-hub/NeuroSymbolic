import logging
import os

import numpy as np

log = logging.getLogger(__name__)

# Set AGENT_LLM_PROVIDER=ollama in .env to use Ollama all-minilm for embeddings.
# Default: sentence_transformers (local PyTorch).
_EMBED_PROVIDER = os.environ.get("AGENT_LLM_PROVIDER", "openai")
_EMBED_MODEL    = os.environ.get("AGENT_EMBED_MODEL", "all-minilm")

# ---------------------------------------------------------------------------
# Prototype sentence sets that define each intent class.
# ---------------------------------------------------------------------------
_FULL_ANALYSIS_PROTOTYPES = [
    # Kinematic / physics queries
    "Did vehicle X brake too hard?",
    "How fast was vehicle 4 going at the time of impact?",
    "What was the speed when the collision happened?",
    "Was the driver exceeding the speed limit?",
    "Calculate the deceleration of vehicle 9.",
    "Verify the physics data for vehicle 3.",
    "What was the acceleration profile of vehicle 7?",
    "Did vehicle 2 perform an emergency stop?",
    "What is the maximum speed recorded for vehicle 5?",
    "How hard did vehicle 6 decelerate?",
    # Rule violation queries
    "Did any vehicle run a red light?",
    "Were any traffic rules violated at the intersection?",
    "Did any vehicle make an illegal turn or maneuver?",
    "Was any vehicle driving on the wrong side of the road?",
    "List all traffic rule violations detected in the video.",
    "Was there a collision between Vehicle 2 and Vehicle 5?",
    "Did Vehicle 2 collide with Vehicle 5?",
    # Zone / gate / flow queries
    "How many vehicles passed through the North gate?",
    "What are the origin-destination pairs for zone crossings?",
    "How many vehicles entered and exited the zone?",
    "What was the average dwell time in the zone?",
    "Show me gate entry and exit counts.",
    # Volume / count queries
    "How many vehicles per hour were observed?",
    "What is the Peak Hour Factor for this site?",
    "When was the peak traffic period?",
    "Give me the AADT estimation for this road section.",
    "How many vehicles pass through per hour?",
    "What is the traffic volume per 15 minutes?",
    # Queue / congestion queries
    "Was there a traffic jam or queue in the video?",
    "How long did the congestion last?",
    "Did a shockwave or queue back-propagation form?",
    "Was there a stationary backup of vehicles?",
    "Identify congestion episodes in the video.",
    # Turning movement / TMC queries
    "How many vehicles turned left from the North approach?",
    "What is the dominant turning movement at the intersection?",
    "Give me the turning movement count matrix.",
    "What is the intersection flow breakdown by approach arm?",
    "Were there any U-turns detected?",
    # TTC / conflict queries
    "What was the time-to-collision between the two vehicles?",
    "How close did Vehicle 4 get to Vehicle 9?",
    "Was there a critical TTC conflict in the video?",
    "What was the minimum gap between the vehicles?",
    "What was the dangerous separation between the following vehicles?",
    # Speed compliance / V85
    "What is the 85th percentile speed on this road?",
    "Are vehicles complying with the speed limit?",
    "What is the V85 design speed for this section?",
    # Vehicle class queries
    "How fast were the motorcycles driving?",
    "What violations did trucks commit?",
    "Were any bicycles detected in the video?",
    # Incident causation
    "What led to the near-miss and what were the vehicles doing beforehand?",
    "Why did the accident happen?",
    "What caused the near-miss between the two vehicles?",
    # Relational queries that need kinematic confirmation
    "Were any vehicles following too closely?",
    "Which vehicles were interacting at t=30 seconds?",
    # Incident causation with narrative phrasing
    "What led to the accident and what were the vehicles doing before the crash?",
    "What sequence of events caused the collision?",
    "What led to the crash before the intersection?",
    "What caused the accident near the junction?",
    "What happened before the collision at the crossing?",
    # Behavioral longitudinal analysis
    "Which vehicle was driving most aggressively throughout the entire video?",
    "Which vehicle exhibited the most dangerous behaviour across the whole recording?",
]

_SEMANTIC_LOOKUP_PROTOTYPES = [
    "What events happened near the intersection?",
    "Describe the traffic scene at 10 seconds.",
    "What was happening at the traffic light?",
    "Summarise the key events in the video.",
    "List all interactions involving pedestrians.",
    "Describe the general situation at the crossroads.",
    "Give me a general overview of the traffic scene.",
    "What was the overall traffic situation in the video?",
    "Was the traffic smooth throughout the video?",
]

# ---------------------------------------------------------------------------
# Embedding back-ends
# ---------------------------------------------------------------------------
_st_model = None          # sentence_transformers model instance
_proto_embeddings: dict = {}


def _embed_sentence_transformers(texts: list) -> np.ndarray:
    """Encode texts with the local sentence-transformers model."""
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        print("Loading intent classifier (all-MiniLM-L6-v2)...")
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    vecs = _st_model.encode(texts)
    return np.array(vecs, dtype=np.float32)


def _embed_ollama(texts: list) -> np.ndarray:
    """Encode texts via Ollama's embedding API (no local PyTorch)."""
    import ollama as _ollama
    vectors = [
        _ollama.embeddings(model=_EMBED_MODEL, prompt=t)["embedding"]
        for t in texts
    ]
    return np.array(vectors, dtype=np.float32)


def _embed(texts: list) -> np.ndarray:
    """Dispatch to the configured embedding back-end."""
    if _EMBED_PROVIDER == "ollama":
        return _embed_ollama(texts)
    return _embed_sentence_transformers(texts)


def _max_cos_sim(query_vec: np.ndarray, proto_mat: np.ndarray) -> float:
    """Max cosine similarity between a query vector and a prototype matrix."""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    p = proto_mat / (np.linalg.norm(proto_mat, axis=1, keepdims=True) + 1e-9)
    return float((p @ q).max())


# ---------------------------------------------------------------------------
# Prototype cache — encoded once on the first query.
# ---------------------------------------------------------------------------

def _get_proto_embeddings() -> dict:
    """Encodes prototype sentences once and caches as numpy arrays."""
    global _proto_embeddings
    if not _proto_embeddings:
        _proto_embeddings = {
            "full_analysis": _embed(_FULL_ANALYSIS_PROTOTYPES),
            "semantic_lookup": _embed(_SEMANTIC_LOOKUP_PROTOTYPES),
        }
    return _proto_embeddings


def _classify_intent(query: str) -> str:
    """
    Returns 'full_analysis' or 'semantic_lookup' based on max cosine
    similarity against the two prototype embedding sets.
    """
    protos    = _get_proto_embeddings()
    query_vec = _embed([query])[0]

    full_score     = _max_cos_sim(query_vec, protos["full_analysis"])
    semantic_score = _max_cos_sim(query_vec, protos["semantic_lookup"])

    intent = "full_analysis" if full_score >= semantic_score else "semantic_lookup"
    log.debug(
        "ROUTING: '%s' → %s (full=%.3f, semantic=%.3f)",
        query[:60], intent, full_score, semantic_score,
    )
    return intent
