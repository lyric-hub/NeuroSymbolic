from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# ---------------------------------------------------------------------------
# Prototype sentence sets that define each intent class.
# The router computes cosine similarity between the live query and every
# prototype, then picks the class whose best prototype score is highest.
# ---------------------------------------------------------------------------
_FULL_ANALYSIS_PROTOTYPES = [
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
    # Fix 7: add traffic rule violation prototypes so queries like "ran a red light"
    # correctly route to full_analysis and reach the evaluate_traffic_rules tool.
    "Did any vehicle run a red light?",
    "Were any traffic rules violated at the intersection?",
    "Did any vehicle make an illegal turn or maneuver?",
    "Was any vehicle driving on the wrong side of the road?",
]

_SEMANTIC_LOOKUP_PROTOTYPES = [
    "What events happened near the intersection?",
    "Describe the traffic scene at 10 seconds.",
    "Were there any near-misses in the video?",
    "What was happening at the traffic light?",
    "Summarise the key events in the video.",
    "What did vehicle 4 do?",
    "List all interactions involving pedestrians.",
    "What occurred between vehicles 3 and 5?",
    "Which vehicles were tailgating?",
    "Describe the general situation at the crossroads.",
]

# ---------------------------------------------------------------------------
# Lazy-loaded embedding model and pre-cached prototype embeddings.
# Prototypes are encoded once on the first query, not on every call.
# ---------------------------------------------------------------------------
_embed_model: SentenceTransformer | None = None
_proto_embeddings: dict = {}


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        print("Loading intent classifier (all-MiniLM-L6-v2)...")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _get_proto_embeddings() -> dict:
    """Encodes prototype sentences once and caches the tensors."""
    global _proto_embeddings
    if not _proto_embeddings:
        model = _get_embed_model()
        _proto_embeddings = {
            "full_analysis": model.encode(
                _FULL_ANALYSIS_PROTOTYPES, convert_to_tensor=True
            ),
            "semantic_lookup": model.encode(
                _SEMANTIC_LOOKUP_PROTOTYPES, convert_to_tensor=True
            ),
        }
    return _proto_embeddings


def _classify_intent(query: str) -> str:
    """
    Returns 'full_analysis' if the query requires physics verification,
    'semantic_lookup' if a Milvus description alone is sufficient.

    Uses max-cosine-similarity against the two prototype embedding sets.
    """
    model = _get_embed_model()
    protos = _get_proto_embeddings()
    query_emb = model.encode(query, convert_to_tensor=True)

    full_score = cos_sim(query_emb, protos["full_analysis"]).max().item()
    semantic_score = cos_sim(query_emb, protos["semantic_lookup"]).max().item()

    intent = "full_analysis" if full_score >= semantic_score else "semantic_lookup"
    print(
        f"--- ROUTING: '{query[:60]}' → {intent} "
        f"(full={full_score:.3f}, semantic={semantic_score:.3f}) ---"
    )
    return intent

