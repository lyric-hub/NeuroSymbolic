import logging
import os
from pathlib import Path
from typing import List, Dict, Any

from pymilvus import MilvusClient

log = logging.getLogger(__name__)

_EMBED_PROVIDER = os.environ.get("AGENT_LLM_PROVIDER", "openai")
_EMBED_MODEL    = os.environ.get("AGENT_EMBED_MODEL", "all-minilm")

# Collection names
_EVENTS_COLLECTION   = "traffic_events"    # frame-level VLM event descriptions
_PROFILES_COLLECTION = "entity_profiles"   # per-vehicle longitudinal summaries


class SemanticVectorStore:
    """
    Manages semantic memory using Milvus Lite (two collections).

    traffic_events (event-level, frame-grained):
        Natural language sentences from VLM triples.
        e.g. "Vehicle 4 tailgating Vehicle 9."
        Enables: "find events where vehicles were close together."

    entity_profiles (entity-level, vehicle-grained):
        Longitudinal behavior summaries accumulated over the full video.
        e.g. "Vehicle 4: mostly speeding, hard-braked near intersection at 10s."
        Enables: "which vehicle was most aggressive throughout the video?"

    Embedding back-end is provider-switchable via AGENT_LLM_PROVIDER:
      - "ollama"  → Ollama HTTP API (all-minilm, no local PyTorch loaded)
      - default   → sentence-transformers all-MiniLM-L6-v2 (lazy-loaded)

    Both paths produce 384-dim vectors in the same embedding space.
    """

    def __init__(
        self,
        db_path: str = "data/milvus_storage/semantic_memory.db",
        collection_name: str = _EVENTS_COLLECTION,
    ):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.client = MilvusClient(db_path)
        self.collection_name = collection_name
        self.dim = 384

        # Sentence-transformers model — loaded lazily only when provider != ollama.
        self._st_model = None

        # Cache last-inserted summary per vehicle to suppress duplicate inserts.
        # upsert_entity_profile is called every macro-loop tick; without this a
        # 5-min video generates hundreds of identical profile vectors per vehicle.
        self._profile_cache: dict = {}

        self._initialize_collection()
        self._initialize_entity_profiles()

    # ------------------------------------------------------------------
    # Embedding back-end
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> list:
        """
        Returns a 384-dim float list for the given text.

        Routes to Ollama HTTP API (no local PyTorch) when
        AGENT_LLM_PROVIDER=ollama, otherwise uses the local
        sentence-transformers model (lazy-loaded on first call).
        """
        if _EMBED_PROVIDER == "ollama":
            import ollama as _ollama
            return _ollama.embeddings(model=_EMBED_MODEL, prompt=text)["embedding"]

        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            log.info("Loading SentenceTransformer embedding model...")
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._st_model.encode(text).tolist()

    # ------------------------------------------------------------------
    # Collection initialisation
    # ------------------------------------------------------------------

    def _initialize_collection(self) -> None:
        """Creates the traffic_events collection if it doesn't already exist."""
        if not self.client.has_collection(self.collection_name):
            log.info("Creating Milvus collection: %s", self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.dim,
                auto_id=True,
            )

    def _initialize_entity_profiles(self) -> None:
        """Creates the entity_profiles collection if it doesn't already exist."""
        if not self.client.has_collection(_PROFILES_COLLECTION):
            log.info("Creating Milvus collection: %s", _PROFILES_COLLECTION)
            self.client.create_collection(
                collection_name=_PROFILES_COLLECTION,
                dimension=self.dim,
                auto_id=True,
            )

    # ------------------------------------------------------------------
    # Event-level methods
    # ------------------------------------------------------------------

    def insert_event_chunk(
        self,
        description: str,
        start_time: float,
        end_time: float,
        frame_id: int,
    ) -> None:
        """
        Embeds and inserts a VLM-generated semantic description into the
        vector store.

        Args:
            description: The raw text from the VLM.
            start_time:  Start of the overlapping time window chunk (seconds).
            end_time:    End / presentation timestamp of the sampled frame (s).
            frame_id:    Integer frame index of the sampled frame.
        """
        if not description.strip():
            return
        self.client.insert(
            collection_name=self.collection_name,
            data=[{
                "vector":               self._encode(description),
                "text":                 description,
                "start_time":           start_time,
                "end_time":             end_time,
                "frame_id":             frame_id,
                "time_window_pointer":  f"{start_time:.1f}-{end_time:.1f}",
            }],
        )

    def search_semantic_events(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Searches for semantic events using natural language.

        Returns:
            List of matching text descriptions and their temporal windows.
        """
        results = self.client.search(
            collection_name=self.collection_name,
            data=[self._encode(query)],
            limit=top_k,
            output_fields=["text", "start_time", "end_time", "frame_id",
                           "time_window_pointer"],
        )
        formatted = []
        for hits in results:
            for hit in hits:
                entity = hit.get("entity", {})
                formatted.append({
                    "similarity_score":         round(
                        1.0 / (1.0 + hit.get("distance", 1.0)), 4
                    ),
                    "description":              entity.get("text"),
                    "time_window":              entity.get("time_window_pointer"),
                    "frame_id":                 entity.get("frame_id"),
                    "presentation_timestamp_s": entity.get("end_time"),
                })
        return formatted

    # ------------------------------------------------------------------
    # Entity profile methods (vehicle-level longitudinal memory)
    # ------------------------------------------------------------------

    def upsert_entity_profile(
        self,
        track_id: int,
        summary: str,
        first_seen: float,
        last_seen: float,
    ) -> None:
        """
        Embeds and stores a longitudinal behavior summary for one vehicle.

        Called in the macro-loop whenever a vehicle's behavior_summary is
        updated.  The profile accumulates over the video; retrieval returns
        the most semantically relevant snapshot for a given query.

        Args:
            track_id:   Integer tracker ID.
            summary:    Natural language longitudinal narrative.
            first_seen: Timestamp when the vehicle first appeared (s).
            last_seen:  Timestamp of the most recent observation (s).
        """
        if not summary.strip():
            return
        if self._profile_cache.get(track_id) == summary:
            return
        self._profile_cache[track_id] = summary
        self.client.insert(
            collection_name=_PROFILES_COLLECTION,
            data=[{
                "vector":     self._encode(summary),
                "track_id":   track_id,
                "summary":    summary,
                "first_seen": first_seen,
                "last_seen":  last_seen,
            }],
        )

    def search_entity_profiles(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Searches entity_profiles for vehicles matching a behavioral description.

        Args:
            query:  Natural language behavioral description.
            top_k:  Number of results to return.

        Returns:
            List of dicts with: similarity_score, track_id, summary,
            first_seen, last_seen.
        """
        results = self.client.search(
            collection_name=_PROFILES_COLLECTION,
            data=[self._encode(query)],
            limit=top_k,
            output_fields=["track_id", "summary", "first_seen", "last_seen"],
        )
        formatted = []
        for hits in results:
            for hit in hits:
                entity = hit.get("entity", {})
                formatted.append({
                    "similarity_score": round(
                        1.0 / (1.0 + hit.get("distance", 1.0)), 4
                    ),
                    "track_id":   entity.get("track_id"),
                    "summary":    entity.get("summary"),
                    "first_seen": entity.get("first_seen"),
                    "last_seen":  entity.get("last_seen"),
                })
        return formatted

    def close(self) -> None:
        """Safely closes the database connection."""
        self.client.close()
