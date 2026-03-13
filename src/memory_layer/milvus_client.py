import logging
from pathlib import Path
from typing import List, Dict, Any
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

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

    Both collections use all-MiniLM-L6-v2 (dim=384) for consistent
    embedding space — queries against either collection use the same
    encoder.
    """

    def __init__(
        self,
        db_path: str = "data/milvus_storage/semantic_memory.db",
        collection_name: str = _EVENTS_COLLECTION,
    ):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.client = MilvusClient(db_path)
        self.collection_name = collection_name

        log.info("Loading SentenceTransformer embedding model...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dim = 384

        self._initialize_collection()
        self._initialize_entity_profiles()

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

    def insert_event_chunk(
        self,
        description: str,
        start_time: float,
        end_time: float,
        frame_id: int,
    ):
        """
        Embeds and inserts a VLM-generated semantic description into the vector store.

        Args:
            description: The raw text from the VLM (e.g., "Vehicle 4 tailgating Vehicle 9").
            start_time:  Start of the overlapping time window chunk (seconds).
            end_time:    End / presentation timestamp of the sampled frame (seconds).
            frame_id:    Integer frame index of the sampled frame.
        """
        if not description.strip():
            return

        # Convert text into a numerical vector embedding
        embedding = self.embedding_model.encode(description).tolist()

        # Prepare data payload. We store the time window and source frame as metadata
        # so the Agent can use them as pointers to query the Graph and Time-Series DBs.
        data = [{
            "vector": embedding,
            "text": description,
            "start_time": start_time,
            "end_time": end_time,
            "frame_id": frame_id,
            "time_window_pointer": f"{start_time:.1f}-{end_time:.1f}",
        }]

        # Insert the vector and metadata into Milvus Lite
        self.client.insert(collection_name=self.collection_name, data=data)

    def search_semantic_events(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Tool for the LangGraph Agent: Searches for semantic events using natural language.
        
        Returns:
            A list of matching text descriptions and their specific temporal windows.
        """
        # Embed the user's natural language query
        query_vector = self.embedding_model.encode(query).tolist()
        
        # Perform Approximate Nearest Neighbor (ANN) similarity search
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=top_k,
            output_fields=["text", "start_time", "end_time", "frame_id", "time_window_pointer"],
        )

        # Format the output into a clean dictionary list for the LLM agent to read
        formatted_results = []
        for hits in results:
            for hit in hits:
                entity = hit.get("entity", {})
                formatted_results.append({
                    "similarity_score": hit.get("distance"),
                    "description": entity.get("text"),
                    "time_window": entity.get("time_window_pointer"),
                    "frame_id": entity.get("frame_id"),
                    "presentation_timestamp_s": entity.get("end_time"),
                })
                
        return formatted_results
        
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

        Called in the macro-loop whenever a vehicle's behavior_summary
        is updated. The profile accumulates over the video: each call
        appends a new embedding snapshot; retrieval returns the most
        semantically relevant snapshot for a given query.

        This enables entity-centric queries that event-level search cannot
        answer — e.g. "which vehicle was most aggressive throughout the
        entire video?" or "find the vehicle that consistently sped".

        Args:
            track_id:   Integer tracker ID (e.g. 4 for "Vehicle 4").
            summary:    Natural language longitudinal narrative, e.g.
                        "Vehicle 4: moving → speeding → hard-braking.
                         Peak speed 18.2 m/s. Hard braking at t=10.5s."
            first_seen: Timestamp when the vehicle first appeared (s).
            last_seen:  Timestamp of the most recent observation (s).
        """
        if not summary.strip():
            return
        embedding = self.embedding_model.encode(summary).tolist()
        self.client.insert(
            collection_name=_PROFILES_COLLECTION,
            data=[{
                "vector":      embedding,
                "track_id":    track_id,
                "summary":     summary,
                "first_seen":  first_seen,
                "last_seen":   last_seen,
            }],
        )

    def search_entity_profiles(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Searches entity_profiles for vehicles matching a behavioral description.

        Enables global behavioral queries such as:
          - "Which vehicle was speeding the most?"
          - "Find the vehicle involved in the near-miss."
          - "Which vehicle behaved most aggressively?"

        Args:
            query:  Natural language behavioral description.
            top_k:  Number of results to return.

        Returns:
            List of dicts with: similarity_score, track_id, summary,
            first_seen, last_seen.
        """
        query_vector = self.embedding_model.encode(query).tolist()
        results = self.client.search(
            collection_name=_PROFILES_COLLECTION,
            data=[query_vector],
            limit=top_k,
            output_fields=["track_id", "summary", "first_seen", "last_seen"],
        )
        formatted = []
        for hits in results:
            for hit in hits:
                entity = hit.get("entity", {})
                formatted.append({
                    "similarity_score": hit.get("distance"),
                    "track_id":   entity.get("track_id"),
                    "summary":    entity.get("summary"),
                    "first_seen": entity.get("first_seen"),
                    "last_seen":  entity.get("last_seen"),
                })
        return formatted

    def close(self):
        """Safely closes the database connection."""
        self.client.close()