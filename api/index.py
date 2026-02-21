"""
Enhanced Football Laws of the Game RAG API
Version 3.0.0 - Gemini Query Understanding + Hybrid RAG
"""
import os
import json
import re
import logging
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from huggingface_hub import InferenceClient
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import time

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path(__file__).parent.parent
CHUNKS_PATH = BASE_DIR / "rag_chunks" / "chunks.jsonl"
EMBEDDINGS_PATH = BASE_DIR / "rag_chunks" / "embeddings.npy"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_FAST_MODEL = "gemini-2.0-flash"   # Used for query understanding (cheaper/faster)
TOP_K = 10

# context controls
MAX_CHARS_PER_CHUNK = 2500
MAX_TOTAL_CONTEXT_CHARS = 15000

# session management
SESSION_TIMEOUT_MINUTES = 30
MAX_CONVERSATION_HISTORY = 5

# MongoDB configuration
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "fcrules")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "query_stats")
MONGODB_SESSIONS_COLLECTION = os.getenv("MONGODB_SESSIONS_COLLECTION", "sessions")

# API keys from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("Warning: HF_TOKEN not found in environment variables")
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not found in environment variables")
if not MONGODB_URI:
    print("Warning: MONGODB_URI not found in environment variables")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("fcrules.api")
counter_lock = threading.Lock()


# -----------------------------
# GEMINI QUERY UNDERSTANDING  (NEW - Step 1)
# -----------------------------
def understand_query(question: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
    """
    Step 1: Use a fast Gemini model to interpret, disambiguate, and enrich
    the user's question before hitting the retriever.

    Returns a dict with:
      - rewritten_query: technically precise version of the question
      - likely_laws: list of law numbers most likely relevant
      - key_concepts: list of key football-law terms to search
      - interpretation_note: brief note on any ambiguity
    """
    if not GEMINI_API_KEY:
        logger.warning("No GEMINI_API_KEY – skipping query understanding")
        return {
            "rewritten_query": question,
            "likely_laws": [],
            "key_concepts": [],
            "interpretation_note": ""
        }

    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        logger.error("Unable to import google-genai for query understanding: %s", e)
        return {
            "rewritten_query": question,
            "likely_laws": [],
            "key_concepts": [],
            "interpretation_note": ""
        }

    # Build a short conversation context string if we have history
    history_text = ""
    if conversation_history:
        recent = conversation_history[-2:]
        lines = []
        for qa in recent:
            lines.append(f"Q: {qa['question']}")
            lines.append(f"A (summary): {qa['answer'][:150]}...")
        history_text = "\nRecent conversation:\n" + "\n".join(lines)

    prompt = f"""You are a Football (Soccer) Laws of the Game expert assistant.

A user asked: "{question}"{history_text}

Your job is to interpret the question and prepare it for a legal text search of the Laws of the Game.

Respond ONLY with a valid JSON object — no markdown, no explanation, no code fences:

{{
  "rewritten_query": "A technically precise version of the question using official Laws of the Game terminology. Expand abbreviations. Clarify ambiguous pronouns using conversation context if needed.",
  "likely_laws": [list of integer law numbers most likely relevant — pick the best 1-4],
  "key_concepts": ["list", "of", "precise", "football-law", "terms", "to", "search"],
  "interpretation_note": "One sentence describing any ambiguity or assumption you made. Empty string if none."
}}

Reference guide for common Laws:
- Law 1: Field of Play
- Law 2: The Ball
- Law 3: The Players (substitutions, team officials, extra persons)
- Law 4: Players Equipment
- Law 5: The Referee
- Law 6: The Other Match Officials
- Law 7: Duration of the Match
- Law 8: The Start and Restart of Play (dropped ball, kick-off)
- Law 9: Ball In and Out of Play
- Law 10: Determining the Outcome (scoring a goal)
- Law 11: Offside
- Law 12: Fouls and Misconduct (fouls, free kicks, cards, handball, charges, tackles)
- Law 13: Free Kicks
- Law 14: The Penalty Kick
- Law 15: The Throw-In
- Law 16: The Goal Kick
- Law 17: The Corner Kick

Key terminology tips:
- "shoulder charge" → "fair charge", "physical challenge", "Law 12", "playing distance", "ball within playing distance"
- "ball hits referee" → "touches a match official", "Law 9", "dropped ball", "promising attack"
- "handball" → "handling the ball", "deliberate handling", "Law 12"
- "own goal" → "kicker's own goal", "thrower's own goal", "corner kick awarded", "Law 10"
- "offside trap" → "offside position", "Law 11", "active play"
- "time wasting" → "delaying restart", "Law 12", "cautionable offense"
"""

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        resp = client.models.generate_content(
            model=GEMINI_FAST_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        text = (resp.text or "").strip()
        # Strip any accidental markdown fences
        text = re.sub(r"```json|```", "", text).strip()
        result = json.loads(text)

        # Validate and sanitise
        return {
            "rewritten_query": str(result.get("rewritten_query", question)),
            "likely_laws": [int(x) for x in result.get("likely_laws", []) if str(x).isdigit()],
            "key_concepts": [str(x) for x in result.get("key_concepts", [])],
            "interpretation_note": str(result.get("interpretation_note", ""))
        }

    except Exception as e:
        logger.warning("Query understanding failed (%s); falling back to raw query", e)
        return {
            "rewritten_query": question,
            "likely_laws": [],
            "key_concepts": [],
            "interpretation_note": ""
        }


# -----------------------------
# SESSION MANAGER
# -----------------------------
class SessionManager:
    """Manages conversation sessions in-memory with MongoDB persistence option"""

    def __init__(self, mongo_db=None):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.mongo_db = mongo_db
        self.sessions_collection = None

        if mongo_db is not None:
            self.sessions_collection = mongo_db[MONGODB_SESSIONS_COLLECTION]
            try:
                self.sessions_collection.create_index(
                    "last_activity",
                    expireAfterSeconds=SESSION_TIMEOUT_MINUTES * 60
                )
                logger.info("Session TTL index created")
            except Exception as e:
                logger.warning("Could not create TTL index: %s", e)

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        session_data = {
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc),
            "last_activity": datetime.now(timezone.utc),
            "conversation_history": []
        }
        self.sessions[session_id] = session_data

        if self.sessions_collection is not None:
            try:
                self.sessions_collection.insert_one(session_data.copy())
            except Exception as e:
                logger.warning("Could not persist session to MongoDB: %s", e)

        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data by ID"""

        def _normalize_datetimes(session: dict) -> dict:
            """Ensure all datetime fields are timezone-aware (MongoDB returns naive datetimes)"""
            for field in ("last_activity", "created_at"):
                val = session.get(field)
                if val and isinstance(val, datetime) and val.tzinfo is None:
                    session[field] = val.replace(tzinfo=timezone.utc)
            # Also fix datetimes nested inside conversation_history
            for qa in session.get("conversation_history", []):
                ts = qa.get("timestamp")
                if ts and isinstance(ts, datetime) and ts.tzinfo is None:
                    qa["timestamp"] = ts.replace(tzinfo=timezone.utc)
            return session

        # Try memory first
        if session_id in self.sessions:
            session = _normalize_datetimes(self.sessions[session_id])
            if datetime.now(timezone.utc) - session["last_activity"] > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
                self.delete_session(session_id)
                return None
            return session

        # Try MongoDB
        if self.sessions_collection is not None:
            try:
                session = self.sessions_collection.find_one({"session_id": session_id})
                if session:
                    session = _normalize_datetimes(session)
                    self.sessions[session_id] = session
                    return session
            except Exception as e:
                logger.warning("Could not load session from MongoDB: %s", e)

        return None

    def update_session(self, session_id: str, question: str, answer: str, evidence: List[Dict]):
        session = self.get_session(session_id)
        if not session:
            return

        session["last_activity"] = datetime.now(timezone.utc)
        qa_pair = {
            "timestamp": datetime.now(timezone.utc),
            "question": question,
            "answer": answer,
            "evidence_laws": list(set([e.get("law_number") for e in evidence if e.get("law_number")]))
        }
        session["conversation_history"].append(qa_pair)

        if len(session["conversation_history"]) > MAX_CONVERSATION_HISTORY:
            session["conversation_history"] = session["conversation_history"][-MAX_CONVERSATION_HISTORY:]

        if self.sessions_collection is not None:
            try:
                self.sessions_collection.update_one(
                    {"session_id": session_id},
                    {
                        "$set": {
                            "last_activity": session["last_activity"],
                            "conversation_history": session["conversation_history"]
                        }
                    }
                )
            except Exception as e:
                logger.warning("Could not update session in MongoDB: %s", e)

    def delete_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

        if self.sessions_collection is not None:
            try:
                self.sessions_collection.delete_one({"session_id": session_id})
            except Exception as e:
                logger.warning("Could not delete session from MongoDB: %s", e)

    def cleanup_expired_sessions(self):
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=SESSION_TIMEOUT_MINUTES)
        expired = [
            sid for sid, session in self.sessions.items()
            if session["last_activity"] < cutoff
        ]
        for sid in expired:
            del self.sessions[sid]
        if expired:
            logger.info("Cleaned up %d expired sessions", len(expired))


# -----------------------------
# MONGODB CLIENT
# -----------------------------
class MongoDBCounter:
    def __init__(self, uri: str, database: str, collection: str):
        self.uri = uri
        self.database_name = database
        self.collection_name = collection
        self.client: Optional[MongoClient] = None
        self.db = None
        self.collection = None
        self._connected = False

    def connect(self):
        try:
            self.client = MongoClient(
                self.uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                retryWrites=True
            )
            self.client.admin.command('ping')
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
            self._connected = True

            result = self.collection.find_one({"_id": "query_counter"})
            if result is None:
                self.collection.insert_one({
                    "_id": "query_counter",
                    "count": 0,
                    "last_updated": time.time()
                })
                logger.info("Initialized new query counter in MongoDB")
            else:
                logger.info("Connected to MongoDB. Current count: %s", result.get("count", 0))

            return True

        except ConnectionFailure as e:
            logger.error("Failed to connect to MongoDB: %s", e)
            self._connected = False
            return False
        except Exception as e:
            logger.error("Unexpected error connecting to MongoDB: %s", e)
            self._connected = False
            return False

    def is_connected(self) -> bool:
        if not self._connected or self.client is None:
            return False
        try:
            self.client.admin.command('ping')
            return True
        except:
            self._connected = False
            return False

    def get_count(self) -> int:
        if not self.is_connected():
            if not self.connect():
                return 0
        try:
            result = self.collection.find_one({"_id": "query_counter"})
            if result:
                return int(result.get("count", 0))
            return 0
        except Exception as e:
            logger.error("Error reading count from MongoDB: %s", e)
            return 0

    def increment(self) -> tuple[int, bool]:
        if not self.is_connected():
            if not self.connect():
                return 0, False
        try:
            result = self.collection.find_one_and_update(
                {"_id": "query_counter"},
                {
                    "$inc": {"count": 1},
                    "$set": {"last_updated": time.time()}
                },
                return_document=True,
                upsert=True
            )
            if result:
                new_count = int(result.get("count", 0))
                logger.info("Query counter incremented to: %s", new_count)
                return new_count, True
            return 0, False
        except OperationFailure as e:
            logger.error("MongoDB operation failed during increment: %s", e)
            return 0, False
        except Exception as e:
            logger.error("Unexpected error incrementing counter: %s", e)
            return 0, False

    def close(self):
        if self.client:
            try:
                self.client.close()
                logger.info("MongoDB connection closed")
            except Exception as e:
                logger.error("Error closing MongoDB connection: %s", e)
            finally:
                self._connected = False


# -----------------------------
# PYDANTIC MODELS
# -----------------------------
class QuestionRequest(BaseModel):
    question: str = Field(..., description="The question to ask about football laws")
    top_k: Optional[int] = Field(10, description="Number of top chunks to retrieve", ge=1, le=20)
    include_raw_chunks: Optional[bool] = Field(False, description="Include raw chunk data in response")


class Evidence(BaseModel):
    rank: int
    citation: str
    text_preview: str
    full_text: Optional[str] = None
    law_number: Optional[int] = None


class ConversationContext(BaseModel):
    timestamp: str
    question: str
    answer_preview: str
    laws_referenced: List[int]


class QueryUnderstanding(BaseModel):
    rewritten_query: str
    likely_laws: List[int]
    key_concepts: List[str]
    interpretation_note: str


class QuestionResponse(BaseModel):
    question: str
    answer: str
    evidence: List[Evidence]
    retrieval_info: Dict[str, Any]
    processing_time_ms: Optional[float] = None
    session_id: str
    conversation_context: Optional[List[ConversationContext]] = None
    query_understanding: Optional[QueryUnderstanding] = None  # NEW – visible in response


class SessionResponse(BaseModel):
    session_id: str
    created_at: str
    conversation_history: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    retriever_loaded: bool
    mongodb_connected: bool
    session_manager_enabled: bool
    model: str
    fast_model: str
    embedding_model: str
    embedding_service: str
    uptime_seconds: float


class StatsResponse(BaseModel):
    total_chunks: int
    embedding_dimension: int
    model: str
    fast_model: str
    embedding_model: str
    embedding_service: str
    max_chars_per_chunk: int
    max_total_context_chars: int
    unique_laws: List[int]
    total_queries_processed: int
    intro_chunks_count: int
    mongodb_connected: bool
    active_sessions: int


# -----------------------------
# HUGGINGFACE EMBEDDING CLIENT
# -----------------------------
class HFEmbeddingClient:
    def __init__(self, token: str, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.client = InferenceClient(token=token)
        self.model = model

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        embeddings = []
        for text in texts:
            try:
                embedding = self.client.feature_extraction(text, model=self.model)
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
                if embedding.ndim > 1:
                    embedding = embedding.mean(axis=0)
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error encoding text: {e}")
                embeddings.append(np.zeros(384))

        embeddings = np.array(embeddings, dtype=np.float32)
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            embeddings = embeddings / norms
        return embeddings


# -----------------------------
# UTILS
# -----------------------------
def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\- ]+", " ", text)
    return [t for t in text.split() if len(t) > 1]


def load_chunks(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"chunks.jsonl not found at: {path.resolve()}")
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def load_embeddings(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found at: {path.resolve()}")
    embeddings = np.load(path)
    print(f"Loaded embeddings with shape: {embeddings.shape}")
    return embeddings


def format_citation(c: Dict[str, Any]) -> str:
    pages = f"pdf pages {c.get('page_start')}–{c.get('page_end')}"
    sub_num = c.get('subsection_number', 0)
    sub_title = c.get('subsection_title', 'Unknown')
    law_num = c.get('law_number')
    law_title = c.get('law_title')

    if sub_num == 0:
        return (
            f"Law {law_num} – {law_title}, "
            f"Introduction ({pages})"
        )
    else:
        return (
            f"Law {law_num} – {law_title}, "
            f"Section {sub_num}: {sub_title} "
            f"({pages})"
        )


def expand_query_with_synonyms(query: str) -> str:
    """Rule-based query expansion (kept as a fallback/supplement to Gemini understanding)"""
    q_lower = query.lower()
    expansions = []

    synonym_map = {
        "referee": ["match official", "touches a match official", "ball out of play", "dropped ball", "law 9"],
        "match official": ["referee", "touches a match official", "ball out of play", "dropped ball", "law 9"],
        "ball hits referee": ["touches a match official", "law 9", "ball out of play", "dropped ball", "promising attack"],
        "throw-in": ["throw in", "Law 15", "awarded", "introduction"],
        "own goal": ["kicker's goal", "thrower's goal", "corner kick awarded", "goal kick awarded", "introduction"],
        "corner kick": ["corner", "Law 17", "introduction", "directly"],
        "goal kick": ["Law 16", "introduction", "directly", "penalty area"],
        "goalkeeper score": ["goal may be scored", "directly", "opposing team", "Law 16"],
        "handball": ["handling the ball", "touches with hand", "law 12"],
        "fan": ["spectator", "outside agent", "interference"],
        "substitute": ["substitution", "law 3", "extra person"],
        "shoulder charge": ["fair charge", "physical challenge", "playing distance", "ball within playing distance", "law 12"],
        "shoulder charged": ["fair charge", "physical challenge", "playing distance", "ball within playing distance", "law 12"],
        "charging": ["fair charge", "physical challenge", "law 12"],
        "away from the ball": ["playing distance", "not within playing distance", "ball not within playing distance"],
        "time wasting": ["delaying restart", "cautionable offense", "law 12"],
        "diving": ["simulation", "deceive", "law 12", "cautionable"],
        "encroachment": ["enters penalty area", "law 14", "inside penalty area before kick"],
        "advantage": ["advantage clause", "law 12", "referee signals"],
    }

    for trigger, expansions_list in synonym_map.items():
        if trigger in q_lower:
            expansions.extend(expansions_list)

    if expansions:
        return query + " " + " ".join(expansions)
    return query


def extract_scenario_context(query: str) -> Dict[str, Any]:
    q = query.lower()
    return {
        "involves_referee": any(term in q for term in ["referee", "match official", "ball hits"]),
        "involves_goalkeeper": any(term in q for term in ["goalkeeper", "goalie", "keeper"]),
        "into_own_goal": any(term in q for term in ["own goal", "into own", "thrower's goal", "kicker's goal"]),
        "directly": "directly" in q or "direct" in q,
        "can_score": any(term in q for term in ["can score", "may score", "score from"]),
    }


def detect_follow_up_question(question: str, conversation_history: List[Dict]) -> Dict[str, Any]:
    if not conversation_history:
        return {"is_follow_up": False}

    q_lower = question.lower()
    follow_up_patterns = [
        r'\b(what about|how about|and if|but what if|what if instead)\b',
        r'\b(also|additionally|furthermore|moreover)\b',
        r'\b(that|this|these|those|it|they)\b',
        r'^(and |but |so |then |however |although )',
        r'\?$'
    ]
    is_follow_up = any(re.search(pattern, q_lower) for pattern in follow_up_patterns)

    recent_laws = set()
    for qa in conversation_history[-2:]:
        recent_laws.update(qa.get("evidence_laws", []))

    return {
        "is_follow_up": is_follow_up,
        "referenced_laws": list(recent_laws),
        "previous_question": conversation_history[-1]["question"] if conversation_history else None,
        "conversation_topic": extract_topic(conversation_history)
    }


def extract_topic(conversation_history: List[Dict]) -> Optional[str]:
    if not conversation_history:
        return None

    recent_questions = [qa["question"] for qa in conversation_history[-2:]]
    combined = " ".join(recent_questions).lower()

    topics = {
        "handball": ["handball", "hand", "arm"],
        "offside": ["offside"],
        "penalty": ["penalty"],
        "referee": ["referee", "match official"],
        "goalkeeper": ["goalkeeper", "keeper", "goalie"],
        "goal kick": ["goal kick"],
        "throw-in": ["throw-in", "throw in"],
        "corner": ["corner"],
        "substitution": ["substitute", "substitution"],
        "foul": ["foul", "charge", "tackle", "shoulder"],
    }

    for topic, keywords in topics.items():
        if any(kw in combined for kw in keywords):
            return topic

    return None


# -----------------------------
# RETRIEVER
# -----------------------------
class HybridRetriever:
    def __init__(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray, hf_client: HFEmbeddingClient):
        self.chunks = chunks
        self.hf_client = hf_client

        self.tokenized = [tokenize(c.get("text", "")) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized)

        self.embeddings = embeddings.astype("float32")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

        print(f"Retriever initialized with {len(chunks)} chunks")

    def route_rules(self, query: str, context_laws: List[int] = None, extra_law_boost: set = None) -> Dict[str, Any]:
        """Determine which laws and subsection terms to boost based on query keywords."""
        q = query.lower()
        law_boost = set()
        subsection_terms = []

        if context_laws:
            law_boost.update(context_laws)
        if extra_law_boost:
            law_boost.update(extra_law_boost)

        # Ball hits referee
        if any(k in q for k in ["referee", "match official", "ball hits"]):
            law_boost.update([9, 8])
            subsection_terms += [
                "touches a match official", "ball out of play", "ball in play",
                "dropped ball", "promising attack", "possession changes",
            ]

        # Charging / physical challenges  (NEW)
        if any(k in q for k in ["shoulder charge", "shoulder charged", "charging", "fair charge"]):
            law_boost.add(12)
            subsection_terms += ["fair charge", "physical challenge", "playing distance",
                                  "ball within playing distance", "direct free kick"]

        if any(k in q for k in ["away from the ball", "not near the ball", "off the ball"]):
            law_boost.add(12)
            subsection_terms += ["playing distance", "ball within playing distance", "direct free kick"]

        # Goal scoring
        if "goal kick" in q or "goalkeeper score" in q:
            law_boost.add(16)
            if any(k in q for k in ["score", "goal may", "directly"]):
                subsection_terms += ["introduction", "goal may be scored", "directly", "opposing team"]

        if "throw-in" in q:
            law_boost.add(15)
            if any(k in q for k in ["own goal", "score", "directly"]):
                subsection_terms += ["introduction", "corner kick", "goal kick", "cannot be scored"]

        if "corner" in q:
            law_boost.add(17)
            if any(k in q for k in ["own goal", "score", "directly"]):
                subsection_terms += ["introduction", "goal may be scored", "directly"]

        # Handball
        if any(k in q for k in ["handball", "hand", "arm"]):
            law_boost.add(12)
            subsection_terms += ["handball", "handling the ball", "direct free kick"]

        # Offside
        if "offside" in q:
            law_boost.add(11)

        # Penalty
        if "penalty" in q:
            law_boost.add(14)

        # Substitution
        if any(k in q for k in ["substitute", "substitution", "sub"]):
            law_boost.add(3)

        # Simulation / diving
        if any(k in q for k in ["dive", "diving", "simulation", "cheat"]):
            law_boost.add(12)
            subsection_terms += ["simulation", "deceive", "caution", "yellow card"]

        # Advantage
        if "advantage" in q:
            law_boost.add(12)
            subsection_terms += ["advantage clause", "advantage", "referee signals"]

        return {"law_boost": law_boost, "subsection_terms": subsection_terms}

    def search(
        self,
        query: str,
        top_k: int = 10,
        context_info: Dict[str, Any] = None,
        extra_law_boost: set = None,
        extra_key_concepts: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid BM25 + dense retrieval with optional Gemini-enriched boosting.
        """
        enhanced_query = query
        context_laws = []

        if context_info and context_info.get("is_follow_up"):
            context_laws = context_info.get("referenced_laws", [])
            prev_question = context_info.get("previous_question")
            topic = context_info.get("conversation_topic")

            context_parts = []
            if topic:
                context_parts.append(topic)
            if prev_question:
                key_terms = [w for w in prev_question.split() if len(w) > 4][:3]
                context_parts.extend(key_terms)
            if context_parts:
                enhanced_query = query + " " + " ".join(context_parts)

        # Append Gemini key concepts to the search query
        if extra_key_concepts:
            enhanced_query = enhanced_query + " " + " ".join(extra_key_concepts)

        expanded_query = expand_query_with_synonyms(enhanced_query)
        q_tokens = tokenize(expanded_query)
        bm25_scores = self.bm25.get_scores(q_tokens)
        bm25_top = np.argsort(bm25_scores)[::-1][: max(80, top_k * 10)]

        q_emb = self.hf_client.encode([expanded_query], normalize=True)
        _, sim_ids = self.index.search(q_emb, k=max(80, top_k * 10))
        sim_ids = sim_ids[0].tolist()

        cand_ids = list(dict.fromkeys(list(bm25_top) + sim_ids))
        routing = self.route_rules(query, context_laws=context_laws, extra_law_boost=extra_law_boost)
        law_boost = routing["law_boost"]
        subsection_terms = routing["subsection_terms"]

        scored = []
        for i in cand_ids:
            c = self.chunks[i]
            text = c.get("text", "")
            score = 0.0

            score += float(bm25_scores[i]) * 0.7
            score += float(np.dot(q_emb[0], self.embeddings[i])) * 4.0

            if law_boost and c.get("law_number") in law_boost:
                score *= 1.5

            scenario = extract_scenario_context(query)
            if c.get("subsection_number") == 0:  # Introduction
                if scenario["into_own_goal"] or scenario["directly"] or scenario["can_score"]:
                    score *= 1.5
                else:
                    score *= 1.2

            if subsection_terms:
                sub_title = (c.get("subsection_title") or "").lower()
                text_lower = text.lower()
                title_hit = sum(1 for t in subsection_terms if t.lower() in sub_title)
                text_hit = sum(1 for t in subsection_terms if t.lower() in text_lower)
                if title_hit:
                    score *= (1.0 + 0.15 * title_hit)
                if text_hit:
                    score *= (1.0 + 0.05 * text_hit)

            L = max(200, len(text))
            score *= 1.0 / np.log(L)
            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)

        out = []
        seen = set()
        for _, c in scored:
            cid = c.get("chunk_id")
            if cid and cid not in seen:
                out.append(c)
                seen.add(cid)
            if len(out) >= top_k:
                break

        return out


# -----------------------------
# CONTEXT BUILDER
# -----------------------------
def build_context(chunks: List[Dict[str, Any]], conversation_history: List[Dict] = None) -> str:
    parts = []

    if conversation_history and len(conversation_history) > 0:
        parts.append("[CONVERSATION CONTEXT]")
        for i, qa in enumerate(conversation_history[-2:], 1):
            parts.append(f"Previous Q{i}: {qa['question']}")
            answer_preview = qa['answer'][:200] + "..." if len(qa['answer']) > 200 else qa['answer']
            parts.append(f"Previous A{i}: {answer_preview}")
        parts.append("\n[CURRENT QUESTION EXTRACTS]\n")

    total = sum(len(p) for p in parts)

    for idx, c in enumerate(chunks, 1):
        text = (c.get("text") or "").strip()
        if len(text) > MAX_CHARS_PER_CHUNK:
            text = text[:MAX_CHARS_PER_CHUNK] + "\n[TRUNCATED]"
        block = (
            f"[EXTRACT {idx}]\n"
            f"CITATION: {format_citation(c)}\n"
            f"TEXT:\n{text}\n"
        )
        if total + len(block) > MAX_TOTAL_CONTEXT_CHARS:
            break
        parts.append(block)
        total += len(block)

    return "\n\n".join(parts)


# -----------------------------
# GEMINI ANSWER  (Step 3)
# -----------------------------
def gemini_answer(
    question: str,
    chunks: List[Dict[str, Any]],
    context_info: Dict[str, Any] = None,
    conversation_history: List[Dict] = None,
    query_understanding: Dict[str, Any] = None
) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY.")

    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        raise RuntimeError(f"Unable to import google-genai: {e}") from e

    client = genai.Client(api_key=GEMINI_API_KEY)
    scenario = extract_scenario_context(question)
    context = build_context(chunks, conversation_history)

    scenario_hints = []
    if scenario["involves_referee"]:
        scenario_hints.append("This question is about the ball touching a match official/referee.")
    if scenario["into_own_goal"]:
        scenario_hints.append("The ball enters the kicker's/thrower's own goal.")
    if scenario["directly"]:
        scenario_hints.append("The ball enters directly (without touching another player).")
    if scenario["can_score"]:
        scenario_hints.append("This is asking whether a goal can be scored in this situation.")

    if context_info and context_info.get("is_follow_up"):
        scenario_hints.append(
            f"This is a follow-up question related to: {context_info.get('conversation_topic', 'previous discussion')}"
        )

    # Add Gemini query understanding hints
    if query_understanding:
        note = query_understanding.get("interpretation_note", "")
        if note:
            scenario_hints.append(f"Query interpretation note: {note}")
        rewritten = query_understanding.get("rewritten_query", "")
        if rewritten and rewritten != question:
            scenario_hints.append(f"Technically precise version of this question: {rewritten}")

    scenario_text = "\n".join(scenario_hints) if scenario_hints else "General scenario."

    system_instruction = (
        "You are a Laws of the Game (football/soccer) assistant.\n"
        "You MUST answer using ONLY the provided EXTRACTS.\n\n"
        "CONVERSATION AWARENESS:\n"
        "- If CONVERSATION CONTEXT is provided, you can reference previous questions/answers\n"
        "- Use phrases like 'As mentioned before', 'Following up on that', 'In addition to the previous answer'\n"
        "- Connect current answer to previous context naturally\n"
        "- If the user says 'that', 'it', 'this situation', refer to the previous context\n\n"
        "IMPORTANT - NO GREETINGS OR PLEASANTRIES:\n"
        "- Do NOT include greetings like 'Hello', 'Hi', 'Great question', etc.\n"
        "- Do NOT include closing pleasantries like 'Hope this helps', 'Let me know if...', etc.\n"
        "- Start DIRECTLY with the answer\n"
        "- Be professional but concise - no conversational fluff\n\n"
        "HANDLING UNANSWERABLE QUESTIONS:\n"
        "If the question cannot be answered from the extracts, respond:\n\n"
        "\"I don't have enough information in the Laws of the Game extracts to answer that specific question. "
        "This could be because:\n"
        "• The scenario might fall outside the official Laws of the Game\n"
        "• The question may require interpretation beyond what's explicitly written\n"
        "• The relevant section might not have been retrieved in my search\n\n"
        "You can rephrase your question or ask about a related topic.\"\n\n"
        "ANSWER FORMAT:\n"
        "Write your answer in clear, flowing English prose. Explain what happens in the scenario naturally, "
        "as if speaking to someone. Include all relevant details about:\n"
        "- What the decision should be\n"
        "- How play restarts\n"
        "- Any disciplinary action (if applicable)\n"
        "- Why this is the correct ruling\n\n"
        "Write this as natural paragraphs - DO NOT use bullet points, numbered lists, or section headers "
        "in the main explanation. Just write the answer naturally in flowing prose.\n\n"
        "After your explanation, add a structured evidence section:\n\n"
        "**Supporting Evidence:**\n\n"
        "**Law X – Title, Subsection** (pdf pages Y–Z)\n"
        "\"Quote from the law that supports the answer\"\n\n"
        "RULES:\n"
        "- NO greetings, NO pleasantries - start directly with the answer\n"
        "- Main answer: Write in clear, natural English prose - like explaining to a friend\n"
        "- Main answer: NO structured breakdowns with headers like \"Decision:\", \"Restart:\", \"Discipline:\"\n"
        "- Main answer: NO bullet points or numbered lists\n"
        "- Main answer: Just write flowing paragraphs that explain what happens\n"
        "- Evidence section: USE the structured format with \"**Supporting Evidence:**\" header\n"
        "- Evidence section: Each law citation should be bolded with **Law X – Title, Subsection**\n"
        "- Evidence section: Include page numbers in parentheses\n"
        "- Evidence section: Quote the relevant text that supports your answer\n"
        "- Do not invent Laws, restarts, or cards\n"
        "- Do not use outside knowledge\n"
        "- Every claim must be supported by the extracts\n"
    )

    user_prompt = (
        f"QUESTION:\n{question}\n\n"
        f"SCENARIO CONTEXT:\n{scenario_text}\n\n"
        f"EXTRACTS:\n{context}"
    )

    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.1,
        ),
    )
    return resp.text or ""


# -----------------------------
# FASTAPI APP
# -----------------------------
retriever: Optional[HybridRetriever] = None
hf_client: Optional[HFEmbeddingClient] = None
mongo_counter: Optional[MongoDBCounter] = None
session_manager: Optional[SessionManager] = None
start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, hf_client, mongo_counter, session_manager

    mongo_db = None
    if MONGODB_URI:
        try:
            mongo_counter = MongoDBCounter(
                uri=MONGODB_URI,
                database=MONGODB_DATABASE,
                collection=MONGODB_COLLECTION
            )
            if mongo_counter.connect():
                logger.info("MongoDB counter initialized successfully")
                mongo_db = mongo_counter.db
            else:
                logger.warning("MongoDB counter failed to initialize")
                mongo_counter = None
        except Exception as e:
            logger.error("Error initializing MongoDB counter: %s", e)
            mongo_counter = None
    else:
        logger.warning("MONGODB_URI not found. Running without persistent counter.")
        mongo_counter = None

    session_manager = SessionManager(mongo_db=mongo_db)
    logger.info("Session manager initialized")

    if not HF_TOKEN:
        logger.warning("HF_TOKEN not found. Starting API without retriever.")
        yield
        if mongo_counter:
            mongo_counter.close()
        return

    try:
        hf_client = HFEmbeddingClient(token=HF_TOKEN, model=EMBED_MODEL_NAME)
        print("HuggingFace client initialised")

        chunks = load_chunks(CHUNKS_PATH)
        print(f"Loaded {len(chunks)} chunks")

        embeddings = load_embeddings(EMBEDDINGS_PATH)

        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings")

        retriever = HybridRetriever(chunks, embeddings, hf_client)
        logger.info("Startup complete. MongoDB connected: %s",
                    mongo_counter is not None and mongo_counter.is_connected())
        print("API ready")
    except Exception as e:
        logger.exception("Error during startup; continuing without retriever")
        print(f"Error during startup (degraded mode): {e}")
        retriever = None
        hf_client = None

    yield

    if mongo_counter:
        mongo_counter.close()


app = FastAPI(
    title="Football Laws of the Game RAG API",
    description="Gemini Query Understanding + Session-aware RAG",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "Football Laws of the Game RAG API",
        "version": "3.0.0",
        "improvements": [
            "Gemini query understanding step before retrieval",
            "Expanded synonym map (shoulder charge, diving, advantage, etc.)",
            "Session-level context awareness for follow-up questions"
        ],
        "storage": "MongoDB Atlas",
        "endpoints": {
            "/ask": "POST - Ask a question (with optional X-Session-ID header)",
            "/session/new": "POST - Create new session",
            "/session/{session_id}": "GET - Get session history",
            "/health": "GET - Health check",
            "/stats": "GET - Statistics"
        }
    }


@app.post("/session/new")
async def create_new_session():
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not available")
    session_id = session_manager.create_session()
    return {"session_id": session_id, "message": "New session created"}


@app.get("/session/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not available")
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return SessionResponse(
        session_id=session["session_id"],
        created_at=session["created_at"].isoformat(),
        conversation_history=session["conversation_history"]
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialised")
    return HealthResponse(
        status="healthy",
        retriever_loaded=True,
        mongodb_connected=mongo_counter is not None and mongo_counter.is_connected(),
        session_manager_enabled=session_manager is not None,
        model=GEMINI_MODEL,
        fast_model=GEMINI_FAST_MODEL,
        embedding_model=EMBED_MODEL_NAME,
        embedding_service="HuggingFace Inference API",
        uptime_seconds=time.time() - start_time
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialised")

    unique_laws = sorted(list(set(c.get("law_number") for c in retriever.chunks if c.get("law_number"))))
    intro_count = sum(1 for c in retriever.chunks if c.get("subsection_number") == 0)

    query_count = 0
    if mongo_counter and mongo_counter.is_connected():
        query_count = mongo_counter.get_count()

    active_sessions = len(session_manager.sessions) if session_manager else 0

    return StatsResponse(
        total_chunks=len(retriever.chunks),
        embedding_dimension=retriever.embeddings.shape[1],
        model=GEMINI_MODEL,
        fast_model=GEMINI_FAST_MODEL,
        embedding_model=EMBED_MODEL_NAME,
        embedding_service="HuggingFace Inference API",
        max_chars_per_chunk=MAX_CHARS_PER_CHUNK,
        max_total_context_chars=MAX_TOTAL_CONTEXT_CHARS,
        unique_laws=unique_laws,
        total_queries_processed=query_count,
        intro_chunks_count=intro_count,
        mongodb_connected=mongo_counter is not None and mongo_counter.is_connected(),
        active_sessions=active_sessions
    )


# -----------------------------------------------------------------------
# /ask  – main endpoint (same interface as before, no frontend changes)
# -----------------------------------------------------------------------
@app.post("/ask", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID")
):
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialised")

    try:
        logger.info("Received /ask request. question=%s, session_id=%s", request.question, x_session_id)
        start = time.time()

        # ── Session management ─────────────────────────────────────────
        session = None
        if session_manager:
            if x_session_id:
                session = session_manager.get_session(x_session_id)
                if not session:
                    x_session_id = session_manager.create_session()
                    session = session_manager.get_session(x_session_id)
            else:
                x_session_id = session_manager.create_session()
                session = session_manager.get_session(x_session_id)

        conversation_history = session.get("conversation_history", []) if session else []

        # ── Follow-up detection ────────────────────────────────────────
        context_info = detect_follow_up_question(request.question, conversation_history)
        logger.info("Context detection: %s", context_info)

        # ── STEP 1: Gemini Query Understanding ─────────────────────────
        t_understand_start = time.time()
        query_understanding = understand_query(request.question, conversation_history)
        t_understand_ms = (time.time() - t_understand_start) * 1000
        logger.info(
            "Query understanding (%.0fms): rewritten='%s', laws=%s, concepts=%s",
            t_understand_ms,
            query_understanding["rewritten_query"],
            query_understanding["likely_laws"],
            query_understanding["key_concepts"]
        )

        # ── STEP 2: Hybrid RAG retrieval with enriched signals ─────────
        top_chunks = retriever.search(
            query=query_understanding["rewritten_query"],   # use enriched query
            top_k=request.top_k,
            context_info=context_info,
            extra_law_boost=set(query_understanding["likely_laws"]),
            extra_key_concepts=query_understanding["key_concepts"]
        )

        # ── STEP 3: Gemini final answer ────────────────────────────────
        answer = gemini_answer(
            question=request.question,          # original question for the user-facing prompt
            chunks=top_chunks,
            context_info=context_info,
            conversation_history=conversation_history,
            query_understanding=query_understanding
        )

        # ── Build evidence list ────────────────────────────────────────
        evidence_list = []
        for i, chunk in enumerate(top_chunks, 1):
            raw_text = chunk.get("text", "")
            text_preview = raw_text[:200] + "..." if len(raw_text) > 200 else raw_text
            evidence_list.append(
                Evidence(
                    rank=i,
                    citation=format_citation(chunk),
                    text_preview=text_preview,
                    full_text=raw_text if request.include_raw_chunks else None,
                    law_number=chunk.get("law_number")
                )
            )

        # ── Update session ─────────────────────────────────────────────
        if session_manager and x_session_id:
            session_manager.update_session(
                x_session_id,
                request.question,
                answer,
                [{"law_number": e.law_number} for e in evidence_list]
            )

        # ── Build retrieval_info ───────────────────────────────────────
        routing = retriever.route_rules(
            request.question,
            extra_law_boost=set(query_understanding["likely_laws"])
        )
        scenario = extract_scenario_context(request.question)

        retrieval_info = {
            "original_query": request.question,
            "rewritten_query": query_understanding["rewritten_query"],
            "query_understanding_ms": round(t_understand_ms, 1),
            "boosted_laws": list(routing["law_boost"]) if routing["law_boost"] else [],
            "scenario_context": scenario,
            "chunks_retrieved": len(top_chunks),
            "context_aware": context_info is not None,
            "is_follow_up": context_info.get("is_follow_up", False) if context_info else False
        }

        processing_time = (time.time() - start) * 1000

        # ── Counter ────────────────────────────────────────────────────
        if mongo_counter and mongo_counter.is_connected():
            latest_count, counter_persisted = mongo_counter.increment()
            retrieval_info["query_counter"] = {
                "value": latest_count,
                "persisted": counter_persisted,
                "storage": "mongodb"
            }
        else:
            retrieval_info["query_counter"] = {
                "value": 0,
                "persisted": False,
                "storage": "none",
                "error": "MongoDB not connected"
            }

        # ── Conversation context for response ──────────────────────────
        conversation_context_response = None
        if conversation_history:
            conversation_context_response = [
                ConversationContext(
                    timestamp=qa["timestamp"].isoformat(),
                    question=qa["question"],
                    answer_preview=qa["answer"][:150] + "..." if len(qa["answer"]) > 150 else qa["answer"],
                    laws_referenced=qa.get("evidence_laws", [])
                )
                for qa in conversation_history[-3:]
            ]

        logger.info(
            "Completed /ask. total=%.0fms (understand=%.0fms) counter=%s session=%s",
            processing_time,
            t_understand_ms,
            retrieval_info["query_counter"]["value"],
            x_session_id
        )

        return QuestionResponse(
            question=request.question,
            answer=answer,
            evidence=evidence_list,
            retrieval_info=retrieval_info,
            processing_time_ms=processing_time,
            session_id=x_session_id or "none",
            conversation_context=conversation_context_response,
            query_understanding=QueryUnderstanding(**query_understanding)
        )

    except Exception as e:
        logger.exception("Error while processing /ask request")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)