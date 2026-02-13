"""
Enhanced Football Laws of the Game RAG API
Version 2.2 - Optimized for New Chunks with Introductions
"""
import os
import sys
import json
import re
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from huggingface_hub import InferenceClient

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import time

# -----------------------------
# CONFIG
# -----------------------------
# Adjust paths for Vercel deployment
BASE_DIR = Path(__file__).parent.parent
CHUNKS_PATH = BASE_DIR / "rag_chunks" / "chunks.jsonl"
EMBEDDINGS_PATH = BASE_DIR / "rag_chunks" / "embeddings.npy"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"
TOP_K = 10

# context controls
MAX_CHARS_PER_CHUNK = 2500
MAX_TOTAL_CONTEXT_CHARS = 15000

# persistent query counter
COUNTER_FILE_PATH = BASE_DIR / "query_counter.json"
COUNTER_KEY = "all_time_queries"

# API keys from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("Warning: HF_TOKEN not found in environment variables")
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not found in environment variables")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("fcrules.api")
counter_lock = threading.Lock()

# -----------------------------
# PYDANTIC MODELS
# -----------------------------
class QuestionRequest(BaseModel):
    question: str = Field(..., description="The question to ask about football laws")
    top_k: Optional[int] = Field(10, description="Number of top chunks to retrieve", ge=1, le=20)
    include_raw_chunks: Optional[bool] = Field(False, description="Include raw chunk data in response")


class BatchQuestionRequest(BaseModel):
    questions: List[str] = Field(..., description="List of questions to ask", max_items=10)
    top_k: Optional[int] = Field(10, description="Number of top chunks to retrieve per question")


class Evidence(BaseModel):
    rank: int
    citation: str
    text_preview: str
    full_text: Optional[str] = None


class QuestionResponse(BaseModel):
    question: str
    answer: str
    evidence: List[Evidence]
    retrieval_info: Dict[str, Any]
    processing_time_ms: Optional[float] = None


class BatchQuestionResponse(BaseModel):
    results: List[QuestionResponse]
    total_processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    retriever_loaded: bool
    model: str
    embedding_model: str
    embedding_service: str
    uptime_seconds: float


class StatsResponse(BaseModel):
    total_chunks: int
    embedding_dimension: int
    model: str
    embedding_model: str
    embedding_service: str
    max_chars_per_chunk: int
    max_total_context_chars: int
    unique_laws: List[int]
    total_queries_processed: int
    intro_chunks_count: int  # NEW


# -----------------------------
# HUGGINGFACE EMBEDDING CLIENT
# -----------------------------
class HFEmbeddingClient:
    """Wrapper for HuggingFace Inference API embeddings"""

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


def load_query_counter_from_file() -> int:
    if not COUNTER_FILE_PATH.exists():
        logger.warning(
            "Query counter file not found at %s. Creating a new counter with 0.",
            COUNTER_FILE_PATH,
        )
        if not persist_query_counter_to_file(0):
            logger.error("Failed to create query counter file during startup.")
        return 0

    try:
        with COUNTER_FILE_PATH.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        counter_value = int(payload.get(COUNTER_KEY, 0))
        if counter_value < 0:
            raise ValueError(f"{COUNTER_KEY} must be >= 0, got {counter_value}")
        logger.info("Loaded query counter from file: %s", counter_value)
        return counter_value
    except Exception:
        logger.exception(
            "Failed to load query counter from %s. Falling back to 0.",
            COUNTER_FILE_PATH,
        )
        return 0


def persist_query_counter_to_file(count: int) -> bool:
    temp_path = COUNTER_FILE_PATH.with_suffix(".tmp")
    payload = {COUNTER_KEY: int(count)}

    try:
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True)

        os.replace(temp_path, COUNTER_FILE_PATH)

        with COUNTER_FILE_PATH.open("r", encoding="utf-8") as f:
            written_payload = json.load(f)
        written_value = int(written_payload.get(COUNTER_KEY, -1))

        if written_value != int(count):
            logger.error(
                "Query counter verification failed. expected=%s written=%s file=%s",
                count,
                written_value,
                COUNTER_FILE_PATH,
            )
            return False

        logger.info(
            "Query counter updated successfully. value=%s file=%s",
            written_value,
            COUNTER_FILE_PATH,
        )
        return True
    except Exception:
        logger.exception(
            "Failed to persist query counter. value=%s file=%s",
            count,
            COUNTER_FILE_PATH,
        )
        return False
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                logger.warning("Could not remove temporary counter file: %s", temp_path)


def increment_and_persist_query_counter() -> tuple[int, bool]:
    global query_count

    with counter_lock:
        query_count += 1
        persisted = persist_query_counter_to_file(query_count)
        if not persisted:
            logger.error(
                "Counter incremented in memory but failed to persist. in_memory=%s file=%s",
                query_count,
                COUNTER_FILE_PATH,
            )
        return query_count, persisted


def format_citation(c: Dict[str, Any]) -> str:
    """Format citation with proper handling of subsection 0 (Introduction)"""
    pages = f"pdf pages {c.get('page_start')}–{c.get('page_end')}"
    sub_num = c.get('subsection_number', 0)
    sub_title = c.get('subsection_title', 'Unknown')

    # Handle subsection 0 (Introduction) specially
    if sub_num == 0:
        return (
            f"Law {c.get('law_number')} – {c.get('law_title')}, "
            f"Introduction ({pages})"
        )
    else:
        return (
            f"Law {c.get('law_number')} – {c.get('law_title')}, "
            f"{sub_num}. {sub_title} "
            f"({pages})"
        )


def expand_query_with_synonyms(query: str) -> str:
    """Enhanced query expansion with new triggers for intro content"""
    q_lower = query.lower()
    expansions = []

    synonym_map = {
        # Existing mappings
        "fan": ["spectator", "outside agent", "person not on team list"],
        "interrupts": ["interference", "interferes with play", "enters field"],
        "kicks away": ["touches ball", "plays ball", "interferes"],
        "shoulder charge": ["charges", "physical challenge", "body contact"],
        "far from ball": ["not within playing distance", "ball not in playing distance"],
        "strikes with hand": ["handball", "handles ball", "touches with hand", "hand ball"],
        "comes on pitch": ["enters field of play", "extra person on field"],
        "what should happen": ["what is the decision", "what is the restart", "what action"],
        "obstacle": ["outside agent", "object", "interference", "dropped ball", "match official", "law 9"],
        "match official": ["referee", "touches a match official", "ball out of play", "dropped ball", "law 9"],
        "referee": ["match official", "touches a match official", "ball out of play", "dropped ball", "law 9"],

        # NEW: Intro-specific expansions for the previously missing content
        "throw-in": ["throw in", "Law 15", "awarded", "introduction"],
        "own goal": ["kicker's goal", "thrower's goal", "team's goal", "goal kick awarded", "corner kick awarded", "introduction"],
        "corner kick": ["corner", "Law 17", "introduction"],
        "goal kick": ["Law 16", "penalty area", "retaken", "introduction"],
        "directly": ["direct", "straight", "without touching", "introduction"],
        "into own goal": ["kicker's goal", "thrower's goal", "corner kick awarded", "goal kick awarded"],
        "into thrower's goal": ["corner kick awarded", "own goal", "Law 15 introduction"],
        "into kicker's goal": ["corner kick awarded", "own goal", "Law 16 Law 17 introduction"],
    }

    # Multi-word phrase detection for better matching
    if "throw-in" in q_lower and "own goal" in q_lower:
        expansions += ["corner kick awarded", "Law 15", "introduction", "directly"]

    if "corner" in q_lower and "own goal" in q_lower:
        expansions += ["corner kick awarded", "Law 17", "introduction", "directly", "kicker's goal"]

    if "goal kick" in q_lower and ("own goal" in q_lower or "kicker's goal" in q_lower):
        expansions += ["corner kick awarded", "Law 16", "introduction", "directly"]

    if "doesn't leave" in q_lower or "not leave" in q_lower:
        expansions += ["penalty area", "retaken", "in play", "clearly moves"]

    # Apply single-word mappings
    for trigger, expansions_list in synonym_map.items():
        if trigger in q_lower:
            expansions.extend(expansions_list)

    if expansions:
        expanded = query + " " + " ".join(expansions)
        return expanded
    return query


def is_quote_mode(query: str) -> bool:
    q = query.lower()
    triggers = [
        "where exactly", "exact wording", "quote", "verbatim", "exact paragraph",
        "show me the exact", "cite the exact", "written"
    ]
    return any(t in q for t in triggers)


def has_discipline_intent(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in ["card", "caution", "yellow", "red", "send off", "sent off", "discipline", "sanction"])


def has_restart_intent(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in
               ["restart", "free kick", "penalty", "dropped ball", "indirect", "direct", "kick-off", "throw-in",
                "goal kick", "corner kick", "awarded"])  # Added "awarded"


def has_handball_intent(query: str) -> bool:
    q = query.lower()
    handball_terms = ["handball", "hand ball", "hand", "arm", "handle", "handling", "touches with hand",
                      "strikes with hand"]
    return any(term in q for term in handball_terms)


def has_interference_intent(query: str) -> bool:
    q = query.lower()
    interference_terms = [
        "fan", "spectator", "interrupts", "interference", "outside agent",
        "animal", "object enters", "comes on pitch", "enters field", "extra person",
        "obstacle", "object", "outside object", "match official", "referee",
    ]
    return any(term in q for term in interference_terms)


def has_physical_contact_intent(query: str) -> bool:
    q = query.lower()
    contact_terms = [
        "charge", "charging", "shoulder charge", "physical contact",
        "pushes", "push", "holds", "holding", "tackles", "tackle"
    ]
    return any(term in q for term in contact_terms)


def extract_scenario_context(query: str) -> Dict[str, Any]:
    q = query.lower()
    context = {
        "involves_teammate": any(term in q for term in ["teammate", "team mate", "own team"]),
        "involves_goalkeeper": any(term in q for term in ["goalkeeper", "goalie", "keeper"]),
        "involves_outfield_player": not any(term in q for term in ["goalkeeper", "goalie", "keeper"]),
        "own_half": any(term in q for term in ["own half", "own goal", "own side"]),
        "penalty_area": any(term in q for term in ["penalty area", "penalty box", "box"]),
        "deliberate": any(term in q for term in ["deliberate", "intentional"]),
        "far_from_ball": any(term in q for term in ["far from ball", "not near ball", "away from ball"]),
        "outside_interference": has_interference_intent(q),
        "physical_contact": has_physical_contact_intent(q),
        "into_own_goal": any(term in q for term in ["own goal", "into own", "thrower's goal", "kicker's goal"]),  # NEW
        "directly": "directly" in q or "direct" in q,  # NEW
    }
    return context


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

    def route_rules(self, query: str) -> Dict[str, Any]:
        q = query.lower()
        law_boost = set()
        subsection_terms = []

        if has_handball_intent(q):
            law_boost.add(12)
            subsection_terms += ["Handling the ball", "handball offence", "Direct free kick",
                                 "scores in the opponents' goal"]
            scenario = extract_scenario_context(q)
            if scenario["penalty_area"]:
                subsection_terms += ["penalty area", "penalty kick"]
            if scenario["involves_teammate"]:
                subsection_terms += ["deliberately kicked to the goalkeeper by a team-mate"]
            if scenario["own_half"] or scenario["penalty_area"]:
                subsection_terms += ["inside their penalty area", "penalty kick", "direct free kick"]

        if has_interference_intent(q):
            law_boost.update([5, 9, 3])
            subsection_terms += [
                "outside interference", "outside agent", "extra person", "spectator",
                "dropped ball", "enters the field of play", "interferes with play",
                "stops play", "animal", "object", "unauthorized person",
                "ball out of play", "ball in play"
            ]

        if any(k in q for k in ["referee", "match official"]):
            law_boost.update([9, 5, 8])
            subsection_terms += [
                "touches a match official",
                "ball out of play",
                "dropped ball",
                "outside interference",
            ]

        if has_physical_contact_intent(q):
            law_boost.add(12)
            subsection_terms += [
                "charges", "physical challenge", "careless", "reckless", "excessive force",
                "playing distance", "within playing distance", "shield the ball",
                "fairly charged", "impedes the progress", "Direct free kick"
            ]
            scenario = extract_scenario_context(q)
            if scenario["far_from_ball"]:
                subsection_terms += [
                    "not within playing distance", "without any contact",
                    "impeding the progress", "indirect free kick"
                ]

        # NEW: Enhanced routing for intro-specific queries
        if "throw-in" in q:
            law_boost.add(15)
            if "own goal" in q or "thrower's goal" in q or "directly" in q:
                subsection_terms += ["introduction", "corner kick", "goal kick", "directly", "cannot be scored"]

        if "corner" in q:
            law_boost.add(17)
            if "own goal" in q or "kicker's goal" in q or "directly" in q:
                subsection_terms += ["introduction", "corner kick", "directly", "opposing team"]

        if "goal kick" in q:
            law_boost.add(16)
            if "own goal" in q or "kicker's goal" in q or "directly" in q:
                subsection_terms += ["introduction", "corner kick", "directly", "opposing team"]
            if "doesn't leave" in q or "not leave" in q or "penalty area" in q:
                subsection_terms += ["penalty area", "in play", "retaken", "clearly moves"]

        if "offside" in q:
            law_boost.add(11)
        if "dropped ball" in q:
            law_boost.update([8, 9])
        if "penalty kick" in q or "penalty" in q:
            law_boost.add(14)
        if "technical area" in q:
            law_boost.add(1)
        if any(k in q for k in ["ball pressure", "circumference", "ball size", "size of the ball", "pressure"]):
            law_boost.add(2)
        if any(k in q for k in ["number of players", "minimum seven", "seven players", "how many players"]):
            law_boost.add(3)

        if has_discipline_intent(q):
            law_boost.update([12, 3])
            subsection_terms += ["Disciplinary", "Offences and sanctions", "sending-off", "caution", "sent off"]

        if has_restart_intent(q):
            subsection_terms += ["Restart", "Restart of play", "Offences and sanctions", "introduction"]

        if any(k in q for k in ["substitute", "extra person", "team official", "enters the pitch"]):
            law_boost.add(3)
            subsection_terms += ["Extra persons", "Substitution procedure", "Offences and sanctions"]

        return {"law_boost": law_boost, "subsection_terms": subsection_terms}

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        expanded_query = expand_query_with_synonyms(query)
        q_tokens = tokenize(expanded_query)
        bm25_scores = self.bm25.get_scores(q_tokens)
        bm25_top = np.argsort(bm25_scores)[::-1][: max(80, top_k * 10)]

        q_emb = self.hf_client.encode([expanded_query], normalize=True)

        _, sim_ids = self.index.search(q_emb, k=max(80, top_k * 10))
        sim_ids = sim_ids[0].tolist()

        cand_ids = list(dict.fromkeys(list(bm25_top) + sim_ids))
        routing = self.route_rules(query)
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
                boost_mult = 1.5 if has_handball_intent(query) or has_interference_intent(query) else 1.25
                score *= boost_mult

            # NEW: Boost introduction chunks (subsection 0) for "own goal" / "directly" queries
            scenario = extract_scenario_context(query)
            if c.get("subsection_number") == 0:  # Introduction section
                if scenario["into_own_goal"] or scenario["directly"]:
                    score *= 1.4  # Significant boost for intros when asking about "own goal" scenarios
                else:
                    score *= 1.15  # Mild boost for intros in general

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
# GEMINI ANSWER
# -----------------------------
def build_context(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    total = 0
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


def gemini_answer(question: str, chunks: List[Dict[str, Any]]) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY.")

    try:
        from google import genai
        from google.genai import types
    except MemoryError as e:
        raise RuntimeError(
            "Unable to import google-genai due to memory limits. "
            "Use Python 3.11 with updated dependencies or increase available memory."
        ) from e
    except Exception as e:
        raise RuntimeError(f"Unable to import google-genai: {e}") from e

    client = genai.Client(api_key=GEMINI_API_KEY)
    quote_mode = is_quote_mode(question)
    scenario = extract_scenario_context(question)
    context = build_context(chunks)

    scenario_hints = []
    if scenario["involves_teammate"]:
        scenario_hints.append("The scenario involves a teammate passing/kicking to another teammate.")
    if scenario["own_half"]:
        scenario_hints.append("The incident occurs in the team's own half.")
    if scenario["penalty_area"]:
        scenario_hints.append("The incident may occur in the penalty area.")
    if scenario["far_from_ball"]:
        scenario_hints.append("The player is far from the ball (not within playing distance).")
    if scenario["outside_interference"]:
        scenario_hints.append("This involves outside interference (fan, spectator, or unauthorised person).")
    if scenario["physical_contact"]:
        scenario_hints.append("This involves physical contact between players.")
    if scenario["into_own_goal"]:
        scenario_hints.append("The ball enters the kicker's/thrower's own goal.")
    if scenario["directly"]:
        scenario_hints.append("The ball enters directly (without touching another player).")

    scenario_text = "\n".join(scenario_hints) if scenario_hints else "General scenario."

    system_instruction = (
        "You are a Laws of the Game (football/soccer) assistant.\n"
        "You MUST answer using ONLY the provided EXTRACTS.\n"
        "If any part of the answer is not clearly supported by the extracts, write:\n"
        "\"Not found in the provided extracts\" for that part.\n\n"
        "IMPORTANT RULES:\n"
        "- When the question involves a SPECIFIC SCENARIO (e.g., teammate passing to teammate, "
        "player far from ball, fan interference, ball into own goal), apply the general rules to that specific scenario.\n"
        "- If the extract provides criteria/tests (e.g., 'it is an offence if ...'), "
        "explain how those criteria apply to the specific scenario described.\n"
        "- Consider WHERE the offence occurs (own half vs penalty area) when determining the restart.\n"
        "- For physical contact offences, consider whether the ball was within playing distance.\n"
        "- For outside interference, explain when play should stop and how it should restart.\n"
        "- Pay special attention to INTRODUCTION sections (subsection 0) which contain important rules "
        "about restarts when the ball enters own goal or specific scenarios.\n\n"
        "For every answer, follow this structure exactly:\n\n"
        "1) Answer (Direct response to the question)\n"
        "   - Decision: [What should happen - be specific to the scenario]\n"
        "   - Restart: [Specify the exact restart based on location]\n"
        "   - Discipline: [If applicable, otherwise state 'None']\n"
        "   - Explanation: [Brief plain English explanation applied to the specific scenario]\n\n"
        "2) Supporting Law References\n"
        "   - Relevant Law & subsection: [Law number + subsection title]\n"
        "   - Evidence: [Provide 1–2 short quotes max unless QUOTE MODE is ON]\n"
        "   - Citation: [Include the exact CITATION line after each quote]\n\n"
        "Rules:\n"
        "- Lead with the ANSWER - what actually happens in this scenario\n"
        "- Then provide the legal basis and evidence\n"
        "- Do not invent Laws, restarts, or cards.\n"
        "- Do not use outside knowledge.\n"
        "- Every Decision/Restart/Discipline claim must be supported by the Evidence quotes.\n"
        "- When multiple extracts are relevant, synthesise them into one coherent answer.\n"
        "- Pay special attention to conditions like 'within playing distance' for physical offences.\n"
        "- Introduction sections often contain the most direct rules for 'own goal' scenarios.\n"
    )

    user_prompt = (
        f"QUESTION:\n{question}\n\n"
        f"SCENARIO CONTEXT:\n{scenario_text}\n\n"
        f"QUOTE MODE: {'ON' if quote_mode else 'OFF'}\n\n"
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
# Global state
retriever: Optional[HybridRetriever] = None
hf_client: Optional[HFEmbeddingClient] = None
start_time = time.time()
query_count = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, hf_client, query_count
    if not HF_TOKEN:
        logger.warning("HF_TOKEN not found. Starting API without retriever.")
        query_count = load_query_counter_from_file()
        yield
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
        query_count = load_query_counter_from_file()
        logger.info("Startup complete. query_count=%s", query_count)
        print("API ready")
    except Exception as e:
        logger.exception("Error during startup; continuing without retriever")
        print(f"Error during startup (degraded mode): {e}")
        retriever = None
        hf_client = None
        query_count = load_query_counter_from_file()

    yield


app = FastAPI(
    title="Football Laws of the Game RAG API",
    description="Vercel Serverless Deployment with HuggingFace API - Optimized for chunks with introductions",
    version="2.2.0",
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
        "version": "2.2.0",
        "endpoints": {
            "/ask": "POST - Ask a question",
            "/health": "GET - Health check",
            "/stats": "GET - Statistics"
        },
        "improvements": [
            "Enhanced support for Law introductions (subsection 0)",
            "Better handling of 'own goal' scenarios",
            "Improved query expansion for throw-in, corner kick, goal kick",
            "Optimized retrieval for directly-into-goal questions"
        ]
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialised")

    return HealthResponse(
        status="healthy",
        retriever_loaded=True,
        model=GEMINI_MODEL,
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

    return StatsResponse(
        total_chunks=len(retriever.chunks),
        embedding_dimension=retriever.embeddings.shape[1],
        model=GEMINI_MODEL,
        embedding_model=EMBED_MODEL_NAME,
        embedding_service="HuggingFace Inference API",
        max_chars_per_chunk=MAX_CHARS_PER_CHUNK,
        max_total_context_chars=MAX_TOTAL_CONTEXT_CHARS,
        unique_laws=unique_laws,
        total_queries_processed=query_count,
        intro_chunks_count=intro_count
    )


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    global query_count

    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialised")

    try:
        logger.info("Received /ask request. question=%s", request.question)
        start = time.time()

        top_chunks = retriever.search(request.question, top_k=request.top_k)
        answer = gemini_answer(request.question, top_chunks)

        evidence_list = []
        for i, chunk in enumerate(top_chunks, 1):
            text_preview = chunk.get("text", "")[:200] + "..." if len(chunk.get("text", "")) > 200 else chunk.get("text", "")
            evidence_list.append(
                Evidence(
                    rank=i,
                    citation=format_citation(chunk),
                    text_preview=text_preview,
                    full_text=chunk.get("text", "") if request.include_raw_chunks else None
                )
            )

        routing = retriever.route_rules(request.question)
        scenario = extract_scenario_context(request.question)

        retrieval_info = {
            "expanded_query": expand_query_with_synonyms(request.question),
            "quote_mode": is_quote_mode(request.question),
            "boosted_laws": list(routing["law_boost"]) if routing["law_boost"] else [],
            "scenario_context": scenario,
            "chunks_retrieved": len(top_chunks)
        }

        processing_time = (time.time() - start) * 1000
        latest_count, counter_persisted = increment_and_persist_query_counter()
        retrieval_info["query_counter"] = {
            "value": latest_count,
            "persisted": counter_persisted,
            "file_path": str(COUNTER_FILE_PATH),
        }
        if not counter_persisted:
            retrieval_info["query_counter"]["error"] = "Counter was not persisted. Check server logs."
        logger.info(
            "Completed /ask request. processing_time_ms=%.2f counter=%s persisted=%s",
            processing_time,
            latest_count,
            counter_persisted,
        )

        return QuestionResponse(
            question=request.question,
            answer=answer,
            evidence=evidence_list,
            retrieval_info=retrieval_info,
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.exception("Error while processing /ask request")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")