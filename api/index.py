"""
Enhanced Football Laws of the Game RAG API
Version 2.1 - With HuggingFace Inference API (FREE)
"""
import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from huggingface_hub import InferenceClient

from google import genai
from google.genai import types

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import time
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# CONFIG
# -----------------------------
CHUNKS_PATH = Path(__file__).parent.parent / "rag_chunks" / "chunks.jsonl"
EMBEDDINGS_PATH = Path(__file__).parent.parent / "rag_chunks" / "embeddings.npy"  # Pre-computed embeddings
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"
TOP_K = 10

# context controls
MAX_CHARS_PER_CHUNK = 2500
MAX_TOTAL_CONTEXT_CHARS = 15000

# API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")  # Get free token from huggingface.co/settings/tokens

if not HF_TOKEN:
    print("⚠️  Warning: HF_TOKEN not found. Get your free token from https://huggingface.co/settings/tokens")


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


class Citation(BaseModel):
    law_number: Optional[int]
    law_title: Optional[str]
    subsection_number: Optional[str]
    subsection_title: Optional[str]
    page_start: Optional[int]
    page_end: Optional[int]
    chunk_id: Optional[str]


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


# -----------------------------
# HUGGINGFACE EMBEDDING CLIENT
# -----------------------------
class HFEmbeddingClient:
    """Wrapper for HuggingFace Inference API embeddings"""

    def __init__(self, token: str, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.client = InferenceClient(token=token)
        self.model = model

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        Encode texts using HuggingFace Inference API

        Args:
            texts: List of texts to encode
            normalize: Whether to normalize embeddings

        Returns:
            numpy array of embeddings
        """
        embeddings = []

        for text in texts:
            try:
                # Call HuggingFace Inference API
                embedding = self.client.feature_extraction(text, model=self.model)

                # Convert to numpy array
                if isinstance(embedding, list):
                    embedding = np.array(embedding)

                # Handle nested lists (batch dimension)
                if embedding.ndim > 1:
                    # Take mean pooling if needed
                    embedding = embedding.mean(axis=0)

                embeddings.append(embedding)

            except Exception as e:
                print(f"Error encoding text: {e}")
                # Return zero vector on error
                embeddings.append(np.zeros(384))  # all-MiniLM-L6-v2 has 384 dimensions

        embeddings = np.array(embeddings, dtype=np.float32)

        # Normalize if requested
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
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
    """Load pre-computed embeddings from .npy file"""
    if not path.exists():
        raise FileNotFoundError(
            f"Embeddings file not found at: {path.resolve()}\n"
            f"Please run the embedding generation script first to create embeddings.npy"
        )
    embeddings = np.load(path)
    print(f"✓ Loaded embeddings with shape: {embeddings.shape}")
    return embeddings


def format_citation(c: Dict[str, Any]) -> str:
    pages = f"pdf pages {c.get('page_start')}–{c.get('page_end')}"
    return (
        f"Law {c.get('law_number')} – {c.get('law_title')}, "
        f"{c.get('subsection_number')}. {c.get('subsection_title')} "
        f"({pages})"
    )


def expand_query_with_synonyms(query: str) -> str:
    q_lower = query.lower()
    expansions = []
    synonym_map = {
        "fan": ["spectator", "outside agent", "person not on team list"],
        "interrupts": ["interference", "interferes with play", "enters field"],
        "kicks away": ["touches ball", "plays ball", "interferes"],
        "shoulder charge": ["charges", "physical challenge", "body contact"],
        "far from ball": ["not within playing distance", "ball not in playing distance"],
        "strikes with hand": ["handball", "handles ball", "touches with hand", "hand ball"],
        "comes on pitch": ["enters field of play", "extra person on field"],
        "what should happen": ["what is the decision", "what is the restart", "what action"],
    }
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
               ["restart", "free kick", "penalty", "dropped ball", "indirect", "direct", "kick-off", "throw-in"])


def has_handball_intent(query: str) -> bool:
    q = query.lower()
    handball_terms = ["handball", "hand ball", "hand", "arm", "handle", "handling", "touches with hand",
                      "strikes with hand"]
    return any(term in q for term in handball_terms)


def has_interference_intent(query: str) -> bool:
    q = query.lower()
    interference_terms = [
        "fan", "spectator", "interrupts", "interference", "outside agent",
        "animal", "object enters", "comes on pitch", "enters field", "extra person"
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
    }
    return context


# -----------------------------
# RETRIEVER (Updated for HF API)
# -----------------------------
class HybridRetriever:
    def __init__(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray, hf_client: HFEmbeddingClient):
        self.chunks = chunks
        self.hf_client = hf_client

        # BM25 setup
        self.tokenized = [tokenize(c.get("text", "")) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized)

        # Use pre-computed embeddings
        self.embeddings = embeddings.astype("float32")

        # FAISS index setup
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

        print(f"✓ Retriever initialized with {len(chunks)} chunks")
        print(f"✓ Embedding dimension: {dim}")

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
                "stops play", "animal", "object", "unauthorized person"
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

        if "offside" in q:
            law_boost.add(11)
        if "dropped ball" in q:
            law_boost.update([8, 9])
        if "penalty kick" in q or "penalty" in q:
            law_boost.add(14)
        if "throw-in" in q:
            law_boost.add(15)
        if "goal kick" in q:
            law_boost.add(16)
        if "corner" in q:
            law_boost.add(17)
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
            subsection_terms += ["Restart", "Restart of play", "Offences and sanctions"]

        if any(k in q for k in ["substitute", "extra person", "team official", "enters the pitch"]):
            law_boost.add(3)
            subsection_terms += ["Extra persons", "Substitution procedure", "Offences and sanctions"]

        return {"law_boost": law_boost, "subsection_terms": subsection_terms}

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        expanded_query = expand_query_with_synonyms(query)
        q_tokens = tokenize(expanded_query)
        bm25_scores = self.bm25.get_scores(q_tokens)
        bm25_top = np.argsort(bm25_scores)[::-1][: max(80, top_k * 10)]

        # Use HuggingFace API to encode query
        q_emb = self.hf_client.encode([expanded_query], normalize=True)

        # Search FAISS index
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
        scenario_hints.append("This involves outside interference (fan, spectator, or unauthorized person).")
    if scenario["physical_contact"]:
        scenario_hints.append("This involves physical contact between players.")

    scenario_text = "\n".join(scenario_hints) if scenario_hints else "General scenario."

    system_instruction = (
        "You are a Laws of the Game (football/soccer) assistant.\n"
        "You MUST answer using ONLY the provided EXTRACTS.\n"
        "If any part of the answer is not clearly supported by the extracts, write:\n"
        "\"Not found in the provided extracts\" for that part.\n\n"
        "IMPORTANT RULES:\n"
        "- When the question involves a SPECIFIC SCENARIO (e.g., teammate passing to teammate, "
        "player far from ball, fan interference), apply the general rules to that specific scenario.\n"
        "- If the extract provides criteria/tests (e.g., 'it is an offence if ...'), "
        "explain how those criteria apply to the specific scenario described.\n"
        "- Consider WHERE the offence occurs (own half vs penalty area) when determining the restart.\n"
        "- For physical contact offences, consider whether the ball was within playing distance.\n"
        "- For outside interference, explain when play should stop and how it should restart.\n\n"
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
        "- When multiple extracts are relevant, synthesize them into one coherent answer.\n"
        "- Pay special attention to conditions like 'within playing distance' for physical offences.\n"
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
app = FastAPI(
    title="Football Laws of the Game RAG API",
    description="Enhanced API with HuggingFace Inference API for embeddings (FREE)",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
retriever: Optional[HybridRetriever] = None
hf_client: Optional[HFEmbeddingClient] = None
start_time = time.time()
query_count = 0


@app.on_event("startup")
async def startup_event():
    global retriever, hf_client
    try:
        # Initialize HuggingFace client
        if not HF_TOKEN:
            raise ValueError(
                "HF_TOKEN not found in environment variables.\n"
                "Get your free token from https://huggingface.co/settings/tokens\n"
                "Add it to your .env file: HF_TOKEN=your_token_here"
            )

        hf_client = HFEmbeddingClient(token=HF_TOKEN, model=EMBED_MODEL_NAME)
        print(f"✓ HuggingFace client initialized with model: {EMBED_MODEL_NAME}")

        # Load chunks
        chunks = load_chunks(CHUNKS_PATH)
        print(f"✓ Loaded {len(chunks)} chunks from {CHUNKS_PATH.resolve()}")

        # Load pre-computed embeddings
        embeddings = load_embeddings(EMBEDDINGS_PATH)

        # Verify dimensions match
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings.\n"
                "Please regenerate embeddings."
            )

        # Initialize retriever
        retriever = HybridRetriever(chunks, embeddings, hf_client)

        print(f"✓ API ready on http://localhost:8000")
        print(f"✓ Using HuggingFace Inference API (FREE)")

    except Exception as e:
        print(f"✗ Error during startup: {e}")
        raise


@app.get("/")
async def root():
    return {
        "message": "Football Laws of the Game RAG API v2.1",
        "version": "2.1.0",
        "embedding_service": "HuggingFace Inference API (FREE)",
        "endpoints": {
            "/ask": "POST - Ask a question about football laws",
            "/batch": "POST - Ask multiple questions at once",
            "/health": "GET - Health check with uptime",
            "/stats": "GET - Enhanced system statistics"
        },
        "features": [
            "HuggingFace Inference API for embeddings",
            "No local model download required",
            "Batch query processing",
            "Processing time metrics",
            "Enhanced statistics"
        ]
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

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
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    # Get unique law numbers
    unique_laws = sorted(list(set(c.get("law_number") for c in retriever.chunks if c.get("law_number"))))

    return StatsResponse(
        total_chunks=len(retriever.chunks),
        embedding_dimension=retriever.embeddings.shape[1],
        model=GEMINI_MODEL,
        embedding_model=EMBED_MODEL_NAME,
        embedding_service="HuggingFace Inference API",
        max_chars_per_chunk=MAX_CHARS_PER_CHUNK,
        max_total_context_chars=MAX_TOTAL_CONTEXT_CHARS,
        unique_laws=unique_laws,
        total_queries_processed=query_count
    )


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    global query_count

    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    try:
        start = time.time()

        # Retrieve relevant chunks
        top_chunks = retriever.search(request.question, top_k=request.top_k)

        # Generate answer with Gemini
        answer = gemini_answer(request.question, top_chunks)

        # Build evidence list
        evidence_list = []
        for i, chunk in enumerate(top_chunks, 1):
            text_preview = chunk.get("text", "")[:200] + "..." if len(chunk.get("text", "")) > 200 else chunk.get(
                "text", "")
            evidence_list.append(
                Evidence(
                    rank=i,
                    citation=format_citation(chunk),
                    text_preview=text_preview,
                    full_text=chunk.get("text", "") if request.include_raw_chunks else None
                )
            )

        # Extract retrieval info
        routing = retriever.route_rules(request.question)
        scenario = extract_scenario_context(request.question)

        retrieval_info = {
            "expanded_query": expand_query_with_synonyms(request.question),
            "quote_mode": is_quote_mode(request.question),
            "boosted_laws": list(routing["law_boost"]) if routing["law_boost"] else [],
            "scenario_context": scenario,
            "chunks_retrieved": len(top_chunks)
        }

        processing_time = (time.time() - start) * 1000  # Convert to milliseconds
        query_count += 1

        return QuestionResponse(
            question=request.question,
            answer=answer,
            evidence=evidence_list,
            retrieval_info=retrieval_info,
            processing_time_ms=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.post("/batch", response_model=BatchQuestionResponse)
async def batch_questions(request: BatchQuestionRequest):
    """Process multiple questions in a batch"""
    global query_count

    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    if not request.questions:
        raise HTTPException(status_code=400, detail="No questions provided")

    try:
        batch_start = time.time()
        results = []

        for question in request.questions:
            start = time.time()

            top_chunks = retriever.search(question, top_k=request.top_k)
            answer = gemini_answer(question, top_chunks)

            evidence_list = []
            for i, chunk in enumerate(top_chunks, 1):
                text_preview = chunk.get("text", "")[:200] + "..." if len(chunk.get("text", "")) > 200 else chunk.get(
                    "text", "")
                evidence_list.append(
                    Evidence(
                        rank=i,
                        citation=format_citation(chunk),
                        text_preview=text_preview
                    )
                )

            routing = retriever.route_rules(question)
            scenario = extract_scenario_context(question)

            retrieval_info = {
                "expanded_query": expand_query_with_synonyms(question),
                "quote_mode": is_quote_mode(question),
                "boosted_laws": list(routing["law_boost"]) if routing["law_boost"] else [],
                "scenario_context": scenario,
                "chunks_retrieved": len(top_chunks)
            }

            processing_time = (time.time() - start) * 1000

            results.append(
                QuestionResponse(
                    question=question,
                    answer=answer,
                    evidence=evidence_list,
                    retrieval_info=retrieval_info,
                    processing_time_ms=processing_time
                )
            )

            query_count += 1

        total_processing_time = (time.time() - batch_start) * 1000

        return BatchQuestionResponse(
            results=results,
            total_processing_time_ms=total_processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)