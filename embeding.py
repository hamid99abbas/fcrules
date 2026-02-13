"""
Complete FIFA Laws of the Game Chunker & Embedder - FINAL WORKING VERSION
==========================================================================
Two-step approach:
1. Extract each Law using TOC (PyMuPDF)
2. Split into subsections with introductions
3. Generate embeddings

This combines your working code with improvements for:
- Introduction detection (subsection 0)
- Better text cleaning
- Embedding generation
- Sub-chunking for long sections

Install:
  pip install pymupdf sentence-transformers

Run:
  python laws_chunker_final.py "Laws of the Game 2025_26.pdf"
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import argparse
import warnings

import numpy as np

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("ERROR: PyMuPDF not installed. Run: pip install pymupdf")

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: sentence-transformers not available")

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ChunkConfig:
    """Configuration for chunking"""
    # Text cleaning
    min_intro_chars: int = 30
    min_chunk_chars: int = 80
    max_subsections_per_law: int = 15  # Safety limit

    # Sub-chunking
    enable_subchunking: bool = True
    target_tokens: int = 170
    overlap_tokens: int = 35

    # Embedding
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    skip_embeddings: bool = False


@dataclass
class LawChunk:
    """Chunk with metadata"""
    chunk_id: str
    law_number: int
    law_title: str
    subsection_number: int  # 0 for intro
    subsection_title: str
    text: str
    page_start: int
    page_end: int
    char_count: int
    is_introduction: bool = False
    is_subchunk: bool = False
    parent_chunk_id: Optional[str] = None
    subchunk_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


# ============================================================================
# STEP 1: EXTRACT LAWS FROM PDF
# ============================================================================

class LawExtractor:
    """Extract individual laws from PDF using TOC"""

    # Matches TOC lines like: "1 The Field of Play 41"
    TOC_LINE_RE = re.compile(r"^\s*(\d{1,2})\s+(.+?)\s+(\d{1,4})\s*$")

    # Matches body header
    BODY_HEADER_RE = re.compile(
        r"Laws\s+of\s+the\s+Game\s+2025/26\s*\|\s*Law\s*(\d{1,2})\s*\|\s*(.+?)(?:\s+(\d{1,4}))?\s*$",
        re.IGNORECASE,
    )

    def __init__(self, pdf_path: Path):
        if not PYMUPDF_AVAILABLE:
            raise RuntimeError("PyMuPDF not installed")
        self.pdf_path = pdf_path
        self.pages = []

    def extract_pages(self) -> List[Dict[str, Any]]:
        """Extract all page texts"""
        doc = fitz.open(self.pdf_path)
        self.pages = []

        for i in range(doc.page_count):
            text = doc.load_page(i).get_text("text") or ""
            self.pages.append({
                "page_index": i,
                "page_number": i + 1,
                "text": text
            })

        doc.close()
        print(f"  → Extracted {len(self.pages)} pages")
        return self.pages

    def find_law_starts_from_toc(self, max_scan_pages: int = 25) -> Dict[int, Dict[str, Any]]:
        """Parse TOC to find law start pages"""
        law_starts = {}
        contents_found = False

        scan_pages = self.pages[:min(len(self.pages), max_scan_pages)]

        for page in scan_pages:
            text = page["text"]

            if not contents_found:
                if re.search(r"(?i)\bContents\b", text):
                    contents_found = True

            if not contents_found:
                continue

            for line in text.split("\n"):
                line = line.strip()
                m = self.TOC_LINE_RE.match(line)
                if not m:
                    continue

                law_num = int(m.group(1))
                title = m.group(2).strip()
                start_page = int(m.group(3))

                if 1 <= law_num <= 17 and law_num not in law_starts:
                    law_starts[law_num] = {
                        "title": title,
                        "start_page": start_page
                    }

        return law_starts

    def find_law_starts_from_headers(self) -> Dict[int, Dict[str, Any]]:
        """Fallback: detect starts from body headers"""
        law_starts = {}

        for page in self.pages:
            for line in page["text"].split("\n"):
                line = line.strip()
                m = self.BODY_HEADER_RE.match(line)
                if m:
                    law_num = int(m.group(1))
                    title = m.group(2).strip()
                    start_page = int(m.group(3)) if m.group(3) else page["page_number"]

                    if 1 <= law_num <= 17 and law_num not in law_starts:
                        law_starts[law_num] = {
                            "title": title,
                            "start_page": start_page
                        }

        return law_starts

    def extract_laws(self) -> Dict[int, Dict[str, Any]]:
        """Extract all 17 laws"""
        # Try TOC first
        law_starts = self.find_law_starts_from_toc()

        # Fallback to headers if TOC incomplete
        if len(law_starts) < 10:
            print("  → TOC incomplete, using body headers...")
            law_starts = self.find_law_starts_from_headers()

        if not law_starts:
            raise RuntimeError("Could not detect law start pages")

        # Sort by page number
        starts_sorted = sorted(
            [(ln, info["start_page"], info["title"])
             for ln, info in law_starts.items()],
            key=lambda x: x[1]
        )

        laws = {}
        page_count = len(self.pages)

        for idx, (law_num, start_page, title) in enumerate(starts_sorted):
            # Determine end page
            next_start = starts_sorted[idx + 1][1] if idx + 1 < len(starts_sorted) else (page_count + 1)
            end_page = next_start - 1

            # Convert to 0-based indexes
            start_i = max(0, start_page - 1)
            end_i = min(page_count - 1, end_page - 1)

            # Concatenate pages
            chunk_pages = self.pages[start_i:end_i + 1]
            combined = "\n\n".join([p["text"] for p in chunk_pages])

            laws[law_num] = {
                "law_number": law_num,
                "title": title,
                "page_start": start_page,
                "page_end": end_page,
                "raw_text": combined,
            }

        print(f"  → Extracted {len(laws)}/17 laws")
        if len(laws) < 17:
            missing = set(range(1, 18)) - set(laws.keys())
            print(f"  → WARNING: Missing laws: {sorted(missing)}")

        return laws


# ============================================================================
# STEP 2: SPLIT INTO SUBSECTIONS
# ============================================================================

class SubsectionParser:
    """Parse laws into subsections with introductions"""

    # Matches subsection headings like: "1. Field surface"
    SUBSECTION_RE = re.compile(r"(?m)^\s*(\d{1,2})\.\s+([^\n]+?)\s*$")

    # Header/footer to remove
    HEADER_FOOTER_RE = re.compile(
        r"(?im)^\s*Laws?\s+of\s+the\s+Game\s+2025/26\s*\|\s*Law\s*\d{1,2}\s*\|\s*.+?\s*$"
    )

    # Page numbers
    PAGE_ONLY_RE = re.compile(r"(?m)^\s*\d{1,3}\s*$")

    # Appendix markers - must be clear section headers with newlines, not words in text
    APPENDIX_HEADER_PATTERNS = [
        re.compile(r'\n\s*video\s+assistant\s+referee\s*\(?\s*var\s*\)?\s*protocol\s*\n', re.IGNORECASE),
        re.compile(r'\n\s*var\s+protocol\s*\n', re.IGNORECASE),
        re.compile(r'\n\s*practical\s+guidelines\s+for\s+match\s+officials\s*\n', re.IGNORECASE),
        re.compile(r'\n\s*fifa\s+quality\s+programme\s*\n', re.IGNORECASE),
        re.compile(r'\n\s*glossary\s*\n', re.IGNORECASE),
        re.compile(r'\n\s*football\s+terms\s*\n', re.IGNORECASE),
        re.compile(r'\n\s*law\s+changes\s+\d{4}/\d{2}\s*\n', re.IGNORECASE),
        re.compile(r'\n\s*details\s+of\s+all\s+law\s+changes\s*\n', re.IGNORECASE),
        re.compile(r'\n\s*notes\s+and\s+modifications\s*\n', re.IGNORECASE),
    ]

    def __init__(self, config: ChunkConfig):
        self.config = config

    def cut_at_appendix(self, text: str) -> str:
        """Remove everything after appendix section headers (not just keywords in text)"""
        earliest_cut = len(text)

        for pattern in self.APPENDIX_HEADER_PATTERNS:
            match = pattern.search(text)
            if match:
                earliest_cut = min(earliest_cut, match.start())

        if earliest_cut < len(text):
            return text[:earliest_cut].strip()
        return text

    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        if not text:
            return ""

        # Remove control characters
        text = text.replace("\u0007", " ")
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", text)

        # Normalize newlines
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove headers/footers
        text = self.HEADER_FOOTER_RE.sub("", text)
        text = self.PAGE_ONLY_RE.sub("", text)

        # Collapse whitespace
        lines = []
        for line in text.split("\n"):
            line = re.sub(r"[ \t]+", " ", line).strip()
            if line:
                lines.append(line)

        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def is_critical_intro(self, text: str) -> bool:
        """Check if intro contains critical phrases"""
        critical_phrases = [
            'goal may be scored',
            'goal cannot be scored',
            'cannot be scored directly',
            'may be scored directly',
            'ball is out of play',
            'ball is in play',
            'touches a match official',
            'awarded when',
            'awarded if',
        ]
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in critical_phrases)

    def parse_law(self, law: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split law into subsections"""
        law_num = law["law_number"]
        law_title = law["title"]
        page_start = law["page_start"]
        page_end = law["page_end"]

        # First cut at appendix markers, then clean
        raw_text = self.cut_at_appendix(law["raw_text"])
        text = self.clean_text(raw_text)

        # Find subsection headings
        matches = list(self.SUBSECTION_RE.finditer(text))

        sections = []

        # Extract introduction (before first subsection)
        if matches:
            intro_text = text[:matches[0].start()].strip()

            # Check if intro is substantial or critical
            if intro_text and (len(intro_text) >= self.config.min_intro_chars or
                              self.is_critical_intro(intro_text)):
                sections.append({
                    "law_number": law_num,
                    "law_title": law_title,
                    "subsection_number": 0,
                    "subsection_title": "Introduction",
                    "page_start": page_start,
                    "page_end": page_end,
                    "text": intro_text,
                    "is_introduction": True,
                })
                print(f"    → Law {law_num}: Added intro ({len(intro_text)} chars, "
                      f"critical={self.is_critical_intro(intro_text)})")
            elif not intro_text:
                # No intro text - law starts immediately with subsection 1
                # This is fine (e.g., Law 9)
                pass

        # Extract numbered subsections (with limit)
        for i, match in enumerate(matches):
            # Safety: stop if we've exceeded reasonable limit
            if i >= self.config.max_subsections_per_law:
                print(f"    → WARNING: Law {law_num} has {len(matches)} subsections, "
                      f"limiting to {self.config.max_subsections_per_law}")
                break

            sub_num = int(match.group(1))
            sub_title = match.group(2).strip()

            start_idx = match.start()
            end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            chunk_text = text[start_idx:end_idx].strip()

            if len(chunk_text) < self.config.min_chunk_chars:
                continue

            sections.append({
                "law_number": law_num,
                "law_title": law_title,
                "subsection_number": sub_num,
                "subsection_title": sub_title,
                "page_start": page_start,
                "page_end": page_end,
                "text": chunk_text,
                "is_introduction": False,
            })

        # If no subsections found, treat entire law as one section
        if not sections:
            sections.append({
                "law_number": law_num,
                "law_title": law_title,
                "subsection_number": 1,
                "subsection_title": "Complete Law",
                "page_start": page_start,
                "page_end": page_end,
                "text": text,
                "is_introduction": False,
            })

        return sections


# ============================================================================
# STEP 3: SUB-CHUNKING
# ============================================================================

class SubChunker:
    """Split large sections into smaller chunks"""

    def __init__(self, config: ChunkConfig):
        self.config = config

    def estimate_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3)

    def should_subchunk(self, text: str) -> bool:
        if not self.config.enable_subchunking:
            return False
        tokens = self.estimate_tokens(text)
        return tokens > self.config.target_tokens * 1.5

    def create_subchunks(self, chunk: LawChunk) -> List[LawChunk]:
        """Split chunk into sub-chunks with overlap"""
        if not self.should_subchunk(chunk.text):
            return [chunk]

        text = chunk.text
        words = text.split()

        if not words:
            return [chunk]

        target = self.config.target_tokens
        overlap = self.config.overlap_tokens

        subchunks = []
        start = 0
        idx = 1

        while start < len(words):
            end = min(len(words), start + target)
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)

            subchunk = LawChunk(
                chunk_id=f"{chunk.chunk_id}_s{idx:02d}",
                law_number=chunk.law_number,
                law_title=chunk.law_title,
                subsection_number=chunk.subsection_number,
                subsection_title=chunk.subsection_title,
                text=chunk_text,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                char_count=len(chunk_text),
                is_introduction=chunk.is_introduction,
                is_subchunk=True,
                parent_chunk_id=chunk.chunk_id,
                subchunk_index=idx,
            )
            subchunks.append(subchunk)

            if end >= len(words):
                break

            start = end - overlap
            idx += 1

        return subchunks


# ============================================================================
# STEP 4: EMBEDDING GENERATION
# ============================================================================

class EmbeddingGenerator:
    """Generate embeddings"""

    def __init__(self, config: ChunkConfig):
        self.config = config
        self.model = None

    def load_model(self):
        if not EMBEDDINGS_AVAILABLE:
            raise RuntimeError("sentence-transformers not installed")

        print(f"  Loading model: {self.config.embedding_model}")
        self.model = SentenceTransformer(self.config.embedding_model)
        print("  Model loaded")

    def generate_embeddings(self, chunks: List[LawChunk]) -> np.ndarray:
        if self.model is None:
            self.load_model()

        texts = [chunk.text for chunk in chunks]

        print(f"  Generating embeddings for {len(texts)} chunks...")

        try:
            from tqdm import tqdm
            show_progress = True
        except ImportError:
            show_progress = False

        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.astype('float32')


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class LawsChunkerPipeline:
    """Complete pipeline"""

    def __init__(self, pdf_path: Path, output_dir: Path, config: ChunkConfig):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.config = config

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.chunks_file = output_dir / "chunks.jsonl"
        self.embeddings_file = output_dir / "embeddings.npy"
        self.stats_file = output_dir / "chunking_stats.json"

    def run(self):
        """Execute pipeline"""
        print("=" * 80)
        print("FIFA LAWS CHUNKER - FINAL WORKING VERSION")
        print("=" * 80)
        print(f"PDF: {self.pdf_path}")
        print(f"Output: {self.output_dir}")
        print("=" * 80)

        # Step 1: Extract laws from PDF
        print("\n[1/4] Extracting laws from PDF...")
        extractor = LawExtractor(self.pdf_path)
        extractor.extract_pages()
        laws = extractor.extract_laws()

        # Step 2: Parse into subsections
        print("\n[2/4] Parsing subsections...")
        parser = SubsectionParser(self.config)
        all_sections = []

        for law_num in sorted(laws.keys()):
            sections = parser.parse_law(laws[law_num])
            all_sections.extend(sections)

            intro_count = sum(1 for s in sections if s.get('is_introduction'))
            subsection_count = len(sections) - intro_count
            print(f"  → Law {law_num:2d}: {intro_count} intro + {subsection_count} subsections")

        print(f"\n  Total sections: {len(all_sections)}")

        # Step 3: Create chunks with sub-chunking
        print("\n[3/4] Creating chunks...")
        chunker = SubChunker(self.config)
        chunks = []

        for section in all_sections:
            chunk_id = f"law{section['law_number']:02d}_sec{section['subsection_number']:02d}"
            chunk = LawChunk(
                chunk_id=chunk_id,
                law_number=section['law_number'],
                law_title=section['law_title'],
                subsection_number=section['subsection_number'],
                subsection_title=section['subsection_title'],
                text=section['text'],
                page_start=section['page_start'],
                page_end=section['page_end'],
                char_count=len(section['text']),
                is_introduction=section.get('is_introduction', False),
            )

            subchunks = chunker.create_subchunks(chunk)
            chunks.extend(subchunks)

        print(f"  → Created {len(chunks)} chunks")
        subchunk_count = sum(1 for c in chunks if c.is_subchunk)
        intro_count = sum(1 for c in chunks if c.is_introduction)
        print(f"  → {intro_count} introductions, {subchunk_count} sub-chunks")

        # Save chunks
        self._save_chunks(chunks)

        # Step 4: Generate embeddings
        embeddings = None
        if not self.config.skip_embeddings and EMBEDDINGS_AVAILABLE:
            print("\n[4/4] Generating embeddings...")
            try:
                generator = EmbeddingGenerator(self.config)
                embeddings = generator.generate_embeddings(chunks)
                print(f"  → Shape: {embeddings.shape}")
                self._save_embeddings(embeddings)
            except Exception as e:
                print(f"  ✗ Failed: {e}")
        else:
            print("\n[4/4] Skipping embeddings")

        # Save stats
        self._save_stats(chunks, embeddings, laws)

        print("\n" + "=" * 80)
        print("✅ COMPLETE")
        print("=" * 80)
        print(f"Chunks: {self.chunks_file} ({len(chunks)} chunks)")
        if embeddings is not None:
            print(f"Embeddings: {self.embeddings_file} (shape: {embeddings.shape})")
        print(f"Stats: {self.stats_file}")
        print("=" * 80)

        return chunks, embeddings

    def _save_chunks(self, chunks: List[LawChunk]):
        with open(self.chunks_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                json.dump(chunk.to_dict(), f, ensure_ascii=False)
                f.write('\n')
        print(f"  → Saved to {self.chunks_file}")

    def _save_embeddings(self, embeddings: np.ndarray):
        np.save(self.embeddings_file, embeddings)
        print(f"  → Saved to {self.embeddings_file}")

    def _save_stats(self, chunks: List[LawChunk], embeddings: Optional[np.ndarray], laws: Dict):
        from collections import defaultdict

        law_chunks = defaultdict(int)
        law_intros = defaultdict(int)

        for chunk in chunks:
            law_chunks[chunk.law_number] += 1
            if chunk.is_introduction:
                law_intros[chunk.law_number] += 1

        stats = {
            'total_chunks': len(chunks),
            'total_subchunks': sum(1 for c in chunks if c.is_subchunk),
            'total_introductions': sum(1 for c in chunks if c.is_introduction),
            'laws_found': sorted(list(laws.keys())),
            'laws_missing': sorted(list(set(range(1, 18)) - set(laws.keys()))),
            'embedding_dimension': embeddings.shape[1] if embeddings is not None else None,
            'chunks_per_law': dict(sorted(law_chunks.items())),
            'introductions_per_law': dict(sorted(law_intros.items())),
            'avg_chunk_length': int(np.mean([c.char_count for c in chunks])),
        }

        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Final working chunker for FIFA Laws'
    )

    parser.add_argument('pdf_path', type=Path, help='Path to PDF')
    parser.add_argument('-o', '--output-dir', type=Path, default=Path('./rag_chunks'),
                       help='Output directory (default: ./rag_chunks)')
    parser.add_argument('--no-subchunk', action='store_true',
                       help='Disable sub-chunking')
    parser.add_argument('--no-embeddings', action='store_true',
                       help='Skip embeddings')
    parser.add_argument('--min-intro-chars', type=int, default=30,
                       help='Min chars for intro (default: 30)')

    args = parser.parse_args()

    if not args.pdf_path.exists():
        print(f"Error: PDF not found: {args.pdf_path}")
        return 1

    if not PYMUPDF_AVAILABLE:
        print("Error: PyMuPDF not installed. Run: pip install pymupdf")
        return 1

    config = ChunkConfig(
        enable_subchunking=not args.no_subchunk,
        skip_embeddings=args.no_embeddings,
        min_intro_chars=args.min_intro_chars,
    )

    try:
        pipeline = LawsChunkerPipeline(args.pdf_path, args.output_dir, config)
        chunks, embeddings = pipeline.run()

        # Final check
        laws_found = set(c.law_number for c in chunks)
        if len(laws_found) == 17:
            print("\n✅ SUCCESS: All 17 laws found!")
        else:
            missing = set(range(1, 18)) - laws_found
            print(f"\n⚠️ WARNING: Missing laws {sorted(missing)}")

        return 0
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())