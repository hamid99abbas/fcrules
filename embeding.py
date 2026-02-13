"""
Complete FIFA Laws of the Game Chunker & Embedder
==================================================
Standalone version using pdftotext (no PyMuPDF needed)

This script performs end-to-end processing:
1. Extracts text from PDF using pdftotext
2. Identifies Laws 1-17 and their subsections
3. Creates hierarchical chunks with intro paragraphs
4. Performs intelligent sub-chunking for long sections
5. Generates embeddings using sentence-transformers
6. Outputs chunks.jsonl and embeddings.npy

Key Features:
- ✅ Captures law introductions (the missing "own goal" rules!)
- ✅ Preserves section hierarchy with subsection 0 for intros
- ✅ Handles edge cases and validates chunk quality
- ✅ Creates semantic sub-chunks with overlap
- ✅ Works offline with pdftotext

Author: Enhanced for FIFA Laws 2025/26
"""

import os
import re
import json
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# Try to import sentence-transformers, fall back to placeholder
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with:")
    print("  pip install sentence-transformers")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ChunkConfig:
    """Configuration for chunking behavior"""
    # Sub-chunking
    enable_subchunking: bool = True
    target_tokens: int = 170
    overlap_tokens: int = 35
    min_subchunk_tokens: int = 50

    # Quality filters
    min_chunk_chars: int = 80
    remove_headers_footers: bool = True

    # Embedding
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    normalize_embeddings: bool = True
    skip_embeddings: bool = False


# Law structure from IFAB 2025/26
LAW_STRUCTURE = {
    1: {"title": "The Field of Play", "max_subsections": 14},
    2: {"title": "The Ball", "max_subsections": 3},
    3: {"title": "The Players", "max_subsections": 10},
    4: {"title": "The Players' Equipment", "max_subsections": 6},
    5: {"title": "The Referee", "max_subsections": 7},
    6: {"title": "The Other Match Officials", "max_subsections": 6},
    7: {"title": "The Duration of the Match", "max_subsections": 5},
    8: {"title": "The Start and Restart of Play", "max_subsections": 2},
    9: {"title": "The Ball in and out of Play", "max_subsections": 2},
    10: {"title": "Determining the Outcome of a Match", "max_subsections": 3},
    11: {"title": "Offside", "max_subsections": 4},
    12: {"title": "Fouls and Misconduct", "max_subsections": 5},
    13: {"title": "Free Kicks", "max_subsections": 3},
    14: {"title": "The Penalty Kick", "max_subsections": 3},
    15: {"title": "The Throw-in", "max_subsections": 2},
    16: {"title": "The Goal Kick", "max_subsections": 2},
    17: {"title": "The Corner Kick", "max_subsections": 2},
}

# Stop words for appendices - content after these should be ignored
APPENDIX_MARKERS = [
    'video assistant referee (var) protocol',
    'var protocol',
    'practical guidelines for match officials',
    'fifa quality programme',
    'glossary',
    'football terms',
    'law changes 2025/26',
]


# ============================================================================
# CHUNK DATA STRUCTURE
# ============================================================================

@dataclass
class LawChunk:
    """Represents a chunk of law text"""
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
    section_type: str = "core_law"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        d = asdict(self)
        # Remove None values
        return {k: v for k, v in d.items() if v is not None}


# ============================================================================
# PDF EXTRACTION
# ============================================================================

class PDFTextExtractor:
    """Extract text from PDF using pdftotext"""

    def __init__(self, pdf_path: Path, config: ChunkConfig):
        self.pdf_path = pdf_path
        self.config = config

    def extract_full_text(self) -> str:
        """Extract complete text from PDF"""
        try:
            result = subprocess.run(
                ['pdftotext', '-layout', str(self.pdf_path), '-'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to extract PDF text: {e}")

    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        if not text:
            return ""

        # Normalize whitespace
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'\x00-\x08\x0B\x0C\x0E-\x1F]', '', text)
        text = re.sub(r'\u0007', ' ', text)

        # Fix common PDF extraction issues
        text = text.replace('â€¢', '•')
        text = text.replace('â€"', '–')
        text = text.replace('â€™', "'")
        text = text.replace('\u2022', '•')
        text = text.replace('\u2013', '-')
        text = text.replace('\u2014', '-')

        # Remove page numbers and headers/footers
        if self.config.remove_headers_footers:
            text = re.sub(r'(?m)^Laws of the Game 2025/26.*$', '', text, flags=re.IGNORECASE)
            text = re.sub(r'(?m)^\s*\d{1,3}\s*$', '', text)
            text = re.sub(r'(?m)^Law \d+\s*\|.*$', '', text, flags=re.IGNORECASE)

        # Normalize spaces and newlines
        lines = []
        for line in text.split('\n'):
            line = re.sub(r'[ \t]{2,}', ' ', line).strip()
            if line:
                lines.append(line)

        text = '\n'.join(lines)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()


class LawParser:
    """Parse the full text into law sections"""

    def __init__(self, config: ChunkConfig):
        self.config = config

    def find_law_start_positions(self, text: str) -> Dict[int, int]:
        """
        Find the starting position of each law in the text.
        Returns {law_number: text_position}
        """
        positions = {}

        # Pattern to match law headers: "Law 1 – The Field of Play" or just "1" followed by law title
        patterns = [
            re.compile(r'\n\s*Law\s+(\d{1,2})\s*[-–—]\s*The\s+([^\n]+)', re.IGNORECASE),
            re.compile(r'\n\s*(\d{1,2})\s*\n\s*Law\s*\n', re.IGNORECASE),
            re.compile(r'\n\s*The\s+(Field of Play|Ball|Players|Players\' Equipment|Referee|'
                      r'Other Match Officials|Duration of the Match|Start and Restart of Play|'
                      r'Ball in and out of Play|Determining the Outcome|Offside|Fouls and Misconduct|'
                      r'Free Kicks|Penalty Kick|Throw-in|Goal Kick|Corner Kick)\s*\n', re.IGNORECASE)
        ]

        # Try to match each pattern
        for law_num, law_info in LAW_STRUCTURE.items():
            # Look for "Law X - Title" format
            pattern1 = re.compile(
                rf'\n\s*Law\s+{law_num}\s*[-–—]\s*{re.escape(law_info["title"])}\s*\n',
                re.IGNORECASE
            )
            match = pattern1.search(text)
            if match:
                positions[law_num] = match.start()
                continue

            # Look for just the title after a law number
            pattern2 = re.compile(
                rf'\n\s*{law_num}\s*\n\s*{re.escape(law_info["title"])}\s*\n',
                re.IGNORECASE
            )
            match = pattern2.search(text)
            if match:
                positions[law_num] = match.start()
                continue

            # Look for just the title
            pattern3 = re.compile(
                rf'\n\s*{re.escape(law_info["title"])}\s*\n',
                re.IGNORECASE
            )
            match = pattern3.search(text)
            if match and law_num not in positions:
                # Make sure it's not just a random mention
                context = text[max(0, match.start()-50):match.end()+50]
                if law_num not in positions:  # Only use if we haven't found it yet
                    positions[law_num] = match.start()

        return positions

    def cut_at_appendix(self, text: str) -> str:
        """Remove everything after appendix markers"""
        text_lower = text.lower()
        earliest_cut = len(text)

        for marker in APPENDIX_MARKERS:
            pos = text_lower.find(marker)
            if pos != -1:
                earliest_cut = min(earliest_cut, pos)

        if earliest_cut < len(text):
            return text[:earliest_cut]
        return text

    def extract_law_text(self, text: str, law_num: int, start_pos: int,
                        next_start: Optional[int]) -> str:
        """Extract text for a specific law"""
        if next_start:
            law_text = text[start_pos:next_start]
        else:
            law_text = text[start_pos:]

        # Cut at appendix
        law_text = self.cut_at_appendix(law_text)

        return law_text.strip()

    def parse_subsections(self, law_num: int, law_text: str) -> List[Dict[str, Any]]:
        """
        Parse subsections from law text.
        Returns list of dictionaries with intro as subsection 0.
        """
        law_info = LAW_STRUCTURE[law_num]
        law_title = law_info['title']

        # Remove the law header
        law_text = re.sub(
            rf'^\s*Law\s+{law_num}\s*[-–—]?\s*{re.escape(law_title)}\s*\n',
            '',
            law_text,
            count=1,
            flags=re.IGNORECASE
        )
        law_text = re.sub(rf'^\s*{law_num}\s*\n\s*{re.escape(law_title)}\s*\n', '', law_text, count=1, flags=re.IGNORECASE)
        law_text = re.sub(rf'^\s*{re.escape(law_title)}\s*\n', '', law_text, count=1, flags=re.IGNORECASE)
        law_text = re.sub(rf'^\s*{law_num}\s*\n', '', law_text, count=1)

        sections = []

        # Find all numbered subsections: "1. Procedure", "2. Offences and sanctions", etc.
        subsection_pattern = re.compile(r'^\s*(\d{1,2})\.\s+([^\n]{3,80})\s*$', re.MULTILINE)
        matches = list(subsection_pattern.finditer(law_text))

        # Extract introduction (everything before first subsection)
        if matches:
            intro_text = law_text[:matches[0].start()].strip()

            # Clean up intro
            intro_text = re.sub(r'^\s*\d{1,2}\s*\n', '', intro_text)
            intro_text = intro_text.strip()

            # Only include if substantial
            if len(intro_text) >= self.config.min_chunk_chars:
                sections.append({
                    'law_number': law_num,
                    'law_title': law_title,
                    'subsection_number': 0,
                    'subsection_title': 'Introduction',
                    'text': intro_text,
                    'is_introduction': True,
                })

        # Extract numbered subsections
        for i, match in enumerate(matches):
            try:
                sub_num = int(match.group(1))
                sub_title = match.group(2).strip()

                # Validate subsection number
                if sub_num < 1 or sub_num > law_info['max_subsections']:
                    continue

                # Get text until next subsection or end
                start = match.end()
                if i + 1 < len(matches):
                    end = matches[i + 1].start()
                else:
                    end = len(law_text)

                section_text = law_text[start:end].strip()

                # Skip if too short
                if len(section_text) < self.config.min_chunk_chars:
                    continue

                # Cut at appendix markers
                section_text = self.cut_at_appendix(section_text)

                if len(section_text) < self.config.min_chunk_chars:
                    continue

                sections.append({
                    'law_number': law_num,
                    'law_title': law_title,
                    'subsection_number': sub_num,
                    'subsection_title': sub_title,
                    'text': section_text,
                    'is_introduction': False,
                })

            except (ValueError, IndexError):
                continue

        return sections


# ============================================================================
# SUB-CHUNKING
# ============================================================================

class SubChunker:
    """Create semantic sub-chunks from large sections"""

    def __init__(self, config: ChunkConfig):
        self.config = config

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation"""
        return int(len(text.split()) * 1.3)

    def should_subchunk(self, text: str) -> bool:
        """Determine if text should be sub-chunked"""
        if not self.config.enable_subchunking:
            return False
        tokens = self.estimate_tokens(text)
        return tokens > self.config.target_tokens * 1.5

    def create_subchunks(self, chunk: LawChunk) -> List[LawChunk]:
        """Split a chunk into smaller sub-chunks with overlap"""
        if not self.should_subchunk(chunk.text):
            return [chunk]

        text = chunk.text
        lines = text.split('\n')

        # Extract heading if present
        heading = ""
        body = text

        if lines and len(lines[0]) < 100:
            heading_match = re.match(r'^(\d+)\.\s+(.+)$', lines[0])
            if heading_match:
                heading = lines[0]
                body = '\n'.join(lines[1:]).strip()

        # Split by words
        words = body.split()
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

            # Prepend heading for context
            if heading:
                chunk_text = f"{heading}\n{chunk_text}"

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
# EMBEDDING GENERATION
# ============================================================================

class EmbeddingGenerator:
    """Generate embeddings using sentence-transformers"""

    def __init__(self, config: ChunkConfig):
        self.config = config
        self.model = None

    def load_model(self):
        """Load the embedding model"""
        if not EMBEDDINGS_AVAILABLE:
            raise RuntimeError("sentence-transformers not installed")

        print(f"Loading embedding model: {self.config.embedding_model}")
        self.model = SentenceTransformer(self.config.embedding_model)
        print("Model loaded successfully")

    def generate_embeddings(self, chunks: List[LawChunk]) -> np.ndarray:
        """Generate embeddings for all chunks"""
        if self.model is None:
            self.load_model()

        texts = [chunk.text for chunk in chunks]

        print(f"Generating embeddings for {len(texts)} chunks...")

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
            normalize_embeddings=self.config.normalize_embeddings,
        )

        return embeddings.astype('float32')


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class LawsChunkerPipeline:
    """Complete end-to-end chunking pipeline"""

    def __init__(self, pdf_path: Path, output_dir: Path, config: ChunkConfig):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.config = config

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.chunks_file = output_dir / "chunks.jsonl"
        self.embeddings_file = output_dir / "embeddings.npy"
        self.stats_file = output_dir / "chunking_stats.json"

    def run(self) -> Tuple[List[LawChunk], Optional[np.ndarray]]:
        """Execute the complete pipeline"""

        print("=" * 80)
        print("FIFA LAWS OF THE GAME - CHUNKING & EMBEDDING PIPELINE")
        print("=" * 80)
        print(f"PDF: {self.pdf_path}")
        print(f"Output: {self.output_dir}")
        print(f"Sub-chunking: {'Enabled' if self.config.enable_subchunking else 'Disabled'}")
        print(f"Embeddings: {'Enabled' if not self.config.skip_embeddings else 'Disabled'}")
        print("=" * 80)

        # Step 1: Extract text from PDF
        print("\n[1/4] Extracting text from PDF...")
        extractor = PDFTextExtractor(self.pdf_path, self.config)
        full_text = extractor.extract_full_text()
        cleaned_text = extractor.clean_text(full_text)
        print(f"  → Extracted {len(cleaned_text):,} characters")

        # Step 2: Parse into laws and sections
        print("\n[2/4] Parsing laws and sections...")
        parser = LawParser(self.config)
        positions = parser.find_law_start_positions(cleaned_text)
        print(f"  → Found {len(positions)} laws")

        all_sections = []
        sorted_laws = sorted(positions.items())

        for i, (law_num, start_pos) in enumerate(sorted_laws):
            # Get next law start or end of text
            next_start = sorted_laws[i + 1][1] if i + 1 < len(sorted_laws) else None

            # Extract law text
            law_text = parser.extract_law_text(cleaned_text, law_num, start_pos, next_start)

            # Parse subsections
            sections = parser.parse_subsections(law_num, law_text)
            all_sections.extend(sections)

            intro_count = sum(1 for s in sections if s.get('is_introduction', False))
            subsection_count = len(sections) - intro_count
            print(f"  → Law {law_num:2d}: {intro_count} intro + {subsection_count} subsections = {len(sections)} total")

        print(f"\n  Total sections: {len(all_sections)}")

        # Step 3: Create chunks with sub-chunking
        print("\n[3/4] Creating chunks and sub-chunks...")
        chunks = []
        subchunker = SubChunker(self.config)

        for section in all_sections:
            # Estimate page numbers (rough)
            page_num = section['law_number'] * 5  # Rough estimate

            chunk_id = f"law{section['law_number']:02d}_sec{section['subsection_number']:02d}"
            chunk = LawChunk(
                chunk_id=chunk_id,
                law_number=section['law_number'],
                law_title=section['law_title'],
                subsection_number=section['subsection_number'],
                subsection_title=section['subsection_title'],
                text=section['text'],
                page_start=page_num,
                page_end=page_num + 2,
                char_count=len(section['text']),
                is_introduction=section.get('is_introduction', False),
            )

            # Apply sub-chunking
            subchunks = subchunker.create_subchunks(chunk)
            chunks.extend(subchunks)

        print(f"  → Created {len(chunks)} total chunks")
        subchunk_count = sum(1 for c in chunks if c.is_subchunk)
        intro_count = sum(1 for c in chunks if c.is_introduction)
        print(f"  → Includes {intro_count} introductions and {subchunk_count} sub-chunks")

        # Save chunks
        self._save_chunks(chunks)

        # Step 4: Generate embeddings
        embeddings = None
        if not self.config.skip_embeddings and EMBEDDINGS_AVAILABLE:
            print("\n[4/4] Generating embeddings...")
            try:
                generator = EmbeddingGenerator(self.config)
                embeddings = generator.generate_embeddings(chunks)
                print(f"  → Generated embeddings with shape: {embeddings.shape}")
                self._save_embeddings(embeddings)
            except Exception as e:
                print(f"  ✗ Failed to generate embeddings: {e}")
                print("  → Continuing without embeddings")
        else:
            print("\n[4/4] Skipping embeddings (disabled or not available)")

        # Save stats
        self._save_stats(chunks, embeddings)

        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        print(f"Chunks:     {self.chunks_file} ({len(chunks)} chunks)")
        if embeddings is not None:
            print(f"Embeddings: {self.embeddings_file} (shape: {embeddings.shape})")
        print(f"Stats:      {self.stats_file}")
        print("=" * 80)

        # Show sample chunks
        print("\nSample chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n{i+1}. {chunk.chunk_id}")
            print(f"   Law {chunk.law_number}: {chunk.law_title}")
            print(f"   Subsection {chunk.subsection_number}: {chunk.subsection_title}")
            print(f"   Text preview: {chunk.text[:100]}...")

        return chunks, embeddings

    def _save_chunks(self, chunks: List[LawChunk]):
        """Save chunks to JSONL"""
        with open(self.chunks_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                json.dump(chunk.to_dict(), f, ensure_ascii=False)
                f.write('\n')
        print(f"\n  → Saved {len(chunks)} chunks to {self.chunks_file}")

    def _save_embeddings(self, embeddings: np.ndarray):
        """Save embeddings to NPY"""
        np.save(self.embeddings_file, embeddings)
        print(f"  → Saved embeddings to {self.embeddings_file}")

    def _save_stats(self, chunks: List[LawChunk], embeddings: Optional[np.ndarray]):
        """Save statistics"""
        law_chunks = defaultdict(int)
        law_subsections = defaultdict(set)
        law_intros = defaultdict(int)

        for chunk in chunks:
            law_chunks[chunk.law_number] += 1
            law_subsections[chunk.law_number].add(chunk.subsection_number)
            if chunk.is_introduction:
                law_intros[chunk.law_number] += 1

        stats = {
            'total_chunks': len(chunks),
            'total_subchunks': sum(1 for c in chunks if c.is_subchunk),
            'total_introductions': sum(1 for c in chunks if c.is_introduction),
            'embedding_dimension': embeddings.shape[1] if embeddings is not None else None,
            'chunks_per_law': dict(sorted(law_chunks.items())),
            'subsections_per_law': {k: len(v) for k, v in sorted(law_subsections.items())},
            'introductions_per_law': dict(sorted(law_intros.items())),
            'avg_chunk_length': int(np.mean([c.char_count for c in chunks])),
            'config': {
                'target_tokens': self.config.target_tokens,
                'overlap_tokens': self.config.overlap_tokens,
                'min_chunk_chars': self.config.min_chunk_chars,
                'embedding_model': self.config.embedding_model,
            }
        }

        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Complete chunking and embedding pipeline for FIFA Laws of the Game',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python laws_chunker_standalone.py Laws_of_the_Game_2025_26.pdf
  
  # Custom output directory
  python laws_chunker_standalone.py Laws_of_the_Game_2025_26.pdf -o ./my_chunks
  
  # Disable sub-chunking
  python laws_chunker_standalone.py Laws_of_the_Game_2025_26.pdf --no-subchunk
  
  # Skip embeddings (chunks only)
  python laws_chunker_standalone.py Laws_of_the_Game_2025_26.pdf --no-embeddings
        """
    )

    parser.add_argument('pdf_path', type=Path, help='Path to the Laws of the Game PDF')
    parser.add_argument('-o', '--output-dir', type=Path, default=Path('./rag_chunks'),
                       help='Output directory (default: ./rag_chunks)')

    # Chunking options
    parser.add_argument('--no-subchunk', action='store_true',
                       help='Disable sub-chunking')
    parser.add_argument('--target-tokens', type=int, default=170,
                       help='Target tokens per sub-chunk (default: 170)')
    parser.add_argument('--overlap', type=int, default=35,
                       help='Overlap tokens between sub-chunks (default: 35)')
    parser.add_argument('--min-chunk-chars', type=int, default=80,
                       help='Minimum characters per chunk (default: 80)')

    # Embedding options
    parser.add_argument('--no-embeddings', action='store_true',
                       help='Skip embedding generation')
    parser.add_argument('--embedding-model', type=str,
                       default='sentence-transformers/all-MiniLM-L6-v2',
                       help='Embedding model name')

    args = parser.parse_args()

    # Validate
    if not args.pdf_path.exists():
        print(f"Error: PDF file not found: {args.pdf_path}")
        return 1

    # Create config
    config = ChunkConfig(
        enable_subchunking=not args.no_subchunk,
        target_tokens=args.target_tokens,
        overlap_tokens=args.overlap,
        min_chunk_chars=args.min_chunk_chars,
        embedding_model=args.embedding_model,
        skip_embeddings=args.no_embeddings,
    )

    # Run pipeline
    try:
        pipeline = LawsChunkerPipeline(args.pdf_path, args.output_dir, config)
        chunks, embeddings = pipeline.run()
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())