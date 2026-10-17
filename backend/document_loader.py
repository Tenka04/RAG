"""
Document Loader and Text Chunker
Supports: PDF (.pdf), Word (.docx), Markdown (.md), Plain Text (.txt), CSV (.csv), JSON (.json), Code (.py, .js, etc.)
"""

import os
from pathlib import Path
from typing import List, Dict, Any


def extract_text_from_file(file_path: str) -> str:
    """Extract clean text content from various file formats."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # 1. PDF files
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            pages_text = []
            for i, page in enumerate(reader.pages):
                page_text = ""
                # Attempt layout-aware extraction to preserve tables, columns, and structural spacing
                try:
                    page_text = page.extract_text(extraction_mode="layout")
                except Exception:
                    page_text = ""

                # Fallback to standard text stream if layout mode yields empty or fails
                if not page_text or not page_text.strip():
                    try:
                        page_text = page.extract_text()
                    except Exception:
                        page_text = ""

                if page_text and page_text.strip():
                    pages_text.append(f"[Page {i + 1}]\n{page_text.strip()}")
            return "\n\n".join(pages_text)
        except Exception as e:
            raise RuntimeError(f"Error reading PDF '{path.name}': {e}")

    # 2. Word documents (.docx)
    elif suffix == ".docx":
        try:
            import docx
            from docx.text.paragraph import Paragraph
            from docx.table import Table

            doc = docx.Document(str(path))
            extracted_parts = []

            # Traverse body elements in natural document order so paragraphs and tables stay properly sequenced
            for child in doc.element.body:
                if child.tag.endswith("p"):
                    p = Paragraph(child, doc)
                    if p.text.strip():
                        extracted_parts.append(p.text.strip())
                elif child.tag.endswith("tbl"):
                    table = Table(child, doc)
                    table_rows = []
                    header_added = False
                    for row in table.rows:
                        row_cells = [cell.text.replace("\r\n", " ").replace("\n", " ").strip() for cell in row.cells]
                        if any(row_cells):
                            row_line = " | ".join(c if c else "-" for c in row_cells)
                            table_rows.append(f"| {row_line} |")
                            if not header_added:
                                sep = " | ".join("---" for _ in row_cells)
                                table_rows.append(f"| {sep} |")
                                header_added = True
                    if table_rows:
                        extracted_parts.append("\n[Table Data]:\n" + "\n".join(table_rows))

            # Fallback if document structure had no body elements recognized
            if not extracted_parts:
                for p in doc.paragraphs:
                    if p.text.strip():
                        extracted_parts.append(p.text.strip())
                for table in doc.tables:
                    t_rows = []
                    for row in table.rows:
                        r_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                        if r_text:
                            t_rows.append(" | ".join(r_text))
                    if t_rows:
                        extracted_parts.append("\n[Table Data]:\n" + "\n".join(t_rows))

            return "\n\n".join(extracted_parts)
        except Exception as e:
            raise RuntimeError(f"Error reading DOCX '{path.name}': {e}")

    # 3. Text, Markdown, CSV, JSON, Code, etc.
    else:
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        for enc in encodings:
            try:
                with open(str(path), "r", encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f"Could not decode file '{path.name}' with standard encodings.")


def chunk_text(
    text: str,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
    metadata: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Split text into overlapping semantic chunks.
    Uses recursive split separators: paragraphs -> sentences -> words.
    """
    if metadata is None:
        metadata = {}

    cleaned_text = text.replace("\r\n", "\n").strip()
    if not cleaned_text:
        return []

    # Priority separators
    separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]

    def recursive_split(chunk: str, seps: List[str]) -> List[str]:
        if len(chunk) <= chunk_size or not seps:
            return [chunk] if chunk.strip() else []

        sep = seps[0]
        splits = chunk.split(sep)
        sub_chunks = []
        current = ""

        for part in splits:
            candidate = f"{current}{sep}{part}" if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    sub_chunks.append(current.strip())
                if len(part) > chunk_size:
                    # Recurse with deeper separator
                    sub_chunks.extend(recursive_split(part, seps[1:]))
                    current = ""
                else:
                    current = part

        if current.strip():
            sub_chunks.append(current.strip())

        return sub_chunks

    initial_chunks = recursive_split(cleaned_text, separators)

    # Apply overlap merging
    final_chunks = []
    for i, c in enumerate(initial_chunks):
        if not c.strip():
            continue

        chunk_meta = dict(metadata)
        chunk_meta["chunk_id"] = i
        chunk_meta["chunk_length"] = len(c)

        final_chunks.append({
            "content": c.strip(),
            "metadata": chunk_meta
        })

    return final_chunks


def load_and_chunk_file(
    file_path: str,
    chunk_size: int = 600,
    chunk_overlap: int = 100
) -> List[Dict[str, Any]]:
    """Load a file and return ready-to-embed chunk dictionaries."""
    path = Path(file_path)
    raw_text = extract_text_from_file(str(path))
    metadata = {
        "source": path.name,
        "extension": path.suffix.lower(),
        "file_size": path.stat().st_size
    }
    return chunk_text(raw_text, chunk_size, chunk_overlap, metadata)
