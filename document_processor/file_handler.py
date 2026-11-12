import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import hashlib
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Union, Any, Optional, Callable
from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.schema import Document
from config import constants
from config.settings import settings
from utils.logging import logger

class DocumentProcessor:
    def __init__(self, headers: List = None, cache_dir: Union[str, Path] = None, cache_expire_days: int = None):
        self.headers = headers or [("#", "Header 1"), ("##", "Header 2")]
        self.cache_dir = Path(cache_dir or settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expire_days = cache_expire_days if cache_expire_days is not None else settings.CACHE_EXPIRE_DAYS
        self.last_errors: List[tuple] = []

    def _resolve_file_path(self, file_item: Any) -> Path:
        if isinstance(file_item, Path):
            return file_item
        if isinstance(file_item, str):
            return Path(file_item)
        if hasattr(file_item, "name") and isinstance(file_item.name, str):
            return Path(file_item.name)
        raise ValueError(f"Unable to resolve file path for object: {file_item}")

    def validate_files(self, files: List[Any]) -> None:
        total_size = 0
        for file_item in files:
            path = self._resolve_file_path(file_item)
            if not path.exists():
                raise FileNotFoundError(f"File not found on disk: {path}")
            file_size = path.stat().st_size
            if file_size > settings.MAX_FILE_SIZE:
                raise ValueError(
                    f"File {path.name} ({file_size // 1024 // 1024}MB) exceeds "
                    f"single file limit of {settings.MAX_FILE_SIZE // 1024 // 1024}MB"
                )
            total_size += file_size

        if total_size > settings.MAX_TOTAL_SIZE:
            raise ValueError(
                f"Total size ({total_size // 1024 // 1024}MB) exceeds "
                f"limit of {settings.MAX_TOTAL_SIZE // 1024 // 1024}MB"
            )

    def process(self, files: List[Any], progress_callback: Optional[Callable[[str], None]] = None) -> List:
        self.validate_files(files)
        self.last_errors = []
        all_chunks = []
        seen_hashes = set()

        for file_item in files:
            path = self._resolve_file_path(file_item)
            try:
                with open(path, "rb") as f:
                    file_hash = self._generate_hash(f.read())

                cache_path = self.cache_dir / f"{file_hash}.pkl"

                if self._is_cache_valid(cache_path):
                    if progress_callback:
                        progress_callback(f"Loading '{path.name}' from cache...")
                    logger.info(f"Loading from cache: {path.name}")
                    chunks = self._load_from_cache(cache_path)
                else:
                    if progress_callback:
                        progress_callback(f"Parsing and chunking '{path.name}'...")
                    logger.info(f"Processing and caching: {path.name}")
                    chunks = self._process_file(path, progress_callback=progress_callback)
                    self._save_to_cache(chunks, cache_path)

                for chunk in chunks:
                    chunk_hash = self._generate_hash(chunk.page_content.encode("utf-8"))
                    if chunk_hash not in seen_hashes:
                        all_chunks.append(chunk)
                        seen_hashes.add(chunk_hash)

            except Exception as e:
                logger.error(f"Failed to process {path.name}: {str(e)}")
                self.last_errors.append((path.name, str(e)))
                continue

        logger.info(f"Total unique chunks: {len(all_chunks)}")
        return all_chunks

    def _process_file(self, file_path: Path, progress_callback: Optional[Callable[[str], None]] = None) -> List:
        path_str = str(file_path)
        valid_extensions = tuple(settings.ALLOWED_TYPES)
        if not path_str.lower().endswith(valid_extensions):
            logger.warning(f"Skipping unsupported file type: {path_str}")
            return []

        if path_str.lower().endswith((".txt", ".md")):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    markdown = f.read()

                splitter = MarkdownHeaderTextSplitter(self.headers)
                chunks = splitter.split_text(markdown)
                if not chunks and markdown.strip():
                    chunks = [Document(page_content=markdown.strip(), metadata={"source": file_path.name})]

                for chunk in chunks:
                    chunk.metadata["source"] = file_path.name
                return chunks
            except Exception as e:
                logger.warning(f"Direct text parsing fallback to Docling for {file_path.name}: {e}")

        # Fast PDF parser for digital PDFs to avoid long OCR model downloads
        if path_str.lower().endswith(".pdf"):
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(file_path))
                pages_content = []
                for idx, page in enumerate(reader.pages):
                    pt = (page.extract_text() or "").strip()
                    if pt:
                        pages_content.append(f"## Page {idx + 1}\n\n{pt}")

                if pages_content:
                    full_md = "\n\n".join(pages_content)
                    splitter = MarkdownHeaderTextSplitter(self.headers)
                    chunks = splitter.split_text(full_md)
                    if not chunks and full_md.strip():
                        chunks = [Document(page_content=full_md.strip(), metadata={"source": file_path.name})]
                    for chunk in chunks:
                        chunk.metadata["source"] = file_path.name
                    logger.info(f"Fast PDF parser extracted {len(chunks)} chunks from {file_path.name}.")
                    return chunks
            except Exception as e:
                logger.warning(f"Fast PDF extraction failed for {file_path.name}, falling back to Docling: {e}")

        # Fast DOCX parser using python-docx
        if path_str.lower().endswith(".docx"):
            try:
                import docx
                doc = docx.Document(str(file_path))
                elements = []
                for p in doc.paragraphs:
                    text = p.text.strip()
                    if text:
                        if p.style and p.style.name and p.style.name.startswith("Heading 1"):
                            elements.append(f"# {text}")
                        elif p.style and p.style.name and p.style.name.startswith("Heading 2"):
                            elements.append(f"## {text}")
                        elif p.style and p.style.name and p.style.name.startswith("Heading 3"):
                            elements.append(f"### {text}")
                        else:
                            elements.append(text)
                for t in doc.tables:
                    for row in t.rows:
                        row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                        if row_text:
                            elements.append(row_text)

                if elements:
                    full_docx_text = "\n\n".join(elements)
                    splitter = MarkdownHeaderTextSplitter(self.headers)
                    chunks = splitter.split_text(full_docx_text)
                    if not chunks and full_docx_text.strip():
                        chunks = [Document(page_content=full_docx_text.strip(), metadata={"source": file_path.name})]

                    # Subdivide any oversized chunks so they fit within vector embedding token limits
                    from langchain_text_splitters import RecursiveCharacterTextSplitter
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
                    refined_chunks = []
                    for c in chunks:
                        c.metadata["source"] = file_path.name
                        if len(c.page_content) > 1000:
                            sub_texts = text_splitter.split_text(c.page_content)
                            for sub in sub_texts:
                                refined_chunks.append(Document(page_content=sub, metadata=dict(c.metadata)))
                        else:
                            refined_chunks.append(c)

                    logger.info(f"Fast DOCX parser extracted {len(refined_chunks)} chunks from {file_path.name}.")
                    return refined_chunks
            except Exception as e:
                logger.warning(f"Fast DOCX extraction failed for {file_path.name}, falling back to Docling: {e}")

        # Fallback to Docling parser
        if progress_callback:
            progress_callback(f"Analyzing layout with Docling for '{file_path.name}'...")
        converter = DocumentConverter()
        converted = converter.convert(path_str)
        markdown = converted.document.export_to_markdown()

        splitter = MarkdownHeaderTextSplitter(self.headers)
        chunks = splitter.split_text(markdown)
        if not chunks and markdown.strip():
            chunks = [Document(page_content=markdown.strip(), metadata={"source": file_path.name})]

        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
        final_chunks = []
        for chunk in chunks:
            chunk.metadata["source"] = file_path.name
            if len(chunk.page_content) > 1000:
                sub_texts = text_splitter.split_text(chunk.page_content)
                for sub in sub_texts:
                    final_chunks.append(Document(page_content=sub, metadata=dict(chunk.metadata)))
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _generate_hash(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def _save_to_cache(self, chunks: List, cache_path: Path) -> None:
        with open(cache_path, "wb") as f:
            pickle.dump({
                "timestamp": datetime.now().timestamp(),
                "chunks": chunks
            }, f)

    def _load_from_cache(self, cache_path: Path) -> List:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data["chunks"]

    def _is_cache_valid(self, cache_path: Path) -> bool:
        if not cache_path.exists():
            return False
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return cache_age < timedelta(days=self.cache_expire_days)

    def clear_cache(self) -> int:
        deleted = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
                deleted += 1
            except Exception as e:
                logger.error(f"Failed to delete {cache_file}: {e}")
        return deleted
