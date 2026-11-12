import os
import hashlib
import pickle
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from config import constants
from config.settings import settings
from utils.logging import logger

class DocumentProcessor:
    def __init__(self, headers: List = None, cache_expire_days: int = None):
        self.headers = headers or [("#", "Header 1"), ("##", "Header 2")]
        self.cache_expire_days = cache_expire_days or settings.CACHE_EXPIRE_DAYS
        self.cache_dir = Path(settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_files(self, files: List) -> None:
        total_size = 0
        for f in files:
            if hasattr(f, "size"):
                total_size += f.size
            elif hasattr(f, "name") and os.path.exists(f.name):
                total_size += os.path.getsize(f.name)
            elif isinstance(f, (str, Path)) and os.path.exists(f):
                total_size += os.path.getsize(f)
        if total_size > constants.MAX_TOTAL_SIZE:
            raise ValueError(f"Total size exceeds {constants.MAX_TOTAL_SIZE // 1024 // 1024}MB limit")
    
    def process(self, files: List) -> List:
        self.validate_files(files)
        all_chunks = []
        seen_hashes = set()
        
        for file in files:
            temp_path = None
            try:
                # Handle file-like objects or direct paths
                if hasattr(file, "read"):
                    content = file.read()
                    if hasattr(file, "seek"):
                        file.seek(0)
                    filename = getattr(file, "name", "document")
                    file_hash = self._generate_hash(content)
                    cache_path = self.cache_dir / f"{file_hash}.pkl"
                    
                    if self._is_cache_valid(cache_path):
                        logger.info(f"Loading from cache: {filename}")
                        chunks = self._load_from_cache(cache_path)
                    else:
                        logger.info(f"Processing and caching: {filename}")
                        suffix = Path(filename).suffix or ".txt"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(content)
                            temp_path = tmp.name
                        chunks = self._process_path(temp_path, filename)
                        self._save_to_cache(chunks, cache_path)
                else:
                    path_str = str(file.name if hasattr(file, "name") else file)
                    filename = Path(path_str).name
                    with open(path_str, "rb") as f:
                        file_hash = self._generate_hash(f.read())
                    cache_path = self.cache_dir / f"{file_hash}.pkl"
                    
                    if self._is_cache_valid(cache_path):
                        logger.info(f"Loading from cache: {filename}")
                        chunks = self._load_from_cache(cache_path)
                    else:
                        logger.info(f"Processing and caching: {filename}")
                        chunks = self._process_path(path_str, filename)
                        self._save_to_cache(chunks, cache_path)
                
                for chunk in chunks:
                    chunk_hash = self._generate_hash(chunk.page_content.encode())
                    if chunk_hash not in seen_hashes:
                        all_chunks.append(chunk)
                        seen_hashes.add(chunk_hash)
                        
            except Exception as e:
                logger.error(f"Failed to process {getattr(file, 'name', file)}: {str(e)}")
                continue
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
                
        logger.info(f"Total unique chunks: {len(all_chunks)}")
        return all_chunks
    
    def _process_path(self, file_path: str, filename: str) -> List:
        if not filename.lower().endswith(('.pdf', '.docx', '.txt', '.md')):
            logger.warning(f"Skipping unsupported file type: {filename}")
            return []

        converter = DocumentConverter()
        markdown = converter.convert(file_path).document.export_to_markdown()
        splitter = MarkdownHeaderTextSplitter(self.headers)
        return splitter.split_text(markdown)
    
    def _generate_hash(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def _save_to_cache(self, chunks: List, cache_path: Path):
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
