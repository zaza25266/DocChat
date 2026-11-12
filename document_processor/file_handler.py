import os
import hashlib
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Union, Any
from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from config import constants
from config.settings import settings
from utils.logging import logger

class DocumentProcessor:
    def __init__(self, headers: List = None, cache_dir: Union[str, Path] = None, cache_expire_days: int = None):
        self.headers = headers or [("#", "Header 1"), ("##", "Header 2")]
        self.cache_dir = Path(cache_dir or settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expire_days = cache_expire_days if cache_expire_days is not None else settings.CACHE_EXPIRE_DAYS

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

    def process(self, files: List[Any]) -> List:
        self.validate_files(files)
        all_chunks = []
        seen_hashes = set()

        for file_item in files:
            path = self._resolve_file_path(file_item)
            try:
                with open(path, "rb") as f:
                    file_hash = self._generate_hash(f.read())

                cache_path = self.cache_dir / f"{file_hash}.pkl"

                if self._is_cache_valid(cache_path):
                    logger.info(f"Loading from cache: {path.name}")
                    chunks = self._load_from_cache(cache_path)
                else:
                    logger.info(f"Processing and caching: {path.name}")
                    chunks = self._process_file(path)
                    self._save_to_cache(chunks, cache_path)

                for chunk in chunks:
                    chunk_hash = self._generate_hash(chunk.page_content.encode("utf-8"))
                    if chunk_hash not in seen_hashes:
                        all_chunks.append(chunk)
                        seen_hashes.add(chunk_hash)

            except Exception as e:
                logger.error(f"Failed to process {path.name}: {str(e)}")
                continue

        logger.info(f"Total unique chunks: {len(all_chunks)}")
        return all_chunks

    def _process_file(self, file_path: Path) -> List:
        path_str = str(file_path)
        valid_extensions = tuple(settings.ALLOWED_TYPES)
        if not path_str.lower().endswith(valid_extensions):
            logger.warning(f"Skipping unsupported file type: {path_str}")
            return []

        converter = DocumentConverter()
        converted = converter.convert(path_str)
        markdown = converted.document.export_to_markdown()

        splitter = MarkdownHeaderTextSplitter(self.headers)
        chunks = splitter.split_text(markdown)

        for chunk in chunks:
            chunk.metadata["source"] = file_path.name

        return chunks

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
