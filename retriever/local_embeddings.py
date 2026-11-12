import logging
from typing import List, Optional
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

class LocalHuggingFaceEmbeddings(Embeddings):
    """Local offline embedding model using PyTorch and HuggingFace Transformers.
    Downloads once from Hugging Face and permanently caches in ~/.cache/huggingface.
    Runs locally on CPU/GPU with zero API cost and zero external network calls after download.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", cache_folder: Optional[str] = None):
        import torch
        from transformers import AutoTokenizer, AutoModel

        self.model_name = model_name
        logger.info(f"Loading local embedding model '{model_name}' from cache...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_folder)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_folder)
        self.model.eval()
        logger.info(f"Local embedding model '{model_name}' successfully loaded and ready.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import torch
        import torch.nn.functional as F

        if not texts:
            return []

        # Process in batches to prevent memory spikes on large sets of chunks
        batch_size = 32
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            with torch.no_grad():
                output = self.model(**encoded)
                attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(output.last_hidden_state.size()).float()
                sum_embeddings = torch.sum(output.last_hidden_state * attention_mask, 1)
                sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
                pooled = sum_embeddings / sum_mask
                normalized = F.normalize(pooled, p=2, dim=1)
                all_embeddings.extend(normalized.tolist())

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
