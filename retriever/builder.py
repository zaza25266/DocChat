import os
import logging
from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from config.settings import settings

logger = logging.getLogger(__name__)

class RetrieverBuilder:
    def __init__(
        self,
        provider: str = "watsonx",
        model_id: Optional[str] = None,
        url: Optional[str] = None,
        project_id: Optional[str] = None,
        api_key: Optional[str] = None,
        truncate_input_tokens: int = 512
    ):
        self.provider = provider.lower()

        if self.provider in ("openai", "custom", "custom api"):
            from langchain_openai import OpenAIEmbeddings
            emb_model = model_id or "text-embedding-3-small"
            target_api_key = (
                api_key
                or (getattr(settings, "CUSTOM_API_KEY", "") if self.provider in ("custom", "custom api") else "")
                or getattr(settings, "OPENAI_API_KEY", "")
                or os.environ.get("CUSTOM_API_KEY", "")
                or os.environ.get("OPENAI_API_KEY", "")
            ).strip()
            init_kwargs = {
                "model": emb_model,
                "api_key": target_api_key or "dummy-key"
            }
            target_url = url or (getattr(settings, "CUSTOM_API_URL", "") if self.provider in ("custom", "custom api") else getattr(settings, "OPENAI_BASE_URL", ""))
            if target_url:
                init_kwargs["base_url"] = target_url
            self.embeddings = OpenAIEmbeddings(**init_kwargs)
            logger.info(f"Initialized {self.provider} embeddings with model '{emb_model}' (base_url: {target_url or 'default'}).")
        else:
            from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
            from langchain_ibm import WatsonxEmbeddings

            default_model = model_id or "ibm/granite-embedding-278m-multilingual"
            clean_model_id = default_model.replace("ibm/ibm/", "ibm/")
            target_url = url or getattr(settings, "WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
            target_project_id = project_id or getattr(settings, "WATSONX_PROJECT_ID", "skills-network")
            target_api_key = (api_key or getattr(settings, "WATSONX_APIKEY", "") or os.environ.get("WATSONX_APIKEY", "")).strip()

            embed_params = {
                EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: truncate_input_tokens,
                EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
            }

            init_kwargs = {
                "model_id": clean_model_id,
                "url": target_url,
                "project_id": target_project_id,
                "params": embed_params
            }
            if target_api_key:
                init_kwargs["apikey"] = target_api_key

            self.embeddings = WatsonxEmbeddings(**init_kwargs)
            logger.info(f"Initialized WatsonX embeddings with model '{clean_model_id}'.")

    def build_hybrid_retriever(
        self,
        docs: List,
        k: Optional[int] = None,
        weights: Optional[List[float]] = None,
        persist_directory: Optional[str] = None
    ) -> EnsembleRetriever:
        target_k = k if k is not None else settings.VECTOR_SEARCH_K
        target_weights = weights if weights is not None else settings.HYBRID_RETRIEVER_WEIGHTS
        target_dir = persist_directory or settings.CHROMA_DB_PATH

        try:
            vector_store = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory=target_dir
            )
            logger.info("Chroma vector store initialized.")

            bm25 = BM25Retriever.from_documents(docs)
            bm25.k = target_k
            logger.info("BM25 retriever created.")

            vector_retriever = vector_store.as_retriever(search_kwargs={"k": target_k})
            logger.info("Vector retriever created.")

            hybrid_retriever = EnsembleRetriever(
                retrievers=[bm25, vector_retriever],
                weights=target_weights
            )
            logger.info(f"Hybrid retriever assembled with weights {target_weights} and k={target_k}.")
            return hybrid_retriever
        except Exception as e:
            logger.error(f"Failed to build hybrid retriever: {e}")
            raise
