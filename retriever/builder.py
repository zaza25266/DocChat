import logging
from typing import List, Optional
from langchain_community.vectorstores import Chroma
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from config.settings import settings

logger = logging.getLogger(__name__)

class RetrieverBuilder:
    def __init__(
        self,
        embedding_model_id: str = "ibm/ibm/granite-embedding-278m-multilingual",
        project_id: str = "skills-network",
        truncate_tokens: int = 512,
    ):
        embed_params = {
            EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: truncate_tokens,
            EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
        }

        self.embeddings = WatsonxEmbeddings(
            model_id=embedding_model_id,
            url="https://us-south.ml.cloud.ibm.com",
            project_id=project_id,
            params=embed_params,
        )

    def build_hybrid_retriever(
        self,
        docs: List,
        k: Optional[int] = None,
        weights: Optional[List[float]] = None,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> EnsembleRetriever:
        try:
            target_k = k if k is not None else settings.VECTOR_SEARCH_K
            target_weights = weights if weights is not None else settings.HYBRID_RETRIEVER_WEIGHTS
            target_dir = persist_directory or settings.CHROMA_DB_PATH
            target_collection = collection_name or settings.CHROMA_COLLECTION_NAME

            vector_store = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory=target_dir,
                collection_name=target_collection,
            )
            logger.info("Vector store created successfully.")

            bm25 = BM25Retriever.from_documents(docs)
            logger.info("BM25 retriever created successfully.")

            vector_retriever = vector_store.as_retriever(search_kwargs={"k": target_k})
            logger.info(f"Vector retriever created with k={target_k}.")

            hybrid_retriever = EnsembleRetriever(
                retrievers=[bm25, vector_retriever],
                weights=target_weights,
            )
            logger.info(f"Hybrid retriever created with weights={target_weights}.")
            return hybrid_retriever
        except Exception as e:
            logger.error(f"Failed to build hybrid retriever: {e}")
            raise
