import os
from typing import Dict, List, Optional
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials, APIClient
from langchain.schema import Document
from config.settings import settings
from utils.logging import logger

class ResearchAgent:
    def __init__(
        self,
        model_id: str = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        temperature: float = 0.3,
        max_tokens: int = 300,
        project_id: Optional[str] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None
    ):
        target_url = url or getattr(settings, "WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
        target_project_id = project_id or getattr(settings, "WATSONX_PROJECT_ID", "skills-network")
        target_key = api_key or getattr(settings, "WATSONX_APIKEY", "") or os.environ.get("WATSONX_APIKEY", "")

        cred_kwargs = {"url": target_url}
        if target_key:
            cred_kwargs["api_key"] = target_key

        credentials = Credentials(**cred_kwargs)
        self.client = APIClient(credentials)
        self.model = ModelInference(
            model_id=model_id,
            credentials=credentials,
            project_id=target_project_id,
            params={
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )

    def sanitize_response(self, response_text: str) -> str:
        return response_text.strip()

    def generate_prompt(self, question: str, context: str, custom_instructions: Optional[str] = None) -> str:
        instructions = custom_instructions or (
            "- Answer the following question using only the provided context.\n"
            "- Be clear, concise, and factual.\n"
            "- Return as much relevant information as you can get from the context."
        )

        return f"""You are an AI assistant designed to provide precise and factual answers based on the given context.

**Instructions:**
{instructions}

**Question:** {question}

**Context:**
{context}

**Provide your answer below:**"""

    def generate(
        self,
        question: str,
        documents: List[Document],
        custom_instructions: Optional[str] = None
    ) -> Dict:
        context = "\n\n".join([doc.page_content for doc in documents])
        prompt = self.generate_prompt(question, context, custom_instructions=custom_instructions)

        try:
            response = self.model.chat(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
        except Exception as e:
            logger.error(f"Error during research model inference: {e}")
            raise RuntimeError(f"Failed to generate answer due to model error: {e}") from e

        try:
            llm_response = response["choices"][0]["message"]["content"].strip()
        except (IndexError, KeyError, TypeError) as e:
            logger.warning(f"Unexpected response structure: {e}")
            llm_response = "I cannot answer this question based on the provided documents."

        draft_answer = self.sanitize_response(llm_response) if llm_response else "I cannot answer this question based on the provided documents."

        return {
            "draft_answer": draft_answer,
            "context_used": context
        }
