import os
from typing import Dict, List, Optional
from langchain.schema import Document
from config.settings import settings
from utils.logging import logger

class ResearchAgent:
    def __init__(
        self,
        provider: str = "watsonx",
        model_id: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        project_id: Optional[str] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None
    ):
        self.provider = provider.lower()

        if self.provider in ("openai", "custom", "custom api"):
            from langchain_openai import ChatOpenAI
            target_model = (model_id or "gpt-4o-mini").encode("ascii", "ignore").decode("ascii").strip()
            raw_key = (
                api_key
                or (getattr(settings, "CUSTOM_API_KEY", "") if self.provider in ("custom", "custom api") else "")
                or getattr(settings, "OPENAI_API_KEY", "")
                or os.environ.get("CUSTOM_API_KEY", "")
                or os.environ.get("OPENAI_API_KEY", "")
            )
            target_key = raw_key.encode("ascii", "ignore").decode("ascii").strip() if raw_key else "dummy-key"
            raw_url = url or (getattr(settings, "CUSTOM_API_URL", "") if self.provider in ("custom", "custom api") else getattr(settings, "OPENAI_BASE_URL", ""))
            target_url = raw_url.encode("ascii", "ignore").decode("ascii").strip() if raw_url else ""
            init_kwargs = {
                "model": target_model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "api_key": target_key
            }
            if target_url:
                init_kwargs["base_url"] = target_url
            self.model = ChatOpenAI(**init_kwargs)
            logger.info(f"ResearchAgent initialized with {self.provider} model '{target_model}' (base_url: {target_url or 'default'}).")
        else:
            from ibm_watsonx_ai.foundation_models import ModelInference
            from ibm_watsonx_ai import Credentials, APIClient

            target_model = model_id or "ibm/granite-3-8b-instruct"
            target_url = url or getattr(settings, "WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
            target_project_id = project_id or getattr(settings, "WATSONX_PROJECT_ID", "skills-network")
            target_key = (api_key or getattr(settings, "WATSONX_APIKEY", "") or os.environ.get("WATSONX_APIKEY", "")).strip()

            cred_kwargs = {"url": target_url}
            if target_key:
                cred_kwargs["api_key"] = target_key

            credentials = Credentials(**cred_kwargs)
            self.client = APIClient(credentials)
            self.model = ModelInference(
                model_id=target_model,
                credentials=credentials,
                project_id=target_project_id,
                params={
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
            )
            logger.info(f"ResearchAgent initialized with WatsonX model '{target_model}'.")

    def sanitize_response(self, response_text: str) -> str:
        return response_text.strip()

    def generate_prompt(self, question: str, context: str, custom_instructions: Optional[str] = None) -> str:
        instructions = custom_instructions or (
            "- Answer the following question strictly and solely using the provided context.\n"
            "- Be direct, factual, and concise.\n"
            "- CRITICAL RULE: If the context does not explicitly contain the answer, or if the document does not discuss the topic (e.g. how to win, victory conditions, rules, or gameplay objectives not in the text), you MUST explicitly state: 'The provided document(s) do not contain information on [topic].'\n"
            "- Do NOT extrapolate, speculate, assume, or invent recommendations, army strategies, or game rules that are not explicitly stated in the context."
        )

        return f"""You are an AI assistant designed to provide precise and factual answers strictly grounded in the given context.

**Instructions:**
{instructions}

**Question:** {question}

**Context:**
{context}

**Provide your answer below (strictly grounded in context; if not present, state clearly that it is not in the document):**"""

    def generate(
        self,
        question: str,
        documents: List[Document],
        custom_instructions: Optional[str] = None
    ) -> Dict:
        context = "\n\n".join([doc.page_content for doc in documents])
        prompt = self.generate_prompt(question, context, custom_instructions=custom_instructions)

        try:
            if self.provider in ("openai", "custom", "custom api"):
                response = self.model.invoke([{"role": "user", "content": prompt}])
                llm_response = str(response.content).strip()
            else:
                response = self.model.chat(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                llm_response = response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Error during research model inference ({self.provider}): {e}")
            raise RuntimeError(f"Failed to generate answer due to model error: {e}") from e

        draft_answer = self.sanitize_response(llm_response) if llm_response else "I cannot answer this question based on the provided documents."

        return {
            "draft_answer": draft_answer,
            "context_used": context
        }
