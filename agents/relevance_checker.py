import os
import logging
from typing import Optional
from config.settings import settings

logger = logging.getLogger(__name__)

class RelevanceChecker:
    def __init__(
        self,
        provider: str = "watsonx",
        model_id: Optional[str] = None,
        temperature: float = 0.0,
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
                "max_tokens": 150,
                "api_key": target_key
            }
            if target_url:
                init_kwargs["base_url"] = target_url
            self.model = ChatOpenAI(**init_kwargs)
            logger.info(f"RelevanceChecker initialized with {self.provider} model '{target_model}' (base_url: {target_url or 'default'}).")
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
                params={"temperature": temperature, "max_tokens": 150},
            )
            logger.info(f"RelevanceChecker initialized with WatsonX model '{target_model}'.")

    @staticmethod
    def is_overview_question(question: str) -> bool:
        q = question.lower().strip()
        overview_patterns = [
            "what the doc is about",
            "what is the doc about",
            "what is this doc about",
            "what is this document about",
            "what is the document about",
            "what is this file about",
            "what's this document about",
            "what's this doc about",
            "what does this document say",
            "what does this document do",
            "what does this file do",
            "what is this about",
            "what is this",
            "summarize",
            "summary",
            "overview",
            "main points",
            "explain this document",
            "tell me about this document",
            "tell me about this doc",
            "tell me about this",
        ]
        return any(p in q for p in overview_patterns)

    def check(self, question: str, retriever=None, documents=None, k: int = 3) -> str:
        logger.debug(f"RelevanceChecker.check called with question='{question}' and k={k}")

        if documents is not None:
            top_docs = documents
        elif retriever is not None:
            top_docs = retriever.invoke(question)
        else:
            top_docs = []

        if not top_docs:
            logger.debug("No documents available. Classifying as NO_MATCH.")
            return "NO_MATCH"

        # Fast-path for summary / overview queries when documents are loaded
        if self.is_overview_question(question):
            logger.info("Overview/summary query detected with active documents. Fast-passing as CAN_ANSWER.")
            return "CAN_ANSWER"

        document_content = "\n\n".join(doc.page_content for doc in top_docs[:k])
        doc_lower = document_content.lower()

        # Check if any significant terms from the user query appear in the retrieved passages
        cleaned_words = [
            w.strip("?,.:;'\"!@#$%^&*()[]{}")
            for w in question.lower().split()
            if len(w.strip("?,.:;'\"!@#$%^&*()[]{}")) >= 3
        ]
        has_matching_words = any(w in doc_lower for w in cleaned_words) if cleaned_words else False

        prompt = f"""You are an AI relevance checker between a user's question and provided document content.

**Instructions:**
- Classify how well the document content addresses the user's question.
- Respond with one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH.

**Special Guidelines:**
- If the question asks for an overview, summary, main topics, purpose, or what the document is about, and the passages are from the document, classify as CAN_ANSWER.
- If the passages discuss or mention the topic, keywords, or subject of the question in any way, even if partial, classify as PARTIAL.
- Only classify as NO_MATCH if the question asks about something completely alien and unrelated to the document content (e.g. asking for cooking recipes when the document is about software).

**Labels:**
1) "CAN_ANSWER": The passages contain enough explicit information to answer the question.
2) "PARTIAL": The passages mention or discuss the question's topic or contain related details.
3) "NO_MATCH": The passages do not discuss or relate to the question's topic at all.

**Question:** {question}
**Passages:** {document_content}

**Respond with your label (CAN_ANSWER, PARTIAL, or NO_MATCH):**"""

        try:
            if self.provider in ("openai", "custom", "custom api"):
                response = self.model.invoke([{"role": "user", "content": prompt}])
                llm_response = str(response.content).strip().upper()
            else:
                response = self.model.chat(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                llm_response = response["choices"][0]["message"]["content"].strip().upper()
        except Exception as e:
            err_msg = str(e)
            logger.error(f"Error during relevance model inference ({self.provider}): {err_msg}")
            raise RuntimeError(f"Relevance checking failed due to model error: {err_msg}") from e

        classification = "PARTIAL" if top_docs else "NO_MATCH"
        if "CAN_ANSWER" in llm_response:
            classification = "CAN_ANSWER"
        elif "PARTIAL" in llm_response:
            classification = "PARTIAL"
        elif "NO_MATCH" in llm_response and "CAN_ANSWER" not in llm_response and "PARTIAL" not in llm_response:
            if has_matching_words:
                logger.info("Overriding model NO_MATCH to PARTIAL because query words appear directly in retrieved passages.")
                classification = "PARTIAL"
            else:
                classification = "NO_MATCH"

        logger.info(f"Relevance classification result: {classification}")
        return classification
