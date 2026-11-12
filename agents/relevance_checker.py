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

        if self.provider == "openai":
            from langchain_openai import ChatOpenAI
            target_model = model_id or "gpt-4o-mini"
            target_key = (api_key or getattr(settings, "OPENAI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")).strip()
            self.model = ChatOpenAI(
                model=target_model,
                temperature=temperature,
                max_tokens=10,
                api_key=target_key
            )
            logger.info(f"RelevanceChecker initialized with OpenAI model '{target_model}'.")
        else:
            from ibm_watsonx_ai.foundation_models import ModelInference
            from ibm_watsonx_ai import Credentials, APIClient

            target_model = model_id or "ibm/granite-4-h-small"
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
                params={"temperature": temperature, "max_tokens": 10},
            )
            logger.info(f"RelevanceChecker initialized with WatsonX model '{target_model}'.")

    def check(self, question: str, retriever, k: int = 3) -> str:
        logger.debug(f"RelevanceChecker.check called with question='{question}' and k={k}")

        top_docs = retriever.invoke(question)
        if not top_docs:
            logger.debug("No documents returned from retriever.invoke(). Classifying as NO_MATCH.")
            return "NO_MATCH"

        document_content = "\n\n".join(doc.page_content for doc in top_docs[:k])

        prompt = f"""You are an AI relevance checker between a user's question and provided document content.

**Instructions:**
- Classify how well the document content addresses the user's question.
- Respond with only one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH.
- Do not include any additional text or explanation.

**Labels:**
1) "CAN_ANSWER": The passages contain enough explicit information to fully answer the question.
2) "PARTIAL": The passages mention or discuss the question's topic but do not provide all the details needed for a complete answer.
3) "NO_MATCH": The passages do not discuss or mention the question's topic at all.

**Important:** If the passages mention or reference the topic or timeframe of the question in any way, even if incomplete, respond with "PARTIAL" instead of "NO_MATCH".

**Question:** {question}
**Passages:** {document_content}

**Respond ONLY with one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH**"""

        try:
            if self.provider == "openai":
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
            logger.error(f"Error during relevance model inference ({self.provider}): {e}")
            return "NO_MATCH"

        valid_labels = {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}
        classification = "NO_MATCH"
        for label in valid_labels:
            if label in llm_response:
                classification = label
                break

        logger.info(f"Relevance classification result: {classification}")
        return classification
