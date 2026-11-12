import os
import logging
from typing import Dict, List, Optional
from langchain.schema import Document
from config.settings import settings

logger = logging.getLogger(__name__)

class VerificationAgent:
    def __init__(
        self,
        provider: str = "watsonx",
        model_id: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 200,
        project_id: Optional[str] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None
    ):
        self.provider = provider.lower()

        if self.provider in ("openai", "custom", "custom api"):
            from langchain_openai import ChatOpenAI
            target_model = model_id or "gpt-4o-mini"
            target_key = (
                api_key
                or (getattr(settings, "CUSTOM_API_KEY", "") if self.provider in ("custom", "custom api") else "")
                or getattr(settings, "OPENAI_API_KEY", "")
                or os.environ.get("CUSTOM_API_KEY", "")
                or os.environ.get("OPENAI_API_KEY", "")
            ).strip()
            target_url = url or (getattr(settings, "CUSTOM_API_URL", "") if self.provider in ("custom", "custom api") else getattr(settings, "OPENAI_BASE_URL", ""))
            init_kwargs = {
                "model": target_model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "api_key": target_key or "dummy-key"
            }
            if target_url:
                init_kwargs["base_url"] = target_url
            self.model = ChatOpenAI(**init_kwargs)
            logger.info(f"VerificationAgent initialized with {self.provider} model '{target_model}' (base_url: {target_url or 'default'}).")
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
                params={
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
            )
            logger.info(f"VerificationAgent initialized with WatsonX model '{target_model}'.")

    def sanitize_response(self, response_text: str) -> str:
        return response_text.strip()

    def generate_prompt(self, answer: str, context: str) -> str:
        return f"""You are an AI assistant designed to verify the accuracy and relevance of answers based on provided context.

**Instructions:**
- Verify the following answer against the provided context.
- Check for:
1. Direct/indirect factual support (YES/NO)
2. Unsupported claims (list any if present)
3. Contradictions (list any if present)
4. Relevance to the question (YES/NO)
- Provide additional details or explanations where relevant.
- Respond in the exact format specified below without adding any unrelated information.

**Format:**
Supported: YES/NO
Unsupported Claims: [item1, item2, ...]
Contradictions: [item1, item2, ...]
Relevant: YES/NO
Additional Details: [Any extra information or explanations]

**Answer:** {answer}
**Context:**
{context}

**Respond ONLY with the above format.**"""

    def parse_verification_response(self, response_text: str) -> Dict:
        try:
            lines = response_text.split("\n")
            verification = {}
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower()
                    value = value.strip()

                    if key == "supported":
                        verification["Supported"] = "YES" if "YES" in value.upper() else "NO"
                    elif key == "relevant":
                        verification["Relevant"] = "YES" if "YES" in value.upper() else "NO"
                    elif key in ("unsupported claims", "unsupported_claims"):
                        if value.startswith("[") and value.endswith("]"):
                            items = [item.strip().strip("\"'") for item in value[1:-1].split(",") if item.strip()]
                            verification["Unsupported Claims"] = items
                        else:
                            verification["Unsupported Claims"] = [] if value.lower() in ("none", "[]", "") else [value]
                    elif key == "contradictions":
                        if value.startswith("[") and value.endswith("]"):
                            items = [item.strip().strip("\"'") for item in value[1:-1].split(",") if item.strip()]
                            verification["Contradictions"] = items
                        else:
                            verification["Contradictions"] = [] if value.lower() in ("none", "[]", "") else [value]
                    elif key in ("additional details", "additional_details"):
                        verification["Additional Details"] = value

            if "Supported" not in verification:
                verification["Supported"] = "NO"
            if "Relevant" not in verification:
                verification["Relevant"] = "NO"
            if "Unsupported Claims" not in verification:
                verification["Unsupported Claims"] = []
            if "Contradictions" not in verification:
                verification["Contradictions"] = []
            if "Additional Details" not in verification:
                verification["Additional Details"] = ""

            return verification
        except Exception as e:
            logger.error(f"Error parsing verification response: {e}")
            return None

    def format_verification_report(self, verification: Dict) -> str:
        supported = verification.get("Supported", "NO")
        unsupported_claims = verification.get("Unsupported Claims", [])
        contradictions = verification.get("Contradictions", [])
        relevant = verification.get("Relevant", "NO")
        additional_details = verification.get("Additional Details", "")

        unsupported_text = ", ".join(unsupported_claims) if unsupported_claims else "None"
        contradictions_text = ", ".join(contradictions) if contradictions else "None"
        details_text = additional_details if additional_details else "None"

        return (
            f"**Supported:** {supported}\n"
            f"**Unsupported Claims:** {unsupported_text}\n"
            f"**Contradictions:** {contradictions_text}\n"
            f"**Relevant:** {relevant}\n"
            f"**Additional Details:** {details_text}"
        )

    def check(self, answer: str, documents: List[Document]) -> Dict:
        context = "\n\n".join([doc.page_content for doc in documents])
        prompt = self.generate_prompt(answer, context)

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
            logger.error(f"Error during verification model inference ({self.provider}): {e}")
            raise RuntimeError(f"Failed to verify answer due to model error: {e}") from e

        sanitized_response = self.sanitize_response(llm_response)
        parsed = self.parse_verification_response(sanitized_response)
        if not parsed:
            parsed = {
                "Supported": "NO",
                "Unsupported Claims": [],
                "Contradictions": [],
                "Relevant": "NO",
                "Additional Details": "Could not parse model verification output."
            }

        formatted = self.format_verification_report(parsed)
        return {
            "verification_report": formatted,
            "context_used": context,
            "parsed": parsed
        }
