from typing import Dict, List, Optional
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials, APIClient
from langchain.schema import Document

credentials = Credentials(url="https://us-south.ml.cloud.ibm.com")
client = APIClient(credentials)

class ResearchAgent:
    def __init__(
        self,
        model_id: str = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        temperature: float = 0.3,
        max_tokens: int = 300,
        project_id: str = "skills-network",
    ):
        self.model = ModelInference(
            model_id=model_id,
            credentials=credentials,
            project_id=project_id,
            params={
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )

    def sanitize_response(self, response_text: str) -> str:
        return response_text.strip()

    def generate_prompt(self, question: str, context: str) -> str:
        return f"""
        You are an AI assistant designed to provide precise and factual answers based on the given context.

        **Instructions:**
        - Answer the following question using only the provided context.
        - Be clear, concise, and factual.
        - Return as much information as you can get from the context.
        
        **Question:** {question}
        **Context:**
        {context}

        **Provide your answer below:**
        """

    def generate(self, question: str, documents: List[Document]) -> Dict:
        context = "\n\n".join([doc.page_content for doc in documents])
        prompt = self.generate_prompt(question, context)

        try:
            response = self.model.chat(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            )
        except Exception as e:
            raise RuntimeError("Failed to generate answer due to a model error.") from e

        try:
            llm_response = response["choices"][0]["message"]["content"].strip()
        except (IndexError, KeyError):
            llm_response = "I cannot answer this question based on the provided documents."

        draft_answer = self.sanitize_response(llm_response) if llm_response else "I cannot answer this question based on the provided documents."

        return {
            "draft_answer": draft_answer,
            "context_used": context,
        }
