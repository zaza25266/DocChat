import logging
from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END
from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever
from .research_agent import ResearchAgent
from .verification_agent import VerificationAgent
from .relevance_checker import RelevanceChecker

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    question: str
    documents: List[Document]
    draft_answer: str
    verification_report: str
    is_relevant: bool
    retriever: EnsembleRetriever
    loop_count: int

class AgentWorkflow:
    def __init__(
        self,
        research_model_id: str = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        research_temperature: float = 0.3,
        research_max_tokens: int = 300,
        verification_model_id: str = "ibm/granite-4-h-small",
        verification_temperature: float = 0.0,
        verification_max_tokens: int = 200,
        relevance_model_id: str = "ibm/granite-4-h-small",
        relevance_k: int = 20,
        enable_verification: bool = True,
        max_research_loops: int = 2,
    ):
        self.relevance_k = relevance_k
        self.enable_verification = enable_verification
        self.max_research_loops = max_research_loops

        self.researcher = ResearchAgent(
            model_id=research_model_id,
            temperature=research_temperature,
            max_tokens=research_max_tokens,
        )
        self.verifier = VerificationAgent(
            model_id=verification_model_id,
            temperature=verification_temperature,
            max_tokens=verification_max_tokens,
        ) if enable_verification else None
        self.relevance_checker = RelevanceChecker(model_id=relevance_model_id)
        self.compiled_workflow = self.build_workflow()

    def build_workflow(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("check_relevance", self._check_relevance_step)
        workflow.add_node("research", self._research_step)

        workflow.set_entry_point("check_relevance")
        workflow.add_conditional_edges(
            "check_relevance",
            self._decide_after_relevance_check,
            {
                "relevant": "research",
                "irrelevant": END,
            },
        )

        if self.enable_verification and self.verifier is not None:
            workflow.add_node("verify", self._verification_step)
            workflow.add_edge("research", "verify")
            workflow.add_conditional_edges(
                "verify",
                self._decide_next_step,
                {
                    "re_research": "research",
                    "end": END,
                },
            )
        else:
            workflow.add_edge("research", END)

        return workflow.compile()

    def _check_relevance_step(self, state: AgentState) -> Dict:
        retriever = state["retriever"]
        classification = self.relevance_checker.check(
            question=state["question"],
            retriever=retriever,
            k=self.relevance_k,
        )

        if classification in {"CAN_ANSWER", "PARTIAL"}:
            return {"is_relevant": True}

        return {
            "is_relevant": False,
            "draft_answer": "This question is not related to the uploaded document(s). Please ask a question relevant to the documents.",
        }

    def _decide_after_relevance_check(self, state: AgentState) -> str:
        return "relevant" if state["is_relevant"] else "irrelevant"

    def full_pipeline(self, question: str, retriever: EnsembleRetriever) -> Dict:
        try:
            documents = retriever.invoke(question)
            logger.info(f"Retrieved {len(documents)} relevant documents")

            initial_state = AgentState(
                question=question,
                documents=documents,
                draft_answer="",
                verification_report="",
                is_relevant=False,
                retriever=retriever,
                loop_count=0,
            )

            final_state = self.compiled_workflow.invoke(initial_state)

            return {
                "draft_answer": final_state.get("draft_answer", ""),
                "verification_report": final_state.get("verification_report", ""),
                "is_relevant": final_state.get("is_relevant", False),
                "documents": documents,
            }
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise

    def _research_step(self, state: AgentState) -> Dict:
        result = self.researcher.generate(state["question"], state["documents"])
        return {
            "draft_answer": result["draft_answer"],
            "loop_count": state.get("loop_count", 0) + 1,
        }

    def _verification_step(self, state: AgentState) -> Dict:
        if not self.verifier:
            return {"verification_report": ""}
        result = self.verifier.check(state["draft_answer"], state["documents"])
        return {"verification_report": result["verification_report"]}

    def _decide_next_step(self, state: AgentState) -> str:
        loop_count = state.get("loop_count", 0)
        if loop_count >= self.max_research_loops:
            logger.info("Maximum research loops reached, ending workflow.")
            return "end"

        verification_report = state.get("verification_report", "")
        if "Supported: NO" in verification_report or "Relevant: NO" in verification_report:
            logger.info("Verification indicates re-research needed.")
            return "re_research"
        return "end"
