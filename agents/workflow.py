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
    iteration_count: int
    custom_instructions: Optional[str]

class AgentWorkflow:
    def __init__(
        self,
        researcher: Optional[ResearchAgent] = None,
        verifier: Optional[VerificationAgent] = None,
        relevance_checker: Optional[RelevanceChecker] = None,
        max_re_research_cycles: int = 1,
        relevance_k: int = 5
    ):
        self.researcher = researcher or ResearchAgent()
        self.verifier = verifier or VerificationAgent()
        self.relevance_checker = relevance_checker or RelevanceChecker()
        self.max_re_research_cycles = max_re_research_cycles
        self.relevance_k = relevance_k
        self.compiled_workflow = self.build_workflow()

    def build_workflow(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("check_relevance", self._check_relevance_step)
        workflow.add_node("research", self._research_step)
        workflow.add_node("verify", self._verification_step)

        workflow.set_entry_point("check_relevance")
        workflow.add_conditional_edges(
            "check_relevance",
            self._decide_after_relevance_check,
            {
                "relevant": "research",
                "irrelevant": END
            }
        )
        workflow.add_edge("research", "verify")
        workflow.add_conditional_edges(
            "verify",
            self._decide_next_step,
            {
                "re_research": "research",
                "end": END
            }
        )
        return workflow.compile()

    def _check_relevance_step(self, state: AgentState) -> Dict:
        retriever = state["retriever"]
        classification = self.relevance_checker.check(
            question=state["question"],
            retriever=retriever,
            k=self.relevance_k
        )

        if classification in ("CAN_ANSWER", "PARTIAL"):
            return {"is_relevant": True}
        else:
            return {
                "is_relevant": False,
                "draft_answer": (
                    "This question is not related to the uploaded document(s), or the documents "
                    "do not contain sufficient data to answer it. Please ask a question relevant "
                    "to your uploaded documents."
                )
            }

    def _decide_after_relevance_check(self, state: AgentState) -> str:
        return "relevant" if state.get("is_relevant", False) else "irrelevant"

    def _research_step(self, state: AgentState) -> Dict:
        custom_instructions = state.get("custom_instructions")
        result = self.researcher.generate(
            state["question"],
            state["documents"],
            custom_instructions=custom_instructions
        )
        current_iterations = state.get("iteration_count", 0) + 1
        return {
            "draft_answer": result["draft_answer"],
            "iteration_count": current_iterations
        }

    def _verification_step(self, state: AgentState) -> Dict:
        result = self.verifier.check(state["draft_answer"], state["documents"])
        return {"verification_report": result["verification_report"]}

    def _decide_next_step(self, state: AgentState) -> str:
        verification_report = state.get("verification_report", "")
        iterations = state.get("iteration_count", 1)

        if iterations > self.max_re_research_cycles:
            logger.info("Maximum re-research cycles reached. Ending workflow.")
            return "end"

        if "Supported: NO" in verification_report or "Relevant: NO" in verification_report:
            logger.info("Verification indicates re-research needed.")
            return "re_research"

        logger.info("Verification passed. Ending workflow.")
        return "end"

    def full_pipeline(
        self,
        question: str,
        retriever: EnsembleRetriever,
        custom_instructions: Optional[str] = None
    ) -> Dict:
        try:
            documents = retriever.invoke(question)
            logger.info(f"Retrieved {len(documents)} document chunks for question.")

            initial_state: AgentState = {
                "question": question,
                "documents": documents,
                "draft_answer": "",
                "verification_report": "",
                "is_relevant": False,
                "retriever": retriever,
                "iteration_count": 0,
                "custom_instructions": custom_instructions
            }

            final_state = self.compiled_workflow.invoke(initial_state)

            return {
                "draft_answer": final_state.get("draft_answer", ""),
                "verification_report": final_state.get("verification_report", ""),
                "is_relevant": final_state.get("is_relevant", False),
                "retrieved_documents": documents
            }
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise
