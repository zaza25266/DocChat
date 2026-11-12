import logging
from typing import TypedDict, List, Dict, Optional, Callable
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
        custom_instructions: Optional[str] = None,
        step_callback: Optional[Callable[[str], None]] = None,
        enable_verification: bool = True,
        enable_relevance_check: bool = True,
        max_context_chunks: int = 5
    ) -> Dict:
        try:
            if step_callback:
                step_callback("Step 1: Retrieving relevant passages via hybrid search (BM25 + Dense Vector)...")

            documents = retriever.invoke(question)
            logger.info(f"Retrieved {len(documents)} document chunks for question.")

            if not documents:
                return {
                    "draft_answer": "No relevant content found in the uploaded documents to answer this question.",
                    "verification_report": "**Relevance:** NO\n**Supported:** NO\n**Additional Details:** No documents retrieved.",
                    "is_relevant": False,
                    "retrieved_documents": []
                }

            # Select top chunks for prompt context to keep LLM response fast and focused
            context_docs = documents[:max_context_chunks]

            if enable_relevance_check:
                if step_callback:
                    step_callback(f"Retrieved {len(documents)} candidate chunks. Evaluating question relevance...")

                classification = self.relevance_checker.check(
                    question=question,
                    documents=context_docs,
                    k=self.relevance_k
                )

                if classification not in ("CAN_ANSWER", "PARTIAL"):
                    if step_callback:
                        step_callback("Relevance check borderline. Verifying with Research Agent...")
                    logger.info("Relevance checker flagged NO_MATCH. Calling Research Agent with strict grounding.")
                    research_res = self.researcher.generate(
                        question,
                        context_docs,
                        custom_instructions=f"{custom_instructions or ''}\nNote: Answer strictly based on the context. If the document does not contain the answer, state that clearly."
                    )
                    draft_answer = research_res["draft_answer"]
                    return {
                        "draft_answer": draft_answer,
                        "verification_report": "**Relevance:** PARTIAL\n**Supported:** YES\n**Additional Details:** Evaluated directly with research agent.",
                        "is_relevant": True,
                        "retrieved_documents": documents
                    }

            if step_callback:
                step_callback("Step 2: Synthesizing draft answer with Research Agent...")

            research_res = self.researcher.generate(
                question,
                context_docs,
                custom_instructions=custom_instructions
            )
            draft_answer = research_res["draft_answer"]

            if not enable_verification:
                if step_callback:
                    step_callback("Verification skipped (Fast Mode enabled).")
                return {
                    "draft_answer": draft_answer,
                    "verification_report": "**Status:** Verification skipped (Fast Mode).",
                    "is_relevant": True,
                    "retrieved_documents": documents
                }

            if step_callback:
                step_callback("Step 3: Auditing draft answer for factual consistency with Verification Agent...")

            verification_res = self.verifier.check(draft_answer, context_docs, question=question)
            verification_report = verification_res["verification_report"]
            parsed = verification_res.get("parsed") or {}

            iterations = 1
            def check_re_research_needed(p, report_str):
                return (
                    p.get("Supported") == "NO"
                    or p.get("Relevant") == "NO"
                    or "**Supported:** NO" in report_str
                    or "Supported: NO" in report_str
                )

            while iterations <= self.max_re_research_cycles and check_re_research_needed(parsed, verification_report):
                if step_callback:
                    step_callback(f"Verification flagged unsupported claims. Re-researching (cycle {iterations}/{self.max_re_research_cycles})...")

                research_res = self.researcher.generate(
                    question,
                    context_docs,
                    custom_instructions=(
                        f"{custom_instructions or ''}\n"
                        f"CRITICAL: Previous draft contained unsupported claims or speculation flagged by verification:\n"
                        f"{verification_report}\n"
                        f"Rule: If the context does not explicitly describe how to answer this question (e.g. win conditions, rules not in the text), you MUST explicitly state that the document does not contain this information. Do NOT extrapolate or assume."
                    )
                )
                draft_answer = research_res["draft_answer"]
                verification_res = self.verifier.check(draft_answer, context_docs, question=question)
                verification_report = verification_res["verification_report"]
                parsed = verification_res.get("parsed") or {}
                iterations += 1

            if step_callback:
                step_callback("Step 4: Answer generation and verification complete.")

            return {
                "draft_answer": draft_answer,
                "verification_report": verification_report,
                "is_relevant": True,
                "retrieved_documents": documents
            }
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise
