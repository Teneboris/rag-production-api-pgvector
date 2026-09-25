from typing import Optional
from typing_extensions import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_ollama import ChatOllama
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage,
)

from langsmith import traceable

from src.app.config import get_settings

from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Agent State
# ============================================================

class AgentState(TypedDict):
    """
    State for the production agent.

    add_messages ensures that new messages are appended
    instead of replacing the existing conversation.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    error: Optional[str]
    retry_count: int
    model_used: str


# ============================================================
# Production Agent
# ============================================================

class ProductionAgent:
    """
    Production LangGraph agent with:

    - Primary model
    - Fallback model
    - Retry/fallback routing
    - Graceful error handling
    - LangSmith tracing
    """

    def __init__(self):
        settings = get_settings()

        ollama_kwargs = {
            "base_url": settings.ollama_base_url,
        }

        if settings.ollama_api_key:
            ollama_kwargs["client_kwargs"] = {
                "headers": {
                    "Authorization": (
                        f"Bearer {settings.ollama_api_key}"
                    ),
                },
            }

        self.primary_llm = ChatOllama(
            model=settings.primary_model,
            temperature=0,
            **ollama_kwargs,
        )

        self.fallback_llm = ChatOllama(
            model=settings.fallback_model,
            temperature=0,
            **ollama_kwargs,
        )

        self.max_retries = settings.max_retries

        self.graph = self._build_graph()

    # ========================================================
    # Build Graph
    # ========================================================

    def _build_graph(self):
        """Build the LangGraph state machine."""

        def process_message(state: AgentState) -> dict:
            """
            Process the request using the primary model.
            """

            try:
                response = self.primary_llm.invoke(
                    state["messages"]
                )

                return {
                    "messages": [response],
                    "error": None,
                    "model_used": "primary",
                }

            except Exception as e:
                return {
                    "error": str(e),
                    "retry_count": state["retry_count"] + 1,
                    "model_used": "",
                }

        def try_fallback(state: AgentState) -> dict:
            """
            Try the fallback model.
            """

            try:
                response = self.fallback_llm.invoke(
                    state["messages"]
                )

                return {
                    "messages": [response],
                    "error": None,
                    "model_used": "fallback",
                }

            except Exception as e:
                return {
                    "error": str(e),
                    "retry_count": state["retry_count"] + 1,
                    "model_used": "",
                }

        def handle_error(state: AgentState) -> dict:
            """
            Return a graceful user-facing error.
            """

            response = AIMessage(
                content=(
                    "I'm sorry, I'm having trouble processing "
                    "your request right now. Please try again "
                    "in a moment."
                )
            )

            return {
                "messages": [response],
                "model_used": "error_handler",
            }

        def route_after_process(state: AgentState) -> str:
            """
            Decide what to do after the primary model.
            """

            if state.get("error") is None:
                return "done"

            if state["retry_count"] <= self.max_retries:
                return "fallback"

            return "error"

        def route_after_fallback(state: AgentState) -> str:
            """
            Decide what to do after fallback.
            """

            if state.get("error") is None:
                return "done"

            return "error"

        # ----------------------------------------------------
        # Build Graph
        # ----------------------------------------------------

        graph = StateGraph(AgentState)

        graph.add_node(
            "process",
            process_message,
        )

        graph.add_node(
            "fallback",
            try_fallback,
        )

        graph.add_node(
            "error",
            handle_error,
        )

        # START -> primary model
        graph.add_edge(
            START,
            "process",
        )

        # Primary routing
        graph.add_conditional_edges(
            "process",
            route_after_process,
            {
                "done": END,
                "fallback": "fallback",
                "error": "error",
            },
        )

        # Fallback routing
        graph.add_conditional_edges(
            "fallback",
            route_after_fallback,
            {
                "done": END,
                "error": "error",
            },
        )

        # Error handler -> END
        graph.add_edge(
            "error",
            END,
        )

        return graph.compile()

    # ========================================================
    # Public API
    # ========================================================

    @traceable(name="production_agent")
    def invoke(self, message: str) -> dict:
        """
        Invoke the agent with a user message.

        Returns:
            {
                "response": str,
                "model_used": str,
                "error": str | None
            }
        """

        initial_state: AgentState = {
            "messages": [
                HumanMessage(
                    content=message
                )
            ],
            "error": None,
            "retry_count": 0,
            "model_used": "",
        }

        result = self.graph.invoke(
            initial_state
        )

        return {
            "response": result["messages"][-1].content,
            "model_used": result.get(
                "model_used",
                "unknown"
            ),
            "error": result.get("error"),
        }


# ============================================================
# Standalone Test
# ============================================================

def test_graph_agent():
    """
    Test the ProductionAgent with several queries.
    """

    agent = ProductionAgent()

    print("=" * 70)
    print("PRODUCTION AGENT - STANDALONE TEST")
    print("=" * 70)

    queries = [
        "What is LangGraph in one sentence?",
        "What is 2 + 2?",
        "Explain the difference between RAG and fine-tuning in 2 sentences.",
    ]

    for index, query in enumerate(
        queries,
        start=1,
    ):
        print()
        print("-" * 70)
        print(f"TEST {index}")
        print("-" * 70)

        print(f"Query: {query}")

        try:
            result = agent.invoke(query)

            print(
                f"Model used: "
                f"{result['model_used']}"
            )

            print(
                f"Response: "
                f"{result['response']}"
            )

            if result["error"]:
                print(
                    f"Error: "
                    f"{result['error']}"
                )
            else:
                print("Status: SUCCESS")

        except Exception as e:
            print("Status: FAILED")
            print(f"Exception: {e}")

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_graph_agent()