
import os
import json
from typing import TypedDict, Optional, Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END

load_dotenv()


class ResearchState(TypedDict, total=False):
    topic: str
    research_data: str
    analysis: str
    report: str
    review: dict
    final_output: dict
    revision_count: int
    max_revisions: int
    status: str


# Initialize models and tools
def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing.")
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )


def get_search_tool():
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY is missing.")
    return TavilySearch(
        max_results=5,
        topic="general"
    )


# Agent 1: Research
def research_agent(state: ResearchState):
    llm = get_llm()
    search = get_search_tool()

    topic = state["topic"]

    results = search.invoke({"query": topic})

    prompt = f"""
    You are an expert research agent.

    Research topic: {topic}

    Web search results:
    {results}

    Prepare a structured research summary including:
    1. Introduction
    2. Important facts
    3. Current developments
    4. Statistics and evidence
    5. Sources and references

    Do not invent facts or sources.
    """

    response = llm.invoke(prompt)

    return {
        "research_data": response.content
    }


# Agent 2: Analysis
def analysis_agent(state: ResearchState):
    llm = get_llm()

    prompt = f"""
    You are an expert research analyst.

    Topic: {state['topic']}

    Research:
    {state['research_data']}

    Analyze the research and identify:
    1. Major insights
    2. Trends
    3. Opportunities
    4. Challenges
    5. Risks
    6. Impact
    7. Recommendations
    """

    response = llm.invoke(prompt)

    return {
        "analysis": response.content
    }


# Agent 3: Writer
def writer_agent(state: ResearchState):
    llm = get_llm()

    prompt = f"""
    You are a professional research report writer.

    Topic: {state['topic']}

    Research:
    {state['research_data']}

    Analysis:
    {state['analysis']}

    Write a detailed professional research report in Markdown.

    Include:
    - Title
    - Executive summary
    - Introduction
    - Research findings
    - Detailed analysis
    - Opportunities and challenges
    - Recommendations
    - Conclusion
    - References

    Clearly distinguish verified information from analysis.
    """

    response = llm.invoke(prompt)

    return {
        "report": response.content
    }


# Agent 4: Reviewer
def reviewer_agent(state: ResearchState):
    llm = get_llm()

    prompt = f"""
    Review the following research report.

    Topic: {state['topic']}

    Report:
    {state['report']}

    Evaluate:
    - Accuracy
    - Completeness
    - Structure
    - Clarity
    - Evidence
    - Professional quality

    Return ONLY valid JSON:
    {{
        "approved": true,
        "score": 9,
        "feedback": "Review feedback",
        "issues": []
    }}

    Set approved to false if important improvements are required.
    """

    response = llm.invoke(prompt)

    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        review = json.loads(content)
    except (json.JSONDecodeError, IndexError):
        review = {
            "approved": False,
            "score": 0,
            "feedback": "Reviewer returned invalid JSON.",
            "issues": ["Unable to parse reviewer response"]
        }

    return {"review": review}


# Agent 5: Revision
def revision_agent(state: ResearchState):
    llm = get_llm()

    prompt = f"""
    You are a professional report revision agent.

    Original report:
    {state['report']}

    Reviewer feedback:
    {json.dumps(state['review'], indent=2)}

    Improve the report according to the feedback.
    Return the complete revised Markdown report only.
    """

    response = llm.invoke(prompt)

    return {
        "report": response.content,
        "revision_count": state.get("revision_count", 0) + 1
    }


# Routing after review
def review_router(state: ResearchState):
    review = state.get("review", {})
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 2)

    if review.get("approved", False):
        return "approved"

    if revision_count < max_revisions:
        return "revise"

    return "max_revisions_reached"


def finalize_agent(state: ResearchState):
    return {
        "status": "approved",
        "final_output": {
            "topic": state["topic"],
            "report": state["report"],
            "review": state.get("review", {}),
            "revision_count": state.get("revision_count", 0)
        }
    }


def finalize_unapproved_agent(state: ResearchState):
    return {
        "status": "completed_with_review_issues",
        "final_output": {
            "topic": state["topic"],
            "report": state["report"],
            "review": state.get("review", {}),
            "revision_count": state.get("revision_count", 0)
        }
    }


# Build LangGraph workflow
def build_workflow():
    graph = StateGraph(ResearchState)

    graph.add_node("research", research_agent)
    graph.add_node("analysis", analysis_agent)
    graph.add_node("writer", writer_agent)
    graph.add_node("reviewer", reviewer_agent)
    graph.add_node("revision", revision_agent)
    graph.add_node("approved", finalize_agent)
    graph.add_node("max_revisions_reached", finalize_unapproved_agent)

    graph.set_entry_point("research")

    graph.add_edge("research", "analysis")
    graph.add_edge("analysis", "writer")
    graph.add_edge("writer", "reviewer")

    graph.add_conditional_edges(
        "reviewer",
        review_router,
        {
            "approved": "approved",
            "revise": "revision",
            "max_revisions_reached": "max_revisions_reached"
        }
    )

    graph.add_edge("revision", "reviewer")
    graph.add_edge("approved", END)
    graph.add_edge("max_revisions_reached", END)

    return graph.compile()


# Main function
def run_research_assistant(
    topic: str,
    max_revisions: int = 2
) -> dict[str, Any]:

    if not topic or not topic.strip():
        raise ValueError("Please provide a research topic.")

    app = build_workflow()

    initial_state = {
        "topic": topic.strip(),
        "revision_count": 0,
        "max_revisions": max_revisions
    }

    return app.invoke(initial_state)
