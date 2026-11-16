import os
import json
import streamlit as st

# Load API keys from Streamlit Cloud Secrets
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

    if "TAVILY_API_KEY" in st.secrets:
        os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
except Exception:
    pass

from agents import run_research_assistant

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 AI Research Assistant")
st.caption("Multi-agent research and report generation powered by LangGraph")

st.markdown("""
Generate structured research reports using five specialized AI agents:
Research, Analysis, Writer, Reviewer, and Revision.
""")

with st.sidebar:
    st.header("Settings")
    max_revisions = st.slider(
        "Maximum revisions",
        min_value=0,
        max_value=5,
        value=2
    )
    st.divider()
    st.info("Powered by Groq, Tavily and LangGraph")

topic = st.text_area(
    "Enter your research topic",
    placeholder="Example: Impact of artificial intelligence on education"
)

if st.button("Generate Research Report", type="primary"):
    if not topic.strip():
        st.warning("Please enter a research topic.")
    else:
        with st.spinner("AI agents are researching and preparing your report..."):
            try:
                result = run_research_assistant(
                    topic.strip(),
                    max_revisions=max_revisions
                )

                st.success("Research process completed!")

                final_output = result.get("final_output", {})
                review = result.get("review", {})

                st.subheader("Research Report")
                report_text = (
                    final_output.get("report")
                    or result.get("report", "No report generated.")
                )
                st.markdown(report_text)

                st.subheader("Review")
                st.write("Status:", result.get("status", "Unknown"))
                st.write("Score:", review.get("score", "N/A"))
                st.write("Revisions:", result.get("revision_count", 0))

                with st.expander("Reviewer feedback"):
                    st.write(review)

                st.download_button(
                    "Download Report",
                    data=report_text,
                    file_name="research_report.md",
                    mime="text/markdown"
                )

                st.download_button(
                    "Download Full Results (JSON)",
                    data=json.dumps(result, indent=2, default=str),
                    file_name="research_results.json",
                    mime="application/json"
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")
