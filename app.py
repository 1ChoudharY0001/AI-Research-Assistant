
import json
import streamlit as st

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
    placeholder="Example: Impact of Artificial Intelligence on Indian Education",
    height=120
)

if st.button("🚀 Generate Research Report", type="primary"):
    if not topic.strip():
        st.warning("Please enter a research topic.")
    else:
        try:
            with st.spinner("AI agents are researching and preparing your report..."):
                result = run_research_assistant(
                    topic=topic,
                    max_revisions=max_revisions
                )

            final_output = result.get("final_output", {})
            report = final_output.get("report", "")
            review = final_output.get("review", {})

            st.success("Research report generated!")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Review Score",
                f"{review.get('score', 'N/A')}/10"
            )

            col2.metric(
                "Revisions",
                final_output.get("revision_count", 0)
            )

            col3.metric(
                "Status",
                result.get("status", "Unknown")
            )

            st.divider()

            st.subheader("📄 Final Research Report")
            st.markdown(report)

            st.download_button(
                "⬇️ Download Markdown Report",
                data=report,
                file_name="research_report.md",
                mime="text/markdown"
            )

            st.download_button(
                "⬇️ Download JSON Summary",
                data=json.dumps(result, indent=2, default=str),
                file_name="research_summary.json",
                mime="application/json"
            )

            with st.expander("Reviewer Feedback"):
                st.json(review)

        except Exception as error:
            st.error(f"An error occurred: {error}")
            st.exception(error)
