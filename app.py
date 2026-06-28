import os
from dotenv import load_dotenv
load_dotenv()  

import streamlit as st
import docx
from pypdf import PdfReader
from graph import resume_filter_app
from langchain_groq import ChatGroq


llm = ChatGroq(
    groq_api_key=os.environ.get("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0,
    response_format={"type": "json_object"}
)

st.set_page_config(page_title="Institutional AI Faculty Filter", layout="wide")

st.title("🎓 Agentic AI Faculty Hiring & Screening System")
st.caption("Deterministic Multi-Agent Processing Framework with Multiformat Document Ingestion")

job_criteria = {
    "min_teaching_years": 3,
    "required_specialization": "Computer Science"
}

st.sidebar.header("System Hiring Parameters")
st.sidebar.markdown(f"""
- **Required Degree:** Ph.D.
- **Min Teaching Experience:** {job_criteria['min_teaching_years']} Years
- **Target Specialization:** {job_criteria['required_specialization']}
""")

def extract_text_from_pdf(file_bytes):
    reader = PdfReader(file_bytes)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_text_from_docx(file_bytes):
    doc = docx.Document(file_bytes)
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    return "\n".join(text)

tab1, tab2 = st.tabs(["📥 Upload & Process Applications", "🔍 Semantic Talent Search"])



# TAB 1: DATA PROCESSING INGESTION FRAMEWORK
with tab1:
    st.subheader("Ingest Faculty Resumes (.pdf or .docx)")
    uploaded_file = st.file_uploader("Drop a candidate document file here:", type=["pdf", "docx"])

    if uploaded_file is not None:
        file_type = uploaded_file.name.split(".")[-1]
        raw_text = ""
        
        if file_type == "pdf":
            raw_text = extract_text_from_pdf(uploaded_file)
        elif file_type == "docx":
            raw_text = extract_text_from_docx(uploaded_file)
            
        if raw_text.strip():
            st.success(f"Successfully read raw characters from {uploaded_file.name}!")
            with st.expander("Preview Raw Extracted Text Segment"):
                st.text(raw_text[:1000] + "...")
                
            if st.button("Trigger Multi-Agent System Audit", type="primary"):
                initial_state = {
                    "raw_resume_text": raw_text,
                    "hiring_criteria": job_criteria,
                    "candidate_profile": {},
                    "candidate_id": None,
                    "is_eligible": False,
                    "research_evaluation": {},
                    "final_report": "",
                    "current_agent": "ScreeningAgent"
                }
                
                with st.spinner("Executing Deterministic Agent Graph Nodes Across Cloud Clusters..."):
                    try:
                        
                        resume_filter_app.invoke(initial_state)
                        
                        
                        from graph import PIPELINE_SHARED_DATA
                        
                        st.balloons()
                        st.success("🎉 Comprehensive Multi-Agent Pipeline Completed Successfully!")
                        
                        st.markdown("### 🏆 System Audit Evaluation Report")
                        
                        
                        is_eligible = PIPELINE_SHARED_DATA.get("is_eligible", False)
                        report_details = PIPELINE_SHARED_DATA.get("final_report", "Processed.")
                        
                        if is_eligible:
                            st.success("✅ **Hiring Assessment:** CANDIDATE MEETS MINIMUM MANDATES")
                        else:
                            st.error("❌ **Hiring Assessment:** REJECTED - DOES NOT MEET BASIC MANDATES")
                            
                        st.info(f"**Executive System Analysis:** {report_details}")
                        
                    except Exception as e:
                        st.error(f"Graph Pipeline Runtime Error: {e}")
        else:
            st.error("Could not parse any clear string characters from the uploaded file structure.")


with tab2:
    st.header("Natural Language Vector Query Portal")
    search_query = st.text_input("Type an institutional talent profile search query:", 
                                 placeholder="e.g., Show me a faculty profile who has authored textbooks and visited Argentina")

    if st.button("Query Local Semantic Storage Indexes"):
        if search_query.strip():
            with st.spinner("Analyzing semantic vectors across cloud records..."):
                try:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                    embeddings_engine = HuggingFaceEmbeddings(
                        model_name="nomic-ai/nomic-embed-text-v1",
                        model_kwargs={'trust_remote_code': True}
                    )
                    
                    query_vector = embeddings_engine.embed_query(search_query)
                    
                    from graph import supabase
                    if supabase:
                        search_results = supabase.rpc("match_resume_embeddings", {
                            "query_embedding": query_vector,
                            "match_threshold": 0.10, 
                            "match_count": 10
                        }).execute()
                        
                        if search_results.data:
                            st.success(f"🎯 Discovered {len(search_results.data)} matching candidate matches:")
                            for row in search_results.data:
                                confidence = row.get("similarity", 0) * 100
                                with st.expander(f"👤 {row['name']} - Match Confidence: {confidence:.1f}%"):
                                    st.write(f"**Specialization:** {row.get('specialization', 'N/A')}")
                                    st.write(f"**Executive System Report:** {row.get('final_evaluation_report', 'No report compiled.')}")
                        else:
                            st.info("No candidates matched that specific semantic query signature.")
                    else:
                        st.error("Supabase client context could not be loaded cleanly.")
                except Exception as e:
                    st.error(f"Execution Failure: {e}")