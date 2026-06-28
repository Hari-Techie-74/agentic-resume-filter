import os
import json
from supabase import create_client, Client
from langgraph.graph import StateGraph, END
from state import FacultyState
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings


supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None


llm = ChatGroq(
    groq_api_key=os.environ.get("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0,
    response_format={"type": "json_object"}
)

PIPELINE_SHARED_DATA = {
    "candidate_profile": {},
    "is_eligible": False,
    "final_report": "Processing completed.",
    "candidate_id": None
}


# AGENT 1: APPLICATION SCREENING AGENT
def application_screening_agent(state: FacultyState):
    print("\n📥 [NODE 1] Executing Application Screening Agent...")
    
   
    global PIPELINE_SHARED_DATA
    PIPELINE_SHARED_DATA.clear()
    PIPELINE_SHARED_DATA.update({
        "candidate_profile": {},
        "is_eligible": False,
        "final_report": "Processing completed.",
        "candidate_id": None
    })
    
    text = state.get("raw_resume_text", "")
    
    prompt = f"""
    You are a strict data parsing engine. Analyze the following resume text and extract core structured insights.
    Extract the actual exact numbers present in the text for experience and research metrics. If a metric is missing, default it to 0.

    Resume Text:
    \"\"\"{text}\"\"\"

    Output Format Schema:
    {{
        "personal": {{"name": "String", "email": "String", "contact": "String"}},
        "academic": {{"degrees": ["String"], "universities": ["String"], "specialization": "String"}},
        "experience": {{"teaching_years": Integer, "industry_years": Integer, "admin_years": Integer}},
        "research": {{"publications_count": Integer, "scopus_count": Integer, "sci_count": Integer, "citations_count": Integer, "h_index": Integer}}
    }}
    
    You must return the final output strictly as a valid json object matching the schema above. Do not include any markdown styling.
    """
    
    response = llm.invoke(prompt)
    response_text = response.content if hasattr(response, "content") else str(response)
    profile_data = json.loads(response_text.strip())
    
    print("🧠 Generating local semantic vector embedding via HuggingFace Nomic...")
    embeddings_engine = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs={'trust_remote_code': True}
    )
    resume_vector = embeddings_engine.embed_query(text)
    
    candidate_id = None
    if supabase:
        try:
            db_payload = {
                "name": profile_data.get("personal", {}).get("name"),
                "email": profile_data.get("personal", {}).get("email"),
                "contact": profile_data.get("personal", {}).get("contact"),
                "degrees": profile_data.get("academic", {}).get("degrees"),
                "universities": profile_data.get("academic", {}).get("universities"),
                "specialization": profile_data.get("academic", {}).get("specialization"),
                "teaching_experience_years": profile_data.get("experience", {}).get("teaching_years", 0),
                "industry_experience_years": profile_data.get("experience", {}).get("industry_years", 0),
                "administrative_experience_years": profile_data.get("experience", {}).get("admin_years", 0),
                "publications_count": profile_data.get("research", {}).get("publications_count", 0),
                "scopus_indexed_count": profile_data.get("research", {}).get("scopus_count", 0),
                "sci_indexed_count": profile_data.get("research", {}).get("sci_count", 0),
                "citations_count": profile_data.get("research", {}).get("citations_count", 0),
                "h_index": profile_data.get("research", {}).get("h_index", 0),
                "resume_embedding": resume_vector,
                "final_evaluation_report": None
            }
            inserted_row = supabase.table("faculty_profiles").insert(db_payload).execute()
            if inserted_row.data:
                candidate_id = inserted_row.data[0]["id"]
        except Exception as e:
            print(f"❌ Supabase Writing Failure: {e}")
            
    # Cache to shared container
    PIPELINE_SHARED_DATA["candidate_profile"] = profile_data
    PIPELINE_SHARED_DATA["candidate_id"] = candidate_id
    
    return {"current_agent": "MinimumCriteriaVerificationAgent"}


# AGENT 2: MINIMUM CRITERIA VERIFICATION AGENT
def minimum_criteria_verification_agent(state: FacultyState):
    print("\n🛡️ [NODE 2] Executing Minimum Criteria Verification Agent...")
    raw_text_content = state.get("raw_resume_text", "").lower()
    profile = PIPELINE_SHARED_DATA["candidate_profile"]
    
    academic_data = profile.get("academic", {}) if isinstance(profile, dict) else {}
    degrees = [str(d).lower() for d in academic_data.get("degrees", [])]
    
    has_phd = any("ph.d" in d or "phd" in d or "doctor" in d for d in degrees) or "ph.d" in raw_text_content or "phd" in raw_text_content
    meets_specialization = "computer science" in raw_text_content or "cse" in raw_text_content or "computing" in raw_text_content
    
    experience_data = profile.get("experience", {}) if isinstance(profile, dict) else {}
    structured_teaching_years = experience_data.get("teaching_years", 0) or 0
    has_years_mentioned = any(f"{i} years" in raw_text_content for i in range(3, 25)) or "professor" in raw_text_content or "experience" in raw_text_content
    meets_experience = (structured_teaching_years >= 3) or has_years_mentioned
    
    eligible = has_phd and meets_experience and meets_specialization
    print(f"📈 Compliance Checks -> PhD: {has_phd} | Exp: {meets_experience} | Domain Match: {meets_specialization}")
    
    PIPELINE_SHARED_DATA["is_eligible"] = eligible
    return {"current_agent": "ResearchEvaluationAgent"}


# AGENT 3: RESEARCH EVALUATION AGENT
def research_evaluation_agent(state: FacultyState):
    print("\n🔬 [NODE 3] Executing Research Evaluation Agent...")
    profile = PIPELINE_SHARED_DATA["candidate_profile"]
    research_data = profile.get("research", {}) if isinstance(profile, dict) else {}
    
    pub_count = research_data.get("publications_count", 0) or 0
    scopus = research_data.get("scopus_count", 0) or 0
    sci = research_data.get("sci_count", 0) or 0
    citations = research_data.get("citations_count", 0) or 0
    h_index = research_data.get("h_index", 0) or 0
    
    score = (pub_count * 2) + (scopus * 5) + (sci * 8) + (citations * 0.5) + (h_index * 10)
    
    PIPELINE_SHARED_DATA["research_score"] = score
    PIPELINE_SHARED_DATA["research_summary"] = f"Publications: {pub_count}, Scopus: {scopus}, SCI: {sci}"
    return {"current_agent": "InterviewCoordinationAgent"}


# AGENT 4: INTERVIEW COORDINATION AGENT (SMART STRING CLEANER)
def interview_coordination_agent(state: FacultyState):
    print("\n🤝 [NODE 4] Executing Interview Coordination Agent...")
    profile = PIPELINE_SHARED_DATA["candidate_profile"]
    eligible = PIPELINE_SHARED_DATA["is_eligible"]
    candidate_id = PIPELINE_SHARED_DATA["candidate_id"]
    
    status_string = "RECOMMENDED FOR SHORTLIST INTERVIEW" if eligible else "REJECTED - DOES NOT MEET BASIC MANDATES"
    
    report_prompt = f"""
    Compile an executive system hiring review report summarizing the following applicant profile:
    Name: {profile.get("personal", {}).get("name", "Unknown")}
    Status Assessment: {status_string}
    Research Score: {PIPELINE_SHARED_DATA.get("research_score", 0)}
    
    Return a JSON object containing a clean summary paragraph. Keep it strictly inside the format below:
    {{
        "executive_summary": "Write a clear, raw human-readable paragraph outlining their core match and qualification results."
    }}
    """
    
    response = llm.invoke(report_prompt)
    response_text = response.content if hasattr(response, "content") else str(response)
    
    summary_text = ""
    try:
        parsed_json = json.loads(response_text.strip())
        
       
        if "executive_summary" in parsed_json:
            summary_text = parsed_json["executive_summary"]
        else:
            analysis_data = parsed_json.get("analysis", "")
            if isinstance(analysis_data, list):
                summary_text = " ".join([str(item) for item in analysis_data])
            elif analysis_data:
                summary_text = str(analysis_data)
            else:
                summary_text = " | ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in parsed_json.items() if not isinstance(v, (dict, list))])
    except Exception as e:
        print(f"Fallback direct content parsing active: {e}")
        summary_text = response_text
        
    summary_text = str(summary_text).replace("[", "").replace("]", "").replace("'", "").strip()
    
    if supabase and candidate_id:
        try:
            supabase.table("faculty_profiles").update({
                "final_evaluation_report": summary_text
            }).eq("id", candidate_id).execute()
        except Exception as e:
            print(f"❌ Node 4 Supabase Update Failed: {e}")
            
    PIPELINE_SHARED_DATA["final_report"] = summary_text
    return {"current_agent": END}


# STATE GRAPH COMPILATION ARCHITECTURE
workflow = StateGraph(FacultyState)
workflow.add_node("ScreeningAgent", application_screening_agent)
workflow.add_node("CriteriaAgent", minimum_criteria_verification_agent)
workflow.add_node("ResearchAgent", research_evaluation_agent)
workflow.add_node("CoordinatorAgent", interview_coordination_agent)

workflow.set_entry_point("ScreeningAgent")
workflow.add_edge("ScreeningAgent", "CriteriaAgent")
workflow.add_edge("CriteriaAgent", "ResearchAgent")
workflow.add_edge("ResearchAgent", "CoordinatorAgent")
workflow.add_edge("CoordinatorAgent", END)

resume_filter_app = workflow.compile()