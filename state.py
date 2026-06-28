from typing import Annotated, Dict, List, Any, TypedDict
from langgraph.graph.message import add_messages

class FacultyState(TypedDict):
    
    messages: Annotated[list, add_messages]
    raw_resume_text: str        
    job_requirements: Dict[str, Any]  
    
    
    candidate_id: str         
    db_sync_status: str       

    faculty_profile: Dict[str, Any]
    
   
    eligibility_status: str     
    eligibility_reason: str    
    
    
    research_score: float
    research_breakdown: Dict[str, float]
    
    
    teaching_score: float
    final_evaluation_report: str
    
    
    next_node: str              