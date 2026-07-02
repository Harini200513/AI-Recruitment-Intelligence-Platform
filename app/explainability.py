import logging
from typing import Dict, List, Any, Tuple
import pandas as pd
from app.config import (
    STRONG_HIRE_THRESHOLD,
    GOOD_FIT_THRESHOLD,
    CONSIDER_THRESHOLD,
    NEEDS_IMPROVEMENT_THRESHOLD,
    EDUCATION_MAPPING
)
from app.preprocessing import normalize_education_level

# Setup Logging
logger = logging.getLogger(__name__)

def compute_confidence_score(candidate_row: pd.Series, scores_dict: Dict[str, float]) -> Tuple[float, str]:
    """
    Computes an AI Confidence Score (percentage) based on:
    1. Data completeness (are Projects, Certifications, Experience details present?)
    2. Model alignment (agreement between semantic relevance score and structured keyword skills score).
    """
    base_confidence = 80.0 # Base confidence
    
    # 1. Check data completeness (max +10%)
    completeness_score = 0.0
    fields_to_check = ["Experience_Details", "Projects", "Certifications", "Behavioral_Signals", "Platform_Activity"]
    for field in fields_to_check:
        if field in candidate_row and not pd.isna(candidate_row[field]) and str(candidate_row[field]).strip() != "":
            completeness_score += 2.0
            
    # 2. Check model consensus alignment (max +10%)
    # If semantic relevance and technical skills match are close, confidence is higher
    semantic = scores_dict.get("Semantic_Score", 0.0)
    skills = scores_dict.get("Skills_Score", 0.0)
    variance = abs(semantic - skills)
    
    consensus_bonus = max(0.0, 10.0 - (variance * 20.0)) # Closer scores = higher bonus
    
    final_confidence = base_confidence + completeness_score + consensus_bonus
    final_confidence = min(max(final_confidence, 50.0), 99.0) # Clamp between 50% and 99%
    
    # Generate textual explanation
    reasons = []
    if completeness_score >= 8.0:
        reasons.append("high profile data completeness")
    else:
        reasons.append("moderate profile details")
        
    if variance < 0.15:
        reasons.append("strong alignment between semantic profile and technical keywords")
    else:
        reasons.append("slight variance between general resume text and specific keywords")
        
    explanation = f"Confidence score set at {final_confidence:.0f}% due to " + " and ".join(reasons) + "."
    
    return float(round(final_confidence, 1)), explanation

def generate_decision_report(
    candidate_row: pd.Series,
    scores_dict: Dict[str, float],
    required_skills: List[str],
    required_exp: float,
    required_edu: str
) -> Dict[str, Any]:
    """
    Generates a full explainable AI Decision Report for a candidate, including recommendations,
    hiring summaries, key strengths, potential risks, and suggested next hiring step.
    """
    final_score = scores_dict["Final_Score"]
    cand_name = candidate_row.get("Candidate_Name", "Candidate")
    
    # 1. Recommendation Level & Suggested Next Step
    if final_score >= STRONG_HIRE_THRESHOLD:
        rec_rating = "★★★★★ Strong Hire"
        next_step = "Proceed to Technical Interview"
    elif final_score >= GOOD_FIT_THRESHOLD:
        rec_rating = "★★★★ Good Fit"
        next_step = "Proceed to Technical Interview"
    elif final_score >= CONSIDER_THRESHOLD:
        rec_rating = "★★★ Consider"
        next_step = "Proceed to HR Screening Round"
    elif final_score >= NEEDS_IMPROVEMENT_THRESHOLD:
        rec_rating = "★★ Needs Improvement"
        next_step = "Keep in Talent Pool for Future Roles"
    else:
        rec_rating = "★ Not Recommended"
        next_step = "Do Not Proceed"

    # 2. Confidence calculation
    confidence, confidence_reason = compute_confidence_score(candidate_row, scores_dict)

    # 3. Identify Matched & Missing Skills
    candidate_skills = [s.strip().lower() for s in str(candidate_row.get("Skills", "")).replace(";", ",").split(",") if s.strip()]
    req_skills_lower = [s.strip().lower() for s in required_skills]
    
    matched_skills = [s for s in req_skills_lower if s in candidate_skills]
    missing_skills = [s for s in req_skills_lower if s not in candidate_skills]
    
    # 4. Experience & Education Assessments
    try:
        cand_exp = float(candidate_row.get("Experience_Years", 0.0))
    except ValueError:
        cand_exp = 0.0
    exp_diff = cand_exp - required_exp
    if exp_diff >= 0:
        exp_assessment = f"Exceeds requirement by {exp_diff:.1f} years (Has {cand_exp:.1f} yrs, Required {required_exp:.1f} yrs)"
    else:
        exp_assessment = f"Deficit of {abs(exp_diff):.1f} years (Has {cand_exp:.1f} yrs, Required {required_exp:.1f} yrs)"

    edu_assessment = f"Highest degree: '{candidate_row.get('Education', 'Not Provided')}' (Required level: '{required_edu}')"

    # 5. Extract Strengths & Risks
    strengths = []
    risks = []
    
    # Analyze experience
    if exp_diff >= 2:
        strengths.append(f"Substantial experience surplus (+{exp_diff:.1f} years above requirement).")
    elif exp_diff >= 0:
        strengths.append("Meets required years of professional experience.")
    else:
        risks.append(f"Experience deficit (-{abs(exp_diff):.1f} years below requirement).")

    # Analyze skills
    if len(matched_skills) == len(required_skills) and required_skills:
        strengths.append("Possesses 100% of the required technical skills.")
    elif len(matched_skills) >= len(required_skills) * 0.7 and required_skills:
        strengths.append(f"Strong skill alignment, matching {len(matched_skills)} out of {len(required_skills)} key technologies.")
    elif len(matched_skills) < len(required_skills) * 0.4 and required_skills:
        risks.append(f"Significant skill gap. Missing {len(missing_skills)} required technical skills.")
        
    # Analyze semantic relevance
    if scores_dict["Semantic_Score"] >= 0.80:
        strengths.append("Exceptional semantic alignment with the job context and role responsibilities.")
    elif scores_dict["Semantic_Score"] < 0.60:
        risks.append("Low contextual/semantic relevance to the job description details.")

    # Analyze education
    cand_edu_val = EDUCATION_MAPPING.get(normalize_education_level(candidate_row.get("Education", "")), 0.0)
    req_edu_val = EDUCATION_MAPPING.get(normalize_education_level(required_edu), 0.0)
    if cand_edu_val >= req_edu_val:
        strengths.append("Meets or exceeds the educational level requirements.")
    else:
        risks.append("Education level is below the preferred minimum degree requirement.")

    # Fallbacks for empty lists
    if not strengths:
        strengths.append("Technical background shows basic competencies.")
    if not risks:
        risks.append("No major risks identified. Credentials meet standard benchmarks.")

    # 6. Natural Language Hiring Summary (2-4 sentences)
    summary_parts = []
    if final_score >= GOOD_FIT_THRESHOLD:
        summary_parts.append(f"{cand_name} is a highly qualified candidate who demonstrates a strong overall match score of {final_score*100:.1f}%.")
        if matched_skills:
            summary_parts.append(f"Their core strengths include solid hands-on experience in {', '.join(matched_skills[:3])}.")
        summary_parts.append(f"With {cand_exp:.1f} years of experience and a clean educational alignment, they are well-suited for the technical demands of this role.")
    else:
        summary_parts.append(f"{cand_name} is a potential candidate with an overall match score of {final_score*100:.1f}%.")
        if missing_skills:
            summary_parts.append(f"However, key skill gaps in {', '.join(missing_skills[:3])} represent a potential onboarding risk.")
        summary_parts.append("They may require training and upskilling support if selected to proceed.")
        
    hiring_summary = " ".join(summary_parts)

    # 7. Ranking Justification
    components_sorted = sorted(
        [
            ("Semantic Relevance", scores_dict["Semantic_Score"]),
            ("Technical Skills Match", scores_dict["Skills_Score"]),
            ("Experience Match", scores_dict["Experience_Score"]),
            ("Education Match", scores_dict["Education_Score"])
        ],
        key=lambda x: x[1],
        reverse=True
    )
    
    justification = (
        f"This candidate is ranked at this level primarily due to high scores in {components_sorted[0][0]} "
        f"({components_sorted[0][1]*100:.0f}%) and {components_sorted[1][0]} ({components_sorted[1][1]*100:.0f}%). "
        f"Conversely, their score was limited by lower scores in {components_sorted[-1][0]} ({components_sorted[-1][1]*100:.0f}%)."
    )

    return {
        "Overall_Match_Score": float(final_score),
        "Recommendation": rec_rating,
        "Confidence_Score": confidence,
        "Confidence_Reason": confidence_reason,
        "Hiring_Summary": hiring_summary,
        "Matched_Skills": matched_skills,
        "Missing_Skills": missing_skills,
        "Experience_Assessment": exp_assessment,
        "Education_Assessment": edu_assessment,
        "Key_Strengths": strengths,
        "Potential_Risks": risks,
        "Ranking_Justification": justification,
        "Suggested_Next_Step": next_step
    }

def generate_comparison_report(
    target_cand: Dict[str, Any],
    top_cand: Dict[str, Any],
    target_row: pd.Series,
    top_row: pd.Series
) -> Dict[str, Any]:
    """
    Compares the selected target candidate against the top-ranked candidate (Rank 1).
    Highlights missing requirements and scoring variances to increase comparison transparency.
    """
    target_skills = [s.strip().lower() for s in str(target_row.get("Skills", "")).replace(";", ",").split(",") if s.strip()]
    top_skills = [s.strip().lower() for s in str(top_row.get("Skills", "")).replace(";", ",").split(",") if s.strip()]

    # Skill differences
    missing_compared_to_top = [s for s in top_skills if s not in target_skills]
    
    # Experience difference
    try:
        t_exp = float(target_row.get("Experience_Years", 0.0))
        top_exp = float(top_row.get("Experience_Years", 0.0))
    except ValueError:
        t_exp = 0.0
        top_exp = 0.0
    exp_diff = top_exp - t_exp

    # Score comparison
    score_diff = top_cand["Overall_Match_Score"] - target_cand["Overall_Match_Score"]

    comparison_narrative = (
        f"Compared to the top-ranked candidate ({top_row.get('Candidate_Name', 'Rank 1')}), "
        f"the selected candidate has a score variance of -{score_diff*100:.1f}%. "
    )
    
    if exp_diff > 0:
        comparison_narrative += f"The top candidate possesses {exp_diff:.1f} more years of industry experience. "
    if missing_compared_to_top:
        comparison_narrative += f"Additionally, the top candidate has skills in {', '.join(missing_compared_to_top[:3])} which this candidate lacks."

    return {
        "score_difference": float(score_diff),
        "experience_difference_years": float(exp_diff),
        "skills_missing_vs_top": missing_compared_to_top,
        "comparison_narrative": comparison_narrative
    }

# ==========================================
# Recruiter Session Notes Manager
# ==========================================
class RecruiterNotesManager:
    """
    Manages in-memory recruiter session notes. 
    Can be easily connected to Streamlit session_state for persistence.
    """
    def __init__(self):
        self.notes: Dict[str, str] = {}

    def add_note(self, candidate_id: str, note_text: str) -> None:
        """Adds or updates notes for a specific candidate."""
        self.notes[candidate_id] = note_text.strip()

    def get_note(self, candidate_id: str) -> str:
        """Retrieves note for a candidate, returning empty string if none exists."""
        return self.notes.get(candidate_id, "")

    def get_all_notes(self) -> Dict[str, str]:
        """Returns all session notes."""
        return self.notes

    def clear(self) -> None:
        """Clears notes."""
        self.notes = {}
