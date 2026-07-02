import re
import logging
from typing import List, Dict, Any, Tuple
import pandas as pd
from app.config import EDUCATION_MAPPING

# Setup Logging
logger = logging.getLogger(__name__)

def clean_text(text: Any) -> str:
    """
    Cleans raw text by removing extra whitespaces, HTML/markdown formatting, 
    and normalizes spacing.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    # Remove HTML tags if any
    text = re.sub(r'<[^>]*>', ' ', text)
    # Replace non-breaking spaces and formatting characters
    text = text.replace('\xa0', ' ').replace('\r', ' ').replace('\n', ' ')
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def anonymize_text(text: str) -> str:
    """
    Scrubs obvious demographic signals from candidate profile descriptions 
    to mitigate unconscious bias (e.g., gender, nationality, age indicators).
    """
    # Scrub gender terms
    gender_patterns = [
        (r'\b(he/him|she/her|they/them)\b', '[pronouns]'),
        (r'\b(he|she|his|her|him|herself|himself)\b', 'the candidate'),
        (r'\b(male|female|man|woman|gentleman|lady)\b', 'individual')
    ]
    
    anonymized = text
    for pattern, replacement in gender_patterns:
        anonymized = re.sub(pattern, replacement, anonymized, flags=re.IGNORECASE)
        
    # Remove age indicators (e.g., "24-year-old developer")
    anonymized = re.sub(r'\b\d{1,2}-year-old\b', 'professional', anonymized, flags=re.IGNORECASE)
    
    return anonymized

def normalize_skills(skills_input: Any) -> List[str]:
    """
    Converts comma-separated skill list or string into normalized, deduplicated lowercase tokens.
    """
    if pd.isna(skills_input) or not skills_input:
        return []
    
    if isinstance(skills_input, str):
        # Split by comma or semicolon
        tokens = re.split(r'[,;]+', skills_input)
    elif isinstance(skills_input, list):
        tokens = skills_input
    else:
        tokens = [str(skills_input)]
        
    normalized = []
    for t in tokens:
        clean_token = t.strip().lower()
        if clean_token and clean_token not in normalized:
            normalized.append(clean_token)
            
    return normalized

def normalize_education_level(education_text: Any) -> str:
    """
    Maps raw education description to a standard standardized key in EDUCATION_MAPPING.
    """
    if pd.isna(education_text) or not isinstance(education_text, str):
        return "none"
        
    text_lower = education_text.lower()
    
    # Check hierarchy from top to bottom
    for edu_key in EDUCATION_MAPPING.keys():
        if edu_key == "none":
            continue
        # Use regex to find whole words or prefixes
        if re.search(r'\b' + re.escape(edu_key) + r'\b', text_lower) or edu_key in text_lower:
            return edu_key
            
    return "none"

def extract_experience_requirement(job_desc: str) -> float:
    """
    Attempts to extract required years of experience from the job description text using regex.
    Examples matched: "5+ years", "3 years of experience", "minimum 2 years"
    """
    if not job_desc:
        return 0.0
        
    # Patterns like "5+ years", "3-5 years", "2 years"
    patterns = [
        r'(\d+)\s*\+?\s*years?\b',
        r'minimum\s*of\s*(\d+)\s*years?\b',
        r'(\d+)\s*years?\s*of\s*experience\b',
        r'(\d+)\s*to\s*\d+\s*years?\b'
    ]
    
    job_desc_clean = clean_text(job_desc).lower()
    for pattern in patterns:
        match = re.search(pattern, job_desc_clean)
        if match:
            try:
                years = float(match.group(1))
                logger.info(f"Extracted required experience from job description: {years} years")
                return years
            except ValueError:
                continue
                
    return 0.0

def extract_skills_from_text(text: str, skill_vocabulary: List[str]) -> List[str]:
    """
    Extracts known skills from job description or details using a predefined skill vocabulary list.
    """
    if not text or not skill_vocabulary:
        return []
        
    text_clean = " " + clean_text(text).lower() + " "
    found_skills = []
    
    for skill in skill_vocabulary:
        skill_clean = skill.strip().lower()
        # Find skill as a discrete boundary word/phrase
        escaped_skill = re.escape(skill_clean)
        # Handle special characters in programming languages (C++, C#, .NET)
        pattern = r'\b' + escaped_skill + r'\b'
        if "c++" in skill_clean:
            pattern = r'\bc\+\+\b'
        elif "c#" in skill_clean:
            pattern = r'\bc#\b'
        elif ".net" in skill_clean:
            pattern = r'\b\.net\b'
            
        if re.search(pattern, text_clean):
            if skill_clean not in found_skills:
                found_skills.append(skill_clean)
                
    return found_skills

# ============================================================================
# JSON Parser for Redrob Candidate Profile Schema
# ============================================================================
def parse_candidate_json(cand: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses a single nested candidate dictionary (per candidate_schema.json)
    and flattens it into standard flat columns matching our scoring interface.
    """
    cand_id = cand.get("candidate_id", "UNKNOWN")
    profile = cand.get("profile", {})
    
    # Names & Basic Demographics (scrubbed during profiling, kept for mapping)
    name = profile.get("anonymized_name", "Anonymized Candidate")
    headline = clean_text(profile.get("headline", ""))
    summary = clean_text(profile.get("summary", ""))
    country = clean_text(profile.get("country", ""))
    location = clean_text(profile.get("location", ""))
    
    # Numerical Experience
    try:
        years_exp = float(profile.get("years_of_experience", 0.0))
    except (ValueError, TypeError):
        years_exp = 0.0

    # Career History Description Merge
    career_history = cand.get("career_history", [])
    history_descriptions = []
    companies = []
    titles = []
    
    for job in career_history:
        comp = clean_text(job.get("company", ""))
        title = clean_text(job.get("title", ""))
        desc = clean_text(job.get("description", ""))
        duration = job.get("duration_months", 0)
        
        companies.append(comp)
        titles.append(title)
        
        hist_entry = f"Role: {title} at {comp} ({duration} months). Description: {desc}."
        history_descriptions.append(hist_entry)
        
    merged_history = " ".join(history_descriptions)
    current_title = clean_text(profile.get("current_title", ""))
    current_company = clean_text(profile.get("current_company", ""))

    # Education Summary
    education_entries = cand.get("education", [])
    edu_summaries = []
    for edu in education_entries:
        deg = clean_text(edu.get("degree", ""))
        field = clean_text(edu.get("field_of_study", ""))
        inst = clean_text(edu.get("institution", ""))
        tier = clean_text(edu.get("tier", "unknown"))
        edu_summaries.append(f"{deg} in {field} from {inst} ({tier})")
        
    merged_edu = "; ".join(edu_summaries)
    
    # Skills list extracting names
    skills_entries = cand.get("skills", [])
    skills_names = [clean_text(s.get("name", "")) for s in skills_entries if s.get("name")]
    merged_skills = ", ".join(skills_names)

    # Certifications
    certifications_entries = cand.get("certifications", [])
    cert_summaries = []
    for cert in certifications_entries:
        cert_name = clean_text(cert.get("name", ""))
        issuer = clean_text(cert.get("issuer", ""))
        year = cert.get("year", "")
        cert_summaries.append(f"{cert_name} issued by {issuer} in {year}")
        
    merged_certs = "; ".join(cert_summaries)

    # Extract Redrob signals sub-dictionary directly
    signals = cand.get("redrob_signals", {})
    
    # Platforms Activity & github score representation
    git_score = signals.get("github_activity_score", -1)
    profile_complete = signals.get("profile_completeness_score", 0)
    last_active = signals.get("last_active_date", "")
    
    behavioral_text = (
        f"Preferred work mode: {signals.get('preferred_work_mode', 'flexible')}. "
        f"Willing to relocate: {signals.get('willing_to_relocate', False)}. "
        f"Notice period: {signals.get('notice_period_days', 30)} days. "
        f"Connection count: {signals.get('connection_count', 0)}."
    )

    return {
        "Candidate_ID": cand_id,
        "Candidate_Name": name,
        "Skills": merged_skills,
        "Experience_Years": years_exp,
        "Experience_Details": f"Headline: {headline}. Summary: {summary}. Career Details: {merged_history}",
        "Education": merged_edu,
        "Projects": "",  # Extracted contextual history serves as project descriptions
        "Certifications": merged_certs,
        "Behavioral_Signals": behavioral_text,
        "Platform_Activity": git_score if git_score >= 0 else (profile_complete * 0.5), # composite activity
        
        # Raw original columns stored for special filters
        "current_title": current_title,
        "companies": companies,
        "titles": titles,
        "raw_skills": skills_entries,
        "raw_education": education_entries,
        "redrob_signals": signals,
        "location": location,
        "country": country
    }

def build_semantic_profile(row: pd.Series) -> str:
    """
    Assembles non-sensitive profile columns into a single unified textual representation.
    Scrubs protected attributes to enforce demographic neutrality.
    """
    # Normalize fields
    skills = ", ".join(normalize_skills(row.get("Skills", "")))
    exp_years = str(row.get("Experience_Years", "0"))
    exp_details = clean_text(row.get("Experience_Details", ""))
    education = clean_text(row.get("Education", ""))
    certifications = clean_text(row.get("Certifications", ""))
    
    # Grab optional fields
    behavior = clean_text(row.get("Behavioral_Signals", ""))
    
    # Construct demographic-free text representation
    profile_parts = [
        f"Skills: {skills}.",
        f"Years of Experience: {exp_years}.",
        f"Experience Details: {exp_details}.",
        f"Education: {education}.",
        f"Certifications: {certifications}."
    ]
    
    if behavior:
        profile_parts.append(f"Behavioral Indicators: {behavior}.")
        
    full_profile = " ".join(profile_parts)
    
    # Apply demographic anonymization (scrub pronouns, ages, etc.)
    return anonymize_text(full_profile)

def get_docx_text(path: str) -> str:
    """
    Dependency-free parser to extract raw text content from DOCX (zipped XML) files.
    """
    import zipfile
    import xml.etree.ElementTree as ET
    try:
        with zipfile.ZipFile(path) as docx:
            xml_content = docx.read('word/document.xml')
            tree = ET.fromstring(xml_content)
        
        paragraphs = []
        for paragraph in tree.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
            texts = [node.text for node in paragraph.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if node.text]
            if texts:
                paragraphs.append("".join(texts))
        return "\n".join(paragraphs)
    except Exception as e:
        logger.error(f"Error reading docx {path}: {e}")
        return ""
