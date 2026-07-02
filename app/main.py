import sys
import os
# Ensure workspace root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import logging
import pandas as pd
import streamlit as st

from app.config import (
    HIRING_MODES,
    PRIMARY_EMBEDDING_MODEL,
    FALLBACK_EMBEDDING_MODEL,
    CROSS_ENCODER_MODEL,
    DEFAULT_TOP_N,
    CANDIDATES_CSV,
    OUTPUT_CSV,
    STRONG_HIRE_THRESHOLD,
    GOOD_FIT_THRESHOLD,
    EDUCATION_MAPPING
)
import gzip
import json
from app.preprocessing import (
    clean_text,
    normalize_skills,
    extract_experience_requirement,
    extract_skills_from_text,
    build_semantic_profile,
    parse_candidate_json
)
from app.embeddings import EmbeddingManager
from app.retrieval import build_faiss_index, load_faiss_index, search_faiss_index
from app.reranking import CrossEncoderReranker
from app.scoring import score_candidate, generate_fairness_report, evaluate_rankings
from app.explainability import generate_decision_report, generate_comparison_report, RecruiterNotesManager
from app.interview_generator import generate_interview_questions
import app.analytics as analytics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Streamlit Page Config
st.set_page_config(
    page_title="AI Recruitment Intelligence Platform",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# Initialize Session State Variables
# ==========================================
if "notes_manager" not in st.session_state:
    st.session_state.notes_manager = RecruiterNotesManager()

if "ranked_data" not in st.session_state:
    st.session_state.ranked_data = None

if "telemetry" not in st.session_state:
    st.session_state.telemetry = {}

if "job_requirements" not in st.session_state:
    st.session_state.job_requirements = {"skills": [], "experience": 3.0, "education": "bachelor"}

if "hiring_mode" not in st.session_state:
    st.session_state.hiring_mode = "Product Company"

if "weights" not in st.session_state:
    st.session_state.weights = HIRING_MODES["Product Company"].copy()

# ==========================================
# Premium CSS Styling Injection
# ==========================================
st.markdown(
    """
    <style>
    /* Light Theme Core Adjustments */
    .stApp {
        background-color: #f8fafc; /* Crisp Slate-50 background */
        color: #0f172a; /* Deep Slate-900 text */
    }
    
    /* Metrics / KPI Cards Styling */
    .kpi-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .kpi-card {
        background: #ffffff; /* Clean white card */
        border: 1px solid #e2e8f0; /* Light slate-200 border */
        padding: 1.25rem;
        border-radius: 10px;
        flex: 1 1 200px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        text-align: center;
        color: #0f172a;
    }
    .kpi-title {
        color: #64748b; /* Slate-500 */
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .kpi-value {
        color: #0284c7; /* Ocean Blue */
        font-size: 1.75rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .kpi-subtitle {
        color: #94a3b8; /* Slate-400 */
        font-size: 0.75rem;
    }

    /* Decision Summary Card */
    .decision-card {
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%); /* Clean light gradient */
        border-left: 5px solid #4f46e5; /* Deep Indigo accent */
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        color: #0f172a;
    }
    .decision-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.75rem;
    }

    /* Interactive lists */
    ul.checklist {
        list-style-type: none;
        padding-left: 0;
    }
    ul.checklist li::before {
        content: "✓ ";
        color: #10b981; /* Emerald */
        font-weight: bold;
        display: inline-block;
        width: 1.2em;
    }
    ul.crosslist {
        list-style-type: none;
        padding-left: 0;
    }
    ul.crosslist li::before {
        content: "✗ ";
        color: #ef4444; /* Red */
        font-weight: bold;
        display: inline-block;
        width: 1.2em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# Sidebar Configuration Interface
# ==========================================
st.sidebar.markdown("<h1 style='text-align: center; margin-top: -30px; margin-bottom: 0px; font-size: 3.5rem;'>💼</h1>", unsafe_allow_html=True)
st.sidebar.title("Recruitment Control Panel")

st.sidebar.subheader("Hiring Presets")
# Sync dynamic hiring mode selection
selected_mode = st.sidebar.selectbox(
    "Hiring Objective Mode",
    options=list(HIRING_MODES.keys()),
    index=list(HIRING_MODES.keys()).index(st.session_state.hiring_mode)
)

if selected_mode != st.session_state.hiring_mode:
    st.session_state.hiring_mode = selected_mode
    st.session_state.weights = HIRING_MODES[selected_mode].copy()

# Advanced Scoring Weight Overrides
st.sidebar.subheader("Custom Scoring Weights")
with st.sidebar.expander("Adjust Sliders", expanded=False):
    w_sem = st.slider("Semantic Relevance Weight", 0.0, 1.0, st.session_state.weights.get("semantic", 0.20), 0.05)
    w_skills = st.slider("Technical Skills Match Weight", 0.0, 1.0, st.session_state.weights.get("skills", 0.30), 0.05)
    w_exp = st.slider("Experience Match Weight", 0.0, 1.0, st.session_state.weights.get("experience", 0.20), 0.05)
    w_edu = st.slider("Education Match Weight", 0.0, 1.0, st.session_state.weights.get("education", 0.05), 0.05)
    w_beh = st.slider("Behavioral Alignment Weight", 0.0, 1.0, st.session_state.weights.get("behavior", 0.05), 0.05)
    w_plat = st.slider("Platform Activity Weight", 0.0, 1.0, st.session_state.weights.get("platform", 0.20), 0.05)

    # Normalize weights so they sum to 1.0
    tot = w_sem + w_skills + w_exp + w_edu + w_beh + w_plat
    if tot > 0.0:
        st.session_state.weights = {
            "semantic": w_sem / tot,
            "skills": w_skills / tot,
            "experience": w_exp / tot,
            "education": w_edu / tot,
            "behavior": w_beh / tot,
            "platform": w_plat / tot
        }

# Model and search parameter configs
st.sidebar.subheader("Semantic Pipeline Configurations")
model_choice = st.sidebar.selectbox("Sentence Transformer Model", [PRIMARY_EMBEDDING_MODEL, FALLBACK_EMBEDDING_MODEL])
top_n = st.sidebar.slider("FAISS Retrieval Top-N", 10, 200, 100)
req_exp_input = st.sidebar.number_input("Target Experience (Years)", 0.0, 20.0, 6.0, 0.5)
req_edu_input = st.sidebar.selectbox("Target Education Level", list(EDUCATION_MAPPING.keys())[:-1], index=2)

if st.sidebar.button("Clear Caches & Reset"):
    try:
        EmbeddingManager().clear_cache()
        if os.path.exists(OUTPUT_CSV):
            os.remove(OUTPUT_CSV)
        st.session_state.ranked_data = None
        st.success("Embedding cache cleared. Index states reset.")
        st.rerun()
    except Exception as cache_err:
        st.error(f"Error resetting caches: {cache_err}")

# Main Page Headings
st.title("💼 AI Recruitment Intelligence Platform")
st.markdown("### Semantic Candidate Matching, Explainable Decisions & Fair Hiring Intelligence")

# ==========================================
# Core Processing Pipeline Execution
# ==========================================
def run_pipeline(job_desc_text: str, candidate_csv_file) -> None:
    logger.info("Initializing Ranking Pipeline...")
    total_start = time.time()
    
    # 1. Load Candidates Dataset (supports CSV, JSON, and JSONL)
    try:
        if isinstance(candidate_csv_file, str):
            is_json = candidate_csv_file.endswith(".json") or candidate_csv_file.endswith(".jsonl") or candidate_csv_file.endswith(".jsonl.gz")
            file_name = candidate_csv_file
        else:
            is_json = candidate_csv_file.name.endswith(".json") or candidate_csv_file.name.endswith(".jsonl") or candidate_csv_file.name.endswith(".jsonl.gz")
            file_name = candidate_csv_file.name

        if is_json:
            if isinstance(candidate_csv_file, str):
                if file_name.endswith(".gz"):
                    with gzip.open(file_name, "rt", encoding="utf-8") as f:
                        content = f.read().strip()
                else:
                    with open(file_name, "r", encoding="utf-8") as f:
                        content = f.read().strip()
            else:
                content = candidate_csv_file.getvalue().decode("utf-8").strip()
                
            if content.startswith("["):
                raw_candidates = json.loads(content)
            else:
                raw_candidates = [json.loads(line) for line in content.splitlines() if line.strip()]
                
            parsed_candidates = [parse_candidate_json(c) for c in raw_candidates]
            cand_df = pd.DataFrame(parsed_candidates)
        else:
            cand_df = pd.read_csv(candidate_csv_file)
    except Exception as e:
        st.error(f"Failed to read Candidate file: {e}")
        return

    # Check candidate schema requirements
    mandatory_cols = ["Candidate_ID", "Candidate_Name", "Skills", "Experience_Years", "Education"]
    missing_cols = [c for c in mandatory_cols if c not in cand_df.columns]
    if missing_cols:
        st.error(f"Candidate file is missing required columns: {', '.join(missing_cols)}")
        return

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    # 2. Preprocess & Extract Job Requirements
    status_text.text("1. Parsing job description & extracting technical requirements...")
    progress_bar.progress(0.15)
    
    # Vocabulary mapping for skill matching
    all_skills_vocab = []
    for skill_str in cand_df["Skills"].dropna().unique():
        all_skills_vocab.extend(normalize_skills(skill_str))
    all_skills_vocab = list(set(all_skills_vocab))
    
    req_skills = extract_skills_from_text(job_desc_text, all_skills_vocab)
    req_exp = extract_experience_requirement(job_desc_text)
    if req_exp == 0.0:
        req_exp = req_exp_input # fallback to slider config
        
    req_edu = req_edu_input # from dropdown selection
    
    st.session_state.job_requirements = {
        "skills": req_skills,
        "experience": req_exp,
        "education": req_edu
    }

    # 3. L1 Heuristic Filter & Skills Overlap
    status_text.text("2. Running L1 Heuristic Filter (traps, consulting, titles, skills overlap)...")
    progress_bar.progress(0.30)
    
    from app.scoring import is_honeypot_candidate, is_consulting_only, is_irrelevant_title
    
    l1_survived = []
    for idx, row in cand_df.iterrows():
        # Disqualifier checks
        if is_honeypot_candidate(row) or is_consulting_only(row) or is_irrelevant_title(row):
            continue
            
        # Calculate skills overlap
        cand_skills = normalize_skills(row.get("Skills", ""))
        intersection = set(cand_skills).intersection(set(req_skills))
        skills_score = len(intersection) / max(1, len(req_skills))
        
        # Calculate L1 score
        try:
            exp_val = float(row.get("Experience_Years", 0.0))
        except ValueError:
            exp_val = 0.0
            
        l1_composite = 0.7 * skills_score + 0.3 * min(1.0, exp_val / req_exp)
        
        row_dict = row.to_dict()
        row_dict["l1_score"] = l1_composite
        l1_survived.append(row_dict)
        
    if not l1_survived:
        st.error("No candidates survived L1 heuristic filtering. Cannot generate ranking.")
        return
        
    df_l1 = pd.DataFrame(l1_survived)
    df_l1 = df_l1.sort_values(by="l1_score", ascending=False).reset_index(drop=True)
    df_l2 = df_l1.head(1000).copy()
    
    # Construct semantic representations
    df_l2["Semantic_Doc"] = df_l2.apply(build_semantic_profile, axis=1)

    # 4. Generate Embeddings (cached)
    status_text.text("3. Generating semantic embeddings for top candidates & job requirements...")
    progress_bar.progress(0.45)
    
    emb_mgr = EmbeddingManager()
    
    emb_start = time.time()
    initial_cache_len = len(emb_mgr.cache)
    
    job_emb = emb_mgr.get_embeddings([job_desc_text], model_name=model_choice, is_query=True)
    cand_embs = emb_mgr.get_embeddings(df_l2["Semantic_Doc"].tolist(), model_name=model_choice)
    
    emb_time = time.time() - emb_start
    final_cache_len = len(emb_mgr.cache)
    cache_hit_status = "HIT" if final_cache_len == initial_cache_len else f"MISS ({final_cache_len - initial_cache_len} new)"

    # 5. FAISS Vector Retrieval
    status_text.text("4. Executing semantic search inside FAISS IndexFlatIP...")
    progress_bar.progress(0.60)
    
    ret_start = time.time()
    index = build_faiss_index(cand_embs)
    
    # Retrieve top-N matches
    ret_sims, ret_indices = search_faiss_index(job_emb, index, top_n=min(len(df_l2), top_n))
    ret_time = time.time() - ret_start
    
    # Slice matched candidate frames
    matched_df = df_l2.iloc[ret_indices].copy()
    matched_df["Retrieval_Score"] = ret_sims
    matched_semantic_docs = matched_df["Semantic_Doc"].tolist()
    matched_ids = matched_df["Candidate_ID"].tolist()

    # 6. Cross-Encoder Reranking
    status_text.text("5. Performing deep context-aware Cross-Encoder reranking...")
    progress_bar.progress(0.75)
    
    ce_start = time.time()
    reranker = CrossEncoderReranker()
    reranked_results = reranker.rerank(
        job_desc_text, 
        matched_ids, 
        matched_semantic_docs, 
        matched_df["Retrieval_Score"].tolist()
    )
    ce_time = time.time() - ce_start

    # Merge rerank scores back to matches
    rerank_scores_map = {item["Candidate_ID"]: item["Semantic_Score"] for item in reranked_results}
    matched_df["CE_Semantic_Score"] = matched_df["Candidate_ID"].map(rerank_scores_map)

    # 7. Hybrid Multi-Objective Scoring & Explanation
    status_text.text("6. Computing hybrid multi-objective scores & decision insights...")
    progress_bar.progress(0.90)
    
    ranked_records = []
    
    for idx, row in matched_df.iterrows():
        cand_id = row["Candidate_ID"]
        sem_score = row["CE_Semantic_Score"]
        
        # Calculate weights & composite scores
        score_details = score_candidate(
            row, 
            sem_score, 
            req_skills, 
            req_exp, 
            req_edu, 
            hiring_mode=st.session_state.hiring_mode
        )
        
        # Generate explanations & decision report
        decision = generate_decision_report(
            row, 
            score_details, 
            req_skills, 
            req_exp, 
            req_edu
        )
        
        # Combine items
        record = {
            "Candidate_ID": cand_id,
            "Candidate_Name": row["Candidate_Name"],
            "Skills": row["Skills"],
            "Experience_Years": row["Experience_Years"],
            "Education": row["Education"],
            "Projects": row.get("Projects", ""),
            "Certifications": row.get("Certifications", ""),
            "Behavioral_Signals": row.get("Behavioral_Signals", ""),
            "Platform_Activity": row.get("Platform_Activity", ""),
            
            # Scores
            "Final_Score": score_details["Final_Score"],
            "Semantic_Score": score_details["Semantic_Score"],
            "Skills_Score": score_details["Skills_Score"],
            "Experience_Score": score_details["Experience_Score"],
            "Education_Score": score_details["Education_Score"],
            "Behavior_Score": score_details["Behavior_Score"],
            "Platform_Score": score_details["Platform_Score"],
            
            # Decision elements
            "Confidence": decision["Confidence_Score"],
            "Recommendation": decision["Recommendation"],
            "Hiring_Summary": decision["Hiring_Summary"],
            "Matched_Skills": ", ".join(decision["Matched_Skills"]),
            "Missing_Skills": ", ".join(decision["Missing_Skills"]),
            "Experience_Assessment": decision["Experience_Assessment"],
            "Education_Assessment": decision["Education_Assessment"],
            "Key_Strengths": decision["Key_Strengths"],
            "Potential_Risks": decision["Potential_Risks"],
            "Ranking_Justification": decision["Ranking_Justification"],
            "Suggested_Next_Step": decision["Suggested_Next_Step"]
        }
        
        # If ground-truth labels exist
        if "Relevance_Label" in cand_df.columns:
            record["Relevance_Label"] = row["Relevance_Label"]
            
        ranked_records.append(record)

    # Sort final DataFrame by score descending
    final_df = pd.DataFrame(ranked_records)
    final_df = final_df.sort_values(by="Final_Score", ascending=False).reset_index(drop=True)
    final_df.insert(0, "Rank", final_df.index + 1)
    
    # Save to CSV
    final_df.to_csv(OUTPUT_CSV, index=False)
    
    progress_bar.progress(1.0)
    status_text.empty()
    progress_bar.empty()
    
    total_time = time.time() - total_start
    
    # Persist data in session state
    st.session_state.ranked_data = final_df
    st.session_state.telemetry = {
        "embedding_time": emb_time,
        "retrieval_time": ret_time,
        "cross_encoder_time": ce_time,
        "total_ranking_time": total_time,
        "processed_candidates": len(cand_df),
        "embedding_cache_status": cache_hit_status,
        "faiss_index_status": "LOADED" if os.path.exists(CANDIDATES_CSV) else "BUILT"
    }
    
    st.success("Ranking completed successfully!")
    st.balloons()

# ==========================================
# File Upload UI Segment
# ==========================================
col_up_1, col_up_2 = st.columns([2, 1])

# Default JD matching rank.py
from app.preprocessing import get_docx_text
JD_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "India_runs_data_and_ai_challenge", "job_description.docx")
DEFAULT_JD = get_docx_text(JD_PATH)
if not DEFAULT_JD:
    DEFAULT_JD = (
        "Senior AI/ML Engineer to own the intelligence layer of Redrob's product: ranking, retrieval, matching, recommendation. "
        "Highly skilled in Python, embeddings-based retrieval systems (sentence-transformers, BGE, E5, OpenAI), "
        "vector databases or hybrid search (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS), "
        "and evaluation frameworks (NDCG, MRR, MAP, offline-to-online correlation, A/B testing). "
        "Product company experience preferred; no pure research or pure consulting firm background. Able to ship production code."
    )

with col_up_1:
    job_desc = st.text_area(
        "Paste Job Description / Requirements", 
        value=DEFAULT_JD,
        height=220,
        placeholder="We are looking for a Senior NLP Engineer with 5+ years of experience in PyTorch, transformers, and SQL..."
    )

with col_up_2:
    candidate_file = st.file_uploader("Upload Candidates Dataset (CSV, JSON, JSONL)", type=["csv", "json", "jsonl"])
    
    # Offer template download or generate sample data
    st.markdown("⚠️ **Don't have a file?** A sample file `data/sample_candidates.csv` will be loaded by default if no file is uploaded.")
    
    run_btn = st.button("🚀 Run AI Intelligent Ranking", use_container_width=True)

# Run process triggers
if run_btn:
    if not job_desc.strip():
        st.warning("Please provide a Job Description to start ranking.")
    else:
        # Load sample candidates if no file uploaded
        if candidate_file is not None:
            run_pipeline(job_desc, candidate_file)
        else:
            if os.path.exists(CANDIDATES_CSV):
                run_pipeline(job_desc, CANDIDATES_CSV)
            else:
                st.error("No candidate file uploaded and sample candidate database not found at `data/sample_candidates.csv`. Please upload a file.")

# Check if we have ranked data to show
if st.session_state.ranked_data is not None:
    ranked_df = st.session_state.ranked_data
    telemetry = st.session_state.telemetry

    # Render Tabs
    tab_leaderboard, tab_deepdive, tab_analytics, tab_settings = st.tabs([
        "🏆 Candidate Leaderboard", 
        "🔍 Candidate Deep Dive", 
        "📊 Analytics Dashboard", 
        "⚙️ Pipeline Settings & Performance"
    ])

    # ==========================================
    # Tab 1: Leaderboard
    # ==========================================
    with tab_leaderboard:
        # Executive Summary KPI Row
        avg_score = ranked_df["Final_Score"].mean() * 100.0
        avg_conf = ranked_df["Confidence"].mean()
        top_cand_name = ranked_df.iloc[0]["Candidate_Name"]
        ready_count = len(ranked_df[ranked_df["Final_Score"] >= GOOD_FIT_THRESHOLD])
        upskill_count = len(ranked_df) - ready_count
        
        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="kpi-card">
                    <div class="kpi-title">Total Processed</div>
                    <div class="kpi-value">{telemetry.get("processed_candidates", 0)}</div>
                    <div class="kpi-subtitle">Candidates Indexed</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Top Recommendation</div>
                    <div class="kpi-value" style="color: #10b981;">{top_cand_name[:12]}...</div>
                    <div class="kpi-subtitle">Rank 1 Match</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Avg Match Score</div>
                    <div class="kpi-value">{avg_score:.1f}%</div>
                    <div class="kpi-subtitle">Score Baseline</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Avg AI Confidence</div>
                    <div class="kpi-value">{avg_conf:.1f}%</div>
                    <div class="kpi-subtitle">Consensus Ratio</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Ready for Interview</div>
                    <div class="kpi-value" style="color: #34d399;">{ready_count}</div>
                    <div class="kpi-subtitle">Score &ge; 75%</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Require Upskilling</div>
                    <div class="kpi-value" style="color: #f59e0b;">{upskill_count}</div>
                    <div class="kpi-subtitle">Score &lt; 75%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Quick Candidate Comparison Panel (Selected Two Candidates Side-by-Side)
        st.markdown("### 🔀 Quick Candidate Comparison")
        col_comp_1, col_comp_2 = st.columns([1, 2])
        with col_comp_1:
            comp_cand_1 = st.selectbox("Select Candidate A", options=ranked_df["Candidate_Name"].tolist(), index=0, key="comp_a")
            comp_cand_2 = st.selectbox("Select Candidate B", options=ranked_df["Candidate_Name"].tolist(), index=min(1, len(ranked_df)-1), key="comp_b")
        
        with col_comp_2:
            row_a = ranked_df[ranked_df["Candidate_Name"] == comp_cand_1].iloc[0]
            row_b = ranked_df[ranked_df["Candidate_Name"] == comp_cand_2].iloc[0]
            
            # Map scoring dictionaries for comparison radar
            scores_a = {
                "Semantic_Score": row_a["Semantic_Score"],
                "Skills_Score": row_a["Skills_Score"],
                "Experience_Score": row_a["Experience_Score"],
                "Education_Score": row_a["Education_Score"],
                "Behavior_Score": row_a["Behavior_Score"],
                "Platform_Score": row_a["Platform_Score"]
            }
            scores_b = {
                "Semantic_Score": row_b["Semantic_Score"],
                "Skills_Score": row_b["Skills_Score"],
                "Experience_Score": row_b["Experience_Score"],
                "Education_Score": row_b["Education_Score"],
                "Behavior_Score": row_b["Behavior_Score"],
                "Platform_Score": row_b["Platform_Score"]
            }
            
            comp_radar = analytics.plot_candidate_comparison_radar(scores_a, comp_cand_1, scores_b, comp_cand_2)
            st.plotly_chart(comp_radar, use_container_width=True)

        # Side-by-side metric table comparison
        st.markdown("#### Comparison Metrics")
        comp_data = {
            "Metric": ["Overall Match Score", "Semantic Similarity", "Technical Skills Match", "Years of Experience", "Highest Education", "Recommendation", "Confidence %"],
            comp_cand_1: [
                f"{row_a['Final_Score']*100:.1f}%", f"{row_a['Semantic_Score']*100:.1f}%", f"{row_a['Skills_Score']*100:.1f}%", 
                f"{row_a['Experience_Years']} yrs", row_a['Education'], row_a['Recommendation'], f"{row_a['Confidence']}%"
            ],
            comp_cand_2: [
                f"{row_b['Final_Score']*100:.1f}%", f"{row_b['Semantic_Score']*100:.1f}%", f"{row_b['Skills_Score']*100:.1f}%", 
                f"{row_b['Experience_Years']} yrs", row_b['Education'], row_b['Recommendation'], f"{row_b['Confidence']}%"
            ]
        }
        st.table(pd.DataFrame(comp_data))

        st.markdown("---")

        # Leaderboard Table View
        st.markdown("### 🏆 Ranked Candidates List")
        
        # Enable recruiter search filter
        search_query = st.text_input("🔍 Search Candidates by Name or Skills", placeholder="Type e.g. Python, Alex...")
        
        display_df = ranked_df.copy()
        if search_query.strip():
            query_clean = search_query.lower()
            display_df = display_df[
                display_df["Candidate_Name"].str.lower().str.contains(query_clean) |
                display_df["Skills"].str.lower().str.contains(query_clean)
            ]

        # Display Data Grid
        grid_cols = ["Rank", "Candidate_ID", "Candidate_Name", "Final_Score", "Semantic_Score", "Skills_Score", "Experience_Years", "Education", "Recommendation", "Confidence"]
        grid_df = display_df[grid_cols].copy()
        grid_df["Final_Score"] = grid_df["Final_Score"].apply(lambda x: f"{x*100:.1f}%")
        grid_df["Semantic_Score"] = grid_df["Semantic_Score"].apply(lambda x: f"{x*100:.1f}%")
        grid_df["Skills_Score"] = grid_df["Skills_Score"].apply(lambda x: f"{x*100:.1f}%")
        
        st.dataframe(grid_df, use_container_width=True, hide_index=True)

        # Download Excel (XLSX)
        col_down_1, col_down_2 = st.columns([1, 1])
        with col_down_1:
            try:
                import io
                buffer = io.BytesIO()
                
                # Format exact submission columns for Excel file
                submission_df = display_df[["Candidate_ID", "Rank", "Final_Score", "Hiring_Summary"]].copy()
                submission_df.columns = ["candidate_id", "rank", "score", "reasoning"]
                
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    submission_df.to_excel(writer, index=False, sheet_name='Ranks')
                excel_data = buffer.getvalue()
                
                st.download_button(
                    label="📥 Export Ranked Candidates Excel (XLSX)",
                    data=excel_data,
                    file_name="submission.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error generating Excel download: {e}")

        # Recruiter Notes Input Area
        st.markdown("#### 📝 Candidate Notes Manager")
        col_note_1, col_note_2 = st.columns([1, 2])
        with col_note_1:
            note_cand_id = st.selectbox("Select Candidate to Note", options=ranked_df["Candidate_ID"].tolist())
        with col_note_2:
            current_note = st.session_state.notes_manager.get_note(note_cand_id)
            note_text = st.text_input(f"Enter note for ID: {note_cand_id}", value=current_note)
            if st.button("Save Note"):
                st.session_state.notes_manager.add_note(note_cand_id, note_text)
                st.success(f"Note saved for candidate {note_cand_id}!")

    # ==========================================
    # Tab 2: Candidate Deep Dive
    # ==========================================
    with tab_deepdive:
        st.markdown("### 🔍 Explainable AI Candidate Assessment Report")
        
        deep_cand_id = st.selectbox("Select Candidate to Inspect", options=ranked_df["Candidate_ID"].tolist())
        
        cand_row = ranked_df[ranked_df["Candidate_ID"] == deep_cand_id].iloc[0]
        
        # Map scoring dict
        cand_scores = {
            "Semantic_Score": cand_row["Semantic_Score"],
            "Skills_Score": cand_row["Skills_Score"],
            "Experience_Score": cand_row["Experience_Score"],
            "Education_Score": cand_row["Education_Score"],
            "Behavior_Score": cand_row["Behavior_Score"],
            "Platform_Score": cand_row["Platform_Score"]
        }

        col_dd_1, col_dd_2 = st.columns([1, 1])

        with col_dd_1:
            # 1. AI Decision Report Card
            st.markdown(
                f"""
                <div class="decision-card">
                    <div class="decision-title">AI Decision Report: {cand_row['Candidate_Name']}</div>
                    <p style="font-size: 1.1rem; color: #1e293b;"><strong>Hiring Recommendation:</strong> {cand_row['Recommendation']}</p>
                    <p style="font-size: 1rem; color: #0284c7;"><strong>AI Match Score:</strong> {cand_row['Final_Score']*100:.1f}% | <strong>Confidence Score:</strong> {cand_row['Confidence']}%</p>
                    <hr style="border-color: rgba(148, 163, 184, 0.4); margin: 0.75rem 0;" />
                    <p style="font-style: italic; color: #334155; font-size: 0.95rem;">"{cand_row['Hiring_Summary']}"</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Recruiter Notes persistence display
            rec_note = st.session_state.notes_manager.get_note(deep_cand_id)
            if rec_note:
                st.info(f"📝 **Recruiter Notes:** {rec_note}")

            # 2. Key Strengths & Risks lists
            st.markdown("#### Strengths & Potential Risks")
            col_dd_list_1, col_dd_list_2 = st.columns(2)
            with col_dd_list_1:
                st.markdown("<p style='color: #10b981; font-weight: bold;'>Key Strengths</p>", unsafe_allow_html=True)
                strengths_html = "".join([f"<li>{s}</li>" for s in eval(str(cand_row["Key_Strengths"]))])
                st.markdown(f"<ul class='checklist'>{strengths_html}</ul>", unsafe_allow_html=True)
                
            with col_dd_list_2:
                st.markdown("<p style='color: #ef4444; font-weight: bold;'>Potential Risks</p>", unsafe_allow_html=True)
                risks_html = "".join([f"<li>{r}</li>" for r in eval(str(cand_row["Potential_Risks"]))])
                st.markdown(f"<ul class='crosslist'>{risks_html}</ul>", unsafe_allow_html=True)

            # 3. Target Assessments
            st.markdown("#### Requirements Match Assessment")
            st.write(f"💼 **Experience Assessment:** {cand_row['Experience_Assessment']}")
            st.write(f"🎓 **Education Assessment:** {cand_row['Education_Assessment']}")
            st.write(f"💡 **Ranking Justification:** {cand_row['Ranking_Justification']}")
            st.write(f"➡️ **Suggested Next Action:** `{cand_row['Suggested_Next_Step']}`")

        with col_dd_2:
            # 4. Radar Chart
            radar_fig = analytics.plot_candidate_radar(cand_scores, cand_row["Candidate_Name"])
            st.plotly_chart(radar_fig, use_container_width=True)

            # 5. Rank Comparison Report ("Why NOT Rank 1?")
            top_cand_row = ranked_df.iloc[0]
            if cand_row["Rank"] != 1:
                st.markdown("### 📊 Rank Comparison Report")
                
                # Fetch target score dict vs top
                top_scores = {
                    "Overall_Match_Score": top_cand_row["Final_Score"],
                    "Semantic_Score": top_cand_row["Semantic_Score"],
                    "Skills_Score": top_cand_row["Skills_Score"]
                }
                target_scores_rep = {
                    "Overall_Match_Score": cand_row["Final_Score"],
                    "Semantic_Score": cand_row["Semantic_Score"],
                    "Skills_Score": cand_row["Skills_Score"]
                }
                
                comp_rep = generate_comparison_report(target_scores_rep, top_scores, cand_row, top_cand_row)
                
                st.warning(f"**Score Gap vs. Rank 1 ({top_cand_row['Candidate_Name']}):** -{comp_rep['score_difference']*100:.1f}%")
                st.markdown(f"*{comp_rep['comparison_narrative']}*")
                if comp_rep["skills_missing_vs_top"]:
                    st.write(f"🔑 **Skills to acquire matching top profile:** `{', '.join(comp_rep['skills_missing_vs_top'])}`")
            else:
                st.success("⭐ This candidate is currently the **Rank #1 Top Recommendation** for this role.")

        st.markdown("---")

        # 6. Technical Interview Questions Panel
        st.markdown("### 🗣️ Custom Technical Interview Question Generator")
        st.markdown("These questions are automatically tailored to probe the candidate's specific skill gaps, experience depth, and project context.")
        
        m_skills = [s.strip() for s in cand_row["Matched_Skills"].split(",") if s.strip()]
        g_skills = [s.strip() for s in cand_row["Missing_Skills"].split(",") if s.strip()]
        
        interview_q = generate_interview_questions(cand_row, m_skills, g_skills)
        
        col_q1, col_q2, col_q3 = st.columns(3)
        with col_q1:
            st.markdown("🟢 **Easy Questions** (Concept checks for Gaps)")
            for i, q in enumerate(interview_q["Easy"], 1):
                st.write(f"**{i}.** {q}")
        with col_q2:
            st.markdown("🟡 **Medium Questions** (Matched Skills & Projects)")
            for i, q in enumerate(interview_q["Medium"], 1):
                st.write(f"**{i}.** {q}")
        with col_q3:
            st.markdown("🔴 **Hard Questions** (System Design & Leadership)")
            for i, q in enumerate(interview_q["Hard"], 1):
                st.write(f"**{i}.** {q}")

    # ==========================================
    # Tab 3: Analytics Dashboard
    # ==========================================
    with tab_analytics:
        st.markdown("### 📊 Candidate Database Analytics")
        
        col_an_1, col_an_2 = st.columns(2)
        
        # 1. Score Distribution
        with col_an_1:
            score_fig = analytics.plot_score_distribution(ranked_df["Final_Score"].tolist())
            st.plotly_chart(score_fig, use_container_width=True)

        # 2. Experience Distribution
        with col_an_2:
            exp_fig = analytics.plot_experience_distribution(ranked_df["Experience_Years"].astype(float).tolist())
            st.plotly_chart(exp_fig, use_container_width=True)

        col_an_3, col_an_4 = st.columns(2)
        
        # 3. Education Pie
        with col_an_3:
            edu_fig = analytics.plot_education_distribution(ranked_df["Education"].tolist())
            st.plotly_chart(edu_fig, use_container_width=True)

        # 4. Top Skills bar
        with col_an_4:
            skills_series = [normalize_skills(s) for s in ranked_df["Skills"].tolist()]
            top_skills_fig = analytics.plot_top_skills(skills_series)
            st.plotly_chart(top_skills_fig, use_container_width=True)

        # 5. Skill Gap Heatmap (Green Match, Red Gap)
        st.markdown("---")
        st.markdown("### 🗺️ Technical Skill Gap Matrix")
        req_skills_heat = st.session_state.job_requirements["skills"]
        if req_skills_heat:
            heatmap_fig = analytics.plot_skill_gap_heatmap(ranked_df, req_skills_heat)
            st.plotly_chart(heatmap_fig, use_container_width=True)
        else:
            st.warning("No skills extracted from job description to compile the gap heatmap.")

        # 6. Evaluation metrics segment
        st.markdown("---")
        st.markdown("### 📈 Evaluation Metrics Report")
        
        eval_metrics = evaluate_rankings(ranked_df, ground_truth_col="Relevance_Label")
        if eval_metrics["metrics_type"] == "GROUND_TRUTH":
            st.success("🎯 **Ground-Truth Evaluation Data Detected!** The metrics below evaluate ranking predictions against manual labels:")
            col_ev_1, col_ev_2, col_ev_3, col_ev_4 = st.columns(4)
            col_ev_1.metric("Precision@5", f"{eval_metrics['Precision@5']*100:.1f}%")
            col_ev_2.metric("Recall@5", f"{eval_metrics['Recall@5']*100:.1f}%")
            col_ev_3.metric("Mean Reciprocal Rank (MRR)", f"{eval_metrics['MRR']:.4f}")
            col_ev_4.metric("NDCG@5", f"{eval_metrics['NDCG@5']:.4f}")
        else:
            st.info("ℹ️ **Operational Telemetry Active** (No ground-truth 'Relevance_Label' column in CSV)")
            col_ev_1, col_ev_2, col_ev_3 = st.columns(3)
            col_ev_1.metric("Average Final Score", f"{eval_metrics['Average_Final_Score']*100:.1f}%")
            col_ev_2.metric("Average Semantic Similarity", f"{eval_metrics['Average_Semantic_Similarity']*100:.1f}%")
            col_ev_3.metric("Total Candidates Processed", eval_metrics['Candidates_Processed'])

    # ==========================================
    # Tab 4: Performance & Fairness Compliance
    # ==========================================
    with tab_settings:
        st.markdown("### ⚙️ Telemetry Performance Dashboard")
        
        col_perf_1, col_perf_2 = st.columns(2)
        with col_perf_1:
            st.markdown("#### Ranking Processing Time Breakdown")
            st.write(f"⏱️ **Embedding Inferences Time:** `{telemetry.get('embedding_time', 0.0):.4f}s`")
            st.write(f"⏱️ **FAISS Search Execution Time:** `{telemetry.get('retrieval_time', 0.0):.4f}s`")
            st.write(f"⏱️ **Cross Encoder Rerank Time:** `{telemetry.get('cross_encoder_time', 0.0):.4f}s`")
            st.write(f"⏱️ **Total Platform Pipeline Time:** `{telemetry.get('total_ranking_time', 0.0):.4f}s`")
            
        with col_perf_2:
            st.markdown("#### System Caching Status")
            st.write(f"💾 **Sentence-Transformer Cache:** `{telemetry.get('embedding_cache_status', 'N/A')}`")
            st.write(f"💾 **FAISS Index Storage State:** `{telemetry.get('faiss_index_status', 'N/A')}`")
            st.write(f"📦 **Currently Loaded Models:**")
            st.write(f"- Semantic Encoder: `{model_choice}`")
            st.write(f"- Deep Contextual Reranker: `{CROSS_ENCODER_MODEL}`")

        st.markdown("---")
        
        # Demographic Fairness Report
        st.markdown("### 🛡️ Demographic Fairness Compliance Report")
        fair_report = generate_fairness_report(ranked_df)
        
        st.success(f"Audit Status: **{fair_report['status']}**")
        st.write(fair_report["verification_statement"])
        st.write(f"Protected fields validated as scrubbed/ignored: `{', '.join(fair_report['protected_attributes_scrubbed'])}`")
else:
    st.info("👋 Welcome! Pastes requirements, upload a CSV or use default mock data, and hit 'Run AI Intelligent Ranking' to launch the recruitment engine!")
