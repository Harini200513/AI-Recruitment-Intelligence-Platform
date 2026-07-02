import logging
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Setup Logging
logger = logging.getLogger(__name__)

# Premium Light Mode Layout Configs
LIGHT_THEME_LAYOUT = dict(
    paper_bgcolor="rgba(255, 255, 255, 1.0)",  # Plain White
    plot_bgcolor="rgba(255, 255, 255, 1.0)",
    font=dict(color="#0f172a", family="Inter, Outfit, sans-serif"),  # Slate-900
    margin=dict(l=40, r=40, t=50, b=40),
    xaxis=dict(
        gridcolor="#e2e8f0",  # Slate-200
        linecolor="#94a3b8",  # Slate-400
        tickfont=dict(color="#475569")  # Slate-600
    ),
    yaxis=dict(
        gridcolor="#e2e8f0",
        linecolor="#94a3b8",
        tickfont=dict(color="#475569")
    )
)

def plot_candidate_radar(scores: Dict[str, float], candidate_name: str) -> go.Figure:
    """
    Plots a radar chart showing the individual component scores of a candidate.
    """
    categories = ['Semantic Match', 'Skills Match', 'Experience Match', 'Education Match', 'Behavioral Alignment', 'Platform Activity']
    
    # Map dictionary keys to match radar categories
    val_map = {
        'Semantic Match': scores.get('Semantic_Score', 0.0),
        'Skills Match': scores.get('Skills_Score', 0.0),
        'Experience Match': scores.get('Experience_Score', 0.0),
        'Education Match': scores.get('Education_Score', 0.0),
        'Behavioral Alignment': scores.get('Behavior_Score', 0.5) if not pd.isna(scores.get('Behavior_Score')) else 0.5,
        'Platform Activity': scores.get('Platform_Score', 0.5) if not pd.isna(scores.get('Platform_Score')) else 0.5
    }
    
    values = [val_map[cat] * 100.0 for cat in categories]
    # Close the radar loop by repeating the first value
    values.append(values[0])
    categories.append(categories[0])

    fig = go.Figure()
    
    # Trace for candidate
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.3)',  # Indigo fill
        line=dict(color='#6366f1', width=2),
        name=candidate_name
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="#e2e8f0",
                linecolor="#94a3b8",
                tickfont=dict(color="#64748b")
            ),
            angularaxis=dict(
                gridcolor="#e2e8f0",
                linecolor="#94a3b8"
            ),
            bgcolor="rgba(241, 245, 249, 0.5)"  # Slate-100
        ),
        showlegend=False,
        title=dict(text=f"Score Breakdown: {candidate_name}", x=0.5, font=dict(size=16)),
        **LIGHT_THEME_LAYOUT
    )
    
    return fig

def plot_candidate_comparison_radar(
    cand1_scores: Dict[str, float], cand1_name: str,
    cand2_scores: Dict[str, float], cand2_name: str
) -> go.Figure:
    """
    Plots an overlay radar chart comparing scores of two candidates.
    """
    categories = ['Semantic Match', 'Skills Match', 'Experience Match', 'Education Match', 'Behavioral Alignment', 'Platform Activity']
    
    def get_values(scores):
        val_map = {
            'Semantic Match': scores.get('Semantic_Score', 0.0),
            'Skills Match': scores.get('Skills_Score', 0.0),
            'Experience Match': scores.get('Experience_Score', 0.0),
            'Education Match': scores.get('Education_Score', 0.0),
            'Behavioral Alignment': scores.get('Behavior_Score', 0.5) if not pd.isna(scores.get('Behavior_Score')) else 0.5,
            'Platform Activity': scores.get('Platform_Score', 0.5) if not pd.isna(scores.get('Platform_Score')) else 0.5
        }
        vals = [val_map[cat] * 100.0 for cat in categories]
        vals.append(vals[0])
        return vals

    vals1 = get_values(cand1_scores)
    vals2 = get_values(cand2_scores)
    radar_cats = categories + [categories[0]]

    fig = go.Figure()

    # Candidate 1 Trace (Indigo)
    fig.add_trace(go.Scatterpolar(
        r=vals1,
        theta=radar_cats,
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.25)',
        line=dict(color='#6366f1', width=2),
        name=cand1_name
    ))

    # Candidate 2 Trace (Emerald/Teal)
    fig.add_trace(go.Scatterpolar(
        r=vals2,
        theta=radar_cats,
        fill='toself',
        fillcolor='rgba(20, 184, 166, 0.25)',
        line=dict(color='#14b8a6', width=2),
        name=cand2_name
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="#e2e8f0",
                tickfont=dict(color="#64748b")
            ),
            bgcolor="rgba(241, 245, 249, 0.5)"
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        title=dict(text=f"Comparison: {cand1_name} vs {cand2_name}", x=0.5, font=dict(size=16)),
        **LIGHT_THEME_LAYOUT
    )
    return fig

def plot_score_distribution(scores: List[float]) -> go.Figure:
    """
    Generates a histogram plot showing candidate score distribution.
    """
    # Convert [0, 1] scores to percentages
    scores_pct = [s * 100.0 for s in scores]
    
    fig = px.histogram(
        x=scores_pct,
        nbins=10,
        labels={'x': 'Overall Match Score (%)'},
        title="Candidate Score Distribution",
        color_discrete_sequence=['#818cf8']  # Light Indigo
    )
    
    fig.update_layout(
        bargap=0.08,
        yaxis_title="Count of Candidates",
        **LIGHT_THEME_LAYOUT
    )
    return fig

def plot_experience_distribution(exps: List[float]) -> go.Figure:
    """
    Plots distribution of experience years in database.
    """
    fig = px.histogram(
        x=exps,
        nbins=8,
        labels={'x': 'Years of Experience'},
        title="Experience Distribution",
        color_discrete_sequence=['#f59e0b']  # Amber/Yellow
    )
    fig.update_layout(
        bargap=0.08,
        yaxis_title="Count of Candidates",
        **LIGHT_THEME_LAYOUT
    )
    return fig

def plot_education_distribution(education_list: List[str]) -> go.Figure:
    """
    Creates a donut pie chart showing education breakdowns.
    """
    # Simple clean counts
    clean_edu = []
    for edu in education_list:
        if pd.isna(edu):
            clean_edu.append("None")
            continue
            
        edu_lower = str(edu).lower()
        if "ph" in edu_lower or "doctor" in edu_lower:
            clean_edu.append("PhD / Doctorate")
        elif "master" in edu_lower or "msc" in edu_lower or "ms" in edu_lower or "mba" in edu_lower:
            clean_edu.append("Master's")
        elif "bachelor" in edu_lower or "bsc" in edu_lower or "btech" in edu_lower or "be " in edu_lower:
            clean_edu.append("Bachelor's")
        elif "associate" in edu_lower or "diploma" in edu_lower:
            clean_edu.append("Associate / Diploma")
        else:
            clean_edu.append("Other / High School")

    df = pd.Series(clean_edu).value_counts().reset_index()
    df.columns = ['Education Level', 'Count']

    fig = px.pie(
        df,
        values='Count',
        names='Education Level',
        hole=0.4,
        title="Education Level Breakdown",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        **LIGHT_THEME_LAYOUT
    )
    return fig

def plot_top_skills(all_skills_list: List[List[str]], max_skills: int = 10) -> go.Figure:
    """
    Counts skill occurrences and plots a horizontal bar chart showing top technical skills.
    """
    # Flatten
    flat_skills = [s.strip().lower() for sublist in all_skills_list for s in sublist if s.strip()]
    if not flat_skills:
        fig = go.Figure()
        fig.update_layout(title="No Skills to Display", **LIGHT_THEME_LAYOUT)
        return fig

    counts = pd.Series(flat_skills).value_counts().head(max_skills)
    df = counts.reset_index()
    df.columns = ['Skill', 'Count']
    
    # Capitalize skills for aesthetic labels
    df['Skill'] = df['Skill'].apply(lambda x: x.upper() if len(x) <= 4 else x.title())

    fig = px.bar(
        df,
        x='Count',
        y='Skill',
        orientation='h',
        title=f"Top {max_skills} Skills Across Candidates",
        color='Count',
        color_continuous_scale='Viridis',
        category_orders={"Skill": df["Skill"].tolist()[::-1]} # Maintain ranking order
    )
    fig.update_layout(
        coloraxis_showscale=False,
        **LIGHT_THEME_LAYOUT
    )
    return fig

def plot_skill_gap_heatmap(candidates_df: pd.DataFrame, required_skills: List[str]) -> go.Figure:
    """
    Plots a binary heatmap showing candidates vs. required skills.
    Value is 1 if candidate has skill, 0 if missing.
    """
    if not required_skills or candidates_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Skills to Analyze", **LIGHT_THEME_LAYOUT)
        return fig

    # Build binary alignment matrix
    matrix_data = []
    candidate_names = []

    for idx, row in candidates_df.iterrows():
        cand_skills = [s.strip().lower() for s in str(row.get("Skills", "")).replace(";", ",").split(",") if s.strip()]
        
        row_indicators = []
        for req_skill in required_skills:
            req_skill_clean = req_skill.strip().lower()
            row_indicators.append(1 if req_skill_clean in cand_skills else 0)
            
        matrix_data.append(row_indicators)
        # Limit name lengths for clean labels
        cand_names_raw = row.get("Candidate_Name", f"Cand_{row.get('Candidate_ID', idx)}")
        candidate_names.append(cand_names_raw[:15])

    # Convert to numpy array
    matrix_np = np.array(matrix_data)
    
    # Build Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix_np,
        x=[s.upper() if len(s) <= 4 else s.title() for s in required_skills],
        y=candidate_names,
        colorscale=[[0, '#ef4444'], [1, '#10b981']],  # Red to Emerald Green
        showscale=False,
        xgap=2,
        ygap=2,
        hovertemplate="Candidate: %{y}<br>Skill: %{x}<br>Status: %{text}<extra></extra>",
        text=[["Matched" if val == 1 else "Missing" for val in row] for row in matrix_data]
    ))

    fig.update_layout(
        title="Required Skills Gap Matrix (Green: Match | Red: Gap)",
        xaxis_title="Required Technologies",
        yaxis_title="Candidates",
        **LIGHT_THEME_LAYOUT
    )
    
    # Set y-axis to draw from top down to keep Rank 1 on top
    fig.update_yaxes(autorange="reversed")
    
    return fig
