import os
import re
import sqlite3
from collections import Counter
import streamlit as st
from dotenv import load_dotenv
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import nltk
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="AI Resume Analyzer", page_icon="🧠", layout="wide")

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
DB_PATH = os.path.join(os.getcwd(), "sra.db")

def _nltk_safe_download():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ]
    fallback_tagger = ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng")
    for rid, pkg in resources:
        try:
            nltk.data.find(rid)
        except LookupError:
            nltk.download(pkg, quiet=True)
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        try:
            nltk.data.find("taggers/averaged_perceptron_tagger_eng")
        except LookupError:
            nltk.download(fallback_tagger[1], quiet=True)

_nltk_safe_download()

if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False
if "resume" not in st.session_state:
    st.session_state.resume = ""
if "job_desc" not in st.session_state:
    st.session_state.job_desc = ""
if "resume_emb" not in st.session_state:
    st.session_state.resume_emb = ""

st.markdown("""
<style>
.main-title {text-align:center;color:#00ADB5;font-size:48px;font-weight:bold;margin-bottom:0;}
.sub-title {text-align:center;color:#EEEEEE;font-size:20px;margin-top:0;}
.card {background-color:#1E1E1E; padding:20px; border-radius:15px; color:#EEE; box-shadow: 0 4px 15px rgba(0,0,0,0.3);}
.metric-box {background-color:#292929; padding:15px; border-radius:15px; text-align:center; color:#EEE; transition: transform 0.2s; box-shadow: 0 3px 12px rgba(0,0,0,0.3);}
.metric-box:hover {transform: scale(1.05);}
.badge {display:inline-block; padding:4px 8px; margin:2px; border-radius:10px; font-size:13px; color:#fff;}
.badge-matched {background-color:#00ADB5;}
.badge-missing {background-color:#FF5722;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>AI Resume Analyzer 🧠</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Powered by Llama 3.3 & BERT — Smart Resume Insights & Skill Recommendations</p>", unsafe_allow_html=True)
st.write(" ")

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True) if os.path.dirname(DB_PATH) else None
    return sqlite3.connect(DB_PATH)

def create_tables():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            resume_text TEXT,
            job_desc TEXT
        )
        """)
        conn.commit()

def save_resume(name: str, resume_text: str, job_desc: str):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO resumes (name, resume_text, job_desc) VALUES (?, ?, ?)",
            (name, resume_text, job_desc)
        )
        conn.commit()

create_tables()

def extract_pdf_text(uploaded_file) -> str:
    if uploaded_file is None:
        return "No file uploaded."
    try:
        uploaded_file.seek(0)
        text = extract_text(uploaded_file)
        return text.strip() if text and text.strip() else "No text detected in PDF."
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return "Could not extract text from PDF."

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

def calculate_similarity_bert(text1: str, text2: str) -> float:
    model = load_model()
    if not model:
        return 0.0
    try:
        emb1 = model.encode([text1 or ""], normalize_embeddings=True)
        emb2 = model.encode([text2 or ""], normalize_embeddings=True)
        sim = cosine_similarity(emb1, emb2)[0][0]
        return max(0.0, min(1.0, float(sim)))
    except Exception as e:
        st.error(f"Similarity calculation failed: {e}")
        return 0.0

def _groq_client():
    if not API_KEY:
        st.warning("GROQ_API_KEY is not set. Please configure it in your .env file.")
        return None
    try:
        return Groq(api_key=API_KEY)
    except Exception as e:
        st.error(f"Error initializing Groq client: {e}")
        return None

def get_report(resume: str, job_desc: str) -> str:
    client = _groq_client()
    if client is None:
        return "❌ Error: Missing or invalid GROQ_API_KEY. Set it in your .env file."
    prompt = f"""
# Context:
You are an AI Resume Analyzer. You will be given a candidate's resume and job description.
# Instructions:
- Analyze based on job requirements, experience, and skills.
- Give a score (out of 5) per point with emoji (✅, ❌, ⚠) and explanation.
- Provide "Suggestions to improve your resume" section.
- Include insights on gender tone (neutral, biased, or clear).
- Suggest relevant missing skills or keywords from the job description.
Resume: {resume}
---
Job Description: {job_desc}
    """.strip()
    try:
        res = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )
        return (res.choices[0].message.content or "").strip()
    except Exception as e:
        return f"❌ Error generating AI report: {e}"

def extract_scores(text: str):
    try:
        return [float(m) for m in re.findall(r'(\d+(?:\.\d+)?)/5\b', text)]
    except Exception:
        return []

def extract_skills(text: str):
    common_skills = [
        'python','java','c++','sql','excel','powerbi','ml','ai','communication',
        'leadership','teamwork','project management','javascript','react','django',
        'flask','tensorflow','pandas','data analysis','html','css','aws','git'
    ]
    found = set()
    for s in common_skills:
        if re.search(rf'\b{re.escape(s)}\b', text or "", re.IGNORECASE):
            found.add(s)
    return sorted(found)

def extract_keywords(text: str):
    try:
        words = nltk.word_tokenize(text or "")
        tagged = nltk.pos_tag(words)
        nouns = [w.lower() for w, pos in tagged if pos in {'NN','NNS','NNP'} and len(w) > 3]
        return [w for w, _ in Counter(nouns).most_common(10)]
    except Exception as e:
        st.warning(f"Keyword extraction warning: {e}")
        return []

def gender_analysis(resume_text: str):
    txt = resume_text or ""
    if re.search(r'\b(he|him|his)\b', txt, re.IGNORECASE):
        return "Male indicators detected 👨"
    if re.search(r'\b(she|her|hers)\b', txt, re.IGNORECASE):
        return "Female indicators detected 👩"
    return "Gender-neutral language detected ⚧"

def radar_skill_values(skills, resume_text: str):
    values = []
    txt = resume_text or ""
    for skill in skills[:5]:
        count = len(re.findall(rf'\b{re.escape(skill)}\b', txt, re.IGNORECASE))
        values.append(min(count, 5))
    return values

if not st.session_state.form_submitted:
    with st.form("upload_form"):
        st.markdown("### 📄 Upload Resume and Job Description")
        col1, col2 = st.columns(2)
        with col1:
            resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
        with col2:
            st.session_state.job_desc = st.text_area("Paste Job Description:", height=200, value=st.session_state.job_desc)
        submitted = st.form_submit_button("🚀 Analyze Resume")
        if submitted:
            if st.session_state.job_desc and resume_file:
                st.session_state.resume = extract_pdf_text(resume_file)
                st.session_state.form_submitted = True
                st.rerun()
            else:
                st.warning("Please upload both resume and job description!")

if st.session_state.form_submitted:
    st.markdown("---")
    with st.spinner("Analyzing Resume... ⏳"):
        ats_score = calculate_similarity_bert(st.session_state.resume, st.session_state.job_desc)
        report = get_report(st.session_state.resume, st.session_state.job_desc)
        scores = extract_scores(report)
        avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0
        skills = extract_skills(st.session_state.resume)
        jd_keywords = extract_keywords(st.session_state.job_desc)
        missing = [k for k in jd_keywords if k not in [s.lower() for s in skills]]
        try:
            save_resume("Candidate", st.session_state.resume, st.session_state.job_desc)
        except Exception as e:
            st.warning(f"Could not save to database: {e}")
        col1, col2, col3 = st.columns(3)
        col1.markdown(
            f"<div class='metric-box'>📝<br>ATS Similarity<br>{round(ats_score*100, 2)}%</div>",
            unsafe_allow_html=True
        )
        col2.markdown(
            f"<div class='metric-box'>🤖<br>AI Avg Score<br>{avg_score}/5</div>",
            unsafe_allow_html=True
        )
        total = len(skills) + len(missing)
        match_display = f"{len(skills)}/{total}" if total > 0 else "0/0"
        col3.markdown(
            f"<div class='metric-box'>💼<br>Skill Match<br>{match_display}</div>",
            unsafe_allow_html=True
        )
        with st.expander("🧾 AI Detailed Report", expanded=True):
            st.markdown(f"<div class='card'>{report}</div>", unsafe_allow_html=True)
        st.divider()
        st.subheader("📊 Skill Insights & Recommendations")
        st.markdown("*Matched Skills:*")
        st.markdown(
            " ".join([f"<span class='badge badge-matched'>{s.title()}</span>" for s in skills]) or "—",
            unsafe_allow_html=True
        )
        st.markdown("*Missing Keywords:*")
        st.markdown(
            " ".join([f"<span class='badge badge-missing'>{s.title()}</span>" for s in missing]) or "—",
            unsafe_allow_html=True
        )
        if skills:
            values = radar_skill_values(skills, st.session_state.resume)
            labels = [s.title() for s in skills[:5]]
            if values and labels:
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=values + values[:1],
                    theta=labels + labels[:1],
                    fill='toself',
                    name='Skill Strength',
                    line_color="#00ADB5",
                    fillcolor="rgba(0,173,181,0.3)"
                ))
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor="#1E1E1E",
                        radialaxis=dict(visible=True, range=[0, 5], tickfont=dict(color="#EEEEEE")),
                        angularaxis=dict(tickfont=dict(color="#EEEEEE"))
                    ),
                    showlegend=False,
                    paper_bgcolor="#121212",
                    font_color="#EEEEEE",
                    title="Skill Strength Radar Chart"
                )
                st.plotly_chart(fig_radar, use_container_width=True)
        st.download_button("📥 Download AI Report", data=report, file_name="AI_Resume_Report.txt")
