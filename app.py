import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import pickle
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import preprocess_pipeline
from src.features import CustomTfidfVectorizer
from src.evaluation import train_test_split_scratch, confusion_matrix_scratch, classification_report_scratch, accuracy_score_scratch, precision_score_scratch, recall_score_scratch, f1_score_scratch
from src.models.knn import KNNClassifier
from src.models.logistic_reg import LogisticRegressionClassifier
from src.models.random_forest import RandomForestClassifier
from src.models.neural_net import SimpleNeuralNetwork

# Setup page layout
st.set_page_config(
    page_title="Fake News Detector | Bauhaus ML",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Bauhaus + Neumorphism Design System CSS
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400..700;1,9..40,400..700&family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@500;700;800&display=swap" rel="stylesheet">

<style>
    :root {
        --bg-base: #1E1E2E;
        --bg-raised: #252535;
        --bg-subtle: #191926;
        --shadow-dark: #12121C;
        --shadow-light: #2D2D40;
        --accent-primary: #E63946;   /* Bauhaus Red */
        --accent-real: #2A9D8F;      /* Bauhaus Teal */
        --accent-fake: #E63946;      /* Bauhaus Red */
        --accent-yellow: #F4A261;    /* Bauhaus Ochre */
        --accent-navy: #1D3557;      /* Bauhaus Navy */
        --text-primary: #E8E8F0;
        --text-secondary: #9090A8;
        --grid-line: #2C2C40;
        --font-display: 'Space Grotesk', -apple-system, sans-serif;
        --font-body: 'DM Sans', -apple-system, sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
    }

    /* Core Application Surface */
    .stApp {
        background-color: var(--bg-base) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-body) !important;
    }

    /* Typography Overrides */
    h1, h2, h3, h4, h5, h6 {
        font-family: var(--font-display) !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: var(--text-primary) !important;
    }

    /* Neumorphic Utilities */
    .nm-raised {
        background: var(--bg-raised);
        box-shadow: 6px 6px 14px var(--shadow-dark),
                   -6px -6px 14px var(--shadow-light);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.03);
    }

    .nm-inset {
        background: var(--bg-subtle);
        box-shadow: inset 4px 4px 8px var(--shadow-dark),
                   inset -4px -4px 8px var(--shadow-light);
        border-radius: 10px;
    }

    .nm-flat-accent {
        background: var(--bg-raised);
        box-shadow: 5px 5px 12px var(--shadow-dark),
                   -5px -5px 12px var(--shadow-light);
        border-left: 4px solid var(--accent-primary);
        border-radius: 0 12px 12px 0;
    }

    /* Bauhaus Section Rule Header */
    .bauhaus-section {
        margin: 28px 0 16px 0;
        border-top: 1px solid var(--grid-line);
        padding-top: 14px;
    }
    .bauhaus-section-label {
        font-family: var(--font-mono);
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        color: var(--text-secondary);
        margin-bottom: 4px;
    }
    .bauhaus-section-title {
        font-family: var(--font-display);
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--text-primary);
        border-left: 4px solid var(--accent-primary);
        padding-left: 12px;
        line-height: 1.2;
    }

    /* Bauhaus Metric Card */
    .metric-card {
        background: var(--bg-raised);
        box-shadow: 6px 6px 14px var(--shadow-dark),
                   -6px -6px 14px var(--shadow-light);
        border-radius: 12px;
        padding: 20px 22px;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.02);
    }
    .metric-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--card-accent, var(--accent-primary));
    }
    .metric-value {
        font-family: var(--font-mono);
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.1;
        margin-top: 6px;
    }
    .metric-label {
        font-family: var(--font-body);
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: var(--text-secondary);
        margin-top: 6px;
    }

    /* Model Decision Cards (Bauhaus + Neumorphic) */
    .model-card-real {
        background: var(--bg-raised);
        box-shadow: 6px 6px 14px var(--shadow-dark),
                   -6px -6px 14px var(--shadow-light);
        border-left: 5px solid var(--accent-real);
        border-radius: 0 12px 12px 0;
        padding: 18px 20px;
        margin: 10px 0;
    }
    .model-card-fake {
        background: var(--bg-raised);
        box-shadow: 6px 6px 14px var(--shadow-dark),
                   -6px -6px 14px var(--shadow-light);
        border-left: 5px solid var(--accent-fake);
        border-radius: 0 12px 12px 0;
        padding: 18px 20px;
        margin: 10px 0;
    }
    .card-title {
        font-family: var(--font-mono);
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--text-secondary);
        margin-bottom: 6px;
    }

    /* Buttons (Neumorphic Tactile) */
    .stButton > button {
        background: var(--bg-raised) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--grid-line) !important;
        border-radius: 8px !important;
        box-shadow: 4px 4px 10px var(--shadow-dark),
                   -4px -4px 10px var(--shadow-light) !important;
        font-family: var(--font-display) !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        padding: 12px 24px !important;
        transition: all 0.15s ease-in-out !important;
    }
    .stButton > button:hover {
        color: var(--accent-primary) !important;
        border-color: var(--accent-primary) !important;
        box-shadow: 2px 2px 6px var(--shadow-dark),
                   -2px -2px 6px var(--shadow-light) !important;
        transform: translateY(-1px);
    }
    .stButton > button:active {
        box-shadow: inset 3px 3px 6px var(--shadow-dark),
                   inset -3px -3px 6px var(--shadow-light) !important;
        transform: translateY(1px);
    }

    /* Sidebar Bauhaus Discipline */
    section[data-testid="stSidebar"] {
        background: var(--bg-raised) !important;
        border-right: 3px solid var(--accent-primary) !important;
        box-shadow: 6px 0 20px rgba(0, 0, 0, 0.4) !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem !important;
    }

    /* Streamlit Tabs */
    button[data-baseweb="tab"] {
        font-family: var(--font-display) !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.12em !important;
        color: var(--text-secondary) !important;
        border: none !important;
        border-bottom: 3px solid transparent !important;
        padding: 12px 24px !important;
        background: transparent !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--text-primary) !important;
        border-bottom-color: var(--accent-primary) !important;
    }
    div[data-testid="stTabs"] > div:first-child {
        border-bottom: 1px solid var(--grid-line) !important;
    }

    /* Inputs: text area, text inputs */
    .stTextArea textarea, .stTextInput input {
        background: var(--bg-subtle) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--grid-line) !important;
        border-radius: 8px !important;
        box-shadow: inset 2px 2px 6px var(--shadow-dark),
                   inset -2px -2px 6px var(--shadow-light) !important;
        font-family: var(--font-body) !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: var(--accent-primary) !important;
        outline: none !important;
    }

    /* Dataframe Table Styling */
    div[data-testid="stDataFrame"] {
        background: var(--bg-raised) !important;
        border-radius: 10px !important;
        border: 1px solid var(--grid-line) !important;
        overflow: hidden !important;
    }

    /* Expander styling */
    div[data-testid="stExpander"] {
        background: var(--bg-raised) !important;
        border: 1px solid var(--grid-line) !important;
        border-radius: 8px !important;
        box-shadow: 3px 3px 8px var(--shadow-dark) !important;
    }

    /* Radio / Selection options */
    div[role="radiogroup"] {
        font-family: var(--font-display) !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function for structured Bauhaus section headings
def section_heading(title: str, label: str = "", accent: str = "#E63946"):
    label_html = f"<div class='bauhaus-section-label'>{label}</div>" if label else ""
    return f"""
    <div class='bauhaus-section'>
        {label_html}
        <div class='bauhaus-section-title' style='border-left-color: {accent};'>{title}</div>
    </div>
    """

# Standard Matplotlib Bauhaus theme
BAUHAUS_CHART_STYLE = {
    "figure.facecolor": "#1E1E2E",
    "axes.facecolor": "#252535",
    "axes.edgecolor": "#2C2C40",
    "axes.labelcolor": "#9090A8",
    "text.color": "#E8E8F0",
    "xtick.color": "#9090A8",
    "ytick.color": "#9090A8",
    "grid.color": "#2C2C40",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "font.family": "sans-serif",
}

# Helper functions for data loading and model training
@st.cache_data(show_spinner=False)
def get_dataset_statistics():
    """
    Loads raw CSV files to get exact lengths and descriptions without caching the whole split.
    """
    true_path = os.path.join("dataset", "True.csv")
    fake_path = os.path.join("dataset", "Fake.csv")
    if not os.path.exists(true_path) or not os.path.exists(fake_path):
        return 0, 0
    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)
    return len(df_true), len(df_fake)

def append_stylistic_features(X_tfidf, raw_texts, stylistic_max=None):
    """
    Appends normalized exclamation mark and uppercase density columns to the TF-IDF matrix.
    Scales features to [0, 1] to match L2-normalized TF-IDF weights without dominating them.
    """
    excl = np.array([t.count('!') / (len(t) + 1) for t in raw_texts], dtype=np.float32).reshape(-1, 1)
    upper = np.array([sum(1 for c in t if c.isupper()) / (len(t) + 1) for t in raw_texts], dtype=np.float32).reshape(-1, 1)
    if stylistic_max is None:
        max_excl, max_upper = 0.05, 0.30
    else:
        max_excl, max_upper = stylistic_max
    excl_norm = np.clip(excl / max(max_excl, 1e-6), 0.0, 1.0)
    upper_norm = np.clip(upper / max(max_upper, 1e-6), 0.0, 1.0)
    return np.hstack((X_tfidf, excl_norm, upper_norm))

def transform_for_prediction(raw_texts, vectorizer):
    """
    Unified feature extraction pipeline for inference so train and predict features match identically.
    """
    clean_texts = [preprocess_pipeline(t) for t in raw_texts]
    X_tfidf = vectorizer.transform(clean_texts)
    stylistic_max = getattr(vectorizer, "stylistic_max_", None)
    return append_stylistic_features(X_tfidf, raw_texts, stylistic_max=stylistic_max)

def generate_interpretability_html(raw_text, vectorizer, log_reg):
    import re
    import html
    from src.preprocessing import STOPWORDS
    
    # Tokenize preserving spaces, tabs, and newlines exactly
    tokens = re.split(r'(\s+)', raw_text)
    
    vocab = vectorizer.vocabulary_
    weights = log_reg.weights
    
    highlighted_tokens = []
    threshold = 0.01  # Minimum weight to trigger highlight
    
    for token in tokens:
        if not token.strip():
            # Whitespace/newline - preserve as-is
            highlighted_tokens.append(html.escape(token))
            continue
            
        # Clean word to match vocabulary preprocessing logic
        cleaned = token.lower()
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', cleaned)
        
        if cleaned and cleaned not in STOPWORDS and cleaned in vocab:
            idx = vocab[cleaned]
            weight = weights[idx, 0]
            
            # Format numbers safely
            escaped_token = html.escape(token)
            if weight > threshold:
                # Fake association (Red shading)
                highlighted_tokens.append(
                    f"<span style='background-color: rgba(239, 68, 68, 0.18); color: #F87171; border-radius: 4px; padding: 1px 3px; font-weight: 600; cursor: help;' title='Weight: {weight:+.4f} (Fake News factor)'>{escaped_token}</span>"
                )
            elif weight < -threshold:
                # Real association (Green shading)
                highlighted_tokens.append(
                    f"<span style='background-color: rgba(16, 185, 129, 0.18); color: #34D399; border-radius: 4px; padding: 1px 3px; font-weight: 600; cursor: help;' title='Weight: {weight:+.4f} (Real News factor)'>{escaped_token}</span>"
                )
            else:
                highlighted_tokens.append(escaped_token)
        else:
            highlighted_tokens.append(html.escape(token))
            
    return "".join(highlighted_tokens)

@st.cache_data(show_spinner=False)
def load_and_preprocess_subset(sample_size, vocab_size):
    """
    Loads, samples, and preprocesses a balanced subset of news.
    Appends accumulated user feedback from user_feedback.csv to ensure
    all models (including Random Forest) train on past corrections.
    """
    true_path = os.path.join("dataset", "True.csv")
    fake_path = os.path.join("dataset", "Fake.csv")
    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)
    
    half_sample = sample_size // 2
    
    # Balanced sample
    df_t = df_true.sample(min(half_sample, len(df_true)), random_state=42)
    df_f = df_fake.sample(min(half_sample, len(df_fake)), random_state=42)
    
    df_t['label'] = 0
    df_f['label'] = 1
    
    df_all_list = [df_t, df_f]
    
    # Load feedback corrections if available
    feedback_file = os.path.join("dataset", "user_feedback.csv")
    if os.path.exists(feedback_file):
        try:
            df_fb = pd.read_csv(feedback_file)
            if not df_fb.empty and "raw_text" in df_fb.columns and "submitted_truth" in df_fb.columns:
                # Map column names to match main dataframe schema
                df_fb_subset = pd.DataFrame({
                    "text": df_fb["raw_text"].fillna(""),
                    "label": df_fb["submitted_truth"].astype(int)
                })
                df_all_list.append(df_fb_subset)
        except Exception:
            pass
            
    df_all = pd.concat(df_all_list, ignore_index=True)
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Preprocess (merge title + text so headline signals are included)
    titles = df_all['title'].fillna("") if 'title' in df_all.columns else pd.Series([""] * len(df_all))
    bodies = df_all['text'].fillna("") if 'text' in df_all.columns else pd.Series([""] * len(df_all))
    texts = (titles + " " + bodies).str.strip().values
    labels = df_all['label'].values
    
    clean_texts = [preprocess_pipeline(t) for t in texts]
    
    # Custom Vectorizer
    vectorizer = CustomTfidfVectorizer(max_features=vocab_size)
    X_tfidf = vectorizer.fit_transform(clean_texts)
    
    # Compute 99th-percentile stylistic bounds for [0, 1] normalization
    excl_vals = np.array([t.count('!') / (len(t) + 1) for t in texts], dtype=np.float32)
    upper_vals = np.array([sum(1 for c in t if c.isupper()) / (len(t) + 1) for t in texts], dtype=np.float32)
    vectorizer.stylistic_max_ = (
        float(max(np.percentile(excl_vals, 99), 0.01)),
        float(max(np.percentile(upper_vals, 99), 0.05))
    )
    X = append_stylistic_features(X_tfidf, texts, stylistic_max=vectorizer.stylistic_max_)
    
    return X, labels, vectorizer, clean_texts, df_all


SAVED_MODELS_DIR = "saved_models"

def save_models_to_disk(vectorizer, knn, log_reg, rf, nn, metrics):
    if not os.path.exists(SAVED_MODELS_DIR):
        os.makedirs(SAVED_MODELS_DIR)
    
    with open(os.path.join(SAVED_MODELS_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(SAVED_MODELS_DIR, "knn.pkl"), "wb") as f:
        pickle.dump(knn, f)
    with open(os.path.join(SAVED_MODELS_DIR, "log_reg.pkl"), "wb") as f:
        pickle.dump(log_reg, f)
    with open(os.path.join(SAVED_MODELS_DIR, "rf.pkl"), "wb") as f:
        pickle.dump(rf, f)
    with open(os.path.join(SAVED_MODELS_DIR, "nn.pkl"), "wb") as f:
        pickle.dump(nn, f)
    with open(os.path.join(SAVED_MODELS_DIR, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)

def load_models_from_disk():
    required_files = ["vectorizer.pkl", "knn.pkl", "log_reg.pkl", "rf.pkl", "nn.pkl", "metrics.pkl"]
    for fname in required_files:
        if not os.path.exists(os.path.join(SAVED_MODELS_DIR, fname)):
            return None
            
    try:
        with open(os.path.join(SAVED_MODELS_DIR, "vectorizer.pkl"), "rb") as f:
            vectorizer = pickle.load(f)
        with open(os.path.join(SAVED_MODELS_DIR, "knn.pkl"), "rb") as f:
            knn = pickle.load(f)
        with open(os.path.join(SAVED_MODELS_DIR, "log_reg.pkl"), "rb") as f:
            log_reg = pickle.load(f)
        with open(os.path.join(SAVED_MODELS_DIR, "rf.pkl"), "rb") as f:
            rf = pickle.load(f)
        with open(os.path.join(SAVED_MODELS_DIR, "nn.pkl"), "rb") as f:
            nn = pickle.load(f)
        with open(os.path.join(SAVED_MODELS_DIR, "metrics.pkl"), "rb") as f:
            metrics = pickle.load(f)
            
        return vectorizer, knn, log_reg, rf, nn, metrics
    except Exception as e:
        return None

def fetch_live_news(query, api_key):
    # Enforce strict search: prefix each word with '+' so NewsAPI requires all terms to be present
    words = [w.strip() for w in query.strip().split() if w.strip()]
    strict_query = " ".join([f"+{w}" for w in words]) if words else query
    
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": strict_query,
        "apiKey": api_key,
        "language": "en",
        "pageSize": 5
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json().get("articles", [])
        else:
            return None
    except Exception as e:
        return None

def fetch_newsdata_io(query, api_key):
    # Enforce strict search: join terms with 'AND'
    words = [w.strip() for w in query.strip().split() if w.strip()]
    strict_query = " AND ".join(words) if words else query
    
    url = "https://newsdata.io/api/1/news"
    params = {
        "apikey": api_key,
        "q": strict_query,
        "language": "en"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json().get("results", [])[:5]
        else:
            return None
    except Exception as e:
        return None

def load_env_api_key(key_name):
    # Support common naming variations / aliases
    aliases = [key_name]
    if "NEWS_API" in key_name or "NEWSAPI" in key_name:
        aliases += ["NEWS_API_KEY", "NEWSAPI_KEY", "NEWS_API", "NEWSAPI", "NEWS_KEY", "NEWSAPIKEY"]
    elif "NEWSDATA" in key_name or "NEWS_DATA" in key_name:
        aliases += ["NEWSDATA_API_KEY", "NEWSDATA_KEY", "NEWS_DATA_API_KEY", "NEWS_DATA_KEY", "NEWSDATA", "NEWSDATA_API"]
    elif "GEMINI" in key_name or "GOOGLE" in key_name:
        aliases += ["GEMINI_API_KEY", "GEMINI_KEY", "GOOGLE_API_KEY", "GEMINI_API", "GEMINI"]

    # 1. Streamlit Secrets (Cloud Deployment & .streamlit/secrets.toml)
    try:
        for alias in aliases:
            # Check direct & case-insensitive keys
            for sec_k, sec_v in st.secrets.items():
                if sec_k.lower() == alias.lower():
                    val = str(sec_v).strip().strip('"\'')
                    if val:
                        return val
                elif isinstance(sec_v, dict):
                    # Check inside nested sections like [general], [api_keys], etc.
                    for sub_k, sub_v in sec_v.items():
                        if sub_k.lower() == alias.lower():
                            val = str(sub_v).strip().strip('"\'')
                            if val:
                                return val
    except Exception:
        pass

    # 2. Environment Variables
    for alias in aliases:
        value = os.getenv(alias) or os.getenv(alias.lower()) or os.getenv(alias.upper())
        if value:
            val = value.strip().strip('"\'')
            if val:
                return val

    # 3. Local .env file
    if os.path.exists(".env"):
        try:
            with open(".env", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        k = parts[0].strip()
                        for alias in aliases:
                            if k.lower() == alias.lower():
                                val = parts[1].strip().strip('"\'')
                                if val:
                                    return val
        except Exception:
            pass

    return ""

def extract_query_with_gemini(text):
    """Use Gemini API to extract the best search keywords from a news article/headline."""
    gemini_key = load_env_api_key("GEMINI_API_KEY")
    if not gemini_key or not text.strip():
        return _fallback_query(text)
    
    prompt = (
        "Extract a specific news search query from the article below.\n\n"
        "Rules:\n"
        "- Return 4 to 6 keywords that describe the SPECIFIC event or claim in the article.\n"
        "- NEVER return just a person's name alone (e.g. 'Trump' or 'Biden'). Always include WHAT happened.\n"
        "- Include key subjects, actions, and objects (e.g. 'Trump Mars space colony plan').\n"
        "- Remove source names (Reuters, AP), datelines, and filler words.\n"
        "- Return ONLY the search query. No quotes, no explanation, no numbering.\n\n"
        "Examples:\n"
        "Article: 'Trump announces plan to build colony on Mars' → Trump Mars colony plan\n"
        "Article: 'Pope Francis endorses Donald Trump for President' → Pope Francis endorses Trump President\n"
        "Article: 'India launches new space mission to study the Sun' → India space mission Sun study\n\n"
        f"Article:\n{text[:600]}"
    )
    
    # Try multiple Gemini models in case one has quota
    models = ["gemini-2.5-flash", "gemini-flash-lite-latest"]
    
    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 300}
        }
        
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    if parts and "text" in parts[0]:
                        query = parts[0]["text"].strip()
                        query = query.strip('"').strip("'").strip()
                        # Reject responses with fewer than 3 words — too generic for meaningful search
                        if query and len(query.split()) >= 3:
                            return query
            else:
                with open("debug.log", "a") as f:
                    f.write(f"Gemini extract_query returned non-200: {resp.status_code}, body: {resp.text}\n")
        except Exception as e:
            with open("debug.log", "a") as f:
                f.write(f"Gemini extract_query exception: {str(e)}\n")
            continue
    
    return _fallback_query(text)

def gemini_fact_check(text, search_query="", coverage_text=""):
    """Ask Gemini to analyze a news article, compare it with retrieved search summaries, 
    and determine if it is likely real or fake.
    Uses headline, extracted keywords, and retrieved search text to save API tokens."""
    gemini_key = load_env_api_key("GEMINI_API_KEY")
    if not gemini_key or not text.strip():
        return None
    
    # Build a compact summary: first sentence + keywords (saves ~70% tokens vs full article)
    first_line = text.strip().split('\n')[0].strip()
    headline = first_line[:200]
    
    prompt = (
        "You are an expert fact-checker. Determine if this news is REAL or FAKE.\n"
        "Evaluate writing style, source credibility, and compare it against the retrieved search results summaries below.\n\n"
        "Respond in EXACTLY this format (3 lines):\n"
        "VERDICT: FAKE or REAL\n"
        "CONFIDENCE: number from 1 to 100\n"
        "REASONING: 2-3 sentence explanation\n\n"
        f"Headline: {headline}\n"
        f"Topic keywords: {search_query}\n"
        f"Retrieved Search Coverage:\n{coverage_text}\n"
    )
    
    models = ["gemini-2.5-flash", "gemini-flash-lite-latest"]
    
    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 800}
        }
        
        try:
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    candidate = candidates[0]
                    content = candidate.get("content", {})
                    parts = content.get("parts", [])
                    if parts and "text" in parts[0]:
                        response_text = parts[0]["text"].strip()
                        
                        # Parse the structured response
                        result = {"verdict": "UNKNOWN", "confidence": 50, "reasoning": "Unable to analyze."}
                        
                        for line in response_text.split('\n'):
                            line = line.strip()
                            if line.upper().startswith("VERDICT:"):
                                v = line.split(":", 1)[1].strip().upper()
                                if "FAKE" in v:
                                    result["verdict"] = "FAKE"
                                elif "REAL" in v:
                                    result["verdict"] = "REAL"
                                else:
                                    result["verdict"] = "UNKNOWN"
                            elif line.upper().startswith("CONFIDENCE:"):
                                try:
                                    result["confidence"] = int(''.join(filter(str.isdigit, line.split(":", 1)[1].strip()))[:3])
                                except (ValueError, IndexError):
                                    pass
                            elif line.upper().startswith("REASONING:"):
                                result["reasoning"] = line.split(":", 1)[1].strip()
                        
                        return result
                    else:
                        finish_reason = candidate.get("finishReason", "UNKNOWN")
                        with open("debug.log", "a") as f:
                            f.write(f"Gemini candidate content empty. finishReason: {finish_reason}\n")
                        if finish_reason == "SAFETY":
                            return {
                                "verdict": "UNKNOWN",
                                "confidence": 50,
                                "reasoning": "Gemini analysis was blocked by safety filters due to sensitive content."
                            }
                else:
                    with open("debug.log", "a") as f:
                        f.write("Gemini returned empty candidates list.\n")
            else:
                with open("debug.log", "a") as f:
                    f.write(f"Gemini API returned non-200 status code: {resp.status_code}, body: {resp.text}\n")
        except Exception as e:
            import traceback
            with open("debug.log", "a") as f:
                f.write(f"Gemini request exception for model {model}: {str(e)}\n{traceback.format_exc()}\n")
            continue
            
    return None

def _fallback_query(text):
    """Smart fallback: strip news prefixes and extract meaningful keywords without AI."""
    if not text.strip():
        return "politics"
    
    # Take first line / first sentence (handle abbreviations like U.S., Dr.)
    first_line = text.strip().split('\n')[0].strip()
    # Only split by period if it looks like a real sentence boundary (followed by space + uppercase)
    import re
    sentences = re.split(r'(?<!\b[A-Z])\.(?=\s+[A-Z])', first_line)
    first_sentence = sentences[0].strip() if sentences else first_line
    
    # Strip common dateline prefixes like "WASHINGTON (Reuters) - " 
    if ' - ' in first_sentence:
        parts = first_sentence.split(' - ', 1)
        # Check if the part before dash looks like a dateline (e.g. WASHINGTON, LONDON (AP), NEW YORK (Reuters))
        before_dash = parts[0].strip()
        # Remove parenthesized source like (Reuters), (AP)
        import re
        dateline_part = re.sub(r'\([^)]*\)', '', before_dash).strip()
        if dateline_part and dateline_part.replace(' ', '').isupper():
            first_sentence = parts[1]
    
    # Strip "BREAKING:" type prefixes
    prefixes_to_remove = ['BREAKING:', 'BREAKING NEWS:', 'UPDATE:', 'FLASH:', 'EXCLUSIVE:', 'OPINION:']
    upper_sentence = first_sentence.upper()
    for prefix in prefixes_to_remove:
        if upper_sentence.startswith(prefix):
            first_sentence = first_sentence[len(prefix):].strip()
            break
    
    # Remove common stopwords and take first 5 meaningful words
    stop = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'has', 'have', 'had', 'in', 'on', 'at', 
            'to', 'for', 'of', 'by', 'with', 'from', 'and', 'or', 'but', 'that', 'this', 'it',
            'its', 'his', 'her', 'their', 'our', 'your', 'today', 'said', 'says', 'say'}
    words = first_sentence.split()
    meaningful = [w.strip('.,!?:;\'"()') for w in words if w.lower().strip('.,!?:;\'"()') not in stop and len(w) > 1]
    
    if meaningful:
        return " ".join(meaningful[:5])
    return "politics"

# State Initialization (Must run before sidebar is drawn)
if "trained" not in st.session_state:
    loaded = load_models_from_disk()
    if loaded is not None:
        vectorizer_disk, knn_disk, log_reg_disk, rf_disk, nn_disk, metrics_disk = loaded
        st.session_state.vectorizer = vectorizer_disk
        st.session_state.knn = knn_disk
        st.session_state.log_reg = log_reg_disk
        st.session_state.rf = rf_disk
        st.session_state.nn = nn_disk
        st.session_state.metrics = metrics_disk
        st.session_state.trained = True
    else:
        st.session_state.trained = False

# Page Header
st.markdown("""
<div style="padding: 12px 0 20px 0; border-bottom: 2px solid var(--grid-line); margin-bottom: 24px;">
    <div style="font-family: var(--font-mono); font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.2em; color: var(--accent-primary); margin-bottom: 6px;">
        LABORATORY SYSTEM // FROM-SCRATCH ML
    </div>
    <h1 style="font-family: var(--font-display); font-size: 2.8rem; font-weight: 800; color: var(--text-primary); margin: 0; line-height: 1.05; letter-spacing: -0.02em;">
        Fake News Detector<span style="color: var(--accent-primary);">.</span>
    </h1>
    <div style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-secondary); margin-top: 8px; letter-spacing: 0.06em;">
        LOGISTIC REGRESSION · MULTI-LAYER PERCEPTRON · RANDOM FOREST · K-NEAREST NEIGHBORS
    </div>
</div>
""", unsafe_allow_html=True)

# Check for files
true_path = os.path.join("dataset", "True.csv")
fake_path = os.path.join("dataset", "Fake.csv")
if not os.path.exists(true_path) or not os.path.exists(fake_path):
    st.error("⚠️ Dataset files (True.csv and Fake.csv) not found in the 'dataset' folder. Please verify the environment.")
    st.stop()

# Sidebar: Hyperparameters and Settings
st.sidebar.markdown("""
<div style="margin-bottom: 14px;">
    <div class="bauhaus-section-label">SYSTEM CONFIGURATION</div>
    <h3 style="margin: 0; font-size: 1.3rem;">Parameters</h3>
</div>
""", unsafe_allow_html=True)

sample_size = st.sidebar.slider("Total Samples (Balanced)", 200, 5000, 2000, step=200, 
                                help="Number of real & fake news articles to use. KNN & RF splits can be slow on larger sample counts.")
vocab_size = st.sidebar.slider("Vocabulary Size (Max Features)", 100, 2500, 1000, step=100,
                               help="Maximum distinct words in the TF-IDF feature space.")
test_split = st.sidebar.slider("Test Split Ratio", 0.1, 0.5, 0.2, step=0.05)

st.sidebar.markdown("<div style='border-top: 1px solid var(--grid-line); margin: 18px 0 14px 0;'></div>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style="margin-bottom: 12px;">
    <div class="bauhaus-section-label">ALGORITHM TUNING</div>
    <h3 style="margin: 0; font-size: 1.15rem;">Hyperparameters</h3>
</div>
""", unsafe_allow_html=True)

# KNN Params
with st.sidebar.expander("K-Nearest Neighbors Parameters"):
    k_neighbors = st.slider("Neighbors (k)", 1, 15, 5, step=2)
    knn_metric = st.selectbox("Distance Metric", ["cosine", "euclidean"])

# Logistic Reg Params
with st.sidebar.expander("Logistic Regression Parameters"):
    log_lr = st.slider("Learning Rate (LR)", 0.01, 1.0, 0.1, step=0.05)
    log_epochs = st.slider("Training Epochs ", 10, 500, 150, step=10)
    log_lambda = st.slider("L2 Penalty (Lambda)", 0.0, 0.5, 0.01, step=0.01)

# Random Forest Params
with st.sidebar.expander("Random Forest Parameters"):
    rf_trees = st.slider("Estimators (Trees)", 2, 30, 10, step=2)
    rf_depth = st.slider("Max Tree Depth", 2, 15, 8, step=1)

# Simple Neural Net Params
with st.sidebar.expander("Simple Neural Net Parameters"):
    nn_hidden = st.slider("Hidden Layer Dimension", 8, 128, 64, step=8)
    nn_lr = st.slider("Learning Rate (NN)", 0.001, 0.5, 0.05, step=0.005)
    nn_epochs = st.slider("NN Epochs", 10, 300, 100, step=10)
    nn_batch = st.select_slider("Batch Size", options=[8, 16, 32, 64, 128], value=32)

# Validate trained model feature shape integrity
is_compatible = True
if st.session_state.trained:
    trained_vocab_size = len(st.session_state.vectorizer.vocabulary_)
    expected_features = trained_vocab_size + 2
    
    # Check weight dimensions
    if hasattr(st.session_state.log_reg, "weights") and st.session_state.log_reg.weights is not None:
        if st.session_state.log_reg.weights.shape[0] != expected_features:
            is_compatible = False

st.sidebar.markdown("<div style='border-top: 1px solid var(--grid-line); margin: 18px 0 14px 0;'></div>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style="margin-bottom: 8px;">
    <div class="bauhaus-section-label">MODEL PERSISTENCE</div>
    <h3 style="margin: 0; font-size: 1.15rem;">Memory Status</h3>
</div>
""", unsafe_allow_html=True)

if st.session_state.trained and is_compatible:
    st.sidebar.success("Ready · Weights Loaded")
else:
    if st.session_state.trained and not is_compatible:
        st.sidebar.warning("Retrain Required · Dimension Mismatch")
    else:
        st.sidebar.warning("Retrain Required · No Model")

st.sidebar.markdown("""
<div style="margin: 14px 0 8px 0;">
    <div class="bauhaus-section-label">CLOUD INFRASTRUCTURE</div>
    <h3 style="margin: 0; font-size: 1.15rem;">Live API Status</h3>
</div>
""", unsafe_allow_html=True)

gemini_detected = bool(load_env_api_key("GEMINI_API_KEY"))
news_detected = bool(load_env_api_key("NEWS_API_KEY"))
newsdata_detected = bool(load_env_api_key("NEWSDATA_API_KEY"))

if gemini_detected and (news_detected or newsdata_detected):
    st.sidebar.success("Fact-Checking APIs Active")
elif gemini_detected or news_detected or newsdata_detected:
    st.sidebar.warning("Partial APIs Connected")
else:
    st.sidebar.info("Offline Mode (Local Models Only)")

st.sidebar.markdown("<div style='border-top: 1px solid var(--grid-line); margin: 18px 0 16px 0;'></div>", unsafe_allow_html=True)
train_trigger = st.sidebar.button("Train All Models", use_container_width=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["DATASET & EDA", "MODEL BENCHMARK", "LIVE VERIFIER"])

# State Initialization completed above.

# TAB 1: DATASET & EDA
with tab1:
    st.markdown(section_heading("Dataset Overview & Statistics", label="DATA CORPUS", accent="#E63946"), unsafe_allow_html=True)
    
    n_real, n_fake = get_dataset_statistics()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class='metric-card' style='--card-accent: #2A9D8F;'>
            <div class='metric-value' style='color:#2A9D8F;'>{n_real:,}</div>
            <div class='metric-label'>Real News Articles</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-card' style='--card-accent: #E63946;'>
            <div class='metric-value' style='color:#E63946;'>{n_fake:,}</div>
            <div class='metric-label'>Fake News Articles</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='metric-card' style='--card-accent: #F4A261;'>
            <div class='metric-value' style='color:#F4A261;'>{n_real + n_fake:,}</div>
            <div class='metric-label'>Total Corpus Size</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(section_heading("Exploratory Word Count Analysis", label="EXPLORATORY DATA ANALYSIS", accent="#2A9D8F"), unsafe_allow_html=True)
    
    # Load a small snippet for EDA preview to keep it fast
    X_eda, y_eda, vectorizer_eda, clean_texts_eda, df_eda = load_and_preprocess_subset(1000, 500)
    
    # Compute text word counts
    df_eda['word_count'] = df_eda['text'].apply(lambda x: len(str(x).split()))
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("<div class='bauhaus-section-label'>DISTRIBUTION</div><h4 style='margin:0 0 10px 0;'>Article Length (Word Count)</h4>", unsafe_allow_html=True)
        with plt.rc_context(BAUHAUS_CHART_STYLE):
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(data=df_eda, x='word_count', hue='label', kde=True, bins=30, palette={0: '#2A9D8F', 1: '#E63946'}, ax=ax, alpha=0.65)
            ax.set_xlabel("Word Count", color='#E8E8F0')
            ax.set_ylabel("Count", color='#E8E8F0')
            ax.set_title("Length Distribution: Real (Teal) vs Fake (Red)", color='#E8E8F0', fontweight="bold")
            ax.tick_params(colors='#9090A8')
            for spine in ax.spines.values():
                spine.set_color('#2C2C40')
            legend = ax.get_legend()
            if legend:
                legend.get_texts()[0].set_text('Real News')
                legend.get_texts()[1].set_text('Fake News')
                legend.get_frame().set_facecolor('#252535')
                legend.get_frame().set_edgecolor('#2C2C40')
                for text in legend.get_texts():
                    text.set_color('#E8E8F0')
            st.pyplot(fig)
        
    with col_chart2:
        st.markdown("<div class='bauhaus-section-label'>VOCABULARY WEIGHTS</div><h4 style='margin:0 0 10px 0;'>Top 10 High-Weight Terms</h4>", unsafe_allow_html=True)
        vocab = vectorizer_eda.feature_names_
        # Sum columns of word counts (excluding appended stylistic features)
        word_frequencies = np.sum(X_eda[:, :len(vocab)], axis=0)
        
        freq_df = pd.DataFrame({'word': vocab, 'tf_idf_weight': word_frequencies}).sort_values(by='tf_idf_weight', ascending=False).head(10)
        
        with plt.rc_context(BAUHAUS_CHART_STYLE):
            fig, ax = plt.subplots(figsize=(10, 5))
            palette_colors = ["#E63946", "#2A9D8F", "#F4A261", "#457B9D"] * 3
            sns.barplot(x='tf_idf_weight', y='word', data=freq_df, palette=palette_colors[:len(freq_df)], ax=ax)
            ax.set_xlabel("Cumulative TF-IDF Score", color='#E8E8F0')
            ax.set_ylabel("Terms", color='#E8E8F0')
            ax.set_title("Top 10 Distinguishing Vocabulary Terms", color='#E8E8F0', fontweight="bold")
            ax.tick_params(colors='#9090A8')
            for spine in ax.spines.values():
                spine.set_color('#2C2C40')
            st.pyplot(fig)

    st.markdown("""
    > [!NOTE]
    > **Structural Distribution Variance:** Notice how **Fake News** and **Real News** display distinctly different word length distributions. 
    > Fabricated reporting is often shorter and more sensationalist, whereas authentic reports present balanced paragraphs. The TF-IDF weights capture the distinguishing vocabulary features.
    """)

# TAB 2: MODEL COMPARISON & TRAINING
with tab2:
    if train_trigger or st.session_state.trained:
        if train_trigger:
            st.cache_data.clear()  # Clear streamlit cache to invalidate old shapes
            with st.spinner("Loading dataset, cleaning text, and building TF-IDF vectors from scratch..."):
                X, y, vectorizer, clean_texts, df_sampled = load_and_preprocess_subset(sample_size, vocab_size)
                
                # Split
                X_train, X_test, y_train, y_test = train_test_split_scratch(X, y, test_size=test_split, random_state=42)
                
                # KNN
                knn = KNNClassifier(k=k_neighbors, metric=knn_metric)
                # Logistic Reg
                log_reg = LogisticRegressionClassifier(lr=log_lr, epochs=log_epochs, lambda_reg=log_lambda, verbose=False)
                # RF
                rf = RandomForestClassifier(n_estimators=rf_trees, max_depth=rf_depth)
                # Neural Net
                nn = SimpleNeuralNetwork(hidden_dim=nn_hidden, lr=nn_lr, epochs=nn_epochs, batch_size=nn_batch, verbose=False)
                
                # Training operations & time tracking
                # 1. KNN
                t0 = time.time()
                knn.fit(X_train, y_train)
                t_knn = time.time() - t0
                
                # 2. Logistic Reg
                t0 = time.time()
                log_reg.fit(X_train, y_train)
                t_log = time.time() - t0
                
                # 3. Random Forest
                t0 = time.time()
                rf.fit(X_train, y_train)
                t_rf = time.time() - t0
                
                # 4. Neural Net
                t0 = time.time()
                nn.fit(X_train, y_train)
                t_nn = time.time() - t0
                
                # Train and Test evaluations
                train_preds_knn = knn.predict(X_train)
                train_preds_log = log_reg.predict(X_train)
                train_preds_rf = rf.predict(X_train)
                train_preds_nn = nn.predict(X_train)

                preds_knn = knn.predict(X_test)
                preds_log = log_reg.predict(X_test)
                preds_rf = rf.predict(X_test)
                preds_nn = nn.predict(X_test)
                
                # Compute metrics
                metrics = {}
                for name, preds, train_preds, t_train, model in [
                    ("KNN", preds_knn, train_preds_knn, t_knn, knn),
                    ("Logistic Regression", preds_log, train_preds_log, t_log, log_reg),
                    ("Random Forest", preds_rf, train_preds_rf, t_rf, rf),
                    ("Simple Neural Network", preds_nn, train_preds_nn, t_nn, nn)
                ]:
                    acc_train = accuracy_score_scratch(y_train, train_preds)
                    acc_test = accuracy_score_scratch(y_test, preds)
                    prec = precision_score_scratch(y_test, preds)
                    rec = recall_score_scratch(y_test, preds)
                    f1 = f1_score_scratch(y_test, preds)
                    tp, fp, tn, fn, mat = confusion_matrix_scratch(y_test, preds)
                    
                    metrics[name] = {
                        'train_accuracy': acc_train,
                        'accuracy': acc_test,
                        'precision': prec,
                        'recall': rec,
                        'f1': f1,
                        'train_time': t_train,
                        'confusion_matrix': mat,
                        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
                    }
                
                # Save to session state
                st.session_state.vectorizer = vectorizer
                st.session_state.knn = knn
                st.session_state.log_reg = log_reg
                st.session_state.rf = rf
                st.session_state.nn = nn
                st.session_state.metrics = metrics
                st.session_state.trained = True
                
                # Save to disk for persistence across restarts
                save_models_to_disk(vectorizer, knn, log_reg, rf, nn, metrics)
                
                # Force immediate rerun to refresh the sidebar status
                st.rerun()
                
        metrics = st.session_state.metrics
        
        st.success("Trained models & evaluation parameters serialized successfully.")
        
        # Display Metrics in beautiful format
        st.markdown(section_heading("Algorithm Performance Comparison", label="EVALUATION BENCHMARK", accent="#E63946"), unsafe_allow_html=True)
        
        compare_data = {
            "Model Name": list(metrics.keys()),
            "Train Accuracy": [metrics[m].get('train_accuracy', 0.0) for m in metrics],
            "Test Accuracy": [metrics[m]['accuracy'] for m in metrics],
            "Precision": [metrics[m]['precision'] for m in metrics],
            "Recall": [metrics[m]['recall'] for m in metrics],
            "F1-Score": [metrics[m]['f1'] for m in metrics],
            "Train Time (s)": [metrics[m]['train_time'] for m in metrics]
        }
        
        compare_df = pd.DataFrame(compare_data)
        st.dataframe(
            compare_df.style.format({
                'Train Accuracy': '{:.2%}',
                'Test Accuracy': '{:.2%}',
                'Precision': '{:.2%}',
                'Recall': '{:.2%}',
                'F1-Score': '{:.2%}',
                'Train Time (s)': '{:.3f}s'
            }), 
            use_container_width=True
        )
        
        # Plot accuracy and time side-by-side
        with plt.rc_context(BAUHAUS_CHART_STYLE):
            fig_comp, (ax_acc, ax_time) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Melt dataframe to compare Train vs. Test Accuracy side-by-side
            melted_acc = compare_df.melt(
                id_vars="Model Name",
                value_vars=["Train Accuracy", "Test Accuracy"],
                var_name="Split",
                value_name="Accuracy"
            )
            
            # Accuracy comparison
            sns.barplot(x="Accuracy", y="Model Name", hue="Split", data=melted_acc, 
                        palette={"Train Accuracy": "#2A9D8F", "Test Accuracy": "#E63946"}, ax=ax_acc)
            ax_acc.set_xlim(0, 1.05)
            ax_acc.set_title("Train (Teal) vs. Test (Red) Accuracy", color="#E8E8F0", fontsize=12, fontweight="bold")
            ax_acc.set_xlabel("Accuracy", color="#E8E8F0")
            ax_acc.set_ylabel("", color="#E8E8F0")
            ax_acc.legend(facecolor='#252535', edgecolor='#2C2C40', labelcolor='#E8E8F0')
            ax_acc.tick_params(colors="#9090A8")
            for spine in ax_acc.spines.values():
                spine.set_color('#2C2C40')
                
            # Training time comparison
            sns.barplot(x="Train Time (s)", y="Model Name", data=compare_df, 
                        palette=["#F4A261", "#2A9D8F", "#E63946", "#457B9D"], ax=ax_time)
            ax_time.set_title("Computation Time per Model (Seconds)", color="#E8E8F0", fontsize=12, fontweight="bold")
            ax_time.set_xlabel("Seconds", color="#E8E8F0")
            ax_time.set_ylabel("", color="#E8E8F0")
            ax_time.tick_params(colors="#9090A8")
            for spine in ax_time.spines.values():
                spine.set_color('#2C2C40')
                
            st.pyplot(fig_comp)
        
        # Learning Curves Tab
        st.markdown(section_heading("Optimization Loss Curves", label="TRAINING CONVERGENCE", accent="#F4A261"), unsafe_allow_html=True)
        col_curve1, col_curve2 = st.columns(2)
        
        with col_curve1:
            st.markdown("<div class='bauhaus-section-label'>LOGISTIC REGRESSION</div><h4 style='margin:0 0 10px 0;'>Gradient Descent Convergence</h4>", unsafe_allow_html=True)
            losses_log = st.session_state.log_reg.loss_history
            with plt.rc_context(BAUHAUS_CHART_STYLE):
                fig_loss1, ax_l1 = plt.subplots(figsize=(8, 4))
                ax_l1.plot(losses_log, color='#2A9D8F', linewidth=2.5)
                ax_l1.set_xlabel("Iteration / Epoch", color="#E8E8F0")
                ax_l1.set_ylabel("BCE Loss + L2 Penalty", color="#E8E8F0")
                ax_l1.set_title("Logistic Regression Cost Curve", color="#E8E8F0", fontweight="bold")
                ax_l1.tick_params(colors="#9090A8")
                ax_l1.grid(True, color="#2C2C40", linestyle="--")
                for spine in ax_l1.spines.values():
                    spine.set_color('#2C2C40')
                st.pyplot(fig_loss1)
            
        with col_curve2:
            st.markdown("<div class='bauhaus-section-label'>NEURAL NETWORK</div><h4 style='margin:0 0 10px 0;'>Backpropagation SGD Convergence</h4>", unsafe_allow_html=True)
            losses_nn = st.session_state.nn.loss_history
            with plt.rc_context(BAUHAUS_CHART_STYLE):
                fig_loss2, ax_l2 = plt.subplots(figsize=(8, 4))
                ax_l2.plot(losses_nn, color='#E63946', linewidth=2.5)
                ax_l2.set_xlabel("Epoch", color="#E8E8F0")
                ax_l2.set_ylabel("BCE Loss + L2 Penalty", color="#E8E8F0")
                ax_l2.set_title("Multi-Layer Perceptron Cost Curve", color="#E8E8F0", fontweight="bold")
                ax_l2.tick_params(colors="#9090A8")
                ax_l2.grid(True, color="#2C2C40", linestyle="--")
                for spine in ax_l2.spines.values():
                    spine.set_color('#2C2C40')
                st.pyplot(fig_loss2)

        # Confusion Matrices Section
        st.markdown(section_heading("Confusion Matrices", label="CLASSIFICATION MATRICES", accent="#2A9D8F"), unsafe_allow_html=True)
        col_cm1, col_cm2, col_cm3, col_cm4 = st.columns(4)
        cms = [
            ("KNN", col_cm1),
            ("Logistic Regression", col_cm2),
            ("Random Forest", col_cm3),
            ("Simple Neural Network", col_cm4)
        ]
        
        for name, col in cms:
            with col:
                st.markdown(f"<div class='card-title'>{name}</div>", unsafe_allow_html=True)
                m = metrics[name]
                cm_data = np.array(m['confusion_matrix'])
                
                with plt.rc_context(BAUHAUS_CHART_STYLE):
                    fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
                    sns.heatmap(cm_data, annot=True, fmt="d", cmap=sns.dark_palette("#2A9D8F", as_cmap=True), cbar=False,
                                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"], ax=ax_cm,
                                annot_kws={"size": 13, "weight": "bold", "color": "#E8E8F0"})
                    
                    ax_cm.set_xlabel("Predicted", color="#E8E8F0", fontsize=9, fontfamily="sans-serif")
                    ax_cm.set_ylabel("Actual", color="#E8E8F0", fontsize=9, fontfamily="sans-serif")
                    ax_cm.tick_params(colors="#9090A8")
                    for spine in ax_cm.spines.values():
                        spine.set_color('#2C2C40')
                    st.pyplot(fig_cm)
                
    else:
        st.info("👈 Please select your preferred configurations and click **🚀 Train All Models** in the sidebar to start!")

# TAB 3: LIVE PREDICTOR
with tab3:
    if "show_toast" in st.session_state and st.session_state.show_toast:
        st.toast(st.session_state.show_toast)
        st.session_state.show_toast = None

    st.markdown(section_heading("Verify Article & Search Live Coverage", label="REAL-TIME FACT-CHECKING ENGINE", accent="#E63946"), unsafe_allow_html=True)
    st.markdown("<p style='color: var(--text-secondary); font-size: 0.95rem; margin-bottom: 18px;'>Input a news headline or complete paragraph. The laboratory system executes inferencing across all 4 custom ML models, cross-references live reports via <strong>NewsAPI</strong> & <strong>NewsData.io</strong>, and performs semantic contradiction analysis with Gemini.</p>", unsafe_allow_html=True)

    # Initialize session state for news input if not present
    if "news_input_area" not in st.session_state:
        st.session_state.news_input_area = ""

    # Load example buttons
    col_ex1, col_ex2 = st.columns(2)
    with col_ex1:
        if st.button("Load Real News Sample", use_container_width=True):
            st.session_state.news_input_area = (
                "WASHINGTON (Reuters) - The U.S. Senate approved a sweeping tax reform bill early Saturday morning, "
                "marking a major legislative victory for President Donald Trump. The Republican-led chamber passed the bill "
                "51-49, following a marathon late-night session. The bill represents the largest overhaul of the U.S. tax code "
                "since the 1980s, cutting rates for corporations and individuals."
            )
    with col_ex2:
        if st.button("Load Fake News Sample", use_container_width=True):
            st.session_state.news_input_area = (
                "BREAKING: Pope Francis has shocked the world today by endorsing Donald Trump for President. "
                "In a statement released by the Vatican, the Pope declared that Donald Trump is the only logical choice "
                "to lead the free world, praising his stances on border control and economic expansion. The statement has "
                "ignited a massive controversy across global religious organizations."
            )

    # Calculate remaining characters dynamically
    current_text = st.session_state.get("news_input_area", "")
    remaining = max(0, 600 - len(current_text))
    
    # Styled remaining counter
    counter_color = "var(--text-secondary)" if remaining > 50 else "var(--accent-primary)"
    st.markdown(
        f"<div style='text-align: right; font-family: var(--font-mono); font-size: 0.78rem; color: {counter_color}; font-weight: 700; margin-bottom: 4px; letter-spacing: 0.08em;'>"
        f"CHARACTERS REMAINING: {remaining} / 600"
        f"</div>", 
        unsafe_allow_html=True
    )

    news_input = st.text_area("News text to verify:", key="news_input_area", height=180, 
                             placeholder="Enter headline or article paragraph here...",
                             max_chars=600)

    # Load API keys silently from .env
    news_api_key = load_env_api_key("NEWS_API_KEY")
    newsdata_api_key = load_env_api_key("NEWSDATA_API_KEY")

    verify_trigger = st.button("Verify Article & Search Live Coverage", use_container_width=True)

    # Initialize request list and results in session state
    if "verification_timestamps" not in st.session_state:
        st.session_state.verification_timestamps = []
    if "verify_results" not in st.session_state:
        st.session_state.verify_results = None

    if verify_trigger:
        if not st.session_state.trained or not is_compatible:
            st.error("Please train the models first using Tab 2 or the sidebar button to align model features.")
        elif not news_input.strip():
            st.warning("Please enter news text to verify.")
        else:
            # Check for Rate Limiting to protect API keys
            current_time = time.time()
            st.session_state.verification_timestamps = [
                t for t in st.session_state.verification_timestamps if current_time - t < 60
            ]
            
            run_verification = True
            
            if run_verification and len(st.session_state.verification_timestamps) >= 5:
                # Calculate exactly how many seconds until the oldest request in the window expires
                oldest_timestamp = st.session_state.verification_timestamps[0]
                wait_time = max(1, int(oldest_timestamp + 60 - current_time))
                st.toast(f"Rate limit reached (5 requests/minute). Please wait {wait_time}s.", icon="⚠️")
                run_verification = False
                
            if run_verification:
                # Accept request and record timestamp
                st.session_state.verification_timestamps.append(current_time)
                
                # Clear previous results during calculation
                st.session_state.verify_results = None
                st.session_state.feedback_success = None
                
                # 1. Classification Phase
                with st.spinner("Analyzing text and running custom model predictions..."):
                    x_input = transform_for_prediction([news_input], st.session_state.vectorizer)

                    p_knn = st.session_state.knn.predict(x_input)[0]
                    p_log = st.session_state.log_reg.predict(x_input)[0]
                    p_rf = st.session_state.rf.predict(x_input)[0]
                    p_nn = st.session_state.nn.predict(x_input)[0]

                    prob_knn = st.session_state.knn.predict_proba(x_input)[0] if hasattr(st.session_state.knn, "predict_proba") else float(p_knn)
                    prob_log = st.session_state.log_reg.predict_proba(x_input)[0]
                    prob_nn = st.session_state.nn.predict_proba(x_input)[0]
                    
                # 2. Extract Query
                with st.spinner("Extracting search keywords from text..."):
                    search_query = extract_query_with_gemini(news_input)
                    
                # 3. Search Live Coverage
                raw_api = []
                raw_data = []
                newsapi_count = 0
                newsdata_count = 0
                
                if news_api_key:
                    with st.spinner("Searching NewsAPI for live coverage..."):
                        raw_api = fetch_live_news(search_query, news_api_key) or []
                        newsapi_count = len(raw_api)
                
                if newsdata_api_key:
                    with st.spinner("Searching NewsData.io for live coverage..."):
                        raw_data = fetch_newsdata_io(search_query, newsdata_api_key) or []
                        newsdata_count = len(raw_data)
                        
                total_sources = newsapi_count + newsdata_count
                
                # Construct a clean text summary of the live news coverage to pass to Gemini
                coverage_items = []
                if raw_api:
                    for art in raw_api[:3]:
                        title = art.get('title', 'No Title')
                        source = art.get('source', {}).get('name', 'Unknown')
                        desc = art.get('description', '') or ''
                        coverage_items.append(f"- [{source}] {title}: {desc[:150]}")
                if raw_data:
                    for art in raw_data[:3]:
                        title = art.get('title', 'No Title')
                        source = art.get('source_id', 'Unknown').upper()
                        desc = art.get('description', '') or ''
                        coverage_items.append(f"- [{source}] {title}: {desc[:150]}")
                
                coverage_text = "\n".join(coverage_items) if coverage_items else "No live news coverage found."
                
                with st.spinner("Gemini AI is analyzing article semantics..."):
                    gemini_result = gemini_fact_check(news_input, search_query, coverage_text)
                    
                # Signal 1: ML Model Ensemble Vote
                predictions = [p_knn, p_log, p_rf, p_nn]
                fake_votes = sum(predictions)
                real_votes = 4 - fake_votes
                model_verdict = "FAKE" if fake_votes > real_votes else "REAL"
                model_confidence = max(fake_votes, real_votes) / 4.0 * 100
                
                # Signal 2: Cross-Reference Score (more sources = more likely real)
                if total_sources >= 5:
                    cross_ref_score = 100  # Well covered — likely real
                elif total_sources >= 2:
                    cross_ref_score = 60   # Some coverage
                elif total_sources == 1:
                    cross_ref_score = 30   # Minimal coverage
                else:
                    cross_ref_score = 0    # No coverage — suspicious
                    
                # Signal 3: Gemini AI Verdict
                gemini_verdict = "UNKNOWN"
                gemini_confidence = 50
                gemini_reasoning = "Gemini API was unavailable for analysis."
                if gemini_result:
                    gemini_verdict = gemini_result["verdict"]
                    gemini_confidence = gemini_result["confidence"]
                    gemini_reasoning = gemini_result["reasoning"]
                    
                # Final Weighted Credibility Score
                model_score = (real_votes / 4.0) * 100
                gemini_score = gemini_confidence if gemini_verdict == "REAL" else (100 - gemini_confidence)
                
                final_credibility = int(model_score * 0.20 + cross_ref_score * 0.40 + gemini_score * 0.40)
                
                # --- REAL-TIME OVERRIDE RULE ---
                override_applied = False
                if total_sources == 0:
                    final_credibility = min(20, final_credibility)
                    override_applied = True
                    
                final_credibility = max(0, min(100, final_credibility))
                
                if final_credibility >= 60:
                    final_verdict = "LIKELY REAL"
                    verdict_color = "#2A9D8F"
                    verdict_emoji = "●"
                    verdict_bg = "var(--bg-raised)"
                elif final_credibility >= 40:
                    final_verdict = "UNCERTAIN"
                    verdict_color = "#F4A261"
                    verdict_emoji = "▲"
                    verdict_bg = "var(--bg-raised)"
                else:
                    final_verdict = "LIKELY FAKE"
                    verdict_color = "#E63946"
                    verdict_emoji = "■"
                    verdict_bg = "var(--bg-raised)"
                    
                highlighted_html = generate_interpretability_html(news_input, st.session_state.vectorizer, st.session_state.log_reg)
                
                # Save results dictionary in session state
                st.session_state.verify_results = {
                    "news_input": news_input,
                    "p_knn": p_knn,
                    "p_log": p_log,
                    "p_rf": p_rf,
                    "p_nn": p_nn,
                    "prob_knn": prob_knn,
                    "prob_log": prob_log,
                    "prob_nn": prob_nn,
                    "search_query": search_query,
                    "raw_api": raw_api,
                    "raw_data": raw_data,
                    "newsapi_count": newsapi_count,
                    "newsdata_count": newsdata_count,
                    "gemini_result": gemini_result,
                    "total_sources": total_sources,
                    "cross_ref_score": cross_ref_score,
                    "gemini_verdict": gemini_verdict,
                    "gemini_confidence": gemini_confidence,
                    "gemini_reasoning": gemini_reasoning,
                    "final_credibility": final_credibility,
                    "override_applied": override_applied,
                    "final_verdict": final_verdict,
                    "verdict_color": verdict_color,
                    "verdict_emoji": verdict_emoji,
                    "verdict_bg": verdict_bg,
                    "model_verdict": model_verdict,
                    "model_confidence": model_confidence,
                    "fake_votes": fake_votes,
                    "real_votes": real_votes,
                    "k_neighbors": st.session_state.knn.k,
                    "rf_trees": st.session_state.rf.n_estimators,
                    "highlighted_html": highlighted_html
                }
                # Force rerun to display immediately
                st.rerun()

    # --- RENDER STAGE (OUTSIDE VERIFY TRIGGER) ---
    if st.session_state.verify_results is not None:
        # Check for changes in input area to avoid showing stale results
        if st.session_state.verify_results["news_input"] != news_input:
            st.session_state.verify_results = None
            st.rerun()
            
        res = st.session_state.verify_results
        
        # 1. Models Decisions
        st.markdown(section_heading("Trained Models Decisions", label="INDIVIDUAL PREDICTOR CONSENSUS", accent="#2A9D8F"), unsafe_allow_html=True)
        col_pred1, col_pred2 = st.columns(2)

        with col_pred1:
            cls_knn = "model-card-fake" if res["p_knn"] == 1 else "model-card-real"
            lbl_knn = "FAKE NEWS" if res["p_knn"] == 1 else "REAL NEWS"
            prob_knn_val = res.get("prob_knn", float(res["p_knn"]))
            conf_knn = prob_knn_val if res["p_knn"] == 1 else (1.0 - prob_knn_val)
            color_knn = "var(--accent-fake)" if res["p_knn"] == 1 else "var(--accent-real)"
            st.markdown(f"""
            <div class='{cls_knn}'>
                <div class='card-title'>K-Nearest Neighbors</div>
                <h2 style='margin:0; font-size:1.8rem; color:{color_knn};'>{lbl_knn}</h2>
                <div style='font-family:var(--font-mono); font-size:0.8rem; color:var(--text-secondary); margin-top:4px;'>Confidence: {conf_knn:.1%} (k={res.get('k_neighbors', 5)} closest vectors)</div>
            </div>
            """, unsafe_allow_html=True)

            cls_rf = "model-card-fake" if res["p_rf"] == 1 else "model-card-real"
            lbl_rf = "FAKE NEWS" if res["p_rf"] == 1 else "REAL NEWS"
            color_rf = "var(--accent-fake)" if res["p_rf"] == 1 else "var(--accent-real)"
            st.markdown(f"""
            <div class='{cls_rf}'>
                <div class='card-title'>Random Forest</div>
                <h2 style='margin:0; font-size:1.8rem; color:{color_rf};'>{lbl_rf}</h2>
                <div style='font-family:var(--font-mono); font-size:0.8rem; color:var(--text-secondary); margin-top:4px;'>Ensemble vote across {res.get('rf_trees', 10)} decision trees</div>
            </div>
            """, unsafe_allow_html=True)

        with col_pred2:
            cls_log = "model-card-fake" if res["p_log"] == 1 else "model-card-real"
            lbl_log = "FAKE NEWS" if res["p_log"] == 1 else "REAL NEWS"
            pct_log = res["prob_log"] if res["p_log"] == 1 else (1.0 - res["prob_log"])
            color_log = "var(--accent-fake)" if res["p_log"] == 1 else "var(--accent-real)"
            st.markdown(f"""
            <div class='{cls_log}'>
                <div class='card-title'>Logistic Regression</div>
                <h2 style='margin:0; font-size:1.8rem; color:{color_log};'>{lbl_log}</h2>
                <div style='font-family:var(--font-mono); font-size:0.8rem; color:var(--text-secondary); margin-top:4px;'>Sigmoid posterior probability: {pct_log:.2%}</div>
            </div>
            """, unsafe_allow_html=True)

            cls_nn = "model-card-fake" if res["p_nn"] == 1 else "model-card-real"
            lbl_nn = "FAKE NEWS" if res["p_nn"] == 1 else "REAL NEWS"
            pct_nn = res["prob_nn"] if res["p_nn"] == 1 else (1.0 - res["prob_nn"])
            color_nn = "var(--accent-fake)" if res["p_nn"] == 1 else "var(--accent-real)"
            st.markdown(f"""
            <div class='{cls_nn}'>
                <div class='card-title'>Simple Neural Network</div>
                <h2 style='margin:0; font-size:1.8rem; color:{color_nn};'>{lbl_nn}</h2>
                <div style='font-family:var(--font-mono); font-size:0.8rem; color:var(--text-secondary); margin-top:4px;'>MLP output probability: {pct_nn:.2%}</div>
            </div>
            """, unsafe_allow_html=True)

        # 2. Live Coverage
        st.markdown(section_heading("Live Search & Supporting Coverage", label="CROSS-REFERENCE SIGNALS", accent="#F4A261"), unsafe_allow_html=True)
        st.markdown(f"""
        <div class='nm-inset' style='padding: 12px 18px; margin-bottom: 16px; border-left: 4px solid var(--accent-yellow);'>
            <span style='font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.12em;'>Gemini Extracted Search Query:</span>
            <strong style='font-family: var(--font-mono); color: var(--accent-yellow); margin-left: 10px;'>{res['search_query']}</strong>
        </div>
        """, unsafe_allow_html=True)

        col_newsapi, col_newsdata = st.columns(2)

        with col_newsapi:
            st.markdown("<div class='bauhaus-section-label'>NEWSAPI ENDPOINT</div><h4 style='margin:0 0 10px 0;'>NewsAPI Coverage</h4>", unsafe_allow_html=True)
            if not news_api_key:
                st.info("NewsAPI key is not configured in Streamlit Secrets or `.env`.")
            elif res["newsapi_count"] == 0:
                st.warning("No matching coverage found on NewsAPI.")
            else:
                for idx, art in enumerate(res["raw_api"]):
                    title = art.get('title', 'No Title')
                    source = art.get('source', {}).get('name', 'Unknown Source')
                    url = art.get('url', '#')
                    desc = art.get('description', '')
                    date = art.get('publishedAt', '')[:10]
                    st.markdown(f"""
                    <div class='nm-raised' style='padding:14px 16px; margin-bottom:12px;'>
                        <div style='font-family:var(--font-mono); font-size:0.75rem; color:var(--accent-yellow); text-transform:uppercase;'>{source} · {date}</div>
                        <div style='font-weight:700; font-size:0.95rem; margin:4px 0;'><a href='{url}' target='_blank' style='color:var(--text-primary); text-decoration:none;'>{title}</a></div>
                        <div style='font-size:0.85rem; color:var(--text-secondary); line-height:1.4;'>{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

        with col_newsdata:
            st.markdown("<div class='bauhaus-section-label'>NEWSDATA ENDPOINT</div><h4 style='margin:0 0 10px 0;'>NewsData.io Coverage</h4>", unsafe_allow_html=True)
            if not newsdata_api_key:
                st.info("NewsData.io key is not configured in Streamlit Secrets or `.env`.")
            elif res["newsdata_count"] == 0:
                st.warning("No matching coverage found on NewsData.io.")
            else:
                for idx, art in enumerate(res["raw_data"]):
                    title = art.get('title', 'No Title')
                    source = art.get('source_id', 'Unknown Source').upper()
                    url = art.get('link', '#')
                    desc = art.get('description', '')
                    date = art.get('pubDate', '')[:10] if art.get('pubDate') else ''
                    st.markdown(f"""
                    <div class='nm-raised' style='padding:14px 16px; margin-bottom:12px;'>
                        <div style='font-family:var(--font-mono); font-size:0.75rem; color:var(--accent-yellow); text-transform:uppercase;'>{source} · {date}</div>
                        <div style='font-weight:700; font-size:0.95rem; margin:4px 0;'><a href='{url}' target='_blank' style='color:var(--text-primary); text-decoration:none;'>{title}</a></div>
                        <div style='font-size:0.85rem; color:var(--text-secondary); line-height:1.4;'>{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # 3. AI Verdict
        st.markdown(section_heading("Unified Fact-Checking Verdict", label="TRI-SIGNAL SYNTHESIS", accent="#E63946"), unsafe_allow_html=True)

        sig1, sig2, sig3 = st.columns(3)
        with sig1:
            verdict_m_color = "var(--accent-fake)" if res["model_verdict"] == "FAKE" else "var(--accent-real)"
            st.markdown(f"""
            <div class='nm-raised' style='padding: 18px; text-align: center; border-top: 3px solid {verdict_m_color};'>
                <div class='bauhaus-section-label'>ML CONSENSUS</div>
                <h3 style='margin: 4px 0; color: {verdict_m_color}; font-size: 1.5rem;'>{res["model_verdict"]}</h3>
                <div style='font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-secondary);'>{res['fake_votes']}/4 Models · {res["model_confidence"]:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with sig2:
            src_label = f"{res['total_sources']} sources" if res["total_sources"] > 0 else "0 sources"
            verdict_c_color = "var(--accent-real)" if res["cross_ref_score"] >= 50 else "var(--accent-fake)"
            cov_status = "COVERED" if res["total_sources"] >= 2 else "UNCONFIRMED"
            st.markdown(f"""
            <div class='nm-raised' style='padding: 18px; text-align: center; border-top: 3px solid {verdict_c_color};'>
                <div class='bauhaus-section-label'>CROSS-REFERENCE</div>
                <h3 style='margin: 4px 0; color: {verdict_c_color}; font-size: 1.5rem;'>{cov_status}</h3>
                <div style='font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-secondary);'>{src_label} · Score {res["cross_ref_score"]}/100</div>
            </div>
            """, unsafe_allow_html=True)

        with sig3:
            verdict_g_color = "var(--accent-fake)" if res["gemini_verdict"] == "FAKE" else "var(--accent-real)" if res["gemini_verdict"] == "REAL" else "var(--accent-yellow)"
            st.markdown(f"""
            <div class='nm-raised' style='padding: 18px; text-align: center; border-top: 3px solid {verdict_g_color};'>
                <div class='bauhaus-section-label'>GEMINI AI REASONING</div>
                <h3 style='margin: 4px 0; color: {verdict_g_color}; font-size: 1.5rem;'>{res["gemini_verdict"]}</h3>
                <div style='font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-secondary);'>Confidence {res["gemini_confidence"]}%</div>
            </div>
            """, unsafe_allow_html=True)

        if res["gemini_result"]:
            st.markdown(f"""
            <div class='nm-inset' style='border-left: 4px solid var(--accent-yellow); padding: 16px 20px; border-radius: 0 10px 10px 0; margin: 18px 0;'>
                <div style='font-family: var(--font-mono); font-size: 0.72rem; color: var(--accent-yellow); text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 4px;'>LLM Semantic Analysis:</div>
                <div style='color: var(--text-primary); font-size: 0.95rem; line-height: 1.5;'>{res["gemini_reasoning"]}</div>
            </div>
            """, unsafe_allow_html=True)

        override_warning_html = ""
        if res["override_applied"]:
            override_warning_html = "<div style='color: var(--accent-fake); font-family: var(--font-mono); font-size: 0.78rem; margin-top: 10px; font-weight: 700; letter-spacing: 0.05em;'>ZERO-COVERAGE PENALTY: Credibility capped at 20% due to zero matching live news reports.</div>"

        st.markdown(
            f"""
            <div class='nm-raised' style='border: 2px solid {res["verdict_color"]}; padding: 28px 24px; text-align: center; margin-top: 20px;'>
                <div style='font-family: var(--font-mono); font-size: 0.75rem; font-weight: 700; letter-spacing: 0.18em; color: var(--text-secondary); margin-bottom: 6px;'>
                    FINAL COMBINED VERDICT
                </div>
                <h1 style='margin: 0; color: {res["verdict_color"]}; font-family: var(--font-display); font-size: 2.8rem; letter-spacing: -0.02em;'>
                    {res["final_verdict"]}
                </h1>
                <div style='font-family: var(--font-mono); font-size: 1.15rem; color: var(--text-primary); margin-top: 8px;'>
                    Credibility Score: <strong style="color: {res['verdict_color']};">{res['final_credibility']} / 100</strong>
                </div>
                <div class='nm-inset' style='height: 18px; margin: 16px auto 8px auto; max-width: 600px; overflow: hidden; padding: 2px;'>
                    <div style='background: {res["verdict_color"]}; height: 100%; width: {res["final_credibility"]}%; border-radius: 6px; transition: width 0.6s ease;'></div>
                </div>
                {override_warning_html}
                <div style='font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-secondary); margin-top: 12px; letter-spacing: 0.06em;'>
                    WEIGHTED FORMULA: ML MODELS (20%) + CROSS-REFERENCE (40%) + GEMINI AI (40%)
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Continuous learning feedback panel
        st.markdown(section_heading("Human-in-the-Loop Continuous Learning", label="ONLINE MODEL ADJUSTMENT", accent="#2A9D8F"), unsafe_allow_html=True)
        
        if st.session_state.get("feedback_success"):
            st.success(st.session_state.feedback_success)
            
        st.markdown("<p style='color: var(--text-secondary); font-size: 0.9rem;'>If the verdict contains a false positive or negative, submit the ground truth below. The system executes immediate incremental online gradient updates to Logistic Regression, Neural Network, and memory vector appending to KNN.</p>", unsafe_allow_html=True)
        
        feed_col1, feed_col2 = st.columns([2, 1])
        with feed_col1:
            true_label_choice = st.radio(
                "Ground truth classification label:",
                options=["Real News", "Fake News"],
                horizontal=True,
                key="feedback_radio"
            )
        with feed_col2:
            st.write("") # Vertical offset spacing
            submit_feedback = st.button("Submit & Train Instantly", use_container_width=True)

        if submit_feedback:
            last_text = news_input.strip()
            if not last_text:
                st.warning("No article text found to train on.")
            else:
                label_val = 0 if true_label_choice == "Real News" else 1
                
                with st.spinner("Executing online parameter refinement..."):
                    # Preprocess and extract features using unified pipeline
                    x_input_fb = transform_for_prediction([last_text], st.session_state.vectorizer)
                    
                    # Online updates
                    # 1. KNN
                    st.session_state.knn.X_train = np.vstack((st.session_state.knn.X_train, x_input_fb))
                    st.session_state.knn.y_train = np.append(st.session_state.knn.y_train, [label_val])
                    
                    # 2. Logistic Regression
                    st.session_state.log_reg.partial_fit(x_input_fb, np.array([label_val]))
                    
                    # 3. Simple Neural Network
                    st.session_state.nn.partial_fit(x_input_fb, np.array([label_val]))
                    
                    # Log to CSV
                    feedback_file = os.path.join("dataset", "user_feedback.csv")
                    feedback_df = pd.DataFrame([{
                        "timestamp": pd.Timestamp.now().isoformat(),
                        "text_length": len(last_text),
                        "submitted_truth": label_val,
                        "raw_text": last_text
                    }])
                    if not os.path.exists(feedback_file):
                        feedback_df.to_csv(feedback_file, index=False)
                    else:
                        feedback_df.to_csv(feedback_file, mode="a", header=False, index=False)
                        
                    # Save models to disk
                    save_models_to_disk(
                        st.session_state.vectorizer,
                        st.session_state.knn,
                        st.session_state.log_reg,
                        st.session_state.rf,
                        st.session_state.nn,
                        st.session_state.metrics
                    )
                    
                    # Run predictions using the newly updated models to update the dashboard instantly!
                    p_knn = st.session_state.knn.predict(x_input_fb)[0]
                    p_log = st.session_state.log_reg.predict(x_input_fb)[0]
                    p_rf = st.session_state.rf.predict(x_input_fb)[0]
                    p_nn = st.session_state.nn.predict(x_input_fb)[0]

                    prob_knn = st.session_state.knn.predict_proba(x_input_fb)[0] if hasattr(st.session_state.knn, "predict_proba") else float(p_knn)
                    prob_log = st.session_state.log_reg.predict_proba(x_input_fb)[0]
                    prob_nn = st.session_state.nn.predict_proba(x_input_fb)[0]
                    
                    # Update the results dictionary in session state
                    res = st.session_state.verify_results
                    res["p_knn"] = p_knn
                    res["p_log"] = p_log
                    res["p_rf"] = p_rf
                    res["p_nn"] = p_nn
                    res["prob_knn"] = prob_knn
                    res["prob_log"] = prob_log
                    res["prob_nn"] = prob_nn
                    res["k_neighbors"] = st.session_state.knn.k
                    res["rf_trees"] = st.session_state.rf.n_estimators
                    
                    # Recalculate combined models consensus
                    predictions = [p_knn, p_log, p_rf, p_nn]
                    fake_votes = sum(predictions)
                    real_votes = 4 - fake_votes
                    res["model_verdict"] = "FAKE" if fake_votes > real_votes else "REAL"
                    res["model_confidence"] = max(fake_votes, real_votes) / 4.0 * 100
                    res["fake_votes"] = fake_votes
                    res["real_votes"] = real_votes
                    
                    # Recalculate final combined credibility score
                    model_score = (real_votes / 4.0) * 100
                    gemini_score = res["gemini_confidence"] if res["gemini_verdict"] == "REAL" else (100 - res["gemini_confidence"])
                    
                    final_credibility = int(model_score * 0.20 + res["cross_ref_score"] * 0.40 + gemini_score * 0.40)
                    
                    # Override rule check
                    if res["total_sources"] == 0:
                        final_credibility = min(20, final_credibility)
                    final_credibility = max(0, min(100, final_credibility))
                    
                    res["final_credibility"] = final_credibility
                    
                    if final_credibility >= 60:
                        res["final_verdict"] = "LIKELY REAL"
                        res["verdict_color"] = "#2A9D8F"
                        res["verdict_emoji"] = "●"
                        res["verdict_bg"] = "var(--bg-raised)"
                    elif final_credibility >= 40:
                        res["final_verdict"] = "UNCERTAIN"
                        res["verdict_color"] = "#F4A261"
                        res["verdict_emoji"] = "▲"
                        res["verdict_bg"] = "var(--bg-raised)"
                    else:
                        res["final_verdict"] = "LIKELY FAKE"
                        res["verdict_color"] = "#E63946"
                        res["verdict_emoji"] = "■"
                        res["verdict_bg"] = "var(--bg-raised)"
                    
                    st.session_state.show_toast = "Models updated in memory and serialized to disk."
                    st.session_state.feedback_success = "Online learning step complete. Predictions updated instantly."
                    st.rerun()
