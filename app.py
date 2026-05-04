"""
ScentMatch — personality quiz → fragrance recommender.

Backend: sentence-transformers embeddings + cosine similarity (unchanged).
Frontend: 9-question personality quiz that compiles answers into a semantic
          search query passed to the embedding backend.
"""

import pandas as pd
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

SAMPLE_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "fragrances.csv")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5


# ─────────────────────────────────────────────
# Backend — unchanged from original
# ─────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """
    Load the fragrance dataset.
    Returns a DataFrame with at minimum: name, brand, notes, accords, description.
    """
    kaggle_path = os.path.join(os.path.dirname(__file__), "data", "fra_cleaned.csv")

    if os.path.exists(kaggle_path):
        df = pd.read_csv(kaggle_path)
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        rename_map = {
            "fragrance_name": "name",
            "fragrance": "name",
            "perfume": "name",
            "main_accords": "accords",
            "top_notes": "notes",
        }
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
        for col in ["name", "brand", "notes", "accords", "description"]:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].fillna("").astype(str)
        return df

    df = pd.read_csv(SAMPLE_DATA_PATH)
    for col in ["name", "brand", "notes", "accords", "description"]:
        df[col] = df[col].fillna("").astype(str)
    return df


def build_corpus(df: pd.DataFrame) -> list[str]:
    """Concatenate key fields per fragrance into one embeddable string."""
    corpus = []
    for _, row in df.iterrows():
        text = (
            f"Fragrance: {row['name']}. "
            f"Brand: {row['brand']}. "
            f"Notes: {row['notes']}. "
            f"Accords: {row['accords']}. "
            f"{row['description']}"
        )
        corpus.append(text)
    return corpus


@st.cache_resource(show_spinner="Loading embedding model…")
def load_model() -> SentenceTransformer:
    """Download (first run) and cache the sentence-transformer model."""
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_data(show_spinner="Computing fragrance embeddings…")
def compute_embeddings(corpus: tuple[str, ...]) -> np.ndarray:
    """
    Embed every fragrance description once and cache the result.
    Accepts a tuple (not list) because st.cache_data needs hashable args.
    Returns shape (n_fragrances, embedding_dim).
    """
    model = load_model()
    return model.encode(list(corpus), convert_to_numpy=True, show_progress_bar=False)


def recommend(query: str, df: pd.DataFrame, embeddings: np.ndarray, top_k: int = 5):
    """
    Embed the user's query and return the top_k most similar fragrances.
    1. Embed query → shape (1, dim)
    2. Cosine similarity vs all fragrance embeddings
    3. Return top_k rows + scores
    """
    model = load_model()
    query_vec = model.encode([query], convert_to_numpy=True)
    scores = cosine_similarity(query_vec, embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = df.iloc[top_indices].copy()
    results["similarity"] = scores[top_indices]
    return results


# ─────────────────────────────────────────────
# Quiz definitions
# ─────────────────────────────────────────────

QUESTIONS = [
    {
        "type": "text",
        "label": "Describe a smell — not a fragrance, just a smell — that instantly makes you feel good.",
        "placeholder": "e.g. petrichor after rain, old bookshops, sunscreen at the beach...",
    },
    {
        "type": "text",
        "label": "You walk into a party where you know maybe 2 people. What are you actually doing for the first 20 minutes?",
        "placeholder": "e.g. find my people and stick close, grab a drink and observe, talk to strangers...",
    },
    {
        "type": "text",
        "label": "Describe your aesthetic in 3 words. Not aspirational — what you actually wear on a regular Tuesday.",
        "placeholder": "e.g. clean quiet minimal... cozy chaotic librarian... dark academia soft...",
    },
    {
        "type": "choice",
        "label": "Pick the environment that feels most like you.",
        "options": [
            "Dense forest",
            "Open ocean",
            "Urban rooftop at night",
            "Desert at sunrise",
            "Garden after rain",
        ],
    },
    {
        "type": "choice",
        "label": "Fresh & bright or warm & rich — which pulls you more?",
        "options": [
            "Fresh & bright (citrus, mint, cucumber)",
            "Warm & rich (vanilla, spice, dark chocolate)",
            "Somewhere in between",
            "Depends on my mood",
        ],
    },
    {
        "type": "text",
        "label": "How do you want people to feel when you enter a room? Not how you want to look — how you want them to feel.",
        "placeholder": "e.g. comfortable, intrigued, like the energy shifted...",
    },
    {
        "type": "choice",
        "label": "Do you want your fragrance to be noticed by others, or is it more something you wear for yourself?",
        "options": [
            "Mostly for others",
            "Mostly for myself",
            "Both equally",
        ],
    },
    {
        "type": "choice",
        "label": "One signature scent, or rotating based on mood?",
        "options": [
            "One signature — I want to be known for a scent",
            "Rotating — different moods need different scents",
            "Open to either",
        ],
    },
    {
        "type": "text",
        "label": "Finish this sentence: \"I want to smell like someone who ___________\"",
        "placeholder": "first instinct, don't overthink it...",
    },
]

# ─────────────────────────────────────────────
# Quiz logic helpers
# ─────────────────────────────────────────────

# Scent profile name derived from Q4 (environment) × Q5 (palate)
_SCENT_PROFILES = {
    ("forest",   "fresh"):  "The Forest Bather",
    ("forest",   "warm"):   "The Dark Wanderer",
    ("forest",   "other"):  "The Woodland Drifter",
    ("ocean",    "fresh"):  "The Sea Drifter",
    ("ocean",    "warm"):   "The Ocean Alchemist",
    ("ocean",    "other"):  "The Coastal Soul",
    ("urban",    "fresh"):  "The City Minimalist",
    ("urban",    "warm"):   "The Night Architect",
    ("urban",    "other"):  "The Urban Mystic",
    ("desert",   "fresh"):  "The Sunrise Nomad",
    ("desert",   "warm"):   "The Desert Mystic",
    ("desert",   "other"):  "The Sand Wanderer",
    ("garden",   "fresh"):  "The Petrichor Soul",
    ("garden",   "warm"):   "The Garden Romantic",
    ("garden",   "other"):  "The Bloom Chaser",
}


def _env_key(answer: str) -> str:
    a = (answer or "").lower()
    if "forest" in a:               return "forest"
    if "ocean" in a:                return "ocean"
    if "urban" in a or "roof" in a: return "urban"
    if "desert" in a:               return "desert"
    if "garden" in a or "rain" in a:return "garden"
    return ""


def _palate_key(answer: str) -> str:
    a = (answer or "").lower()
    if "fresh" in a or "bright" in a: return "fresh"
    if "warm" in a or "rich" in a:    return "warm"
    return "other"


def derive_scent_profile(answers: list) -> str:
    env     = _env_key(answers[3] or "")
    palate  = _palate_key(answers[4] or "")
    return _SCENT_PROFILES.get((env, palate), "The Scent Seeker")


def compile_query(answers: list) -> str:
    """
    Turn the 9 quiz answers into one natural-language paragraph.
    This paragraph is passed directly to the embedding model as the search query.
    Skipped questions get a neutral filler so the sentence still reads naturally.
    """
    a = [x or "something hard to define" for x in answers]

    palate = a[4]
    if "fresh" in palate.lower():
        palate_desc = "fresh and bright"
    elif "warm" in palate.lower():
        palate_desc = "warm and rich"
    elif "in between" in palate.lower():
        palate_desc = "balanced, somewhere between fresh and warm"
    else:
        palate_desc = "varied depending on the mood"

    audience = a[6]
    if "others" in audience.lower():
        audience_desc = "worn for others — to be noticed and leave an impression"
    elif "myself" in audience.lower():
        audience_desc = "worn for themselves, a private and personal pleasure"
    else:
        audience_desc = "worn for both themselves and others equally"

    style = a[7]
    if "signature" in style.lower():
        style_desc = "a single defining signature scent they can be known for"
    elif "rotating" in style.lower():
        style_desc = "a rotating wardrobe of scents tuned to different moods"
    else:
        style_desc = "either a signature or a rotating wardrobe"

    return (
        f"Someone who loves the smell of {a[0]}. "
        f"In social situations they tend to {a[1]}. "
        f"Their everyday aesthetic is {a[2]}. "
        f"They feel most at home in a {a[3]} kind of environment. "
        f"They're drawn to {palate_desc} scents. "
        f"They want people around them to feel {a[5]}. "
        f"Their fragrance is {audience_desc}. "
        f"They want {style_desc}. "
        f"They want to smell like someone who {a[8]}."
    )


def reset_quiz():
    """Clear all quiz state so the app returns to the intro screen."""
    st.session_state.step = 0
    st.session_state.answers = [None] * 9
    st.session_state.pop("results_df", None)
    st.session_state.pop("results_query", None)
    # Clear any lingering widget values so inputs don't pre-fill on restart
    for i in range(1, 10):
        st.session_state.pop(f"q{i}_text", None)
        st.session_state.pop(f"q{i}_choice", None)


# ─────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────

def inject_css():
    """Inject custom CSS for the warm luxury minimal theme."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;1,300;1,400&family=Inter:wght@300;400;500&display=swap');

        /* ── Hide Streamlit chrome ── */
        #MainMenu, footer, header { visibility: hidden; }

        /* ── Warm gradient background, fixed so it doesn't scroll ── */
        html, body {
            background: linear-gradient(160deg, #F5F0EB 0%, #EDE8E2 100%) !important;
            background-attachment: fixed !important;
            min-height: 100vh;
        }
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        section[data-testid="stSidebar"] {
            background: transparent !important;
        }
        .stApp {
            background: linear-gradient(160deg, #F5F0EB 0%, #EDE8E2 100%) fixed !important;
        }

        /* ── Narrow centered column ── */
        .block-container {
            max-width: 560px !important;
            padding-top: 0 !important;
            padding-bottom: 6rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        /* ── Full-width 2px progress line pinned to top ── */
        div[data-testid="stProgressBar"] {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            z-index: 9999;
            padding: 0;
            margin: 0;
            background: transparent;
        }
        div[data-testid="stProgressBar"] > div {
            background-color: #DDD8D2 !important;
            border-radius: 0 !important;
            height: 2px !important;
        }
        div[data-testid="stProgressBar"] > div > div {
            background-color: #1a1a1a !important;
            border-radius: 0 !important;
            height: 2px !important;
            transition: width 0.4s ease;
        }

        /* ── Text input: warm inset field ── */
        div[data-testid="stTextInput"] > div > div {
            background: #EFEBE6 !important;
            border: none !important;
            border-radius: 8px !important;
            box-shadow: none !important;
        }
        div[data-testid="stTextInput"] input {
            background: transparent !important;
            border: none !important;
            border-radius: 8px !important;
            box-shadow: none !important;
            padding: 16px !important;
            font-size: 1rem !important;
            color: #1a1a1a !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 300 !important;
        }
        div[data-testid="stTextInput"] input:focus {
            box-shadow: none !important;
            outline: none !important;
            border: none !important;
        }
        div[data-testid="stTextInput"] input::placeholder {
            color: #b5afa8 !important;
            font-weight: 300 !important;
        }

        /* ── Radio options: clean separator rows ── */
        div[data-testid="stRadio"] > div { gap: 0; }
        div[data-testid="stRadio"] label {
            padding: 15px 0;
            border: none !important;
            border-bottom: 1px solid #DDD8D2 !important;
            border-radius: 0 !important;
            background: transparent !important;
            margin: 0 !important;
            font-size: 0.96rem;
            font-family: 'Inter', sans-serif;
            font-weight: 300;
            color: #2a2520;
            cursor: pointer;
            transition: color 0.15s;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        div[data-testid="stRadio"] label:hover { color: #1a1a1a; }
        div[data-testid="stRadio"] label:first-of-type {
            border-top: 1px solid #DDD8D2 !important;
        }

        /* ── PRIMARY button (Continue / Begin / See my matches) ── */
        button[kind="primary"] {
            background: #1a1a1a !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 4px !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 400 !important;
            font-size: 0.75rem !important;
            letter-spacing: 0.12em !important;
            text-transform: uppercase !important;
            height: 52px !important;
            transition: background 0.2s !important;
            box-shadow: none !important;
        }
        button[kind="primary"]:hover {
            background: #2e2e2e !important;
            color: #ffffff !important;
        }
        button[kind="primary"]:disabled {
            background: #ccc8c3 !important;
            color: #f0ede9 !important;
            cursor: not-allowed !important;
        }

        /* ── SECONDARY button (Skip) — plain text link, no box ── */
        button[kind="secondary"] {
            background: transparent !important;
            color: #9a9490 !important;
            border: none !important;
            border-radius: 0 !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 300 !important;
            font-size: 0.83rem !important;
            letter-spacing: 0.02em !important;
            text-transform: none !important;
            height: auto !important;
            padding: 8px 0 !important;
            box-shadow: none !important;
            text-decoration: underline !important;
            text-decoration-color: #c5bfb8 !important;
            text-underline-offset: 3px !important;
        }
        button[kind="secondary"]:hover {
            color: #1a1a1a !important;
            background: transparent !important;
            text-decoration-color: #1a1a1a !important;
        }

        /* ── Quiz question text ── */
        .quiz-question {
            font-family: 'Cormorant Garamond', Georgia, serif;
            font-size: 1.75rem;
            font-weight: 300;
            line-height: 1.45;
            letter-spacing: 0.01em;
            color: #1a1a1a;
            margin: 2rem 0 2rem 0;
        }

        /* ── Step counter ── */
        .step-label {
            font-family: 'Inter', sans-serif;
            font-size: 0.7rem;
            font-weight: 400;
            letter-spacing: 0.22em;
            color: #9a9490;
            margin-top: 3.5rem;
            margin-bottom: 0;
            display: block;
        }

        /* ── Result card ── */
        .result-card {
            border: none;
            border-bottom: 1px solid #DDD8D2;
            padding: 26px 0;
            background: transparent;
        }
        .result-card:first-of-type { border-top: 1px solid #DDD8D2; }

        .card-top {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 4px;
        }
        .card-name {
            font-family: 'Cormorant Garamond', Georgia, serif;
            font-size: 1.3rem;
            font-weight: 400;
            color: #1a1a1a;
            margin: 0;
            letter-spacing: 0.02em;
        }
        .card-rank {
            font-family: 'Inter', sans-serif;
            font-size: 0.68rem;
            letter-spacing: 0.1em;
            color: #b5afa8;
            font-weight: 300;
            white-space: nowrap;
            margin-left: 16px;
            flex-shrink: 0;
        }
        .card-brand {
            font-family: 'Inter', sans-serif;
            font-size: 0.7rem;
            color: #9a9490;
            margin: 2px 0 10px 0;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-weight: 400;
        }
        .card-notes {
            font-family: 'Inter', sans-serif;
            font-size: 0.83rem;
            color: #5a5550;
            margin: 0 0 7px 0;
            font-weight: 300;
            line-height: 1.65;
        }
        .card-desc {
            font-family: 'Inter', sans-serif;
            font-size: 0.81rem;
            color: #9a9490;
            margin: 0;
            line-height: 1.65;
            font-weight: 300;
        }

        /* ── Results page header ── */
        .profile-eyebrow {
            font-family: 'Inter', sans-serif;
            font-size: 0.7rem;
            font-weight: 400;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            color: #9a9490;
            margin: 4rem 0 1rem 0;
            display: block;
        }
        .profile-name {
            font-family: 'Cormorant Garamond', Georgia, serif;
            font-size: 3rem;
            font-weight: 300;
            color: #1a1a1a;
            margin: 0 0 1.75rem 0;
            line-height: 1.1;
            letter-spacing: 0.03em;
        }
        .profile-quote {
            font-family: 'Cormorant Garamond', Georgia, serif;
            font-size: 1.2rem;
            font-style: italic;
            font-weight: 300;
            color: #9a9490;
            margin: 0 0 2.5rem 0;
            line-height: 1.55;
        }
        .results-eyebrow {
            font-family: 'Inter', sans-serif;
            font-size: 0.7rem;
            font-weight: 400;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            color: #9a9490;
            margin: 0 0 4px 0;
            display: block;
        }

        /* ── Intro hero ── */
        .hero-wordmark {
            font-family: 'Inter', sans-serif;
            font-size: 0.7rem;
            font-weight: 500;
            letter-spacing: 0.3em;
            text-transform: uppercase;
            color: #9a9490;
            margin: 4rem 0 3rem 0;
            display: block;
        }
        .hero-title {
            font-family: 'Cormorant Garamond', Georgia, serif;
            font-size: 3.4rem;
            font-weight: 300;
            letter-spacing: 0.02em;
            line-height: 1.12;
            color: #1a1a1a;
            margin: 0 0 1.5rem 0;
        }
        .hero-sub {
            font-family: 'Inter', sans-serif;
            font-size: 0.92rem;
            font-weight: 300;
            color: #6b6560;
            margin-bottom: 3rem;
            line-height: 1.85;
        }
        .hero-meta {
            font-family: 'Inter', sans-serif;
            font-size: 0.77rem;
            font-weight: 300;
            color: #b0aaa4;
            line-height: 2.1;
            margin-bottom: 3rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(row: pd.Series, rank: int):
    """Render one fragrance result as a minimal editorial row."""
    score_pct = int(row["similarity"] * 100)
    st.markdown(
        f"""
        <div class="result-card">
            <div class="card-top">
                <p class="card-name">{row['name']}</p>
                <span class="card-rank">{score_pct}% match</span>
            </div>
            <p class="card-brand">{row['brand']}</p>
            <p class="card-notes">{row['notes']}</p>
            <p class="card-desc">{row['description']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Screen renderers
# ─────────────────────────────────────────────

def show_intro():
    """Landing / intro screen."""
    st.markdown(
        """
        <span class="hero-wordmark">ScentMatch</span>
        <p class="hero-title">Find your<br>fragrance.</p>
        <p class="hero-sub">
            Answer nine questions about who you are.<br>
            We'll match you to a fragrance using AI.
        </p>
        <p class="hero-meta">
            Nine questions &nbsp;·&nbsp; Two minutes &nbsp;·&nbsp; No account needed
        </p>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Begin", type="primary", use_container_width=True):
        st.session_state.step = 1
        st.session_state.answers = [None] * 9
        st.rerun()


def _submit_text(step: int, idx: int):
    """
    on_change callback for text questions.
    Fires when the user presses Enter (or blurs the field).
    If the field has content, save the answer and advance to the next step.
    """
    val = st.session_state.get(f"q{step}_text", "").strip()
    if val:
        st.session_state.answers[idx] = val
        st.session_state.step = step + 1


def show_question(step: int):
    """Render a single quiz question (step = 1..9)."""
    q = QUESTIONS[step - 1]
    idx = step - 1

    # Thin progress line pinned to top of viewport
    st.progress((step - 1) / 9)

    st.markdown(
        f'<span class="step-label">{step} &nbsp;/&nbsp; 9</span>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<p class="quiz-question">{q["label"]}</p>', unsafe_allow_html=True)

    # ── Input widget ──
    answered = False

    if q["type"] == "text":
        val = st.text_input(
            "your answer",
            placeholder=q["placeholder"],
            key=f"q{step}_text",
            label_visibility="collapsed",
            on_change=_submit_text,
            args=(step, idx),
        )
        answered = bool(val.strip())

    else:  # multiple choice
        choice = st.radio(
            "pick one",
            q["options"],
            index=None,
            key=f"q{step}_choice",
            label_visibility="collapsed",
        )
        answered = choice is not None

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

    # ── Continue button (full width, primary) ──
    label = "See my matches" if step == 9 else "Continue"
    if st.button(label, disabled=not answered, type="primary", use_container_width=True):
        if q["type"] == "text":
            st.session_state.answers[idx] = st.session_state.get(f"q{step}_text", "").strip()
        else:
            st.session_state.answers[idx] = st.session_state.get(f"q{step}_choice")
        st.session_state.step = step + 1
        st.rerun()

    # ── Skip link — centered text below, no box ──
    _, col_mid, _ = st.columns([2, 1, 2])
    with col_mid:
        if st.button("skip", key=f"skip_{step}", use_container_width=True):
            st.session_state.answers[idx] = ""
            st.session_state.step = step + 1
            st.rerun()


def show_results(df: pd.DataFrame, embeddings: np.ndarray):
    """Results page: profile name, Q9 quote, recommendation cards, start-over."""
    answers = st.session_state.answers

    # Compile query and run recommendation (cached in session_state)
    query = compile_query(answers)
    if st.session_state.get("results_query") != query:
        with st.spinner("Finding your matches…"):
            st.session_state.results_df    = recommend(query, df, embeddings, top_k=TOP_K)
            st.session_state.results_query = query

    results = st.session_state.results_df
    profile = derive_scent_profile(answers)
    q9      = (answers[8] or "").strip()

    # ── Profile header ──
    st.markdown(
        f'<span class="profile-eyebrow">Your scent profile</span>'
        f'<p class="profile-name">{profile}</p>',
        unsafe_allow_html=True,
    )

    if q9:
        st.markdown(
            f'<p class="profile-quote">"I want to smell like someone who {q9}"</p>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<span class="results-eyebrow">Your matches</span>',
        unsafe_allow_html=True,
    )

    # ── Result cards ──
    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        render_result_card(row, rank)

    st.markdown("<div style='height:2.5rem'></div>", unsafe_allow_html=True)

    if st.button("Start over", use_container_width=True):
        reset_quiz()
        st.rerun()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="ScentMatch",
        page_icon="🌸",
        layout="centered",
    )
    inject_css()

    # Load data and pre-compute embeddings once (cached globally by Streamlit)
    df         = load_data()
    corpus     = build_corpus(df)
    embeddings = compute_embeddings(tuple(corpus))

    # Initialise quiz state on first load
    if "step" not in st.session_state:
        st.session_state.step    = 0
        st.session_state.answers = [None] * 9

    step = st.session_state.step

    if step == 0:
        show_intro()
    elif 1 <= step <= 9:
        show_question(step)
    else:
        show_results(df, embeddings)


if __name__ == "__main__":
    main()
