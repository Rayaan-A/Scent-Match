"""
ScentMatch — personality quiz → fragrance recommender.

Backend: sentence-transformers embeddings + cosine similarity.
Frontend: 9-question personality quiz that compiles answers into a semantic
          search query passed to the embedding backend.
Dataset: Fragrantica dataset (fra_cleaned.csv).
"""

import ast
import re
import pandas as pd
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import requests
import anthropic
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

FRA_PATH        = os.path.join(os.path.dirname(__file__), "data", "fra_cleaned.csv")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K           = 5


# ─────────────────────────────────────────────
# Backend
# ─────────────────────────────────────────────

def _parse_list_col(val) -> str:
    """Convert a stringified Python list like \"['a', 'b']\" to comma-joined text."""
    if not val or (isinstance(val, float)):
        return ""
    s = str(val).strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return ", ".join(str(x) for x in parsed)
    except Exception:
        pass
    return re.sub(r"[\[\]'\"]", "", s).strip()


def load_data() -> pd.DataFrame:
    """Load and normalise the Fragrantica dataset."""
    if not os.path.exists(FRA_PATH):
        raise FileNotFoundError(
            f"Fragrantica dataset not found at {FRA_PATH}.\n"
            "Run: python precompute_embeddings.py"
        )

    df = pd.read_csv(FRA_PATH, encoding="latin-1", on_bad_lines="skip", sep=None, engine="python")
    df.columns = df.columns.str.strip()

    rename_map = {
        "Perfume":      "name",
        "Brand":        "brand",
        "Gender":       "gender",
        "Rating Value": "rating",
        "Rating Count": "reviews",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Ratings use European comma-decimal format ("3,42" → "3.42")
    if "rating" in df.columns:
        df["rating"] = df["rating"].astype(str).str.replace(",", ".", regex=False).str.strip()

    # Combine Top/Middle/Base into a labelled notes string
    note_cols = [c for c in ["Top", "Middle", "Base"] if c in df.columns]
    df["notes"] = df[note_cols].fillna("").apply(
        lambda row: "  ·  ".join(
            f"{col}: {val.strip()}" for col, val in zip(note_cols, row) if str(val).strip()
        ),
        axis=1,
    )

    # Combine mainaccord columns into a comma-separated string
    accord_cols = [c for c in df.columns if c.startswith("mainaccord")]
    df["accords"] = df[accord_cols].fillna("").apply(
        lambda row: ", ".join(str(v).strip() for v in row if str(v).strip()),
        axis=1,
    )

    # Gender already lowercase in Fragrantica: "women" / "men" / "unisex"
    if "gender" in df.columns:
        df["gender"] = df["gender"].str.lower().str.strip().fillna("unisex")
    else:
        df["gender"] = "unisex"

    for col in ["name", "brand", "rating", "reviews", "notes", "accords", "url"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()

    df = df[df["name"] != ""].reset_index(drop=True)
    print(f"[load_data] Fragrantica products loaded: {len(df):,}")
    return df


def build_corpus(df: pd.DataFrame) -> list[str]:
    """Build embeddable text from name, brand, all note columns, accords, and country."""
    corpus = []
    note_cols = [c for c in ["Top", "Middle", "Base"] if c in df.columns]
    for _, row in df.iterrows():
        parts = [row["name"], row["brand"]]
        # Labeled notes string ("Top: X · Middle: Y · Base: Z")
        if row.get("notes"):
            parts.append(row["notes"])
        # Raw parsed notes — repeat ingredient names for denser keyword signal
        for col in note_cols:
            raw = _parse_list_col(row.get(col, ""))
            if raw:
                parts.append(raw)
        if row.get("accords"):
            parts.append(row["accords"])
        country = str(row.get("Country", "") or "").strip()
        if country:
            parts.append(country)
        corpus.append(" ".join(p for p in parts if p))
    return corpus


CACHE_PATH = os.path.join(os.path.dirname(__file__), "embeddings_cache.npy")
PKL_PATH   = os.path.join(os.path.dirname(__file__), "data", "fra_processed.pkl")


@st.cache_resource(show_spinner="Loading model…")
def load_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource(show_spinner="Loading fragrances…")
def load_data_and_embeddings():
    """
    Load DataFrame + embeddings from pre-computed cache if available,
    otherwise compute on the fly and save for next time.
    """
    if os.path.exists(PKL_PATH) and os.path.exists(CACHE_PATH):
        df         = pd.read_pickle(PKL_PATH)
        embeddings = np.load(CACHE_PATH)
        return df, embeddings

    # First-run fallback: compute and save
    df         = load_data()
    corpus     = build_corpus(df)
    model      = load_model()
    embeddings = model.encode(corpus, convert_to_numpy=True, show_progress_bar=False)
    np.save(CACHE_PATH, embeddings)
    df.to_pickle(PKL_PATH)
    return df, embeddings


def recommend(
    query: str,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    gender: str = "Show me everything",
    top_k: int = 5,
):
    full_query = query

    # ── Gender hard-filter ──
    if gender != "Show me everything" and "gender" in df.columns:
        allowed = {"men"} if gender == "Men's" else {"women"}
        # include unisex always
        mask = df["gender"].isin(allowed | {"unisex"})
        filt_df  = df[mask].reset_index(drop=True)
        filt_emb = embeddings[mask.values]
    else:
        filt_df  = df
        filt_emb = embeddings

    # ── Men's: exclude feminine-skewing unisex products and boost genuine men's ──
    if gender == "Men's":
        _MENS_EXCLUSIONS = [
            "jo malone", "kayali", "blossom", "honey", "juicy", "rose",
            "floral", "peony", "cherry", "lace", "bridal", "nude",
        ]
        combined = (filt_df["name"] + " " + filt_df["brand"]).str.lower()
        exclude_mask = combined.apply(
            lambda s: any(term in s for term in _MENS_EXCLUSIONS)
        ) & (filt_df["gender"] == "unisex")
        filt_emb = filt_emb[~exclude_mask.values]
        filt_df  = filt_df[~exclude_mask].reset_index(drop=True)

    model     = load_model()
    query_vec = model.encode([full_query], convert_to_numpy=True)
    scores    = cosine_similarity(query_vec, filt_emb)[0]

    # ── Men's: boost genuine men's products so they rank above unisex ──
    if gender == "Men's" and "gender" in filt_df.columns:
        scores = scores.copy()
        scores[filt_df["gender"].values == "men"] += 0.05

    # ── Debug: print query + top-10 scores to terminal ──
    print(f"\n[DEBUG] Query ({len(full_query)} chars):\n  {full_query[:300]}")
    print(f"[DEBUG] Gender filter: {gender}  |  Pool size: {len(filt_df)}")
    print("[DEBUG] Top 10 scores:")
    top10 = np.argsort(scores)[::-1][:10]
    for rank, idx in enumerate(top10, 1):
        print(f"  {rank:2d}. {filt_df.iloc[idx]['name'][:40]:40s}  {filt_df.iloc[idx]['brand'][:20]:20s}  {scores[idx]:.4f}")
    print(f"[DEBUG] Score range (top-10): {scores[top10[-1]]:.4f} – {scores[top10[0]]:.4f}\n")

    # ── Diversity: pool top-20, deduplicate by brand, take top_k ──
    pool_size = min(top_k * 4, len(filt_df))
    pool_idx  = np.argsort(scores)[::-1][:pool_size]
    pool_df   = filt_df.iloc[pool_idx].copy()
    pool_df["similarity"] = scores[pool_idx]

    seen_brands: set[str] = set()
    deduped: list[pd.Series] = []
    for _, row in pool_df.iterrows():
        brand = row["brand"].lower().strip()
        if brand not in seen_brands:
            seen_brands.add(brand)
            deduped.append(row)
        if len(deduped) >= top_k:
            break

    results = pd.DataFrame(deduped).reset_index(drop=True)
    return results


# ─────────────────────────────────────────────
# Community insight layer (Reddit → Claude)
# ─────────────────────────────────────────────

def _fetch_reddit_texts(fragrance_name: str, brand: str) -> list[str]:
    """Search r/fragrance for posts about this fragrance and return raw text snippets."""
    query = f"{fragrance_name} {brand}"
    params = {"q": query, "limit": 8, "sort": "top", "t": "all", "restrict_sr": 1}
    headers = {"User-Agent": "ScentMatch/1.0 (fragrance recommendation app)"}
    try:
        r = requests.get(
            "https://www.reddit.com/r/fragrance/search.json",
            params=params, headers=headers, timeout=5,
        )
        r.raise_for_status()
        texts = []
        for post in r.json()["data"]["children"]:
            d = post["data"]
            if d.get("title"):
                texts.append(d["title"])
            body = d.get("selftext", "").strip()
            if body and len(body) > 40:
                texts.append(body[:500])
        return texts
    except Exception:
        return []


def _summarize_with_claude(fragrance_name: str, texts: list[str]) -> str:
    """
    Turn raw community text into one or two natural sentences using Claude Haiku.
    Haiku is intentionally chosen here — this runs once per result card per search,
    so cost matters. Falls back silently if ANTHROPIC_API_KEY is not set.
    """
    if not texts:
        return ""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return ""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        combined = "\n".join(texts[:6])
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=120,
            messages=[{
                "role": "user",
                "content": (
                    f"Based on these fragrance enthusiast discussions about {fragrance_name}, "
                    "write 1-2 natural sentences describing how wearers experience it and what makes it "
                    "distinctive. Third person, present tense. No attribution, no quotes, no 'people say' "
                    "— just flowing descriptive prose.\n\n"
                    f"Discussions:\n{combined}"
                ),
            }],
        )
        return response.content[0].text.strip()
    except Exception:
        return ""


def fetch_all_summaries(results_df: pd.DataFrame) -> dict[str, str]:
    """
    Fetch Reddit data and generate Claude summaries for all result fragrances in parallel.
    Results are stored in st.session_state to avoid re-fetching on Streamlit reruns.
    """
    cache = st.session_state.setdefault("summary_cache", {})

    def fetch_one(name: str, brand: str) -> tuple[str, str]:
        key = f"{name}::{brand}"
        if key in cache:
            return name, cache[key]
        texts  = _fetch_reddit_texts(name, brand)
        summary = _summarize_with_claude(name, texts)
        cache[key] = summary
        return name, summary

    summaries: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {
            pool.submit(fetch_one, row["name"], row["brand"]): row["name"]
            for _, row in results_df.iterrows()
        }
        for future in as_completed(futures):
            name, summary = future.result()
            summaries[name] = summary

    return summaries


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


_ENVIRONMENT_MAP = {
    "Dense forest":          "woody mossy earthy vetiver oakmoss green dark",
    "Open ocean":            "aquatic marine fresh ozonic citrus light clean",
    "Urban rooftop at night":"smoky amber leather dark woody musk urban",
    "Desert at sunrise":     "dry warm spicy resinous amber incense arid",
    "Garden after rain":     "floral green fresh petrichor soft dewy rose",
}

_PALATE_MAP = {
    "Fresh & bright":      "citrus bergamot mint grapefruit light fresh clean",
    "Warm & rich":         "vanilla amber sandalwood spice dark chocolate gourmand",
    "Somewhere in between":"woody musk balanced moderate clean warm",
    "Depends on my mood":  "versatile fresh woody",
}

_MOOD_KEYWORD_MAP = [
    (["comfort", "cozy", "warm"],           "vanilla musk amber soft"),
    (["confident", "bold", "strong"],        "oud leather woody spicy intense"),
    (["fresh", "clean", "light"],            "citrus aquatic bergamot clean"),
    (["mysterious", "dark", "deep"],         "oud smoky resinous dark amber"),
    (["natural", "earthy", "outdoors"],      "vetiver moss green woody earth"),
    (["sweet", "dessert", "food"],           "vanilla gourmand caramel praline"),
    (["floral", "flowers", "garden"],        "rose jasmine peony iris floral"),
    (["ocean", "beach", "sea", "water"],     "marine aquatic salt fresh"),
    (["rain"],                               "petrichor green fresh ozonic"),
    (["coffee", "cafe", "bookshop"],         "warm woody tobacco vanilla"),
    (["wood", "forest", "trees"],            "cedar sandalwood vetiver pine"),
    (["citrus", "fruit", "lemon", "orange"], "bergamot citrus grapefruit"),
]

_AGE_MAP = {
    "18-24": "fresh light citrus clean modern",
    "25-34": "",
    "35-44": "complex woody amber sophisticated",
    "45+":   "rich classic oriental deep sillage",
}

_AUDIENCE_MAP = {
    "Mostly for others": "sillage projection bold lasting",
    "Mostly for myself": "skin scent subtle personal intimate",
    "Both equally":      "moderate balanced",
}


def build_fragrance_query(answers: list, age: str = "") -> str:
    """
    Translate raw quiz answers into a fragrance-vocabulary-enriched query string.
    Returns: "[original compiled answers]. Fragrance profile: [translated terms]"
    """
    a = [x or "" for x in answers]

    # ENVIRONMENT (answers[3])
    env_terms = _ENVIRONMENT_MAP.get(a[3], "")

    # PALATE (answers[4]) — partial match because options carry extra parenthetical text
    palate_terms = ""
    for key, val in _PALATE_MAP.items():
        if key.lower() in a[4].lower():
            palate_terms = val
            break

    # MOOD/INTENTION — scan all open-text answers
    text_blob = " ".join([a[0], a[1], a[2], a[5], a[8]]).lower()
    mood_terms = [
        terms for keywords, terms in _MOOD_KEYWORD_MAP
        if any(kw in text_blob for kw in keywords)
    ]

    # AGE
    age_terms = _AGE_MAP.get(age, "")

    # AUDIENCE (answers[6])
    audience_terms = _AUDIENCE_MAP.get(a[6], "")

    translated = " ".join(
        t for t in [env_terms, palate_terms, " ".join(mood_terms), age_terms, audience_terms]
        if t
    )

    raw_answers = ". ".join(x for x in a if x.strip())
    return f"{raw_answers}. Fragrance profile: {translated}".strip()


def compile_query(answers: list, age: str = "") -> str:
    return build_fragrance_query(answers, age=age)


def reset_quiz():
    """Clear all quiz state so the app returns to the intro screen."""
    st.session_state.step = 0
    st.session_state.answers = [None] * 9
    st.session_state.pop("demo_age", None)
    st.session_state.pop("demo_gender", None)
    st.session_state.pop("results_df", None)
    st.session_state.pop("results_query", None)
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
        /* Hide "Press Enter to apply" hint */
        div[data-testid="stTextInput"] small,
        [data-testid="InputInstructions"] { display: none !important; }

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

        /* ── Result card — white card on cream background ── */
        .result-card {
            background: #ffffff;
            border-radius: 12px;
            padding: 24px 26px 20px 26px;
            margin-bottom: 16px;
            box-shadow: 0 1px 10px rgba(0,0,0,0.06), 0 0 1px rgba(0,0,0,0.04);
        }

        .card-top {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 2px;
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
        .card-meta {
            display: flex;
            align-items: center;
            gap: 16px;
            margin: 0 0 12px 0;
        }
        .card-price {
            font-family: 'Inter', sans-serif;
            font-size: 0.9rem;
            font-weight: 400;
            color: #1a1a1a;
        }
        .card-stars {
            color: #C8A96E;
            font-size: 0.85rem;
            letter-spacing: 0.05em;
        }
        .card-reviews {
            font-family: 'Inter', sans-serif;
            font-size: 0.75rem;
            color: #9a9490;
            font-weight: 300;
        }
        .card-highlights {
            font-family: 'Inter', sans-serif;
            font-size: 0.82rem;
            color: #5a5550;
            margin: 0 0 14px 0;
            font-weight: 300;
            line-height: 1.65;
        }
        .card-community {
            font-family: 'Cormorant Garamond', Georgia, serif;
            font-size: 0.95rem;
            font-style: italic;
            font-weight: 300;
            color: #7a7470;
            margin: 14px 0 14px 0;
            line-height: 1.6;
            border-top: 1px solid #F0EDE8;
            padding-top: 12px;
        }
        .shop-btn {
            display: inline-block;
            background: #1a1a1a;
            color: #ffffff !important;
            text-decoration: none !important;
            font-family: 'Inter', sans-serif;
            font-size: 0.72rem;
            font-weight: 400;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            padding: 11px 20px;
            border-radius: 4px;
            transition: background 0.2s;
        }
        .shop-btn:hover { background: #2e2e2e; }

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

        /* ── Demographics chip grid ── */
        .chip-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 1.5rem 0 2.5rem 0;
        }
        .chip {
            font-family: 'Inter', sans-serif;
            font-size: 0.82rem;
            font-weight: 300;
            color: #5a5550;
            background: transparent;
            border: 1px solid #C8C2BB;
            border-radius: 100px;
            padding: 9px 20px;
            cursor: pointer;
            transition: all 0.15s;
            letter-spacing: 0.02em;
        }
        .chip:hover { border-color: #8a8480; color: #1a1a1a; }
        .chip.selected {
            background: #1a1a1a;
            border-color: #1a1a1a;
            color: #ffffff;
        }
        .demo-section-label {
            font-family: 'Inter', sans-serif;
            font-size: 0.7rem;
            font-weight: 400;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #9a9490;
            margin: 2rem 0 0.5rem 0;
            display: block;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_stars(rating_str: str) -> str:
    try:
        r = float(rating_str)
        if r <= 0:
            return ""
    except (ValueError, TypeError):
        return ""
    stars = ""
    for i in range(5):
        stars += "★" if r >= i + 0.75 else ("½" if r >= i + 0.25 else "☆")
    return stars


def _format_reviews(reviews_str: str) -> str:
    try:
        n = int(float(reviews_str))
        return f"{n/1000:.1f}k reviews" if n >= 1000 else f"{n} reviews"
    except (ValueError, TypeError):
        return ""


def _format_price(price_str: str) -> str:
    try:
        return f"${float(price_str):.2f}"
    except (ValueError, TypeError):
        return price_str


def render_result_card(row: pd.Series, rank: int, summary: str = ""):
    """Render one Fragrantica product as a white card with rating, notes, and accords."""
    import html as _html
    score_pct = int(row["similarity"] * 100)

    stars   = _render_stars(row.get("rating", ""))
    reviews = _format_reviews(row.get("reviews", ""))

    meta_html = ""
    if stars:
        meta_html = (
            f'<div class="card-meta">'
            f'<span class="card-stars">{stars}</span>'
            f'<span class="card-reviews">{reviews}</span>'
            f'</div>'
        )

    notes = _html.escape((row.get("notes", "") or "").strip())
    notes_html = f'<p class="card-highlights">{notes}</p>' if notes else ""

    accords = _html.escape((row.get("accords", "") or "").strip())
    accords_html = (
        f'<p class="card-highlights" style="color:#9a9490;font-size:0.78rem;">{accords}</p>'
        if accords else ""
    )

    safe_summary = _html.escape(summary) if summary else ""
    community_html = f'<p class="card-community">{safe_summary}</p>' if safe_summary else ""

    url = (row.get("url", "") or "").strip()
    link_html = (
        f'<a class="shop-btn" href="{_html.escape(url)}" target="_blank" rel="noopener">View on Fragrantica →</a>'
        if url else ""
    )

    name  = _html.escape(str(row["name"]))
    brand = _html.escape(str(row["brand"]))

    st.markdown(
        f"""
        <div class="result-card">
            <div class="card-top">
                <p class="card-name">{name}</p>
                <span class="card-rank">{score_pct}% match</span>
            </div>
            <p class="card-brand">{brand}</p>
            {meta_html}
            {notes_html}
            {accords_html}
            {community_html}
            {link_html}
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

        # Enable Continue button as soon as user types — no Enter required
        import streamlit.components.v1 as components
        components.html(
            """
            <script>
            (function() {
                function wire(attempts) {
                    if (attempts <= 0) return;
                    var doc = window.parent.document;
                    var inp = doc.querySelector('[data-testid="stTextInput"] input');
                    var btn = Array.from(doc.querySelectorAll('button[kind="primary"]'))
                                  .find(function(b) {
                                      return /Continue|See my matches/.test(b.textContent);
                                  });
                    if (!inp || !btn) { setTimeout(function(){ wire(attempts-1); }, 100); return; }
                    inp.addEventListener('input', function() {
                        btn.disabled = inp.value.trim().length === 0;
                    });
                }
                wire(30);
            })();
            </script>
            """,
            height=0,
        )

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


def show_demographics():
    """Step 10: two quick chip questions before results."""
    st.progress(1.0)
    st.markdown('<span class="step-label">Almost there</span>', unsafe_allow_html=True)
    st.markdown('<p class="quiz-question">One last thing — help us personalise your results.</p>', unsafe_allow_html=True)

    age_options    = ["18-24", "25-34", "35-44", "45+"]
    gender_options = ["Men's", "Women's", "Show me everything"]

    age    = st.session_state.get("demo_age", None)
    gender = st.session_state.get("demo_gender", None)

    # Age chips
    st.markdown('<span class="demo-section-label">How old are you?</span>', unsafe_allow_html=True)
    age_cols = st.columns(len(age_options))
    for i, opt in enumerate(age_options):
        selected = age == opt
        label = f"**{opt}**" if selected else opt
        if age_cols[i].button(opt, key=f"age_{opt}", type="primary" if selected else "secondary", use_container_width=True):
            st.session_state.demo_age = opt
            st.rerun()

    # Gender chips
    st.markdown('<span class="demo-section-label">Which fragrances should we focus on?</span>', unsafe_allow_html=True)
    gender_cols = st.columns(len(gender_options))
    for i, opt in enumerate(gender_options):
        selected = gender == opt
        if gender_cols[i].button(opt, key=f"gender_{opt}", type="primary" if selected else "secondary", use_container_width=True):
            st.session_state.demo_gender = opt
            st.rerun()

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

    ready = age is not None and gender is not None
    if st.button("See my matches", disabled=not ready, type="primary", use_container_width=True):
        st.session_state.step = 11
        st.rerun()


def show_results(df: pd.DataFrame, embeddings: np.ndarray):
    """Results page: profile name, Q9 quote, recommendation cards, start-over."""
    answers = st.session_state.answers

    age    = st.session_state.get("demo_age", "25-34")
    gender = st.session_state.get("demo_gender", "Show me everything")

    # Compile query and run recommendation (cached in session_state)
    query    = compile_query(answers, age=age)
    cache_key = f"{query}|{gender}"
    if st.session_state.get("results_query") != cache_key:
        with st.spinner("Finding your matches…"):
            st.session_state.results_df    = recommend(query, df, embeddings, gender=gender, top_k=TOP_K)
            st.session_state.results_query = cache_key

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
        '<span class="results-eyebrow">Your matches</span>',
        unsafe_allow_html=True,
    )

    # ── Fetch community summaries (parallel, cached in session_state) ──
    with st.spinner("Gathering community insights…"):
        summaries = fetch_all_summaries(results)

    # ── Result cards ──
    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        render_result_card(row, rank, summaries.get(row["name"], ""))

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
    st.markdown("""
<script>
document.querySelectorAll('input, textarea').forEach(el => {
    el.setAttribute('autocomplete', 'off');
});
</script>
""", unsafe_allow_html=True)

    # Load data + embeddings (from cache file if available, else compute once)
    df, embeddings = load_data_and_embeddings()

    # Initialise quiz state on first load
    if "step" not in st.session_state:
        st.session_state.step    = 0
        st.session_state.answers = [None] * 9

    step = st.session_state.step

    if step == 0:
        show_intro()
    elif 1 <= step <= 9:
        show_question(step)
    elif step == 10:
        show_demographics()
    else:
        show_results(df, embeddings)


if __name__ == "__main__":
    main()
