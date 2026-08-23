"""
ScentMatch — personality quiz → fragrance recommender.

Quiz flow:
  step 0  : intro
  step 1  : Phase 0 — occasion multi-select
  steps 2-12 : Phase 1 — 11 personality questions
  step 13 : demographics (age / budget / gender)
  step 14 : email capture (background API call running)
  step 15 : results
"""

import html as _html
import json
import os
import re
import threading
import time
import uuid

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

TOTAL_STEPS = 13  # steps 1-13 are quiz/demo; used for progress bar

# ─────────────────────────────────────────────
# Background result store (module-level, persists across reruns)
# ─────────────────────────────────────────────

# Streamlit exec()s the script into a BRAND NEW module namespace on every rerun
# (ScriptRunner._run_script -> _new_module("__main__")). Plain module-level
# globals therefore do NOT persist across reruns - they are reset to their
# initial value each time, so a background thread would write its result into a
# dict that the next rerun can no longer see, and the results page would poll
# forever. @st.cache_resource returns the same object across reruns, which is
# what actually gives us a stable place to hand results back to.
@st.cache_resource
def _bg_store() -> dict:
    return {"results": {}, "lock": threading.Lock()}


def _run_in_background(store: dict, request_id: str, api_key: str, occasions, answers, age, budget, gender):
    try:
        profile, frags = get_gemini_recommendations(occasions, answers, age, budget, gender, api_key=api_key)
        payload = {"profile": profile, "frags": frags, "error": None}
    except Exception as e:
        payload = {"profile": None, "frags": None, "error": str(e)}
    with store["lock"]:
        store["results"][request_id] = payload


# ─────────────────────────────────────────────
# Quiz definitions
# ─────────────────────────────────────────────

PHASE0_OCCASIONS = [
    "Daily signature",
    "Summer / hot weather",
    "Night outs / dates / parties",
    "Office / professional settings",
    "Casual weekends",
    "Cold weather / winter",
    "Special occasions / formal events",
    "A gift for someone else",
]

PHASE1_QUESTIONS = [
    {
        "type": "text",
        "label": "You walk into a party where you know maybe 2 people. What are you actually doing for the first 20 minutes?",
        "placeholder": "grab a drink and observe, find my people and stick close, talk to strangers...",
    },
    {
        "type": "text",
        "label": "Describe your aesthetic in 3 words. Not aspirational — what you actually wear on a regular Tuesday.",
        "placeholder": "clean quiet minimal... cozy chaotic librarian... dark academia soft...",
    },
    {
        "type": "text",
        "label": "Free Saturday with zero obligations. Walk me through it.",
        "placeholder": "sleep in, long coffee, walk... gym at 6am, brunch, errands... nothing before noon...",
    },
    {
        "type": "text",
        "label": "Describe a smell — not a fragrance, just a smell — that instantly makes you feel good.",
        "placeholder": "petrichor after rain, old bookshops, sunscreen at the beach, fresh laundry...",
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
        "type": "text",
        "label": "How do you want people to feel when you enter a room? Not how you want to look — how you want them to feel.",
        "placeholder": "comfortable, intrigued, like something shifted, calm, energised...",
    },
    {
        "type": "text",
        "label": "Describe a color palette that matches your vibe. Pick 3 colors.",
        "placeholder": "navy, cream, aged leather... terracotta, sand, olive... black, grey, white...",
    },
    {
        "type": "multi",
        "label": "Which scent directions pull you? Pick everything that applies.",
        "options": [
            "Fresh & aquatic (ocean, sea salt, clean water)",
            "Fresh & citrus (lemon, bergamot, grapefruit)",
            "Fresh & green (grass, herbs, cucumber)",
            "Floral (rose, jasmine, peony, iris)",
            "Warm & spicy (cardamom, pepper, ginger)",
            "Warm & sweet (vanilla, caramel, tonka)",
            "Woody & earthy (cedar, sandalwood, vetiver)",
            "Smoky & dark (oud, leather, tobacco, incense)",
            "Powdery & soft (musk, iris, violet)",
        ],
    },
    {
        "type": "choice",
        "label": "Is your fragrance more for others, or for yourself?",
        "options": [
            "Mostly for others — I want to leave an impression",
            "Mostly for myself — it's personal",
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

_Q_LABELS = [
    "Party behavior (first 20 min)",
    "Aesthetic in 3 words",
    "Free Saturday walkthrough",
    "Smell that makes them feel good",
    "Environment that fits them",
    "How they want people to feel when they enter",
    "Color palette (3 colors)",
    "Scent direction preference",
    "Fragrance for others or themselves",
    "One signature or rotating",
    "I want to smell like someone who...",
]


# ─────────────────────────────────────────────
# Gemini
# ─────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert fragrance analyst. A user has completed a quiz on a website and their answers will be passed to you as structured data. Your job is to analyze their answers and return personalized fragrance recommendations immediately — no questions, no back and forth, just your analysis and output.

---

## INPUT FORMAT

You will receive the user's quiz answers in this format:

OCCASION: [what they are shopping for — daily, summer, night out, office, gift, etc.]
AGE: [their age]
BUDGET: [per bottle in USD]
SOCIAL STYLE: [their answer to the party/social question]
AESTHETIC: [3 words describing their style]
FREE DAY: [how they'd spend a free Saturday]
SMELL THEY LOVE: [non-fragrance smell they enjoy]
ENVIRONMENT: [forest, ocean, rooftop, desert, or garden]
ROOM PRESENCE: [how they want people to feel when they enter]
COLOR PALETTE: [their vibe in colors]
FLAVOR PALATE: [fresh/bright or warm/rich]
NOTICEABILITY: [for self or for others]
ROTATION: [one signature or multiple]
IDENTITY STATEMENT: [their "I want to smell like someone who..." answer]

---

## YOUR TASK

Step 1 — BUILD AN INTERNAL PROFILE
Before recommending anything, silently analyze:
- Personality type (introverted/extroverted, bold/understated, energetic/calm)
- What scent families suit them (citrus, aquatic, woody, oriental, green, gourmand, floral, fruity)
- What constraints apply (age maturity, budget, occasion)
- What they actually need vs what they think they want

Step 2 — APPLY THESE HARD RULES before recommending anything:

AGE RULE: If the user is under 24, do not recommend fragrances that skew 30+

BUDGET RULE: Never recommend a fragrance above their stated budget. If a slightly over-budget option is worth flagging, mention it once as an "if you can stretch" aside — never as a primary recommendation.

OCCASION RULE: Every recommendation must be appropriate for the stated occasion. Do not recommend a heavy night-out fragrance for daily summer wear, or a thin aquatic for a cold-weather formal occasion.

Step 3 — RECOMMEND 3 TO 5 FRAGRANCES

Always include at minimum:
- One SAFE pick — accessible, crowd-pleasing, low risk
- One UNIQUE pick — underrated, less talked about, more character
- One SIGNATURE pick — the deepest match to who they are

If they selected multiple occasions, label each pick with its best occasion.

---

## OUTPUT FORMAT

Return a single JSON object with exactly two keys — no markdown, no backticks, nothing else:

{
  "profile_summary": "1-2 sentences. Sharp and specific. Who is this person and what do they need from a fragrance. Reference their actual answers. Second person.",
  "fragrances": [
    {
      "name": "fragrance name",
      "brand": "house name",
      "family": "fragrance family (e.g. woody aromatic, fresh citrus)",
      "top_notes": "top notes",
      "middle_notes": "middle / heart notes",
      "base_notes": "base notes",
      "why": "2-3 sentences tied directly to something they said",
      "price": "price range with $ sign (e.g. '$85' or '$60–$120')",
      "pick_type": "Safe Pick | Unique Pick | Signature Pick",
      "heat_performance": "one sentence on how it performs in heat",
      "caveat": "one honest caveat"
    }
  ]
}

## TONE RULES

- Be direct. No fluff, no filler, no generic statements
- Every sentence must be traceable to something the user actually said
- Do not use fragrance clichés like "versatile for any occasion" or "perfect for day to night"
- Write like a knowledgeable friend who happens to know everything about fragrance — not like a perfume counter salesperson"""


def _build_prompt(occasions, answers, age, budget, gender):
    a = [x or "(skipped)" for x in answers]
    occasion_str = ", ".join(occasions) if occasions else "not specified"
    return (
        f"OCCASION: {occasion_str}\n"
        f"AGE: {age}\n"
        f"BUDGET: {budget} per bottle\n"
        f"SOCIAL STYLE: {a[0]}\n"
        f"AESTHETIC: {a[1]}\n"
        f"FREE DAY: {a[2]}\n"
        f"SMELL THEY LOVE: {a[3]}\n"
        f"ENVIRONMENT: {a[4]}\n"
        f"ROOM PRESENCE: {a[5]}\n"
        f"COLOR PALETTE: {a[6]}\n"
        f"FLAVOR PALATE: {a[7]}\n"
        f"NOTICEABILITY: {a[8]}\n"
        f"ROTATION: {a[9]}\n"
        f"IDENTITY STATEMENT: {a[10]}\n"
        f"GENDER PREFERENCE: {gender}"
    )


# gemini-flash-latest sheds load under demand (503s). The SDK silently retries
# 429/500/503 five times with exponential backoff, which is what turned a ~12s
# call into a 2-3 minute spinner. Pin the model and cap the retries so a bad
# minute fails fast instead of hanging the user.
# Model choice is a quality decision, not just a speed one. flash-lite answers in
# ~3s but ignores the BUDGET RULE in _SYSTEM_PROMPT (~1 in 3 picks came back over
# the user's stated ceiling, rationalised with "buy a decant"). gemini-3.6-flash
# takes ~15s and respected the ceiling on every pick, so it wins here.
# Tradeoff: free tier allows only 20 requests/day for this model.
_MODEL = "gemini-2.0-flash"

_HTTP_OPTIONS = types.HttpOptions(
    timeout=45_000,  # milliseconds
    retry_options=types.HttpRetryOptions(attempts=2, initial_delay=1.0, max_delay=4.0),
)


def _friendly_error(err: str) -> str:
    """Turn raw Gemini API errors into something a visitor can read."""
    e = err or ""
    if "RESOURCE_EXHAUSTED" in e or "429" in e:
        return (
            "We're getting more requests than our current plan allows. "
            "This resets daily - please try again later."
        )
    if "UNAVAILABLE" in e or "503" in e:
        return "The recommendation service is busy right now. Give it a moment and try again."
    if "DEADLINE" in e or "timeout" in e.lower():
        return "That took longer than expected. Please try again."
    if "GEMINI_API_KEY" in e:
        return "The app isn't configured correctly. (Missing API key.)"
    return "Something went wrong finding your matches. Please try again."


def _resolve_api_key(api_key: str = "") -> str:
    if api_key:
        return api_key
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        try:
            key = st.secrets["GEMINI_API_KEY"]
        except (KeyError, FileNotFoundError):
            key = ""
    return key


def get_gemini_recommendations(occasions, answers, age, budget, gender, api_key: str = ""):
    api_key = _resolve_api_key(api_key)
    if not api_key:
        raise ValueError("GEMINI_API_KEY not configured")

    prompt = _build_prompt(occasions, answers, age, budget, gender)
    client = genai.Client(api_key=api_key, http_options=_HTTP_OPTIONS)
    response = client.models.generate_content(
        model=_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            temperature=0.7,
            # Thinking tokens count against max_output_tokens. The JSON payload
            # alone is ~700 tokens, so a 1500 cap left nothing for the answer and
            # every response came back truncated (finish_reason=MAX_TOKENS).
            # Thinking tokens share this budget; ~950 thinking + ~720 output observed.
            max_output_tokens=4096,
            response_mime_type="application/json",
        ),
        contents=prompt,
    )
    if response.candidates and response.candidates[0].finish_reason.name != "STOP":
        raise RuntimeError(f"model stopped early: {response.candidates[0].finish_reason.name}")

    raw = (response.text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    parsed = json.loads(raw)
    profile_summary = parsed.get("profile_summary", "")
    fragrances = parsed.get("fragrances", [])
    return profile_summary, fragrances


# ─────────────────────────────────────────────
# State helpers
# ─────────────────────────────────────────────

def reset_quiz():
    for key in [
        "step", "occasions", "answers",
        "demo_age", "demo_budget", "demo_budget_range", "demo_gender",
        "results_profile", "results_data", "results_key", "results_error",
        "bg_request_id", "user_email",
    ]:
        st.session_state.pop(key, None)
    for i in range(1, 14):
        st.session_state.pop(f"q{i}_text", None)
        st.session_state.pop(f"q{i}_choice", None)
        multi_key = f"q{i}_multi"
        st.session_state.pop(multi_key, None)
        for q in PHASE1_QUESTIONS:
            if q["type"] == "multi":
                for opt in q["options"]:
                    st.session_state.pop(f"{multi_key}_{opt}", None)
    st.session_state.pop("occ_radio", None)


# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────

def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;1,300;1,400&family=Inter:wght@300;400;500&display=swap');

        /* ── Hide Streamlit chrome ── */
        #MainMenu, footer, header { visibility: hidden; }

        /* ── Warm gradient background ── */
        html, body {
            background: linear-gradient(160deg, #F5F0EB 0%, #EDE8E2 100%) !important;
            background-attachment: fixed !important;
            min-height: 100vh;
        }
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        section[data-testid="stSidebar"] { background: transparent !important; }
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
            top: 0; left: 0;
            width: 100vw;
            z-index: 9999;
            padding: 0; margin: 0;
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

        /* ── Text input ── */
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
        div[data-testid="stTextInput"] small,
        [data-testid="InputInstructions"] { display: none !important; }

        /* ── Radio options ── */
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
            display: flex;
            align-items: center;
            gap: 12px;
        }
        div[data-testid="stRadio"] label:hover { color: #1a1a1a; }
        div[data-testid="stRadio"] label:first-of-type {
            border-top: 1px solid #DDD8D2 !important;
        }

        /* ── Checkboxes (Phase 0 occasions) — styled like radio rows ── */
        div[data-testid="stCheckbox"] {
            border-bottom: 1px solid #DDD8D2;
            padding: 13px 0;
        }
        div[data-testid="stCheckbox"]:first-of-type {
            border-top: 1px solid #DDD8D2;
        }
        div[data-testid="stCheckbox"] label {
            font-size: 0.96rem !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 300 !important;
            color: #2a2520 !important;
            gap: 12px !important;
        }
        div[data-testid="stCheckbox"] label:hover { color: #1a1a1a !important; }

        /* ── PRIMARY button ── */
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
        button[kind="primary"]:hover { background: #2e2e2e !important; color: #ffffff !important; }
        button[kind="primary"]:disabled {
            background: #ccc8c3 !important;
            color: #f0ede9 !important;
            cursor: not-allowed !important;
        }

        /* ── SECONDARY button (Skip / Start over) ── */
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
        .card-pick-type {
            font-family: 'Inter', sans-serif;
            font-size: 0.64rem;
            letter-spacing: 0.1em;
            color: #b5afa8;
            font-weight: 400;
            white-space: nowrap;
            margin-left: 16px;
            flex-shrink: 0;
            text-transform: uppercase;
        }
        .card-pick-type.signature { color: #8a7a6a; }
        .card-brand {
            font-family: 'Inter', sans-serif;
            font-size: 0.7rem;
            color: #9a9490;
            margin: 2px 0 10px 0;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-weight: 400;
        }
        .card-family {
            font-family: 'Inter', sans-serif;
            font-size: 0.72rem;
            color: #b5afa8;
            margin: 0 0 12px 0;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 300;
        }
        .card-notes {
            font-family: 'Inter', sans-serif;
            font-size: 0.82rem;
            color: #5a5550;
            margin: 0 0 10px 0;
            font-weight: 300;
            line-height: 1.65;
        }
        .card-meta {
            font-family: 'Inter', sans-serif;
            font-size: 0.78rem;
            color: #9a9490;
            margin: 0 0 6px 0;
            font-weight: 300;
        }
        .card-heat {
            font-family: 'Inter', sans-serif;
            font-size: 0.78rem;
            color: #9a9490;
            margin: 0 0 12px 0;
            font-weight: 300;
            font-style: italic;
        }
        .card-why {
            font-family: 'Cormorant Garamond', Georgia, serif;
            font-size: 0.95rem;
            font-style: italic;
            font-weight: 300;
            color: #7a7470;
            margin: 12px 0 0 0;
            line-height: 1.6;
            border-top: 1px solid #F0EDE8;
            padding-top: 12px;
        }
        .card-caveat {
            font-family: 'Inter', sans-serif;
            font-size: 0.76rem;
            color: #b5afa8;
            margin: 8px 0 0 0;
            font-weight: 300;
            font-style: italic;
        }

        /* ── Results / intro headers ── */
        .profile-eyebrow {
            font-family: 'Inter', sans-serif;
            font-size: 0.7rem;
            font-weight: 400;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            color: #9a9490;
            margin: 4rem 0 0.75rem 0;
            display: block;
        }
        .profile-name {
            font-family: 'Cormorant Garamond', Georgia, serif;
            font-size: 2.6rem;
            font-weight: 300;
            color: #1a1a1a;
            margin: 0 0 1.5rem 0;
            line-height: 1.15;
            letter-spacing: 0.03em;
        }
        .profile-summary {
            font-family: 'Inter', sans-serif;
            font-size: 0.92rem;
            font-weight: 300;
            color: #5a5550;
            margin: 0 0 2.5rem 0;
            line-height: 1.85;
        }
        .profile-quote {
            font-family: 'Cormorant Garamond', Georgia, serif;
            font-size: 1.15rem;
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

        /* ── Budget slider ── */
        [data-testid="stSlider"] {
            padding: 0.5rem 0 1rem 0;
        }
        [data-testid="stSlider"] [data-testid="stTickBarMin"],
        [data-testid="stSlider"] [data-testid="stTickBarMax"] {
            font-family: 'Inter', sans-serif !important;
            font-size: 0.75rem !important;
            font-weight: 300 !important;
            color: #b5afa8 !important;
        }
        [data-testid="stSlider"] div[role="slider"] {
            background: #1a1a1a !important;
            border: none !important;
        }
        [data-testid="stSlider"] .st-emotion-cache-1jmves9 {
            background: #1a1a1a !important;
        }

        /* ── Demographics chip buttons ── */
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

        /* ── Skeleton loading ── */
        @keyframes shimmer {
            0%   { background-position: -600px 0; }
            100% { background-position:  600px 0; }
        }
        .skeleton {
            background: linear-gradient(90deg, #d4cec7 25%, #e8e2db 50%, #d4cec7 75%);
            background-size: 600px 100%;
            animation: shimmer 1.4s ease-in-out infinite;
            border-radius: 6px;
        }
        .skeleton-card {
            background: #ffffff;
            border-radius: 12px;
            padding: 24px 26px 20px 26px;
            margin-bottom: 16px;
            box-shadow: 0 1px 10px rgba(0,0,0,0.06), 0 0 1px rgba(0,0,0,0.04);
        }
        .sk-title    { height: 22px; width: 55%; margin-bottom: 10px; }
        .sk-badge    { height: 10px; width: 18%; margin-bottom: 14px; }
        .sk-brand    { height: 10px; width: 30%; margin-bottom: 16px; }
        .sk-line     { height: 10px; width: 100%; margin-bottom: 8px; }
        .sk-line-sm  { height: 10px; width: 70%; margin-bottom: 8px; }
        .sk-line-med { height: 10px; width: 85%; margin-bottom: 8px; }
        .sk-divider  { height: 1px; background: #F0EDE8; margin: 14px 0; }
        .sk-para     { height: 10px; margin-bottom: 6px; }

        .skeleton-profile-title { height: 16px; width: 40%; margin-bottom: 18px; }
        .skeleton-profile-line  { height: 12px; margin-bottom: 10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Skeleton loaders
# ─────────────────────────────────────────────

def render_skeleton_card():
    st.markdown(
        """
        <div class="skeleton-card">
            <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px">
                <div class="skeleton sk-title"></div>
                <div class="skeleton sk-badge"></div>
            </div>
            <div class="skeleton sk-brand"></div>
            <div class="skeleton sk-line"></div>
            <div class="skeleton sk-line-med"></div>
            <div class="skeleton sk-line-sm"></div>
            <div class="sk-divider"></div>
            <div class="skeleton sk-para" style="width:100%"></div>
            <div class="skeleton sk-para" style="width:88%"></div>
            <div class="skeleton sk-para" style="width:74%"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_skeleton_results():
    st.markdown('<span class="profile-eyebrow">Your fragrance profile</span>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="margin-bottom:2.5rem">
            <div class="skeleton skeleton-profile-line" style="width:100%"></div>
            <div class="skeleton skeleton-profile-line" style="width:82%"></div>
        </div>
        <span class="results-eyebrow">Your matches</span>
        """,
        unsafe_allow_html=True,
    )
    for _ in range(3):
        render_skeleton_card()


# ─────────────────────────────────────────────
# Result card renderer
# ─────────────────────────────────────────────

def render_result_card(frag: dict):
    name      = _html.escape(str(frag.get("name", "")))
    brand     = _html.escape(str(frag.get("brand", "")))
    family    = _html.escape(str(frag.get("family", "")))
    top       = _html.escape(str(frag.get("top_notes", "")))
    mid       = _html.escape(str(frag.get("middle_notes", "")))
    base      = _html.escape(str(frag.get("base_notes", "")))
    why       = _html.escape(str(frag.get("why", "")))
    pick_type = _html.escape(str(frag.get("pick_type", "")))
    heat      = _html.escape(str(frag.get("heat_performance", "")))
    caveat    = _html.escape(str(frag.get("caveat", "")))

    raw_price = str(frag.get("price", "")).strip()
    if raw_price and "$" not in raw_price:
        raw_price = "$" + raw_price
    price = _html.escape(raw_price)

    note_parts = []
    if top:  note_parts.append(f"Top: {top}")
    if mid:  note_parts.append(f"Middle: {mid}")
    if base: note_parts.append(f"Base: {base}")
    notes_html = (
        f'<p class="card-notes">{"  ·  ".join(note_parts)}</p>'
        if note_parts else ""
    )

    meta_html = (
        f'<p class="card-meta">{price}  ·  Usually found on FragranceNet</p>'
        if price else
        '<p class="card-meta">Usually found on FragranceNet</p>'
    )

    pick_class = "signature" if "signature" in pick_type.lower() else ""
    pick_html  = f'<span class="card-pick-type {pick_class}">{pick_type}</span>' if pick_type else ""
    family_html = f'<p class="card-family">{family}</p>' if family else ""
    heat_html   = f'<p class="card-heat">Heat: {heat}</p>' if heat else ""
    why_html    = f'<p class="card-why">{why}</p>' if why else ""
    caveat_html = f'<p class="card-caveat">Note: {caveat}</p>' if caveat else ""

    st.markdown(
        f"""
        <div class="result-card">
            <div class="card-top">
                <p class="card-name">{name}</p>
                {pick_html}
            </div>
            <p class="card-brand">{brand}</p>
            {family_html}
            {notes_html}
            {meta_html}
            {heat_html}
            {why_html}
            {caveat_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Screen renderers
# ─────────────────────────────────────────────

def show_intro():
    st.markdown(
        """
        <span class="hero-wordmark">ScentMatch</span>
        <p class="hero-title">Find your<br>fragrance.</p>
        <p class="hero-sub">
            Answer a few questions about who you are.<br>
            We'll match you to a fragrance using AI.
        </p>
        <p class="hero-meta">
            12 questions &nbsp;·&nbsp; Three minutes &nbsp;·&nbsp; No account needed
        </p>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Begin", type="primary", use_container_width=True):
        st.session_state.step = 1
        st.session_state.occasions = []
        st.session_state.answers = [None] * 11
        st.rerun()


def show_occasions():
    """Phase 0 — single-select occasion question."""
    st.progress(1 / TOTAL_STEPS)
    st.markdown('<span class="step-label">What are you shopping for?</span>', unsafe_allow_html=True)
    st.markdown(
        '<p class="quiz-question">What\'s the occasion?</p>',
        unsafe_allow_html=True,
    )

    if "occasions" not in st.session_state:
        st.session_state.occasions = []

    current = st.session_state.occasions[0] if st.session_state.occasions else None
    current_idx = PHASE0_OCCASIONS.index(current) if current in PHASE0_OCCASIONS else None

    choice = st.radio(
        "occasion",
        PHASE0_OCCASIONS,
        index=current_idx,
        key="occ_radio",
        label_visibility="collapsed",
    )

    import streamlit.components.v1 as components
    components.html(
        """
        <script>
        (function() {
            function setActive(btn, active) {
                btn.disabled = !active;
                if (active) {
                    btn.style.setProperty('background', '#1a1a1a', 'important');
                    btn.style.setProperty('color', '#ffffff', 'important');
                    btn.style.setProperty('cursor', 'pointer', 'important');
                    btn.style.setProperty('opacity', '1', 'important');
                } else {
                    btn.style.setProperty('background', '#ccc8c3', 'important');
                    btn.style.setProperty('color', '#f0ede9', 'important');
                    btn.style.setProperty('cursor', 'not-allowed', 'important');
                }
            }
            function wire(attempts) {
                if (attempts <= 0) return;
                var doc = window.parent.document;
                var radios = doc.querySelectorAll('[data-testid="stRadio"] input[type="radio"]');
                var btn = Array.from(doc.querySelectorAll('button[kind="primary"]'))
                              .find(function(b) { return /Continue/.test(b.textContent); });
                if (!radios.length || !btn) { setTimeout(function(){ wire(attempts-1); }, 100); return; }
                var anyChecked = Array.from(radios).some(function(r) { return r.checked; });
                setActive(btn, anyChecked);
                radios.forEach(function(r) {
                    r.addEventListener('change', function() { setActive(btn, true); });
                });
            }
            wire(30);
        })();
        </script>
        """,
        height=0,
    )

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

    if st.button("Continue", disabled=choice is None, type="primary", use_container_width=True):
        st.session_state.occasions = [choice]
        st.session_state.step = 2
        st.rerun()


def _submit_text(step: int, idx: int):
    val = st.session_state.get(f"q{step}_text", "").strip()
    if val:
        st.session_state.answers[idx] = val
        st.session_state.step = step + 1


def show_question(step: int):
    """Phase 1 questions — steps 2 through 16."""
    q_idx = step - 2          # 0-based index into PHASE1_QUESTIONS
    q     = PHASE1_QUESTIONS[q_idx]
    total_q_steps = len(PHASE1_QUESTIONS)  # 15

    st.progress(step / TOTAL_STEPS)
    display_num = q_idx + 1
    st.markdown(
        f'<span class="step-label">{display_num} &nbsp;/&nbsp; {total_q_steps}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<p class="quiz-question">{q["label"]}</p>', unsafe_allow_html=True)

    answered = False

    if q["type"] == "text":
        val = st.text_input(
            "your answer",
            placeholder=q.get("placeholder", ""),
            key=f"q{step}_text",
            label_visibility="collapsed",
            on_change=_submit_text,
            args=(step, q_idx),
        )
        answered = bool(val.strip())

        import streamlit.components.v1 as components
        components.html(
            """
            <script>
            (function() {
                function setActive(btn, active) {
                    btn.disabled = !active;
                    if (active) {
                        btn.style.setProperty('background', '#1a1a1a', 'important');
                        btn.style.setProperty('color', '#ffffff', 'important');
                        btn.style.setProperty('cursor', 'pointer', 'important');
                        btn.style.setProperty('opacity', '1', 'important');
                    } else {
                        btn.style.setProperty('background', '#ccc8c3', 'important');
                        btn.style.setProperty('color', '#f0ede9', 'important');
                        btn.style.setProperty('cursor', 'not-allowed', 'important');
                    }
                }
                function wire(attempts) {
                    if (attempts <= 0) return;
                    var doc = window.parent.document;
                    var inp = doc.querySelector('[data-testid="stTextInput"] input');
                    var btn = Array.from(doc.querySelectorAll('button[kind="primary"]'))
                                  .find(function(b) {
                                      return /Continue|Almost there/.test(b.textContent);
                                  });
                    if (!inp || !btn) { setTimeout(function(){ wire(attempts-1); }, 100); return; }
                    inp.addEventListener('input', function() {
                        setActive(btn, inp.value.trim().length > 0);
                    });
                    setActive(btn, inp.value.trim().length > 0);
                }
                wire(30);
            })();
            </script>
            """,
            height=0,
        )
    elif q["type"] == "choice":
        choice = st.radio(
            "pick one",
            q["options"],
            index=None,
            key=f"q{step}_choice",
            label_visibility="collapsed",
        )
        answered = choice is not None

    else:  # "multi"
        multi_key = f"q{step}_multi"
        if multi_key not in st.session_state:
            st.session_state[multi_key] = set()
        selected = st.session_state[multi_key]
        for opt in q["options"]:
            if st.checkbox(opt, value=(opt in selected), key=f"{multi_key}_{opt}"):
                selected.add(opt)
            else:
                selected.discard(opt)
        st.session_state[multi_key] = selected
        answered = len(selected) > 0

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

    is_last = (step == 2 + len(PHASE1_QUESTIONS) - 1)
    btn_label = "Almost there" if is_last else "Continue"

    if st.button(btn_label, disabled=not answered, type="primary", use_container_width=True):
        if q["type"] == "text":
            st.session_state.answers[q_idx] = st.session_state.get(f"q{step}_text", "").strip()
        elif q["type"] == "choice":
            st.session_state.answers[q_idx] = st.session_state.get(f"q{step}_choice")
        else:  # multi
            picked = st.session_state.get(f"q{step}_multi", set())
            st.session_state.answers[q_idx] = ", ".join(sorted(picked))
        st.session_state.step = step + 1
        st.rerun()

    if q.get("skippable"):
        _, col_mid, _ = st.columns([2, 1, 2])
        with col_mid:
            if st.button("skip", key=f"skip_{step}", use_container_width=True):
                st.session_state.answers[q_idx] = ""
                st.session_state.step = step + 1
                st.rerun()


def show_demographics():
    """Step 13 — age, budget, gender."""
    st.progress(13 / TOTAL_STEPS)
    st.markdown('<span class="step-label">Almost there</span>', unsafe_allow_html=True)
    st.markdown(
        '<p class="quiz-question">One last thing — help us personalise your results.</p>',
        unsafe_allow_html=True,
    )

    age_options    = ["18–24", "25–34", "35–44", "45+"]
    gender_options = ["Men's", "Women's", "Show me everything"]

    age    = st.session_state.get("demo_age")
    gender = st.session_state.get("demo_gender")

    st.markdown('<span class="demo-section-label">How old are you?</span>', unsafe_allow_html=True)
    age_cols = st.columns(len(age_options))
    for i, opt in enumerate(age_options):
        if age_cols[i].button(opt, key=f"age_{opt}", type="primary" if age == opt else "secondary", use_container_width=True):
            st.session_state.demo_age = opt
            st.rerun()

    st.markdown('<span class="demo-section-label">Budget per bottle (USD)</span>', unsafe_allow_html=True)
    budget_range = st.slider(
        "budget",
        min_value=20,
        max_value=500,
        value=st.session_state.get("demo_budget_range", (50, 150)),
        step=10,
        label_visibility="collapsed",
        format="$%d",
    )
    st.session_state.demo_budget_range = budget_range
    budget_str = f"${budget_range[0]}–${budget_range[1]}"
    st.session_state.demo_budget = budget_str

    st.markdown('<span class="demo-section-label">Which fragrances should we focus on?</span>', unsafe_allow_html=True)
    gender_cols = st.columns(len(gender_options))
    for i, opt in enumerate(gender_options):
        if gender_cols[i].button(opt, key=f"gender_{opt}", type="primary" if gender == opt else "secondary", use_container_width=True):
            st.session_state.demo_gender = opt
            st.rerun()

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

    ready = age is not None and gender is not None
    if st.button("See my matches", disabled=not ready, type="primary", use_container_width=True):
        # Resolve API key on main thread before spawning background thread
        occasions = st.session_state.get("occasions", [])
        answers   = st.session_state.get("answers", [None] * 11)
        api_key   = _resolve_api_key()
        request_id = str(uuid.uuid4())
        st.session_state.bg_request_id = request_id
        t = threading.Thread(
            target=_run_in_background,
            args=(_bg_store(), request_id, api_key, occasions, answers, age, budget_str, gender),
            daemon=True,
        )
        t.start()
        st.session_state.step = 14
        st.rerun()


def show_email():
    """Step 14 — email capture while API runs in background."""
    st.markdown(
        """
        <span class="step-label" style="margin-top:3.5rem;display:block">Almost ready</span>
        <p class="quiz-question">Where should we send your matches?</p>
        """,
        unsafe_allow_html=True,
    )

    email = st.text_input(
        "email",
        placeholder="your@email.com",
        key="email_input",
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    valid = "@" in email and "." in email.split("@")[-1]
    if st.button("See my matches", disabled=not valid, type="primary", use_container_width=True):
        st.session_state.user_email = email.strip()
        st.session_state.step = 15
        st.rerun()


def show_results():
    request_id = st.session_state.get("bg_request_id")
    store = _bg_store()

    with store["lock"]:
        result = store["results"].get(request_id)

    if result is None:
        # Still loading — show skeletons and poll
        render_skeleton_results()
        time.sleep(0.6)
        st.rerun()
        return

    if result["error"]:
        st.markdown('<span class="profile-eyebrow">Your fragrance profile</span>', unsafe_allow_html=True)
        st.error(_friendly_error(result["error"]))
        st.caption(f"Debug: {result['error'][:300]}")
        if st.button("Try again", type="primary", use_container_width=True):
            with store["lock"]:
                store["results"].pop(request_id, None)
            st.session_state.pop("bg_request_id", None)
            st.session_state.step = 13
            st.rerun()
        return

    st.markdown('<span class="profile-eyebrow">Your fragrance profile</span>', unsafe_allow_html=True)

    profile_summary = result["profile"]
    if profile_summary:
        st.markdown(
            f'<p class="profile-summary">{_html.escape(profile_summary)}</p>',
            unsafe_allow_html=True,
        )

    st.markdown('<span class="results-eyebrow">Your matches</span>', unsafe_allow_html=True)

    for frag in result["frags"] or []:
        render_result_card(frag)

    st.markdown("<div style='height:2.5rem'></div>", unsafe_allow_html=True)

    if st.button("Start over", use_container_width=True):
        with store["lock"]:
            store["results"].pop(request_id, None)
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

    if "step" not in st.session_state:
        st.session_state.step    = 0
        st.session_state.answers = [None] * 11
        st.session_state.occasions = []

    step = st.session_state.step

    if step == 0:
        show_intro()
    elif step == 1:
        show_occasions()
    elif 2 <= step <= 12:
        show_question(step)
    elif step == 13:
        show_demographics()
    elif step == 14:
        show_email()
    else:
        show_results()


if __name__ == "__main__":
    main()
