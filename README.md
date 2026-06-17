# ScentMatch

AI-powered fragrance recommender. Answer nine questions about who you are — your aesthetic, your environment, how you want people to feel around you — and ScentMatch finds real Sephora fragrances that match your personality using semantic AI embeddings.

## How it works

1. Each Sephora fragrance's name, brand, highlights, and ingredients are embedded into a vector using `sentence-transformers` (`all-MiniLM-L6-v2`).
2. Your quiz answers are compiled into a natural-language description and embedded the same way.
3. Cosine similarity ranks the fragrances closest to your personality.
4. Results are shown as product cards with price, rating, and a direct link to buy on Sephora.

## Setup

### 1. Download the Sephora dataset from Kaggle

1. Go to: https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews
2. Download the dataset and extract `product_info.csv`
3. Place it at `data/product_info.csv` (next to `app.py`)

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Pre-compute embeddings (run once)

```bash
python precompute_embeddings.py
```

This filters the Sephora dataset to fragrance products, embeds all descriptions, and saves:
- `embeddings_cache.npy` — the embedding matrix
- `data/fra_processed.pkl` — the cleaned DataFrame

Takes ~60–120 seconds the first time. Subsequent app starts load in under 2 seconds.

### 4. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Project structure

```
scentmatch/
├── app.py                    # Streamlit UI + embedding + recommendation logic
├── precompute_embeddings.py  # One-time script to build the embedding cache
├── data/
│   └── product_info.csv      # Sephora dataset (you download this)
├── requirements.txt
└── README.md
```

## Optional: Community insights via Claude

If you set an `ANTHROPIC_API_KEY` environment variable, each result card will include a short community insight pulled from Reddit's r/fragrance community and summarised by Claude Haiku.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
streamlit run app.py
```
