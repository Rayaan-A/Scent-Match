# 🌸 ScentMatch

AI-powered fragrance recommender. Describe a vibe — "fresh citrusy summer", "dark mysterious oud", "cozy vanilla winter" — and ScentMatch finds the best-matching perfumes using semantic embeddings.

## How it works

1. Each fragrance's notes, accords, and description are embedded into a vector using `sentence-transformers` (`all-MiniLM-L6-v2`).
2. Your query is embedded the same way.
3. Cosine similarity ranks the fragrances closest to your query.
4. The top results are displayed with name, brand, notes, and a match score.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** First run will download the ~80 MB `all-MiniLM-L6-v2` model automatically.

### 2. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Dataset

The bundled `data/fragrances.csv` contains 50 hand-curated real fragrances and works out of the box.

### Using the full Kaggle dataset (optional)

1. Search Kaggle for **"fragrantica fragrance dataset"**
2. Download the CSV (look for columns: name, brand, notes, accords, description)
3. Rename/place it as `data/fra_cleaned.csv` next to `app.py`

ScentMatch auto-detects the larger dataset on startup.

## Project structure

```
scentmatch/
├── app.py              # Streamlit UI + embedding + recommendation logic
├── data/
│   └── fragrances.csv  # Bundled 50-fragrance sample dataset
├── requirements.txt
└── README.md
```

## Example queries

- `fresh citrusy summer scent for the beach`
- `warm cozy vanilla for winter evenings`
- `dark mysterious oud and leather`
- `light floral rose for a wedding`
- `clean soapy skin scent for everyday wear`
- `sporty aquatic ocean breeze`
