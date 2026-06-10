# abugoogle-search-engine

A BM25-ranked full-text search engine over a news article dataset. Indexes articles from a CSV into a lexicon, forward index, and a sharded inverted index, then serves ranked results through either a Flask web UI or an interactive CLI.

---

## File Guide

### Read these — they define the code

| File | Why it matters |
|---|---|
| `search-engine-moazam-project.py` | The indexing and CLI search pipeline. Run this first on a new dataset: it reads the CSV, builds the lexicon, forward index, and doc lengths, then writes the inverted index across numbered barrel files. Also contains the BM25 scoring function and the interactive CLI query loop. **Start here to understand the data pipeline.** |
| `backend.py` | The Flask web server. Loads the pre-built indexes at startup, then serves a search API at `POST /search` and the HTML front end at `GET /`. Contains the same BM25 scoring and `search()` logic as the CLI script. |
| `templates/index.html` | The entire front end. Single-page app: search form, paginated results (Pagination.js), and all styling inline. Calls `http://127.0.0.1:5000/search` via `fetch`. |
| `.env` | Sets two required environment variables — `files_path` (directory where index `.pkl` files are read/written) and `CSV_Path` (path to the input CSV). Both scripts load this via `python-dotenv`. |

### Ignore these — generated at runtime, IDE-only, or OS artefacts

| File / Folder | Reason to ignore |
|---|---|
| `lexicon.pkl` | Built by the indexer from the CSV. Regenerated whenever you re-index. |
| `forward_index.pkl` | Built by the indexer. Maps `doc_id → article metadata`. Regenerated on re-index. |
| `doc_lengths.pkl` | Built by the indexer. Maps `doc_id → token count`. Regenerated on re-index. |
| `inverted_indexes/` | Directory of barrel files (`inverted_index_1.pkl`, `inverted_index_2.pkl`, …). Entirely generated output; size grows with the dataset. Never commit. |
| `.idea/` | PyCharm project metadata. References a local Python interpreter. No effect on the code. |
| `BLOG.iml` | IntelliJ/PyCharm module file. IDE bookkeeping only. |
| `__pycache__/` | Python bytecode cache. Regenerated automatically. |

---

## Architecture

```
search-engine-moazam-project.py    Indexer + CLI search
backend.py                         Flask API + serves index.html
templates/index.html               Single-page search UI

.env                               files_path, CSV_Path

[generated at runtime — never commit]
{files_path}/
  lexicon.pkl                      term → term_id
  forward_index.pkl                doc_id → article metadata
  doc_lengths.pkl                  doc_id → token count
  inverted_indexes/
    inverted_index_1.pkl           term_id → {doc_id: freq}  (barrel 1)
    inverted_index_2.pkl           ...                        (barrel 2)
    ...
```

---

## How the Index Works

### Building (run once per dataset)

`search-engine-moazam-project.py` reads the CSV row by row:

1. Concatenates `title + source_name + description + full_content` and tokenises (lowercase, strip punctuation).
2. Assigns each new term a numeric `term_id` and stores it in the **lexicon**.
3. Stores article metadata in the **forward index** keyed by `doc_id`.
4. Accumulates term frequencies per document in an **inverted index** (`term_id → {doc_id: count}`).
5. Saves the inverted index in **barrels** of 1,000 terms each (`inverted_index_N.pkl`) to keep individual file sizes manageable.

### Searching

At query time, both the CLI and Flask backend:

1. Tokenise and look up each query term in the lexicon to get `term_id` values.
2. Walk the inverted index for each `term_id` to find candidate documents.
3. Score each candidate with **BM25** (`k1=1.5`, `b=0.75`).
4. Sort by score descending and return ranked results with article metadata.

### BM25 Parameters

| Parameter | Value | Effect |
|---|---|---|
| `k1` | 1.5 | Term frequency saturation |
| `b` | 0.75 | Document length normalisation |

---

## CSV Schema

The input CSV must have these columns:

| Column | Used for |
|---|---|
| `article_id` | Stored in forward index; returned in results |
| `source_name` | Indexed + stored |
| `title` | Indexed + stored |
| `description` | Indexed + stored |
| `full_content` | Indexed + stored |
| `url` | Stored; used as link in UI |
| `url_to_image` | Stored; displayed in results |

---

## Setup

### 1. Install dependencies

```bash
pip install flask python-dotenv
```

### 2. Create `.env`

```
files_path=/absolute/path/to/index/directory/
CSV_Path=/absolute/path/to/articles.csv
```

`files_path` must end with `/`. The directory will be created if it doesn't exist.

### 3. Build the index (first run only)

```bash
python search-engine-moazam-project.py
```

If index files are already present, the script skips indexing and goes straight to the CLI search prompt.

### 4a. Use the CLI

The indexer script drops into an interactive prompt after building or loading indexes:

```
Welcome to AbuGoogle!
Enter your search query: climate change
```

### 4b. Run the web UI

```bash
python backend.py
```

Open `http://127.0.0.1:5000`. The Flask app loads all indexes at startup (this may take a moment for large datasets), then serves the search UI.

---

## Known Issues

- Both `backend.py` and `search-engine-moazam-project.py` duplicate the `preprocess()`, `bm25_score()`, `search()`, `load_barrel()`, and `load_indexes()` functions exactly. These should live in a shared `indexing.py` module.
- The entire inverted index is loaded into memory in one dict at startup in `backend.py`. For large datasets this can use significant RAM. The barrel structure exists but isn't used for lazy/partial loading.
- `backend.py` calls `app.run(debug=True)` at module level rather than inside `if __name__ == "__main__"`.
- NaN values in result fields are caught with `value != value` (a float NaN idiom). Using `math.isnan()` or `pandas.isna()` would be clearer.
- `templates/index.html` hardcodes `http://127.0.0.1:5000/search` as the fetch target, so it breaks when deployed to any other host or port.
