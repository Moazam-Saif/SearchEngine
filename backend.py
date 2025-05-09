from flask import Flask, request, jsonify, render_template
import os
import pickle
import math
from collections import defaultdict
import string
import csv
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()
files_path=os.getenv("files_path")

app = Flask(__name__)

# Increase the CSV field size limit
csv.field_size_limit(10**7)

# Base directory for barrels
barrels_dir = files_path+"inverted_indexes/"

# Function to preprocess text (lowercase, remove punctuation, tokenize)
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()

# Function to load a specific barrel
def load_barrel(barrel_index):
    barrel_filename = f"inverted_index_{barrel_index}.pkl"
    barrel_path = os.path.join(barrels_dir, barrel_filename)

    if os.path.exists(barrel_path):
        with open(barrel_path, 'rb') as file:
            barrel_data = pickle.load(file)
        return barrel_data
    else:
        return None

# Function to load indexes
def load_indexes(lexicon_path, forward_index_path, doc_lengths_path):
    with open(lexicon_path, 'rb') as file:
        lexicon = pickle.load(file)

    with open(forward_index_path, 'rb') as file:
        forward_index = pickle.load(file)

    with open(doc_lengths_path, 'rb') as file:
        doc_lengths = pickle.load(file)

    return lexicon, forward_index, doc_lengths

# BM25 Scoring function
def bm25_score(query_terms, doc_id, inverted_index, doc_lengths, avg_doc_length, k1=1.5, b=0.75):
    score = 0.0
    for term_id in query_terms:
        if term_id in inverted_index:
            doc_frequency = len(inverted_index[term_id])
            term_frequency = inverted_index[term_id].get(doc_id, 0)

            if term_frequency > 0:
                idf = math.log((len(doc_lengths) - doc_frequency + 0.5) / (doc_frequency + 0.5) + 1.0)
                doc_length = doc_lengths[doc_id]
                tf_norm = (term_frequency * (k1 + 1)) / (term_frequency + k1 * (1 - b + b * doc_length / avg_doc_length))
                score += idf * tf_norm

    return score

# Search function
def search(query, lexicon, forward_index, inverted_index, doc_lengths):
    query_terms = preprocess(query)
    query_term_ids = [lexicon[term] for term in query_terms if term in lexicon]

    if not query_term_ids:
        return []

    avg_doc_length = sum(doc_lengths.values()) / len(doc_lengths)
    scores = defaultdict(float)

    # Calculate BM25 scores for the query terms
    for term_id in query_term_ids:
        if term_id in inverted_index:
            for doc_id in inverted_index[term_id]:
                scores[doc_id] += bm25_score(query_term_ids, doc_id, inverted_index, doc_lengths, avg_doc_length)

    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for doc_id, score in ranked_docs:
        document = forward_index[doc_id]
        results.append({
            'article_id': document['article_id'],
            'source_name': document['source_name'],
            'title': document['title'],
            'description': document['description'],
            'full_content': document['full_content'],
            'url': document['url'],
            'url_to_image': document['url_to_image'],
            'score': score
        })

    return results

# Load indexes
lexicon_path = files_path+"lexicon.pkl"
forward_index_path = files_path+"forward_index.pkl"
doc_lengths_path = files_path+"doc_lengths.pkl"

print("Loading indexes...")
lexicon, forward_index, doc_lengths = load_indexes(lexicon_path, forward_index_path, doc_lengths_path)

# Load inverted index from barrels
inverted_index = {}
barrel_index = 1
while True:
    barrel_data = load_barrel(barrel_index)
    if barrel_data is None:
        break
    inverted_index.update(barrel_data)
    barrel_index += 1

@app.route("/")
def index():
    # Render the HTML file (ensure it's in a 'templates' folder)
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search_query():
    query = request.form.get("query", "").strip()  # Get the raw query string from the form
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    # Process search and get results
    results = search(query, lexicon, forward_index, inverted_index, doc_lengths)

    # Ensure no NaN values are included in the results
    for result in results:
        for key, value in result.items():
            # Check for NaN and replace it with None
            if isinstance(value, float) and (value != value):  # NaN check
                result[key] = None  # Or replace with any other suitable default value

    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)
