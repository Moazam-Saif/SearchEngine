import os
import pickle
import math
import csv
from collections import defaultdict, Counter
import string
import sys
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()
files_path=os.getenv("files_path")

# Increase the CSV field size limit
csv.field_size_limit(10**7)


# Base directory for barrels
barrels_dir = files_path+"inverted_indexes/"

# Create the directory if it doesn't exist
if not os.path.exists(barrels_dir):
    os.makedirs(barrels_dir)

# Function to preprocess text (lowercase, remove punctuation, tokenize)
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()

# Function to create indexes
def create_indexes(csv_path, lexicon_path, forward_index_path, inverted_index_path, doc_lengths_path):
    lexicon = {}
    inverted_index = defaultdict(dict)
    forward_index = {}
    doc_lengths = {}
    term_id_counter = 0
    doc_id = 0

    # Read the CSV file
    with open(csv_path, 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            article_id = row['article_id']
            source_name = row['source_name']
            title = row['title']
            description = row['description']
            full_content = row['full_content']
            url=row['url']
            url_to_image=row['url_to_image']

            # Generate a unique doc_id
            doc_id += 1

            # Store in forward index
            forward_index[doc_id] = {
                'article_id': article_id,
                'source_name': source_name,
                'title': title,
                'description': description,
                'full_content': full_content,
                'url':url,
                'url_to_image': url_to_image
            }

            # Combine title, source_name, description, and full_content for indexing
            content = f"{title} {source_name} {description} {full_content}"
            terms = preprocess(content)
            term_counts = Counter(terms)

            # Update lexicon and inverted index
            for term in term_counts:
                if term not in lexicon:
                    term_id_counter += 1
                    lexicon[term] = term_id_counter

                term_id = lexicon[term]
                inverted_index[term_id][doc_id] = term_counts[term]

            # Update document lengths
            doc_lengths[doc_id] = sum(term_counts.values())

    # Save indexes to disk
    with open(lexicon_path, 'wb') as file:
        pickle.dump(lexicon, file)

    with open(forward_index_path, 'wb') as file:
        pickle.dump(forward_index, file)

    # Save inverted index in barrels
    save_barrels(inverted_index)

    with open(doc_lengths_path, 'wb') as file:
        pickle.dump(doc_lengths, file)

    print("Indexes created and saved.")

# Function to save the inverted index in barrels
def save_barrels(inverted_index, barrel_size=1000):
    num_terms = len(inverted_index)
    num_barrels = math.ceil(num_terms / barrel_size)
    
    for i in range(num_barrels):
        # Get a slice of the inverted index for the current barrel
        start_index = i * barrel_size
        end_index = (i + 1) * barrel_size
        barrel_data = dict(list(inverted_index.items())[start_index:end_index])

        # Generate a barrel filename (e.g., inverted_index_1.pkl, inverted_index_2.pkl)
        barrel_filename = f"inverted_index_{i+1}.pkl"
        barrel_path = os.path.join(barrels_dir, barrel_filename)

        # Save the barrel to disk
        with open(barrel_path, 'wb') as file:
            pickle.dump(barrel_data, file)
        print(f"Barrel {i+1} saved at {barrel_path}")

# Function to load a specific barrel
def load_barrel(barrel_index):
    barrel_filename = f"inverted_index_{barrel_index}.pkl"
    barrel_path = os.path.join(barrels_dir, barrel_filename)

    # Check if the barrel file exists
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
        return "No matching documents found."

    avg_doc_length = sum(doc_lengths.values()) / len(doc_lengths)
    scores = defaultdict(float)

    # Calculate the BM25 scores for the query terms
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
            'score': score,
            'url':document['url'],
            'url_to_image':document['url_to_image']
        })

    return results

# Main function to handle indexing and searching
def main():
    # Paths for files
    csv_path = os.getenv("CSV_Path")  # Input CSV file
    lexicon_path = files_path+"lexicon.pkl"
    forward_index_path = files_path+"forward_index.pkl"
    inverted_index_path = files_path+"inverted_index.pkl"
    doc_lengths_path = files_path+"doc_lengths.pkl"

    # Check if index files already exist
    if all(os.path.exists(path) for path in [lexicon_path, forward_index_path, doc_lengths_path]):
        print("Index files found. Loading indexes...")
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
    else:
        print("Index files not found. Creating indexes...")
        create_indexes(csv_path, lexicon_path, forward_index_path,inverted_index_path, doc_lengths_path)
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

    # Perform a search
    print("\nWelcome to AbuGoogle!\n")
    query = input("Enter your search query: ")
    results = search(query, lexicon, forward_index, inverted_index, doc_lengths)

    print("\nSearch Results:")
    for result in results:
        print(f"Article ID: {result['article_id']}, Source: {result['source_name']}, Title: {result['title']}, Score: {result['score']:.4f},URL:{result['url']},URL_Image:{result['url_to_image']}")

if __name__ == "__main__":
    main()
