from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

# Load the FAISS index
index = faiss.read_index("ncert_faiss.index")

# Load the paragraphs mapping (from Step 1)
with open("ncert_texts.json", "r", encoding="utf-8") as f:
    paragraphs = json.load(f)

# Load the Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def search(query, top_k=5):
    # Convert query to embeddings
    query_embedding = model.encode([query])

    # Search in the FAISS index
    D, I = index.search(np.array(query_embedding), top_k)  # D is distances, I is indices of matching paragraphs

    results = []
    for i in I[0]:  # Loop through the top K indices
        result = {
            "paragraph": paragraphs[i],
            "distance": D[0][I[0].tolist().index(i)]  # Distance between query and paragraph (lower is more relevant)
        }
        results.append(result)

    return results

# Example query:
query = "What is the rise of civilization?"
top_k = 5
results = search(query, top_k)

# Print the top K relevant paragraphs
for idx, result in enumerate(results):
    print(f"Result {idx + 1}:")
    print(f"Paragraph: {result['paragraph']}")
    print(f"Distance: {result['distance']}")
    print("-" * 80)