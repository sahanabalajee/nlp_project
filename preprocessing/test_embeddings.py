import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')
# Load the FAISS index and metadata
# Load the FAISS index
index = faiss.read_index("ncert_faiss.index")
# Load the metadata
with open("ncert_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)
# Define the search function
def search(query, model, index, metadata, top_k=3):
    query_vec = model.encode([query])
    distances, indices = index.search(np.array(query_vec), top_k)

    results = []
    for i in range(top_k):
        idx = indices[0][i]
        results.append({
            "class": metadata[idx][0],
            "chapter": metadata[idx][1],
            "score": float(distances[0][i])
        })
    return results

# Load index & metadata
index = faiss.read_index("ncert_faiss.index")
with open("ncert_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Test
results = search("Who was Hitler?", model, index, metadata)
for r in results:
    print(r)