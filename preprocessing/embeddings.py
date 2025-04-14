import json

with open("ncert_texts.json", "r", encoding="utf-8") as f:
    pdf_data = json.load(f)

texts = [entry["text"] for entry in pdf_data]
metadata = [(entry["class"], entry["chapter"]) for entry in pdf_data]

from sentence_transformers import SentenceTransformer

# Load a pre-trained model (lightweight and effective)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for all chapter texts
embeddings = model.encode(texts, show_progress_bar=True)
import faiss
import numpy as np

# Create FAISS index
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)  # L2 distance
index.add(np.array(embeddings))

# Save index and metadata
faiss.write_index(index, "ncert_faiss.index")

import pickle
with open("ncert_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)
