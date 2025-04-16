import streamlit as st
from PIL import Image
import torch
import faiss
import pickle

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from sentence_transformers import SentenceTransformer

# Load TrOCR model
@st.cache_resource
def load_trocr():
    processor = TrOCRProcessor.from_pretrained("./models/trocr")
    model = VisionEncoderDecoderModel.from_pretrained("./models/trocr")
    return processor, model

# Load PEGASUS model
@st.cache_resource
def load_pegasus():
    tokenizer = PegasusTokenizer.from_pretrained("./models/pegasus")
    model = PegasusForConditionalGeneration.from_pretrained("./models/pegasus")
    return tokenizer, model

# Load FAISS index + NCERT chunks
@st.cache_resource
def load_faiss():
    index = faiss.read_index("ncert_faiss.index")
    with open("texts.pkl", "rb") as f:
        texts = pickle.load(f)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return index, texts, embedder

# Load models
trocr_processor, trocr_model = load_trocr()
pegasus_tokenizer, pegasus_model = load_pegasus()
faiss_index, ncert_texts, embedder = load_faiss()

# UI – File Upload
st.title("📖 Handwritten Notes to NCERT Summary")
uploaded_file = st.file_uploader("📤 Upload a Handwritten Note (JPG/PNG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Handwritten Note", use_column_width=True)

    # OCR Button
    if st.button("🔍 Run OCR with TrOCR"):
        with st.spinner("Extracting text... please wait ⏳"):
            pixel_values = trocr_processor(images=image, return_tensors="pt").pixel_values
            with torch.no_grad():
                generated_ids = trocr_model.generate(pixel_values)
                extracted_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        st.success("✅ OCR completed!")
        st.subheader("📝 Extracted Text")
        st.text_area("OCR Output:", extracted_text, height=150)
        st.session_state["ocr_text"] = extracted_text

        # Fetch relevant NCERT context
        query_embedding = embedder.encode([extracted_text])
        _, top_k_indices = faiss_index.search(query_embedding, k=5)
        retrieved_context = "\n".join([ncert_texts[i] for i in top_k_indices[0]])

        st.subheader("📚 Relevant NCERT Chunks")
        st.text_area("Top Matches from NCERT:", retrieved_context, height=200)

        # PEGASUS Summary
        if st.button("🧠 Summarize with PEGASUS"):
            with st.spinner("Generating summary..."):
                inputs = pegasus_tokenizer(retrieved_context, truncation=True, padding="longest", return_tensors="pt")
                summary_ids = pegasus_model.generate(**inputs)
                summary = pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            st.success("✅ Summary Generated!")
            st.subheader("📘 NCERT-based Summary")
            st.text_area("Summary:", summary, height=150)
