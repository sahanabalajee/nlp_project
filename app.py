import streamlit as st
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, PegasusTokenizer, PegasusForConditionalGeneration

# Load TrOCR model and processor
# Load TrOCR from local
processor = TrOCRProcessor.from_pretrained("./models/trocr")
trocr_model = VisionEncoderDecoderModel.from_pretrained("./models/trocr")

# Load PEGASUS from local
pegasus_tokenizer = PegasusTokenizer.from_pretrained("./models/pegasus")
pegasus_model = PegasusForConditionalGeneration.from_pretrained("./models/pegasus")

# 3️⃣ Upload handwritten image
uploaded_file = st.file_uploader("📤 Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Handwritten Note", use_column_width=True)

    # 4️⃣ Run OCR
    if st.button("🔍 Run OCR with TrOCR"):
        with st.spinner("Extracting text... please wait ⏳"):
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            with torch.no_grad():
                generated_ids = trocr_model.generate(pixel_values)
                extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        st.success("✅ OCR completed successfully!")
        st.subheader("📝 Extracted Text")
        st.text_area("Here’s the text from your handwritten note:", extracted_text, height=200)

        # Save to session
        st.session_state["ocr_text"] = extracted_text

        # Button to run summarization with PEGASUS
        if st.button("🧠 Summarize with PEGASUS"):
            with st.spinner("Generating summary... ✍️"):
                inputs = pegasus_tokenizer(extracted_text, truncation=True, padding="longest", return_tensors="pt")
                summary_ids = pegasus_model.generate(**inputs)
                summary = pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            st.success("✅ Summary Generated!")
            st.subheader("📚 NCERT-Style Summary")
            st.text_area("Summary:", summary, height=150)
