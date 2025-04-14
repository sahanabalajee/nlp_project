import os
import fitz  
import json
def extract_text_from_pdfs(root_dir):
    data = []

    for class_folder in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_folder)
        if os.path.isdir(class_path):
            for chapter_pdf in os.listdir(class_path):
                if chapter_pdf.endswith('.pdf'):
                    chapter_path = os.path.join(class_path, chapter_pdf)
                    doc = fitz.open(chapter_path)
                    full_text = ""
                    for page in doc:
                        full_text += page.get_text()
                    data.append({
                        "class": class_folder,
                        "chapter": os.path.splitext(chapter_pdf)[0],
                        "text": full_text.strip()
                    })
    return data

pdf_data = extract_text_from_pdfs("textbooks")

with open("ncert_texts.json", "w", encoding="utf-8") as f:
    json.dump(pdf_data, f, indent=2, ensure_ascii=False)
