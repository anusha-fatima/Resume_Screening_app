import streamlit as st
import pickle
import docx
import PyPDF2
from io import BytesIO



# Load model, vectorizer, and label encoder
with open("models/best_model.pkl", "rb") as f:
    best_model = pickle.load(f)

with open("models/tfidf_vectorizer.pkl", "rb") as f:   # match file name
    tfidf = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)





# Helper functions
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return " ".join([p.text for p in doc.paragraphs])

def clean_resume_text(text):
    # TODO: 
    return text


def predict_resume_category(resume_text, model, vectorizer, label_encoder, top_n=3):
    cleaned_text = clean_resume_text(resume_text)
    text_vectorized = vectorizer.transform([cleaned_text])

    probs = model.predict_proba(text_vectorized)[0]
    top_indices = probs.argsort()[::-1][:top_n]

    results = [(label_encoder.inverse_transform([i])[0], float(probs[i])) for i in top_indices]
    return results


# Streamlit UI
st.title("Resume Screening App")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".pdf"):
        resume_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        resume_text = extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file type")
        resume_text = ""

    if resume_text:
        st.subheader("Extracted Resume Text")
        st.write(resume_text[:1000] + "...")  

        predictions = predict_resume_category(resume_text, best_model, tfidf, label_encoder)

        st.subheader("Predicted Job Categories")
        for category, score in predictions:
            st.write(f"**{category}** — {score:.2f}")
