import streamlit as st
import torch
from transformers import pipeline

# Fungsi untuk menjalankan sistem Q&A
def run_qa(question, context):
    nlp = pipeline("question-answering", model="cahya/bert-base-indonesian-1.5G", tokenizer="cahya/bert-base-indonesian-1.5G")
    result = nlp(question=question, context=context)
    return result

def main():
    st.title("Sistem Pertanyaan dan Jawaban Bahasa Indonesia")

    # Masukkan teks konteks
    context = st.text_area("Masukkan teks konteks di sini:", value="", height=300)

    # Masukkan pertanyaan
    question = st.text_input("Masukkan pertanyaan Anda di sini:")

    if st.button("Jawab"):
        if not context or not question:
            st.warning("Mohon masukkan teks konteks dan pertanyaan.")
        else:
            with st.spinner("Mencari jawaban..."):
                result = run_qa(question, context)
            st.write(f"Jawaban: {result['answer']}")
            st.write(f"Nilai kepercayaan: {result['score']:.4f}")

if __name__ == "__main__":
    main()
