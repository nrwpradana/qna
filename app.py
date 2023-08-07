# Install the required libraries
# !pip install transformers torch Flask streamlit

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import streamlit as st

def create_qa_model():
    model_name = "google/t5-small-qa-qg"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer

def generate_answer(question):
    model, tokenizer = create_qa_model()
    input_text = "question: {} context: ".format(question)
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        output = model.generate(input_ids)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

def main():
    st.title("Question & Answer System")
    question = st.text_input("Masukkan Pertanyaan:")
    if st.button("Tanya"):
        if question:
            answer = generate_answer(question)
            st.subheader("Jawaban:")
            st.write(answer)

if __name__ == "__main__":
    main()
