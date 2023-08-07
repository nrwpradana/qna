import torch
from transformers import BertTokenizer, BertForQuestionAnswering
import streamlit as st

def create_qa_model():
    model_name = "indobenchmark/indobert-base-p2"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)
    return model, tokenizer

def generate_answer(question, context):
    model, tokenizer = create_qa_model()
    input_text = "[CLS] " + question + " [SEP] " + context + " [SEP]"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    start_scores, end_scores = model(input_ids)
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores) + 1
    answer = tokenizer.decode(input_ids[0, start_index:end_index], skip_special_tokens=True)
    return answer

def main():
    st.title("Question & Answer System (Bahasa Indonesia)")
    question = st.text_input("Masukkan Pertanyaan:")
    context = st.text_area("Masukkan Konteks:")
    if st.button("Tanya"):
        if question and context:
            answer = generate_answer(question, context)
            st.subheader("Jawaban:")
            st.write(answer)

if __name__ == "__main__":
    main()
