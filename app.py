import streamlit as st
import openai

# Streamlit App
def main():
    st.title("Sistem Question & Answer dengan GPT-3.5 Bahasa Indonesia")
    
    # Input API key
    api_key = st.text_input("Masukkan API key OpenAI:", "")

    if api_key:
        openai.api_key = api_key
        model_name = 'gpt-3.5-turbo'  # Model Bahasa Indonesia GPT-3.5

        # Fungsi untuk mendapatkan jawaban dari pertanyaan menggunakan GPT-3.5
        def get_answer(question, context):
            prompt = f"pertanyaan: {question}\nkonteks: {context}\njawaban:"
            response = openai.Completion.create(
                engine=model_name,
                prompt=prompt,
                max_tokens=150,
                stop=["\n"],
                temperature=0.7,
                n=1,
            )
            answer = response.choices[0].text.strip()
            return answer

        context = st.text_area(
            "Masukkan teks konteks di sini:",
            "Pisang adalah buah yang sangat enak dan bergizi. "
            "Buah ini memiliki kulit kuning dan daging yang manis.",
        )
        question = st.text_input("Masukkan pertanyaan Anda di sini:", "Apa warna kulit pisang?")

        if st.button("Jawab"):
            if context and question:
                answer = get_answer(question, context)
                st.markdown(f"**Jawaban:** {answer}")
            else:
                st.warning("Masukkan konteks dan pertanyaan terlebih dahulu.")
    else:
        st.warning("Masukkan API key OpenAI terlebih dahulu.")

if __name__ == "__main__":
    main()
