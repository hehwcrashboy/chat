import openai
import streamlit as st

openai.api_key = st.secrets["OPENAI_API_KEY"]

def chat_with_gpt3(prompt):
    model_engine = "gpt-3.5-turbo"
    
    response = openai.ChatCompletion.create(
        model=model_engine,
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": f"{prompt}"}],
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.8,
    )

    message = response.choices[0].text.strip()
    return message

st.title("ChatGPT with Streamlit")
user_input = st.text_input("Ask a question, enter a conversation or request a translation:")
submit_button = st.button("Submit")

if submit_button:
    response = chat_with_gpt3(user_input)
    st.write(response)
