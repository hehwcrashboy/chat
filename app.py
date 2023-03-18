import openai
import streamlit as st

openai.api_key = st.secrets["OPENAI_API_KEY"]

def chat_with_gpt3(prompt):
    model_engine = "gpt-3.5-turbo"
    
    response = openai.ChatCompletion.create(
        model=model_engine,
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": f"{prompt}"}],
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.8,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    message = response['choices'][0]['message']['content'].strip()
    return message

st.title("ChatGPT with Streamlit")
user_input = st.text_area("Ask a question, enter a conversation or request a translation:", height=150)
submit_button = st.button("Submit")

if submit_button:
    response = chat_with_gpt3(user_input)
    st.write(response)
