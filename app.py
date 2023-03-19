import openai
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "system", "content": "You are a helpful assistant."}]

def chat_with_gpt3(messages):
    model_engine = "gpt-3.5-turbo"

    response = openai.ChatCompletion.create(
        model=model_engine,
        messages=messages,
        max_tokens=2000,
        n=1,
        stop=None,
        temperature=1.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    message = response['choices'][0]['message']['content'].strip()
    return message

st.title("ChatGPT with Streamlit")

user_input = st.text_input("Ask a question, enter a conversation, or request a translation:", value="", key="user_input")
submit_button = st.button("Submit", key="submit_button")

if submit_button:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Limit chat history length and remove oldest message if necessary
    max_history_length = 10
    if len(st.session_state.chat_history) > max_history_length:
        st.session_state.chat_history.pop(0)

    # Get model response and add it to chat history
    response = chat_with_gpt3(st.session_state.chat_history)
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Display the conversation
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.write(f"User: {message['content']}")
        else:
            st.write(f"Assistant: {message['content']}")

    # Clear the user input
    user_input = ""
