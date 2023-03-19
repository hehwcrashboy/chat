import openai
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "system", "content": "You are a helpful assistant."}]

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")

def is_question_related(new_question, chat_history):
    threshold = 0.85
    # Retrieve the last user message from chat history
    last_user_message = ""
    for message in reversed(chat_history):
        if message["role"] == "user":
            last_user_message = message["content"]
            break

    # If there is no previous user message, return False
    if not last_user_message:
        return False

    # Compute the similarity between the new question and the last user message
    text_pair = [new_question, last_user_message]
    encoded_input = tokenizer(text_pair, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        scores = model(**encoded_input)[0]

    similarity_score = torch.softmax(scores, dim=1).tolist()[0][1]

    return similarity_score > threshold

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

def initialize_app():
    st.write("<style>body{margin: 0; padding: 0;}</style>", unsafe_allow_html=True)
    conversation_history = st.empty()
    input_container = st.empty()
    return conversation_history, input_container

conversation_history, input_container = initialize_app()

# Display the conversation
def display_conversation():
    conversation_history.markdown("---")
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            conversation_history.write(f"<div style='background-color: white; padding: 10px; border-radius: 5px;'><strong><img src='https://i.imgur.com/0n7vG8E.png' width='25px' style='vertical-align:middle;'> {message['content']}</strong></div>", unsafe_allow_html=True)
        else:
            conversation_history.write(f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'><img src='https://i.imgur.com/9X45lgs.png' width='25px' style='vertical-align:middle;'> {message['content']}</div>", unsafe_allow_html=True)
    conversation_history.markdown("---")

display_conversation()

input_container = st.empty()
user_input = input_container.text_input("Ask a question, enter a conversation, or request a translation:", value="", key="bottom_user_input")
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

    # Clear the user input
    input_container.empty()
    user_input = ""

    # Update the conversation display
    display_conversation()
