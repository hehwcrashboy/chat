import openai
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

openai.api_key = st.secrets["OPENAI_API_KEY"]

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
model = AutoModelForSequenceClassification.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")

def check_similarity(question1, question2):
    encoded_input = tokenizer.encode_plus(question1, question2, return_tensors='pt', padding=True, truncation=True)
    scores = model(**encoded_input)[0].softmax(1)
    return scores[0][1].item()

def is_question_related(new_question, chat_history):
    threshold = 0.7
    for message in chat_history:
        if message["role"] == "user":
            similarity_score = check_similarity(new_question, message["content"])
            if similarity_score >= threshold:
                return True
    return False

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

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "system", "content": "You are a helpful assistant."}]

st.write("---")

# Display the conversation
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.write(f"<div style='background-color: white; padding: 10px; border-radius: 5px;'><strong><img src='https://i.imgur.com/0n7vG8E.png' width='25px' style='vertical-align:middle;'> {message['content']}</strong></div>", unsafe_allow_html=True)
    else:
        st.write(f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'><img src='https://i.imgur.com/9X45lgs.png' width='25px' style='vertical-align:middle;'> {message['content']}</div>", unsafe_allow_html=True)

st.write("---")

user_input = st.text_area("Ask a question, enter a conversation, or request a translation:", height=150)
submit_button = st.button("Submit")

if submit_button:
    # Check if the new question is related to the previous ones
    if is_question_related(user_input, st.session_state.chat_history):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
    else:
        # Start a new chat history with the new question
        st.session_state.chat_history = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": user_input}]

    # Get model response and add it to chat history
    response = chat_with_gpt3(st.session_state.chat_history)
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Display the new message
    st.write(f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'><img src='https://i.imgur.com/9X45lgs.png' width='25px' style='vertical-align:middle;'> {response}</div>", unsafe_allow_html=True)

    # Clear the user input
    user_input = ""
