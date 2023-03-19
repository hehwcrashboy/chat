import openai
import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch

openai.api_key = st.secrets["OPENAI_API_KEY"]

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-xlm-r-multilingual-v1")
model = AutoModel.from_pretrained("sentence-transformers/paraphrase-xlm-r-multilingual-v1")

def compute_similarity(text1, text2):
    inputs1 = tokenizer(text1, return_tensors="pt")
    inputs2 = tokenizer(text2, return_tensors="pt")
    with torch.no_grad():
        embeddings1 = model(**inputs1).last_hidden_state.mean(dim=1)
        embeddings2 = model(**inputs2).last_hidden_state.mean(dim=1)
    cos_sim = torch.nn.functional.cosine_similarity(embeddings1, embeddings2).item()
    return cos_sim

def is_question_related(new_question, chat_history, similarity_threshold=0.7):
    for message in chat_history:
        if message["role"] == "user":
            similarity = compute_similarity(new_question, message["content"])
            if similarity >= similarity_threshold:
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

st.write("<style>body{margin: 0; padding: 0;}</style>", unsafe_allow_html=True)
conversation_history = st.empty()

# Display the conversation
def display_conversation():
    conversation_history.markdown("---")
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            conversation_history.markdown(
                f"<div style='background-color: white; padding: 10px; border-radius: 5px;'><strong><img src='https://i.imgur.com/0n7vG8E.png' width='25px' style='vertical-align:middle;'> {message['content']}</strong></div>",
                unsafe_allow_html=True,
            )
        else:
            conversation_history.markdown(
                f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'><img src='https://i.imgur.com/9X45lgs.png' width='25px' style='vertical-align:middle;'> {message['content']}</div>",
                unsafe_allow_html=True,
            )
    conversation_history.markdown("---")

display_conversation()

# User input and submit button
with st.form(key='input_form'):
    user_input = st.text_area("Ask a question, enter a conversation, or request a translation:", key='user_input', height=50)
    submit_button = st.form_submit_button("Submit")

if submit_button:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Check if the new question is related to the chat history
    if is_question_related(user_input, st.session_state.chat_history):
        relevant_history = st.session_state.chat_history
    else:
        relevant_history = [{"role": "system", "content": "You are a helpful assistant."}]

    # Get model response and add it to chat history
    response = chat_with_gpt3(relevant_history)
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Display the conversation
    display_conversation()

    # Clear the user input
    st.form(key='input_form').clear_on_submit()
