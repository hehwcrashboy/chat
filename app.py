import openai
import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch

openai.api_key = st.secrets["OPENAI_API_KEY"]

def calculate_similarity(text1, text2):
    model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    inputs1 = tokenizer(text1, return_tensors="pt", truncation=True, padding=True)
    inputs2 = tokenizer(text2, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

    embeddings1 = outputs1.last_hidden_state.mean(dim=1)
    embeddings2 = outputs2.last_hidden_state.mean(dim=1)

    cosine_sim = torch.nn.functional.cosine_similarity(embeddings1, embeddings2).item()
    return cosine_sim

def is_question_related(new_question, previous_question, threshold=0.6):
    similarity = calculate_similarity(new_question, previous_question)
    return similarity >= threshold

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

user_input = st.text_area("Ask a question, enter a conversation, or request a translation:", height=150)
submit_button = st.button("Submit")

if submit_button:
    previous_question = st.session_state.chat_history[-1]['content'] if st.session_state.chat_history[-1]['role'] == "user" else ""
    is_related = is_question_related(user_input, previous_question) if previous_question else True

    if is_related:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
    else:
        # Reset chat history and start with a new context
        st.session_state.chat_history = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": user_input}]

    # Limit chat history length and remove oldest message if necessary
    max_history_length = 10
    if len(st.session_state.chat_history) > max_history_length:
        st.session_state.chat_history.pop(0)

    # Get model response and add it to chat history
    response = chat_with_gpt3(st.session_state.chat_history)
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Clear the user input
    user_input = ""

# Display the conversation
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.write(f"Me: {message['content']}")
    else:
        st.write(f"chatgpt: {message['content']}")
