import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# Load model and tokenizer from directory (not .pkl)
model_path = r"C:/final project/model_path"  # Use forward slashes or raw string
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Set page configuration
st.set_page_config(page_title="Sentivibe 🌈", page_icon="💬", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
        html, body {
            background-color: #f4f7f9;
        }
        .title {
            font-size: 3.5em;
            font-weight: 800;
            text-align: center;
            background: -webkit-linear-gradient(45deg, #021526, #6EACDA);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.3em;
        }
        .subtitle {
            text-align: center;
            font-size: 1.2em;
            color: #5f6f81;
            margin-bottom: 2em;
        }
        .stTextArea textarea {
            border: 2px solid #6EACDA;
            border-radius: 10px;
            font-size: 1.1em;
            padding: 10px;
        }
        .stButton button {
            background: linear-gradient(90deg, #03346E, #6EACDA);
            border: none;
            color: white;
            padding: 10px 25px;
            font-size: 1.1em;
            border-radius: 10px;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton button:hover {
            transform: scale(1.05);
            background: linear-gradient(90deg, #021526, #03346E);
        }
        .emoji {
            font-size: 3em;
            text-align: center;
            margin-top: 1em;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="title">Sentivibe 🌈</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Feel the vibe of your words – powered by AI & Emotion 🎭</div>', unsafe_allow_html=True)

# Prediction function
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    return prediction

# User input
user_input = st.text_area("📝 Enter your review or message:")

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("🚨 Please write something to analyze.")
    else:
        result = predict(user_input)
        if result == 1:
            st.success("💚 Sentiment: Positive 😊")

        else:
            st.error("❤️ Sentiment: Negative 😞")
