import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# -----------------------------
# ⚡ Streamlit page config
# -----------------------------
st.set_page_config(page_title="Gujarati Summarizer", layout="centered")

# -----------------------------
# Device setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load mT5 model + tokenizer
# -----------------------------
@st.cache_resource
def load_model():
    model_name = "google/mt5-small"  # multilingual T5
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------
# Function to summarize
# -----------------------------
def gujarati_abstractive_summary(text, max_input_length=512, max_summary_length=80):
    # Prefix "summarize: " for T5-based models
    input_text = "summarize: " + text

    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        max_length=max_input_length, 
        truncation=True
    ).to(device)

    summary_ids = model.generate(
        inputs["input_ids"], 
        max_length=max_summary_length, 
        min_length=20, 
        length_penalty=2.0, 
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# -----------------------------
# Streamlit UI + CSS
# -----------------------------
st.markdown("""
<style>
body { background: linear-gradient(135deg, #FFDEE9 0%, #B5FFFC 100%); font-family: 'Segoe UI', sans-serif;}
.title { text-align: center; font-size: 40px; font-weight: bold; color: #2C3E50; padding: 10px; text-shadow: 2px 2px #ffffff; }
.marquee { width: 100%; overflow: hidden; white-space: nowrap; box-sizing: border-box; color: #E74C3C; font-size: 20px; font-weight: bold;}
.marquee span { display: inline-block; padding-left: 100%; animation: marquee 10s linear infinite; }
@keyframes marquee { 0% { transform: translate(0,0); } 100% { transform: translate(-100%,0); } }
.summary-box { background: #ffffffcc; border-radius: 15px; padding: 20px; box-shadow: 2px 4px 12px rgba(0,0,0,0.2); margin-top: 20px;}
</style>
""", unsafe_allow_html=True)

# Title + Marquee
st.markdown('<div class="title">📝 Gujarati Text Summarizer (Abstractive)</div>', unsafe_allow_html=True)
st.markdown('<div class="marquee"><span>🚀 Real Gujarati summaries generated with AI! 🚀</span></div>', unsafe_allow_html=True)

# Input text
user_text = st.text_area("👉 Enter Gujarati text here:", height=200)

# Button to summarize
if st.button("✨ Generate Summary"):
    if user_text.strip():
        with st.spinner("Generating summary... 📝"):
            summary = gujarati_abstractive_summary(user_text)

        st.markdown('<div class="summary-box">', unsafe_allow_html=True)
        st.subheader("📖 Original Text")
        st.write(user_text)
        st.subheader("✂️ Generated Summary")
        st.success(summary)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter some Gujarati text to summarize.")
