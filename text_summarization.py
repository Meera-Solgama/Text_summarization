# -----------------------------
# Step 1: Import Streamlit FIRST
# -----------------------------
import streamlit as st

# ⚡ Must be the very first Streamlit command
st.set_page_config(page_title="Gujarati Summarizer", layout="centered")

# -----------------------------
# Step 2: Import other libraries
# -----------------------------
import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# -----------------------------
# Step 3: Device setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Step 4: Load IndicBERT
# -----------------------------
@st.cache_resource
def load_model():
    model_name = "ai4bharat/indic-bert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------
# Step 5: Gujarati sentence splitter
# -----------------------------
def split_sentences_gu(text: str):
    text = re.sub(r'\s+', ' ', text.strip())
    if not text:
        return []
    parts = re.split(r'(?<=[।\.?!])\s+', text)
    sentences = [s.strip() for s in parts if s.strip()]
    return sentences

# -----------------------------
# Step 6: Embeddings with mean pooling
# -----------------------------
def get_sentence_embeddings(sentences, max_length=256):
    if len(sentences) == 0:
        return np.zeros((0, 768), dtype=np.float32)

    enc = tokenizer(
        sentences,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        out = model(**enc)

    last_hidden = out.last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1)

    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    mean_pooled = summed / counts

    return mean_pooled.detach().cpu().numpy()

# -----------------------------
# Step 7: Cosine similarity
# -----------------------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

# -----------------------------
# Step 8: Summarizer (fixed 2 sentences)
# -----------------------------
def gujarati_summarize(text: str, num_sentences: int = 2):
    sentences = split_sentences_gu(text)
    if len(sentences) <= num_sentences:
        return text.strip()

    sent_embs = get_sentence_embeddings(sentences)
    doc_emb = sent_embs.mean(axis=0)
    sims = [cosine_sim(e, doc_emb) for e in sent_embs]

    k = max(1, min(num_sentences, len(sentences)))
    top_idx = np.argsort(sims)[::-1][:k]
    top_idx_sorted = sorted(top_idx.tolist())

    summary_sentences = [sentences[i] for i in top_idx_sorted]
    return " ".join(summary_sentences)

# -----------------------------
# Step 9: Streamlit UI + CSS
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

# Title
st.markdown('<div class="title">📝 Gujarati Text Summarizer</div>', unsafe_allow_html=True)

# Marquee
st.markdown('<div class="marquee"><span>🚀 Summarize your Gujarati text instantly using AI-powered IndicBERT! 🚀</span></div>', unsafe_allow_html=True)

# Input
user_text = st.text_area("👉 Enter Gujarati text here:", height=200)

# Button
if st.button("✨ Generate Summary"):
    if user_text.strip():
        summary = gujarati_summarize(user_text, num_sentences=2)  # fixed
        st.markdown('<div class="summary-box">', unsafe_allow_html=True)
        st.subheader("📖 Original Text")
        st.write(user_text)
        st.subheader("✂️ Generated Summary")
        st.success(summary)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter some Gujarati text to summarize.")
