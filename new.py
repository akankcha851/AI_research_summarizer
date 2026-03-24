import streamlit as st
import fitz 
import requests
import os
import base64
from dotenv import load_dotenv
st.set_page_config(
    page_title="📄 AI Research Paper Summarizer",
    page_icon="📄",
    layout="centered"
)
load_dotenv()
API_KEY = os.getenv("HF_API_KEY")
API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"
headers = {"Authorization": f"Bearer {API_KEY}"}
def load_local_css(css_file):
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_local_css("style.css")
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return f"data:image/jpg;base64,{encoded}"

bg_image = get_base64_image("capture.png")
st.markdown(f"""
    <style>
        .stApp {{
            background-image: url("{bg_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }}
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align:center; margin-top:20px;'>
        <img src="https://img.icons8.com/fluency/96/artificial-intelligence.png" width="80"/>
        <h1>AI Research Paper Summarizer</h1>
        <h3 style="color:blue;">📄 Summarize your research PDFs instantly</h3>
        <hr style="margin-top: -10px; margin-bottom: 30px;">
    </div>
""", unsafe_allow_html=True)

# -- FILE UPLOADER ----
st.markdown("<h4 style='color:black;'>📄 Upload your Research Paper (PDF)</h4>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["pdf"])


# ----- HUGGING FACE SUMMARIZATION ----
def summarize_text(text):
    payload = {
        "inputs": text,
        "parameters": {"max_length": 300, "min_length": 50, "do_sample": False}
    }
    response = requests.post(API_URL, headers=headers, json=payload)

    try:
        result = response.json()
        if isinstance(result, dict) and 'error' in result:
            error_msg = result['error']
            # Friendly messages for common errors
            if "Invalid username or password" in error_msg or "401" in str(response.status_code):
                return "🔑 Invalid or missing API key. Please check your HF_API_KEY in the .env file.\n\nGet a free token at: https://huggingface.co/settings/tokens"
            elif "loading" in error_msg.lower():
                return "⏳ Model is loading on Hugging Face servers. Please wait 20–30 seconds and try again."
            elif "quota" in error_msg.lower() or "rate" in error_msg.lower():
                return "🚫 API rate limit reached. Please wait a few minutes and try again."
            else:
                return f"❌ API Error: {error_msg}"
        return result[0]['summary_text']
    except Exception as e:
        return f"❌ Unexpected Error: {str(e)}"


# ----PROCESS PDF -----
if uploaded_file is not None:
    with st.spinner("🔍 Extracting text from PDF..."):
        pdf_reader = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = "".join([page.get_text() for page in pdf_reader])

    # Success badge
    st.markdown("""
        <div style="
            background-color: #28a745;
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            text-align: center;
            font-weight: bold;
            width: fit-content;
            margin: 10px auto;
            font-size: 1rem;
        ">
            ✅ PDF Loaded Successfully!
        </div>
    """, unsafe_allow_html=True)

    if st.button("📝 Generate Summary"):
        with st.spinner("✨ Summarising your paper..."):
            summary = summarize_text(text[:3000])  # limit for performance
        st.markdown("""
            <div style="
                font-size: 1.2rem;
                font-weight: 700;
                color: #0f172a;
                margin-top: 24px;
                margin-bottom: 8px;
            ">
                🧠 Summary
            </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
            <div style="
                background: rgba(255, 255, 255, 0.92);
                backdrop-filter: blur(12px);
                border: 1.5px solid #4facfe;
                padding: 24px 28px;
                border-radius: 14px;
                margin-top: 4px;
                font-size: 15.5px;
                line-height: 1.75;
                color: #0f172a;
                font-family: 'Poppins', sans-serif;
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.10);
            ">
                {summary}
            </div>
        """, unsafe_allow_html=True)

        st.download_button("📥 Download Summary", summary, file_name="summary.txt")
st.markdown("""
<hr>
<footer>Made with ❤️ by <b>Akankcha</b> | Powered by Hugging Face</footer>
""", unsafe_allow_html=True)