import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import tempfile
import os
from gtts import gTTS
from pylint.lint import Run
from pylint.reporters.text import TextReporter
from io import StringIO
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Ensure NLTK data is available at startup
try:
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    st.warning("Downloading required NLTK resources...")
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    st.success("NLTK resources downloaded successfully.")

# PyMuPDF check
try:
    import fitz  # PyMuPDF
except ImportError:
    st.warning("PyMuPDF not installed. ATS Score Checker will be skipped.")
    fitz = None

# API Key (Hugging Face)
hf_api_key = st.secrets.get("HF_API_KEY", os.getenv("HF_API_KEY"))
if not hf_api_key:
    st.warning("Hugging Face API key missing. Text-to-Image will not work.")

# Custom CSS for transparent background, modern UI, unique title, and tools selection
st.markdown("""
    <style>
    .stApp {
        background: rgba(20, 20, 20, 0.85);
        color: #ffffff;
    }
    .stTextInput, .stTextArea, .stSelectbox, .stFileUploader {
        background-color: rgba(50, 50, 50, 0.9);
        color: #ffffff;
        border-radius: 8px;
    }
    .stButton>button {
        background-color: #1e90ff;
        color: #ffffff;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #4682b4;
    }
    h2, h3 {
        color: #1e90ff;
    }
    .stMarkdown, .stWarning, .stError, .stSuccess {
        color: #ffffff;
    }
    /* Unique styling for the title */
    .unique-title {
        font-size: 5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(45deg, #1e90ff, #c71585);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 15px rgba(30, 144, 255, 0.9), 0 0 25px rgba(199, 21, 133, 0.7);
        margin-bottom: 30px;
        animation: pulse 3s infinite;
    }
    @keyframes pulse {
        0% { text-shadow: 0 0 15px rgba(30, 144, 255, 0.9), 0 0 25px rgba(199, 21, 133, 0.7); }
        50% { text-shadow: 0 0 25px rgba(30, 144, 255, 1), 0 0 35px rgba(199, 21, 133, 1); }
        100% { text-shadow: 0 0 15px rgba(30, 144, 255, 0.9), 0 0 25px rgba(199, 21, 133, 0.7); }
    }
    /* Unique tools selection */
    .tool-button {
        display: inline-block;
        background: rgba(40, 40, 40, 0.9);
        color: #ffffff;
        padding: 15px 20px;
        margin: 5px;
        border: 2px solid #1e90ff;
        border-radius: 12px;
        font-weight: bold;
        text-align: center;
        transition: all 0.3s;
        cursor: pointer;
    }
    .tool-button:hover {
        background: rgba(80, 80, 80, 0.9);
        transform: scale(1.1);
        box-shadow: 0 0 10px rgba(30, 144, 255, 0.7);
    }
    .tool-button.active {
        background: #1e90ff;
        color: #ffffff;
        box-shadow: 0 0 15px rgba(30, 144, 255, 1);
    }
    .tool-container {
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title with unique styling
st.markdown('<h1 class="unique-title">GEN IQ</h1>', unsafe_allow_html=True)

# Tools selection
if 'selected_tool' not in st.session_state:
    st.session_state.selected_tool = "Text-to-Image"

tools = [
    {"name": "Text-to-Image", "icon": "🖼️"},
    {"name": "Text-to-Audio", "icon": "🎵"},
    {"name": "Summarization", "icon": "📝"},
    {"name": "Code Debugger", "icon": "💻"},
    {"name": "ATS Score Checker", "icon": "📄"}
]

# Render tool buttons
st.markdown('<div class="tool-container">', unsafe_allow_html=True)
for tool in tools:
    active_class = "active" if st.session_state.selected_tool == tool["name"] else ""
    st.markdown(
        f'<div class="tool-button {active_class}" '
        f'onclick="document.getElementById(\'{tool["name"]}\').click()">'
        f'{tool["icon"]} {tool["name"]}</div>',
        unsafe_allow_html=True
    )
    # Hidden button for handling clicks
    if st.button(tool["name"], key=tool["name"], help="Select tool", use_container_width=False):
        st.session_state.selected_tool = tool["name"]
st.markdown('</div>', unsafe_allow_html=True)

# Tool content based on selection
if st.session_state.selected_tool == "Text-to-Image":
    st.header("Text-to-Image Generation")
    prompt = st.text_input("Enter a prompt:", "A futuristic city")
    if st.button("Generate Image") and hf_api_key:
        url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
        headers = {"Authorization": f"Bearer {hf_api_key}"}
        payload = {"inputs": prompt}
        with st.spinner("Generating..."):
            try:
                response = requests.post(url, headers=headers, json=payload)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content))
                    st.image(image, caption="Generated Image")
                else:
                    st.error(f"API error: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    elif not hf_api_key:
        st.warning("Hugging Face API key missing.")

elif st.session_state.selected_tool == "Text-to-Audio":
    st.header("Text-to-Audio Conversion")
    text = st.text_area("Enter text:", "Hello, this is a test.")
    lang = st.selectbox("Language:", ["en", "es", "fr"], index=0)
    if st.button("Convert to Audio"):
        with st.spinner("Generating..."):
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                    tts = gTTS(text=text, lang=lang, slow=False)
                    tts.save(tmp.name)
                    st.audio(tmp.name, format="audio/mp3")
                    with open(tmp.name, "rb") as f:
                        st.download_button("Download", f.read(), "output.mp3", "audio/mp3")
                    os.unlink(tmp.name)
            except Exception as e:
                st.error(f"Error: {str(e)}")

elif st.session_state.selected_tool == "Summarization":
    st.header("AI-Powered Summarization")
    text_to_summarize = st.text_area("Enter text to summarize:", "Paste your text here...", height=200)
    st.write(f"Sentence count: {len(sent_tokenize(text_to_summarize))} | Word count: {len(text_to_summarize.split())}")
    summary_sentences = st.slider("Number of sentences in summary:", 1, 5, 2)
    if st.button("Summarize"):
        if text_to_summarize.strip():
            if len(sent_tokenize(text_to_summarize)) < 2:
                st.warning("Please enter at least two sentences to summarize.")
            else:
                with st.spinner("Summarizing..."):
                    try:
                        sentences = sent_tokenize(text_to_summarize)
                        if len(sentences) <= summary_sentences:
                            st.write("**Summary:**")
                            st.write(text_to_summarize)
                        else:
                            stop_words = set(stopwords.words("english"))
                            words = [w.lower() for w in word_tokenize(text_to_summarize) if w.isalnum() and w.lower() not in stop_words]
                            word_freq = Counter(words)
                            sentence_scores = {}
                            for i, sent in enumerate(sentences):
                                score = sum(word_freq[w.lower()] for w in word_tokenize(sent) if w.isalnum() and w.lower() in word_freq)
                                sentence_scores[i] = score / (len(word_tokenize(sent)) + 1)
                            top_sentences = sorted(sorted(sentence_scores.items(), key=lambda x: x[0])[:summary_sentences], key=lambda x: x[1], reverse=True)
                            summary_sentences_list = [sentences[i] for i, _ in top_sentences]
                            st.write("**Summary:**")
                            for idx, sent in enumerate(summary_sentences_list, 1):
                                st.write(f"{idx}. {sent}")
                    except LookupError as e:
                        st.error(f"NLTK resource missing: {str(e)}. Please restart the app.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter some text to summarize.")

elif st.session_state.selected_tool == "Code Debugger":
    st.header("Code Debugger & Explainer")
    code = st.text_area("Your code:", "def example():\n    print(undefined_variable)")
    if st.button("Debug"):
        with st.spinner("Analyzing..."):
            try:
                with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as tmp:
                    tmp.write(code)
                    tmp_path = tmp.name
                output = StringIO()
                reporter = TextReporter(output)
                Run([tmp_path, "--reports=n"], reporter=reporter, exit=False)
                lint_output = output.getvalue()
                output.close()
                if lint_output.strip():
                    st.text("Issues found:")
                    st.text(lint_output)
                else:
                    st.text("No issues detected by pylint.")
                os.unlink(tmp_path)
                if "undefined_variable" in code:
                    st.markdown("**Explanation**: `undefined_variable` is not defined. Define it (e.g., `undefined_variable = 'something'`) before using it.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

elif st.session_state.selected_tool == "ATS Score Checker":
    st.header("ATS Score Checker")
    resume = st.file_uploader("Upload resume (PDF):", type="pdf")
    job_desc = st.text_area("Job description:", "Enter here...")
    if st.button("Check Score") and fitz:
        if resume and job_desc:
            with st.spinner("Analyzing..."):
                try:
                    pdf = fitz.open(stream=resume.read(), filetype="pdf")
                    resume_text = "".join(page.get_text() for page in pdf)
                    resume_words = set(resume_text.lower().split())
                    job_words = set(job_desc.lower().split())
                    common = resume_words.intersection(job_words)
                    score = min(len(common) / len(job_words) * 100, 100)
                    st.write(f"ATS Score: {score:.2f}%")
                    st.write("Matches:", ", ".join(common))
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.error("Upload a resume and enter a job description.")
    elif not fitz:
        st.warning("ATS unavailable due to missing 'pymupdf'.")
