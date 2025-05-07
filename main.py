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

# Custom CSS for unique homepage, centered buttons, and hiding Streamlit branding
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1a0033, #330066);
        color: #ffffff;
        position: relative;
        min-height: 100vh;
        overflow: hidden;
    }
    .stTextInput, .stTextArea, .stSelectbox, .stFileUploader {
        background-color: rgba(50, 50, 50, 0.9);
        color: #ffffff;
        border-radius: 8px;
    }
    .stButton>button {
        background: linear-gradient(45deg, #ff00ff, #00ccff);
        color: #ffffff;
        border-radius: 12px;
        border: none;
        box-shadow: 0 0 15px rgba(255, 0, 255, 0.7);
        display: block;
        margin: 10px auto;
        padding: 12px 24px;
        font-size: 1.1rem;
        font-weight: bold;
        width: 90%;
        max-width: 200px;
        text-align: center;
        transition: transform 0.3s, box-shadow 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.1);
        box-shadow: 0 0 25px rgba(0, 204, 255, 0.9);
    }
    .stButton>button:disabled {
        background: linear-gradient(45deg, #666666, #999999);
        box-shadow: none;
        cursor: not-allowed;
    }
    h2, h3 {
        color: #00ffcc;
    }
    .stMarkdown, .stWarning, .stError, .stSuccess {
        color: #ffffff;
    }
    /* Enhanced particle background */
    .background-particles {
        position: absolute;
        width: 100%;
        height: 100%;
        top: 0;
        left: 0;
        pointer-events: none;
        z-index: 0;
    }
    .particle {
        position: absolute;
        background: radial-gradient(circle, rgba(0, 255, 204, 0.8), transparent);
        border-radius: 50%;
        width: 6px;
        height: 6px;
        animation: float 8s infinite ease-in-out;
        box-shadow: 0 0 10px rgba(0, 255, 204, 0.5);
    }
    @keyframes float {
        0% { transform: translateY(100vh) scale(0.5); opacity: 0.8; }
        50% { opacity: 1; }
        100% { transform: translateY(-20vh) scale(1.2); opacity: 0; }
    }
    .particle:nth-child(1) { left: 5%; animation-duration: 10s; animation-delay: 0s; }
    .particle:nth-child(2) { left: 15%; animation-duration: 12s; animation-delay: 1s; }
    .particle:nth-child(3) { left: 25%; animation-duration: 9s; animation-delay: 2s; }
    .particle:nth-child(4) { left: 35%; animation-duration: 11s; animation-delay: 3s; }
    .particle:nth-child(5) { left: 45%; animation-duration: 13s; animation-delay: 0.5s; }
    .particle:nth-child(6) { left: 55%; animation-duration: 10s; animation-delay: 1.5s; }
    .particle:nth-child(7) { left: 65%; animation-duration: 14s; animation-delay: 2.5s; }
    .particle:nth-child(8) { left: 75%; animation-duration: 8s; animation-delay: 3.5s; }
    .particle:nth-child(9) { left: 85%; animation-duration: 12s; animation-delay: 4s; }
    .particle:nth-child(10) { left: 95%; animation-duration: 11s; animation-delay: 0.2s; }
    /* Neon glowing title */
    .unique-title {
        font-size: 7rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(45deg, #ff00ff, #00ccff, #ff00ff);
        background-size: 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 20px #ff00ff, 0 0 30px #00ccff;
        margin: 30px 0;
        animation: neon-glow 4s ease-in-out infinite;
    }
    @keyframes neon-glow {
        0% { background-position: 0%; text-shadow: 0 0 20px #ff00ff, 0 0 30px #00ccff; }
        50% { background-position: 100%; text-shadow: 0 0 30px #ff00ff, 0 0 40px #00ccff; }
        100% { background-position: 0%; text-shadow: 0 0 20px #ff00ff, 0 0 30px #00ccff; }
    }
    /* 3D tool card grid with neon borders */
    .tool-card-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 25px;
        width: 90%;
        max-width: 1300px;
        margin: 30px auto;
        perspective: 1000px;
    }
    .tool-card {
        background: rgba(20, 20, 30, 0.95);
        color: #ffffff;
        padding: 25px;
        border: 2px solid transparent;
        border-radius: 15px;
        text-align: center;
        transition: transform 0.4s, box-shadow 0.4s, border 0.4s;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 0 15px rgba(0, 204, 255, 0.5);
        transform: rotateY(0deg);
    }
    .tool-card:hover {
        transform: rotateY(5deg) translateY(-10px);
        box-shadow: 0 0 30px rgba(255, 0, 255, 0.7);
        border: 2px solid #ff00ff;
    }
    .tool-card h3 {
        margin: 15px 0;
        font-size: 1.7rem;
        color: #00ffcc;
        text-shadow: 0 0 5px #00ffcc;
    }
    .tool-card p {
        font-size: 1rem;
        color: #e0e0e0;
        margin-bottom: 25px;
    }
    .tool-card .button-container {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
    }
    .back-button-container {
        margin: 20px 0;
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
    }
    /* Hide Streamlit footer, logo, and cloud branding */
    footer, .stApp > footer, .stApp > footer::before, .stApp > footer::after {
        visibility: hidden !important;
        display: none !important;
    }
    [data-testid="stStatusWidget"], [data-testid="stFooter"], .st-emotion-cache-1wmy9hl {
        visibility: hidden !important;
        display: none !important;
    }
    /* Ensure no Streamlit branding elements are visible */
    .stApp > div:last-child {
        visibility: hidden !important;
        display: none !important;
    }
    @media (max-width: 600px) {
        .unique-title {
            font-size: 4.5rem;
        }
        .tool-card-container {
            grid-template-columns: 1fr;
        }
        .stButton>button {
            width: 95%;
        }
    }
    </style>
""", unsafe_allow_html=True)

# JavaScript for single-click enforcement
st.markdown("""
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const buttons = document.querySelectorAll('.stButton button');
        buttons.forEach(button => {
            button.addEventListener('click', function() {
                button.disabled = true;
                setTimeout(() => {
                    button.disabled = false;
                }, 1000); // Re-enable after 1 second
            });
        });
    });
    </script>
""", unsafe_allow_html=True)

# Session state for navigation and tool selection
if 'page' not in st.session_state:
    st.session_state.page = "tools"
if 'selected_tool' not in st.session_state:
    st.session_state.selected_tool = None

# Tools definitions
tools = [
    {
        "name": "Text-to-Image",
        "icon": "🖼️",
        "description": "Generate stunning visuals from text prompts using AI."
    },
    {
        "name": "Text-to-Audio",
        "icon": "🎵",
        "description": "Transform text into speech with multilingual support."
    },
    {
        "name": "Summarization",
        "icon": "📝",
        "description": "Condense lengthy texts into concise summaries."
    },
    {
        "name": "Code Debugger",
        "icon": "💻",
        "description": "Analyze and fix Python code with ease."
    },
    {
        "name": "ATS Score Checker",
        "icon": "📄",
        "description": "Optimize your resume for job applications."
    }
]

# Page rendering
if st.session_state.page == "tools":
    # Unique homepage: Tools selection
    st.markdown('<h1 class="unique-title">GEN IQ</h1>', unsafe_allow_html=True)
    st.markdown("""
        <div class="background-particles">
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="tool-card-container">', unsafe_allow_html=True)
    
    # Enhanced tool cards with centered buttons
    for tool in tools:
        st.markdown(
            f'<div class="tool-card">'
            f'<h3>{tool["icon"]} {tool["name"]}</h3>'
            f'<p>{tool["description"]}</p>'
            f'<div class="button-container">',
            unsafe_allow_html=True
        )
        if st.button(tool["name"], key=tool["name"], help=f"Explore {tool['name']}"):
            st.session_state.selected_tool = tool["name"]
            st.session_state.page = "tool"
        st.markdown('</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "tool" and st.session_state.selected_tool:
    # Second page: Selected tool
    st.markdown('<div class="back-button-container">', unsafe_allow_html=True)
    if st.button("Back to Tools"):
        st.session_state.page = "tools"
        st.session_state.selected_tool = None
    st.markdown('</div>', unsafe_allow_html=True)

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
