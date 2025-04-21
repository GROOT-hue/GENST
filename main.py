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

# Custom CSS for unique UI with animated background and 3D cards
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(45deg, #0a0a1a, #1a1a3a);
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
        background-color: #1e90ff;
        color: #ffffff;
        border-radius: 8px;
        border: none;
        box-shadow: 0 0 10px #1e90ff;
    }
    .stButton>button:hover {
        background-color: #4682b4;
        box-shadow: 0 0 20px #4682b4;
    }
    h2, h3 {
        color: #1e90ff;
    }
    .stMarkdown, .stWarning, .stError, .stSuccess {
        color: #ffffff;
    }
    /* Animated background particles */
    .background-particles {
        position: absolute;
        width: 100%;
        height: 100%;
        top: 0;
        left: 0;
        pointer-events: none;
        overflow: hidden;
    }
    .particle {
        position: absolute;
        background: rgba(30, 144, 255, 0.5);
        border-radius: 50%;
        animation: float 15s infinite;
    }
    @keyframes float {
        0% { transform: translateY(100vh) scale(0); opacity: 1; }
        100% { transform: translateY(-10vh) scale(1.5); opacity: 0; }
    }
    /* Unique holographic title */
    .unique-title {
        font-size: 7rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(45deg, #1e90ff, #ff00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 20px #1e90ff, 0 0 30px #ff00ff;
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        animation: pulse 3s infinite, floatTitle 4s infinite alternate;
        z-index: 1;
    }
    @keyframes pulse {
        0% { text-shadow: 0 0 20px #1e90ff, 0 0 30px #ff00ff; }
        50% { text-shadow: 0 0 30px #1e90ff, 0 0 40px #ff00ff; }
        100% { text-shadow: 0 0 20px #1e90ff, 0 0 30px #ff00ff; }
    }
    @keyframes floatTitle {
        0% { transform: translate(-50%, -50%) translateY(0); }
        100% { transform: translate(-50%, -50%) translateY(-10px); }
    }
    /* 3D Tool cards with neon borders */
    .tool-card {
        background: rgba(40, 40, 40, 0.9);
        color: #ffffff;
        padding: 20px;
        border: 2px solid #1e90ff;
        border-radius: 12px;
        text-align: center;
        transition: all 0.3s;
        cursor: pointer;
        min-height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        position: absolute;
        box-shadow: 0 0 10px #1e90ff, inset 0 0 10px #1e90ff;
        transform-style: preserve-3d;
        transform: perspective(500px) rotateY(0deg);
    }
    .tool-card:hover {
        background: rgba(80, 80, 80, 0.9);
        transform: perspective(500px) rotateY(10deg) scale(1.05);
        box-shadow: 0 0 20px #1e90ff, inset 0 0 20px #1e90ff;
    }
    .tool-card h3 {
        margin: 10px 0;
        font-size: 1.5rem;
    }
    .tool-card p {
        font-size: 0.9rem;
        color: #cccccc;
    }
    .tool-card::after {
        content: "↓";
        font-size: 2rem;
        color: #1e90ff;
        position: absolute;
        bottom: 10px;
        left: 50%;
        transform: translateX(-50%);
        animation: pulseArrow 2s infinite;
    }
    @keyframes pulseArrow {
        0% { text-shadow: 0 0 5px #1e90ff; }
        50% { text-shadow: 0 0 10px #1e90ff; }
        100% { text-shadow: 0 0 5px #1e90ff; }
    }
    /* Custom positioning for tool cards */
    .card-1 { top: 10%; left: 10%; width: 20%; }
    .card-2 { top: 40%; left: 30%; width: 20%; }
    .card-3 { top: 40%; left: 55%; width: 20%; }
    .card-4 { top: 70%; left: 30%; width: 20%; }
    .card-5 { top: 70%; left: 55%; width: 20%; }
    .back-button-container {
        margin: 20px 0;
        text-align: left;
    }
    /* JavaScript for particle animation */
    <script>
        function createParticles() {
            const particleCount = 50;
            const particlesContainer = document.querySelector('.stApp');
            for (let i = 0; i < particleCount; i++) {
                const particle = document.createElement('div');
                particle.className = 'particle';
                particle.style.left = `${Math.random() * 100}vw`;
                particle.style.width = `${Math.random() * 5 + 2}px`;
                particle.style.height = particle.style.width;
                particle.style.animationDuration = `${Math.random() * 10 + 5}s`;
                particle.style.animationDelay = `${Math.random() * 5}s`;
                particlesContainer.appendChild(particle);
            }
        }
        createParticles();
    </script>
    </style>
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
        "description": "Generate images from text prompts using AI."
    },
    {
        "name": "Text-to-Audio",
        "icon": "🎵",
        "description": "Convert text to speech in multiple languages."
    },
    {
        "name": "Summarization",
        "icon": "📝",
        "description": "Summarize long texts into concise points."
    },
    {
        "name": "Code Debugger",
        "icon": "💻",
        "description": "Analyze and debug Python code."
    },
    {
        "name": "ATS Score Checker",
        "icon": "📄",
        "description": "Evaluate resume compatibility with job descriptions."
    }
]

# Page rendering
if st.session_state.page == "tools":
    # First page: Tools selection
    st.markdown('<h1 class="unique-title">GEN IQ</h1>', unsafe_allow_html=True)
    st.markdown('<div class="background-particles"></div>', unsafe_allow_html=True)
    
    # Tool cards with custom positioning
    for i, tool in enumerate(tools):
        card_class = f"card-{i+1}"
        st.markdown(
            f'<div class="tool-card {card_class}" onclick="document.getElementById(\'{tool["name"]}\').click()">'
            f'<h3>{tool["icon"]} {tool["name"]}</h3>'
            f'<p>{tool["description"]}</p>'
            f'</div>',
            unsafe_allow_html=True
        )
        if st.button(tool["name"], key=tool["name"], help="Select tool", use_container_width=False):
            st.session_state.selected_tool = tool["name"]
            st.session_state.page = "tool"

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
