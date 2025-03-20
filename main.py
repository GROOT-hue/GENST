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

# Title
st.title("GEN IQ")

# Tabs
tab_names = ["Text-to-Image", "Text-to-Audio", "Summarization", "Code Debugger", "ATS Score Checker"]
tabs = st.tabs(tab_names)

# 1. Text-to-Image
with tabs[0]:
    st.header("Text-to-Image Generation")
    # Diagnostic outputs
    st.write(f"API Key Status: {'Set' if hf_api_key else 'Not Set'} (Length: {len(hf_api_key) if hf_api_key else 0})")
    if hf_api_key:
        st.write(f"API Key Preview: {hf_api_key[:5]}...{hf_api_key[-5:]}")
    else:
        st.write("No API key detected in st.secrets or os.getenv('HF_API_KEY').")
    prompt = st.text_input("Enter a prompt:", "A futuristic city")
    if st.button("Generate Image"):
        if not hf_api_key:
            st.error("No API key provided. Set 'HF_API_KEY' in secrets or environment variables.")
        else:
            url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
            headers = {"Authorization": f"Bearer {hf_api_key.strip()}"}
            payload = {"inputs": prompt}
            with st.spinner("Generating..."):
                try:
                    st.write(f"Sending request to: {url}")
                    st.write(f"Headers (masked): {{'Authorization': 'Bearer {hf_api_key[:5]}...'}}")
                    response = requests.post(url, headers=headers, json=payload)
                    if response.status_code == 200:
                        image = Image.open(BytesIO(response.content))
                        st.image(image, caption="Generated Image")
                    elif response.status_code == 401:
                        st.error("401 Unauthorized: Invalid or missing API key. Verify your Hugging Face key.")
                        st.write(f"Full API response: {response.text}")
                    else:
                        st.error(f"API error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Request failed: {str(e)}")
# 2. Text-to-Audio
with tabs[1]:
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

# 3. Summarization
with tabs[2]:
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

# 4. Code Debugger
with tabs[3]:
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

# 5. ATS Score Checker
with tabs[4]:
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
