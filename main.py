import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import tempfile
import os
from gtts import gTTS
from pylint.lint import Run  # Updated import
from pylint.reporters.text import TextReporter
from io import StringIO

try:
    import fitz  # PyMuPDF
except ImportError:
    st.warning("PyMuPDF not installed. ATS Score Checker will be skipped.")
    fitz = None

# API Key (only Hugging Face)
hf_api_key = st.secrets.get("HF_API_KEY", os.getenv("HF_API_KEY"))

# Title
st.title("GEN IQ")

# Tabs
tab_names = ["Text-to-Image", "Text-to-Audio", "Summarization", "Code Debugger", "ATS Score Checker"]
tabs = st.tabs(tab_names)

# 1. Text-to-Image
with tabs[0]:
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

# 3. Summarization (Disabled)
with tabs[2]:
    st.header("AI-Powered Summarization")
    st.warning("Summarization unavailable due to dependency issues. Requires 'transformers' and 'torch'.")

# 4. Code Debugger (Fixed with pylint.lint)
with tabs[3]:
    st.header("Code Debugger & Explainer")
    code = st.text_area("Your code:", "def example():\n    print(undefined_variable)")
    if st.button("Debug"):
        with st.spinner("Analyzing..."):
            try:
                # Write code to a temporary file
                with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as tmp:
                    tmp.write(code)
                    tmp_path = tmp.name
                
                # Capture pylint output
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
                
                # Basic rule-based explanation
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
