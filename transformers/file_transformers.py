import streamlit as st
from transformers import pipeline
from PIL import Image

st.set_page_config(page_title="AI Playground", layout="centered")

st.title("🤖 AI Playground with Transformers")

# Sidebar
task = st.sidebar.selectbox(
    "Choose a task:",
    ["Sentiment Analysis", "Text Generation",
     "Image Classification", "Speech to Text"]
)

# ------------------ LOADERS (Lazy Loading) ------------------

@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis")

@st.cache_resource
def load_generator():
    return pipeline("text-generation", model="gpt2")

@st.cache_resource
def load_image_classifier():
    return pipeline("image-classification")

@st.cache_resource
def load_asr():
    return pipeline("automatic-speech-recognition")

# ------------------ SENTIMENT ------------------

if task == "Sentiment Analysis":
    text = st.text_area("Enter text:", "I love using transformers")

    if st.button("Analyze"):
        model = load_sentiment()
        with st.spinner("Analyzing..."):
            result = model(text)[0]

        st.success(f"Sentiment: {result['label']}")
        st.write(f"Confidence: {result['score']:.4f}")

# ------------------ TEXT GENERATION ------------------

elif task == "Text Generation":
    prompt = st.text_area("Enter prompt:", "The quick brown fox")

    if st.button("Generate"):
        model = load_generator()
        with st.spinner("Generating..."):
            result = model(prompt, max_length=50, num_return_sequences=1)

        st.success("Generated Text:")
        st.write(result[0]['generated_text'])

# ------------------ IMAGE CLASSIFICATION ------------------

elif task == "Image Classification":
    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Classify"):
            model = load_image_classifier()
            with st.spinner("Classifying..."):
                results = model(image)

            st.success("Predictions:")
            for r in results[:5]:
                st.write(f"{r['label']} : {r['score']:.4f}")

# ------------------ SPEECH TO TEXT ------------------

elif task == "Speech to Text":
    uploaded_file = st.file_uploader(
        "Upload audio", type=["wav", "mp3", "flac"])

    if uploaded_file:
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes)

        if st.button("Transcribe"):
            model = load_asr()
            with st.spinner("Transcribing..."):
                result = model(audio_bytes)

            st.success("Transcription:")
            st.write(result['text'])
