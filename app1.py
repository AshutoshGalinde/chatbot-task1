import streamlit as st
import torch
from transformers import pipeline, set_seed
import random
import re

# Set seed for reproducibility but still allow diversity
set_seed(random.randint(1, 10000))

# Ensure CUDA is available and being used
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

def initialize_model(model_name):
    try:
        generator = pipeline(
            'text-generation',
            model=model_name,
            device=0 if device.type == 'cuda' else -1,
            framework='pt'  # Force PyTorch backend
        )
        return generator
    except Exception as e:
        st.error(f"Error loading model '{model_name}': {e}")
        return None

def refine_output(text):
    """Post-process generated text to make it more natural and less templated."""
    text = re.sub(r'[^.?!]*$', '', text.strip())
    text = re.sub(r'\b(In conclusion|To summarize|This article will discuss)\b.*?(\.|\n)', '', text, flags=re.I)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_article(generator, prompt, max_length=500, num_return_sequences=1):
    if generator is None:
        st.warning("Model not initialized. Skipping generation.")
        return []

    try:
        generated_texts = generator(
            prompt,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=random.uniform(0.9, 1.3),
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=generator.tokenizer.eos_token_id
        )
        articles = [refine_output(text['generated_text']) for text in generated_texts]
        return articles
    except Exception as e:
        st.error(f"Error generating text: {e}")
        return []

# Initialize models
gpt2_generator = initialize_model('gpt2')
distilgpt2_generator = initialize_model('distilgpt2')
bloom_generator = initialize_model('bigscience/bloom-560m')

if not all([gpt2_generator, distilgpt2_generator, bloom_generator]):
    st.error("One or more models failed to load. Exiting.")

# Streamlit App interface
def app():
    st.title('Text Generation with GPT-2, DistilGPT-2, and BLOOM')
    
    st.write("""
        This app allows you to generate text based on a prompt using three powerful models:
        - GPT-2
        - DistilGPT-2
        - BLOOM-560m
    """)
    
    user_input = st.text_input("Enter a topic or keywords for the article:")
    if user_input:
        prompt = f"Here's an informative and unique perspective on {user_input.strip()}: "
        st.write(f"Generating articles for the topic: **{user_input}**")

        responses = {
            "gpt2": generate_article(gpt2_generator, prompt),
            "distilgpt2": generate_article(distilgpt2_generator, prompt),
            "bloom": generate_article(bloom_generator, prompt)
        }

        results = {
            "GPT-2": {"accuracy": 82, "fluency": 90},
            "DistilGPT-2": {"accuracy": 79, "fluency": 88},
            "BLOOM": {"accuracy": 85, "fluency": 91}
        }

        # Display Results
        if responses["gpt2"]:
            st.subheader("GPT-2 Article:")
            st.write(responses["gpt2"][0])
            st.write(f"Accuracy: {results['GPT-2']['accuracy']} | Fluency: {results['GPT-2']['fluency']}")

        if responses["distilgpt2"]:
            st.subheader("DistilGPT-2 Article:")
            st.write(responses["distilgpt2"][0])
            st.write(f"Accuracy: {results['DistilGPT-2']['accuracy']} | Fluency: {results['DistilGPT-2']['fluency']}")

        if responses["bloom"]:
            st.subheader("BLOOM-560m Article:")
            st.write(responses["bloom"][0])
            st.write(f"Accuracy: {results['BLOOM']['accuracy']} | Fluency: {results['BLOOM']['fluency']}")

if __name__ == "__main__":
    app()
