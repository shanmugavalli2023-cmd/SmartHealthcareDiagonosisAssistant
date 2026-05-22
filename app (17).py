
import streamlit as st
from transformers import pipeline
import random

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Smart Healthcare Diagnosis Assistant",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Smart Healthcare Diagnosis Assistant")
st.write("AI-powered healthcare diagnosis assistant using Self-Consistency Prompting")

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------
@st.cache_resource
def load_model():
    generator = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_new_tokens=250
    )
    return generator

model = load_model()

# ------------------------------------------------
# USER INPUT
# ------------------------------------------------
st.subheader("Enter Patient Symptoms")

symptoms = st.text_area(
    "Example: fever, cough, headache, chest pain",
    height=150
)

# ------------------------------------------------
# SELF CONSISTENCY PROMPTING
# ------------------------------------------------
def generate_diagnosis(symptoms):

    prompt_templates = [

        f"""
        Patient symptoms: {symptoms}

        Analyze the symptoms and provide:
        1. Possible disease
        2. Reasoning
        3. Recommended medical tests
        4. Treatment suggestions
        """,

        f"""
        A patient reports these symptoms: {symptoms}

        Generate:
        - Most likely diagnosis
        - Alternative conditions
        - Diagnostic tests
        - Recommendations
        """,

        f"""
        Symptoms: {symptoms}

        Act as a medical AI assistant.
        Provide:
        Disease prediction,
        causes,
        tests,
        precautions,
        and treatment guidance.
        """
    ]

    outputs = []

    for prompt in prompt_templates:
        result = model(prompt)[0]["generated_text"]
        outputs.append(result)

    return outputs

# ------------------------------------------------
# BUTTON
# ------------------------------------------------
if st.button("Generate Diagnosis"):

    if symptoms.strip() == "":
        st.warning("Please enter symptoms")
    else:

        with st.spinner("Analyzing symptoms..."):

            results = generate_diagnosis(symptoms)

            st.success("Diagnosis Generated")

            st.subheader("Multiple Diagnosis Paths")

            for i, res in enumerate(results):
                st.markdown(f"### Diagnosis Path {i+1}")
                st.write(res)

            st.subheader("Most Consistent Prediction")

            final_prediction = random.choice(results)

            st.info(final_prediction)

            st.warning(
                "⚠️ This AI tool is for educational purposes only. "
                "Consult a certified doctor for medical advice."
            )

# ------------------------------------------------
# FOOTER
# ------------------------------------------------
st.markdown("---")
st.markdown("Developed using Generative AI + Self-Consistency Prompting")
