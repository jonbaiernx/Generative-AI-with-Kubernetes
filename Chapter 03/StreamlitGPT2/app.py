import torch
import streamlit as st
from transformers import pipeline

# Initialize the GPT-2 model pipeline
gpt2_pipeline = pipeline('text-generation', model='gpt2')

def generate_text(prompt, max_length, num_return_sequences):
    # Generate text using the GPT-2 model
    results = gpt2_pipeline(prompt, 
                            max_length=max_length,
                            num_return_sequences=num_return_sequences,
                            do_sample=True)
    # Return just the generated text
    responses = []
    for j in range(len(results)):
        responses.append(results[j]['generated_text'])
    return responses

#Create a Streamlit Form Interface
with st.form('GPT2_form'):
    st.title('Natural Language Generation with GPT-2')
    prompt = st.text_input(label='Prompt', 
                           value='The color of the ocean is')
    num_return_sequences = st.number_input(label='Number of generated sequences', 
                                           min_value=1, 
                                           max_value=100, 
                                           value=15)
    max_length = st.number_input(label='Length of sequences', 
                                 min_value=5, 
                                 max_value=200, 
                                 value=30)
    generate_btn = st.form_submit_button('Generate')
    #When 'Generate' button is clicked
    if generate_btn:
        try:
            st.dataframe(generate_text(prompt, 
                                       max_length, 
                                       num_return_sequences), 
                            use_container_width=True)
        except Exception as e:
            st.exception("Exception: %s\n" % e)

    