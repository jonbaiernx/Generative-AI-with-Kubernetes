import gradio as gr
from transformers import pipeline


# Initialize the GPT-2 model pipeline
gpt2_pipeline = pipeline('text-generation', 
                         model='gpt2')

def generate_text(prompt, max_length=100, num_return_sequences=1):
    """ Generate text using the GPT-2 model """
    results = gpt2_pipeline(prompt, 
                            max_length=max_length,
                            num_return_sequences=num_return_sequences,
                            do_sample=True)
    # Return just the generated text
    return results[0]['generated_text']

# Create a Gradio interface
gpt2_iface = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(lines=2, 
                      placeholder="Enter a prompt..."),
    outputs='text',
    title="GPT-2 Text Generator",
    description="Enter a prompt to generate text using GPT-2."
)

# Launch the application
gpt2_iface.launch(server_name="0.0.0.0",server_port=7800)
