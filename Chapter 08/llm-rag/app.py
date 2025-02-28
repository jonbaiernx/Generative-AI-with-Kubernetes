import os
import sys
import gradio as gr
import pandas as pd
import chromadb
from chromadb import Settings
from langchain_community.llms import LlamaCpp

model_label = "Microsoft’s Phi-2 (Quantised)"
model_file = "phi-2-GGUF/phi-2.Q4_K_M.gguf"

#Defualt configurations
default_context_length = 512
default_token_length = 160
default_tempurature = 0.1

# Setup connection to ChromaDB Service
chroma_token = os.environ['CHROMA_TOKEN']
chroma_host = "chroma-chromadb"
chroma_port = "8000"
client = chromadb.HttpClient(host=chroma_host, port=chroma_port,
                            settings=Settings(allow_reset=True, anonymized_telemetry=False, chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
                                            chroma_client_auth_credentials=chroma_token))

# Create collection to store artwork descriptions, if it doesn't exist
collection = client.get_or_create_collection("art-descriptions")

if collection.count() <= 1:
    print("Error retrieving collection")
    sys.exit()

def generate_text(prompt, max_length=default_token_length, context_length=default_context_length, temp=default_tempurature):
    """ Generate text using the Phi-2 GGUF models """
    
    #Get Vector db matches
    retrevied_descriptions = pd.DataFrame(
        {'IDs': ['-999'],
        'Descriptions': ['No Results']})
    
    try:
        retrevial_matches = collection.query(
            query_texts=[prompt],
            n_results=3)
        #Transform into a DataFrame for easy display
        retrevied_descriptions = pd.DataFrame(
            {'IDs': retrevial_matches.get('ids')[0],
            'Descriptions':  retrevial_matches.get('documents')[0]})
    except ValueError:
        print("Error fecthing collection")

    # Placeholder value to return
    results = "Inference was not run with this model"

    # Prepare prompt
    augmented_prompt = """Instruct: You are a bot that makes recommendations for art. Answer with the best item and description, but only use descriptions from these: 
    {relevant_document}. Make sure to use the correct matching item id. Suggest all matching items based on the recommendations and this request: {user_input}
    Output: Based on your preferences I recommend item """
    augmented_prompt = augmented_prompt.format(user_input=prompt, relevant_document=retrevied_descriptions)
    #Load model 
    phi_llm = LlamaCpp(
        model_path=model_file,
        temperature=temp,
        max_tokens=max_length,
        n_ctx=context_length,
        n_gpu_layers=0,
        verbose=True
    )
    #Run Inference
    results = phi_llm.invoke(augmented_prompt)
    #return responses
    return retrevied_descriptions, results


# Create the Gradio interface with blocks
with gr.Blocks() as prompt_compare_iface:
    gr.Markdown("""# Phi-2 GGUF RAG""")
    gr.Markdown("""This is a simple Gradio app that allows users to generate quantized Microsoft's Phi-2 responses with a simple ChromaDB table augementation.""")
    with gr.Row(equal_height=True):
        with gr.Column():
            #Input for prompt
            prompt_input = gr.Textbox(lines=5,
                                       label="User Prompt",
                                    placeholder='Enter a prompt ...')
            #Maximum length of responses
            token_length = gr.Number(value=default_token_length, 
                                label="Maximum response length (# of Tokens) from model", minimum=0, step=1)
            #Maximum length of Context Window
            context_length = gr.Number(value=default_context_length, 
                                label="Maximum context length (Input+Output)", minimum=0, step=1)
            #Tempurature of responses
            tempurature = gr.Number(value=default_tempurature, 
                                label="Tempurature of response", min_width=50, minimum=0, step=0.1)
        with gr.Column():
            #Textbox for each model output
            output1 = gr.Dataframe(
                label="Art DB Matches",
                headers=["Item #","Description"],
                datatype=["str","str"])
            output2 = gr.Textbox(label=model_label + " with RAG recommendation:")
    with gr.Row(equal_height=True):
        with gr.Column():
            #Examples
            examples = gr.Examples(
                [["I like cars, can you recommend something?", 160], ["I am looking for a painting with clouds.", 160], ["I love the color colbat", 160], ["I don't like clouds, what do you recommend?", 160]],
                [prompt_input, token_length])
            generate_btn = gr.Button("Generate")
            generate_btn.click(fn=generate_text,
                                inputs=[prompt_input, token_length, context_length, tempurature],
                                outputs=[output1, output2])

# Launch the application
prompt_compare_iface.launch(server_name="0.0.0.0", server_port=7801, share=False, debug=True)