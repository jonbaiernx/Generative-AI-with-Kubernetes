import os
import sys
import gradio as gr
import chromadb
from chromadb import Settings

from llama_index.core import VectorStoreIndex,ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core.evaluation import ContextRelevancyEvaluator
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core import PromptTemplate

import torch

model_label = "Microsoft’s Phi-2 (Quantised)"
model_file = "phi-2-GGUF/phi-2.Q4_K_M.gguf"

#Defualt configurations
default_context_length = 1024
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
    """ Generate text using the Phi-2 GGUF models and RAG"""
    
    # Placeholder value to return
    results = "Inference did not run."

    # Prompting template
    text_qa_template_str = (
        "Instruct: You are a bot that makes recommendations for art. "
        "Answer with the best item and description, but only use descriptions from these: \n" 
        "{context_str} \n"
        "Make sure to use the correct matching item id. "
        "Suggest all matching items based on the recommendations and the question below: \n"
        "\nQuestion: {query_str} \n"
        "\n Output: Based on your preferences I recommend item "
    ) 
    text_qa_template = PromptTemplate(text_qa_template_str)

    # Create Phi model using Llama-cpp-python on a CPU
    llm = LlamaCPP(
        model_path=model_file,
        temperature=temp,
        max_new_tokens=max_length,
        context_window=context_length,
        # kwargs to pass to __call__()
        generate_kwargs={},
        # kwargs to pass to __init__()
        # set to 0 for CPU and >= 1 for GPU
        model_kwargs={"n_gpu_layers": 0},
        verbose=True,
    )

    # Load embeddings and set service context
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    #Update to Settings 
    service_context = ServiceContext.from_defaults(
        chunk_size=1024,
        llm=llm,
        embed_model=embed_model
    )

    #Create Vector store object and Indexes
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)

    #Create query engine and run prompt using vector store for additional information
    query_engine = index.as_query_engine(text_qa_template=text_qa_template, llm=llm)
    response = query_engine.query(prompt)

    #Setup Evaluators to judge results
    cre_evaluator = ContextRelevancyEvaluator(llm=llm, service_context=service_context)
    cre_eval_results = cre_evaluator.evaluate_response(query=prompt, response=response)

    return response.response, cre_eval_results.feedback


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
            #Textboxes for results
            results_ = gr.Textbox(label=model_label + " with RAG recommendation:")
            context_feedback = gr.Textbox(label="Context Relevancy Evaluation Feedback")  
    with gr.Row(equal_height=True):
        with gr.Column():
            #Examples
            examples = gr.Examples(
                [["I like candy, can you recommend something?", 160], ["I like cars, can you recommend something?", 160], ["I am looking for a painting with clouds.", 160], ["I love the color colbat", 160], ["I don't like clouds, what do you recommend?", 160]],                
                [prompt_input, token_length])
            generate_btn = gr.Button("Generate")
            generate_btn.click(fn=generate_text,
                                inputs=[prompt_input, token_length, context_length, tempurature],
                                outputs=[results_, context_feedback])

# Launch the application
prompt_compare_iface.launch(server_name="0.0.0.0", server_port=7802, share=False, debug=True)
