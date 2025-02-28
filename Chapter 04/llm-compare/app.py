import gradio as gr
from langchain.prompts import PromptTemplate 
from langchain_community.llms import LlamaCpp

orca_label = "Orca 2 7B"
orca_model_file = "Orca-2-7B-GGUF/orca-2-7b.Q5_K_M.gguf"
mistral_label = "Mistral 7B Instruct"
mistral_model_file = "Mistral-7B-Instruct-v0.2-GGUF/mistral-7b-instruct-v0.2.Q5_K_M.gguf"

default_context_length = 1024
default_token_length = 200
default_tempurature = 0.1


def generate_text(instruction, prompt, models, max_length=default_token_length, temp=default_tempurature):
    """ Generate text using the Orca and Mistral 7B GGUF models """
    #placeholder value to return
    results1 = results2 = "Inference was not run with this model"
 
    # Prepare prompt
    template = """{instructions}

                Question: {query}

                Answer: """

    prompt_template = PromptTemplate(
        input_variables=["instructions","query"],
        template=template
    )
    prompt = prompt_template.format(instructions=instruction, query=prompt)

    #Run inference for selected models
    if orca_label in models:
        #Load model
        orca_llm = LlamaCpp(
            model_path=orca_model_file,
            temperature=default_tempurature,
            max_tokens=default_token_length,
            n_ctx=default_context_length
        )
        
        #Run Inference
        orca_response = orca_llm.invoke(prompt)
            
    if mistral_label in models:
        #Load model 
        mistral_llm = LlamaCpp(
            model_path=mistral_model_file,
            temperature=default_tempurature,
            max_tokens=default_token_length,
            n_ctx=default_context_length
        )

              
        #Run Inference
        mistral_response = mistral_llm.invoke(prompt)
            

    return [orca_response, mistral_response]


# Create the Gradio interface with blocks
with gr.Blocks() as prompt_compare_iface:
    gr.Markdown("""# Orca and Mistral 7B Comparison""")
    gr.Markdown("""This is a simple Gradio app that allows users to compare the Microsoft Orca and Mistral 7B quantized models.""")
    with gr.Row(equal_height=True):
        with gr.Column():
            #Input for instructions
            instruction_input = gr.Textbox(lines=5,
                                           label="Instruction Prompt",
                                    placeholder='Enter instructions for the model ...')
            #Input for prompt
            prompt_input = gr.Textbox(lines=5,
                                       label="User Prompt",
                                    placeholder='Enter a prompt ...')
            #Maximum length of responses
            token_length = gr.Number(value=default_token_length, label="Maximum response length (# of Tokens) from model", minimum=0, step=1)
            #Tempurature of responses
            tempurature = gr.Number(value=default_tempurature, label="Tempurature of response", min_width=50, minimum=0, step=0.1)
            #Model Choice
            model_choice = gr.CheckboxGroup(choices=[orca_label, mistral_label], value=[orca_label, mistral_label], label="Models", info="Which Models Should We Run?")

        with gr.Column():              
            #Textbox for each model output
            output1 = gr.Textbox(label=orca_label)
            output2 = gr.Textbox(label=mistral_label)
    with gr.Row(equal_height=True):
        with gr.Column():
            generate_btn = gr.Button("Generate")
            generate_btn.click(fn=generate_text,
                                inputs=[instruction_input, prompt_input, model_choice, token_length, tempurature],
                                outputs=[output1, output2])


# Launch the application
prompt_compare_iface.launch(server_name="0.0.0.0", server_port=7800, share=False, debug=True)