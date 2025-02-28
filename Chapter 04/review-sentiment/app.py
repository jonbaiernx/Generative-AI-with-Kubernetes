import gradio as gr
from langchain.prompts import PromptTemplate 
from langchain.prompts import FewShotPromptTemplate
from langchain_community.llms import LlamaCpp

orca_label = "Orca 2 7B"
orca_model_file = "Orca-2-7B-GGUF/orca-2-7b.Q5_K_M.gguf"
mistral_label = "Mistral 7B Instruct"
mistral_model_file = "Mistral-7B-Instruct-v0.2-GGUF/mistral-7b-instruct-v0.2.Q5_K_M.gguf"

default_context_length = 1024
default_token_length = 200
default_tempurature = 0.1


def classify_reviews(prompt, models, max_length=default_token_length, temp=default_tempurature):
    """ Classify review sentiment using the Orca and Mistral 7B GPTQ models """

    #placeholder value to return
    results1 = results2 = "Inference was not run with this model"
  
    #Review Sentiment Instructions
    instruction_prompt = "You are to classify statement according to their sentiment as Positive, Negative, or Neutral."
    #Add some additional instructions to user prompt
    user_prompt = "Please list the sentiment of each of the following sentences: " + prompt 

    # Classified review examples
    review_examples = [
        {
            "query": "What is the sentiment of the following: 'I really enjoyed the food!'",
            "answer": "Positive"
        }, {
            "query": "What is the sentiment of the following: 'The lighting was too dim and the music was too loud'",
            "answer": "Negative"
        }, {
            "query": "What is the sentiment of the following: 'The staff was very attentive'",
            "answer": "Positive"
        }
    ]

    # Template for example reviews
    example_review_template = """
    User: {query}
    AI: {answer}
    """

    # Create template for example review section of system prompt
    example_prompts = PromptTemplate(
        input_variables=["query", "answer"],
        template=example_review_template
    )

    # Prepare our main prompt with a place for Preceding Instructions, Examples, and User Prompt
    instructions_template = """{instructions} 
    Here are some
    examples: 
    """
    # and the suffix our user input and output indicator
    user_prompt_template = """
    User: {query}
    AI: """

    # Prepare full template with few-shot examples
    few_shot_prompt_template = FewShotPromptTemplate(
        examples=review_examples,
        example_prompt=example_prompts,
        prefix=instructions_template,
        suffix=user_prompt_template,
        input_variables=["instructions","query"],
        example_separator="\n\n"
    )

    few_shot_prompt = few_shot_prompt_template.format(instructions=instruction_prompt, query=user_prompt)

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
        orca_response = orca_llm.invoke(few_shot_prompt)
            
    if mistral_label in models:
        #Load model 
        mistral_llm = LlamaCpp(
            model_path=mistral_model_file,
            temperature=default_tempurature,
            max_tokens=default_token_length,
            n_ctx=default_context_length
        )

              
        #Run Inference
        mistral_response = mistral_llm.invoke(few_shot_prompt)
            

    return [orca_response, mistral_response]


# Create the Gradio interface with blocks
with gr.Blocks() as prompt_compare_iface:
    gr.Markdown("""# Restaurant Review Sentiment Classification - Orca and Mistral 7B Comparison""")
    gr.Markdown("""This is a simple restaurant review sentiment classification app that allows users to compare the Microsoft Orca and Mistral 7B quantized models.""")
    with gr.Row(equal_height=True):
        with gr.Column():
            #Input for prompt
            prompt_input = gr.Textbox(lines=5,
                                       label="Restaurant Review(s)",
                                    placeholder='Enter a review(s) to classify seperated by commas...')
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
            generate_btn.click(fn=classify_reviews,
                                inputs=[prompt_input, model_choice, token_length, tempurature],
                                outputs=[output1, output2])


# Launch the application
prompt_compare_iface.launch(server_name="0.0.0.0", server_port=7801, share=False, debug=True)