import random
import gradio as gr
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, AutoencoderKL
import torch

#Configuration Settings
IMAGE_HEIGHT = 400
IMAGE_WIDTH = 600
IMAGE_VARIATIONS = 3

# Initialize OpenJourney model pipeline (Creates a StableDiffusionPipeline)
openj_pipeline = DiffusionPipeline.from_pretrained(
    "prompthero/openjourney-v4", 
    torch_dtype=torch.float32,
    use_safetensors=True
)

def generate_image(prompt, num_inference_steps, VAE_choice):
    """ Generate an image from prompt using the PromptHero's OpenJourney model """
    #How closely to match prompt guidance
    prompt_guidance = 9
    random_seeds = [random.randint(0, 65000) for _ in range(IMAGE_VARIATIONS)]
    
    #Configure Pipeline Scheduler
    openj_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(openj_pipeline.scheduler.config)
    #Configure Pipeline VAE
    openj_pipeline.vae = AutoencoderKL.from_pretrained("stabilityai/"+VAE_choice, 
                                                       torch_dtype=torch.float32)
    
    #Enable CUDA
    #openj_pipeline.to("cuda")
    
    images = openj_pipeline(prompt= IMAGE_VARIATIONS * [prompt],
                num_inference_steps=num_inference_steps,
                guidance_scale=prompt_guidance,
                height = IMAGE_HEIGHT,
                width = IMAGE_WIDTH,
                generator = [torch.Generator().manual_seed(i) for i in random_seeds]
                ).images
    return images

# Create the Gradio interface with blocks
with gr.Blocks() as iface:
    gr.Markdown("""# OpenJourney Text to Image Generator""")
    gr.Markdown("""This is a simple Gradio app that uses the OpenJourney model to generate images based on a prompt.""")
    with gr.Row(equal_height=True):
        with gr.Column():
            #Input for prompt
            prompt_input = gr.Textbox(lines=5, 
                                    placeholder='Enter a prompt for ...')
            #Slider for the number of inference steps
            inference_steps = gr.Slider(minimum=1, 
                                        maximum=100, 
                                        value=20, 
                                        step=1, 
                                        label="# of Inference Steps", 
                                        info="Between 1 and 100")
            #VAE choice
            model_VAE = gr.Radio(["sd-vae-ft-mse", "sd-vae-ft-ema"], 
                                    value="sd-vae-ft-mse", 
                                    label="VAE Method", 
                                    info="Which VAE method to use?")
        #Gallery for generated images
        outputs = gr.Gallery(label="Generated images", 
                                show_label=False, 
                                elem_id="gallery",
                                columns=[3], 
                                rows=[1], 
                                object_fit="contain", 
                                height="auto")
    with gr.Row(equal_height=True):
        with gr.Column():
            generate_btn = gr.Button("Generate")
            generate_btn.click(fn=generate_image,
                                inputs=[prompt_input, inference_steps, model_VAE],
                                outputs=outputs)

# Launch the application
iface.launch(server_name="0.0.0.0",server_port=7805)
