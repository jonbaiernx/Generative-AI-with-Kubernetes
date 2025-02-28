import gradio as gr
import torchaudio
from audiocraft.models import AudioGen, MusicGen
from audiocraft.data.audio import audio_write

# Pre-download models
audiogen_model = AudioGen.get_pretrained('facebook/audiogen-medium')
musicgen_model = MusicGen.get_pretrained('facebook/musicgen-medium')

def generate_audio(prompt, model_type="sound", sound_length=2):
    """ Generate text using the AudioGen and MusicGen models """
    
    if model_type=="sound":
        model = audiogen_model
    else:
        model = musicgen_model
    
    #Set length and configuration
    model.set_generation_params(use_sampling=True, top_k=250, top_p=0.0, temperature=1.0, duration=sound_length, cfg_coef=3.0)
    #Generate audio
    wav = model.generate([prompt], progress=True)  
    audio_write(f'output', wav[0].cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
    return "output.wav"


# Create the Gradio interface with blocks
with gr.Blocks() as audiogen_iface:
    gr.Markdown("""# AudioGen Text-to-Audio Generator""")
    gr.Markdown("""This is a simple Gradio app that uses the AudioGen and MusicGen models to generate sounds or music based on a prompt.""")
    with gr.Row(equal_height=True):
        with gr.Column():
            #Input for prompt
            prompt_input = gr.Textbox(lines=5, 
                                    placeholder='Enter a prompt for ...')
            #Slider for the duration of sound
            sound_length = gr.Slider(minimum=1, 
                                        maximum=60, 
                                        value=2, 
                                        step=1, 
                                        label="Length of sounds (seconds)", 
                                        info="Between 1 and 60")
            #Model choice, sound or music
            model_type = gr.Radio([("Sound Effects", "sound"), ("Music","music")], 
                                    value="sound", 
                                    label="Model Type", 
                                    info="Generate sound or music?")
        #Gallery for generated images
        outputs = gr.Audio(label= "Generated Audio",
                           interactive=False)
    with gr.Row(equal_height=True):
        with gr.Column():
            generate_btn = gr.Button("Generate")
            generate_btn.click(fn=generate_audio,
                                inputs=[prompt_input, model_type, sound_length],
                                outputs=outputs)

# Launch the application
audiogen_iface.launch(server_name="0.0.0.0",server_port=7806)
