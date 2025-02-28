import os
import gradio as gr
import pandas as pd
import chromadb
from chromadb import Settings

#Demo descriptions from various sources
artwork_descriptions = [
    {"id":"0001",
    "source":"Gallery 105",
    "description":"The painting captures the essence of tranquility with its vast expanse of clear blue sky, stretching infinitely above."
    },
    {"id":"0002",
    "source":"Gallery 105",
    "description": "A sapphire gemstone with a rich, deep blue hue is reminiscent of the ocean's heart, where light seldom reaches."
    }, 
    {"id":"0003",
    "source":"Gallery 105",
    "description": "The blue morpho butterfly, with its wings wide open, is a living tapestry of vibrant azure."
    }, 
    {"id":"0004",
    "source":"Gallery 105",
    "description": "The brilliance of the blue is achieved through a meticulous layering of paint."
    }, 
    {"id":"0005",
    "source":"Gallery 105",
    "description": "A classic vintage car, painted in a glossy navy blue, sits proudly on display." 
    },
    {"id":"0006",
    "source":"Gallery 105",
    "description": "A cobalt blue glass vase stands prominently on a sunlit windowsill."
    }, 
    {"id":"0007",
    "source":"CityTown Museum of Art",
    "description": "Soft, wispy clouds sporadically dot the canvas, adding a sense of depth and lightness to the serene azure backdrop."
    },
    {"id":"0008",
    "source":"CityTown Museum of Art",
    "description": "The car's curves and lines are accentuated by the rich blue paint."
    }, 
    {"id":"0009",
    "source":"CityTown Museum of Art",
    "description": "The sky, painted in varying shades of blue, transitions seamlessly from a light, almost ethereal hue at the horizon to a rich, deep cobalt as it arches overhead."
    }, 
    {"id":"0010",
    "source":"My Art Blog 108",
    "description": "The subtle gradients of color mimic the way light filters through the atmosphere, providing a realistic depiction of the sky at midday."
    }, 
    {"id":"0011",
    "source":"My Art Blog 108",
    "description": "The vase catches and refracts the sunlight, casting dancing patterns of blue light across the room."
    }         
]

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
    try:
        for art in artwork_descriptions:
            collection.add(
                documents=[art.get('description')], 
                metadatas=[{"source" : art.get('source')}], 
                ids=[art.get('id')]) 
    except ValueError:
        print("Error adding to collection")


def find_similar_text(phrase, num_return_sequences=5, include_blogs=True):
    """ Search artwork descriptions """
    df = pd.DataFrame({'ids':[0], 'descriptions':["No results"]})
    try:
        if include_blogs:
            results = collection.query(
                query_texts=[phrase],
                n_results=num_return_sequences)
        else:
            results = collection.query(
                query_texts=[phrase],
                n_results=num_return_sequences,
                where={"source": { "$ne": "My Art Blog 108"}})
        #Transform into a DataFrame for easy display
        df = pd.DataFrame(
            {'ids': results.get('ids')[0],
            'descriptions':  results.get('documents')[0]})
    except ValueError:
        print("Error fecthing collection")

    # Return results    
    return df

# Create a Gradio interface
with gr.Blocks() as vector_db_iface:
    gr.Markdown("""# ChromaDB Example""")
    gr.Markdown("""This is a simple Gradio app that allows for a similarity search against an example set of descriptions.""")
    with gr.Row(equal_height=True):
        with gr.Column():
            #Input phrase
            phrase_input = gr.Textbox(lines=2, 
                                    placeholder='Enter a phrase to search...')
            num_results =  gr.Number(value=5, visible=False)
            with_blogs = gr.Checkbox(label="Include blogs in search?", 
                                     info="Do you want to find descriptions for artwork from blogs as well?")
            examples = gr.Examples(
                [["Dreaming of blue skies", True], ["I am looking for a painting with clouds", False]],
                [phrase_input, with_blogs])
        with gr.Column():            
            #Gallery for generated images
            outputs = gr.Dataframe(
                headers=["ID", "Description"],
                datatype=["str", "str"]
            )
    with gr.Row(equal_height=True):
        with gr.Column():
            generate_btn = gr.Button("Generate")
            generate_btn.click(fn=find_similar_text,
                                inputs=[phrase_input, num_results, with_blogs],
                                outputs=outputs)

# Launch the application
vector_db_iface.launch(server_name="0.0.0.0",server_port=7800)
