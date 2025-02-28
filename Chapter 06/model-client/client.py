import os
import json
import requests

model_inference_url="http://"+os.environ['MODEL_SERVER_HOST']+":"+os.environ['MODEL_SERVER_PORT']+"/v2/models/"+os.environ['MODEL_SERVER_NAME']+"/infer"
prompt_text = "What is the color of the sky?"
inference_request={
    "inputs": [
        {
          "name": "args",
          "shape": [1],
          "datatype": "BYTES",
          "data": prompt_text
        }
    ]
}
infer_response = requests.post(model_inference_url, json=inference_request).json()

print("# " + os.environ['MODEL_SERVER_NAME'] + " Inference Output \n")
print("\n")
print("**Prompt:** " + prompt_text + " \n")
print("\n")
an_output=infer_response.get('outputs')[0]
a_response=an_output.get('data')[0]
print("**Response:** " + json.loads(a_response).get('generated_text') + "\n")
