import os
import transformers
import pandas as pd
import mlflow
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    GenerationConfig,
)

#Configuration
experiment_name = "Transformers Model Comparison"
task = "text-generation"
tracking_server_ip = os.environ['MLFLOW_SERVER_IP']
tracking_server_port = os.environ['MLFLOW_SERVER_PORT']

#Define models for comparison
gpt2_pipeline = transformers.pipeline(
    task=task,
    model="gpt2"
)
#Compare multiple models by defining a pipeline object above and adding a key-value pair here.
pipelines = {
    'GPT2' : gpt2_pipeline
}

#Input examples to infer model signature
input_example = ["The color of a tree is", "The capital of Germany is", "The biggest ocean is"]

#Set MLFlow Server and experiment name
mlflow.set_tracking_uri(uri="http://"+tracking_server_ip+":"+tracking_server_port)
mlflow.set_experiment(experiment_name)

#Set evaluation data
eval_data = pd.DataFrame(
    {
        "inputs": [            
            "The color of the sky is",
            "The color of the moon is",
        ],
    }
)

# Define the parameters (and their defaults) for optional overrides at inference time.
parameters_sets = [
                {"max_length": 128, "do_sample": True, "temperature": 0.1},
                {"max_length": 128, "do_sample": True, "temperature": 0.4},
                {"max_length": 128, "do_sample": True, "temperature": 0.9}
]

#Loop through models and parameter sets and evaluate each combo
for pipeline_name, pipeline_object in pipelines.items():
    for parameters in parameters_sets:
        # Generate the signature for the model for validation
        signature = mlflow.models.infer_signature(
            input_example,
            mlflow.transformers.generate_signature_output(pipeline_object, input_example),
            parameters,
        )

        #Create run name with the current model and current temperature
        run_name = pipeline_name + '(temp:'+str(parameters.get('temperature', '--')) + ')'

        with mlflow.start_run(run_name = run_name):
            #Log parameters used for each run
            for key, val  in parameters.items():
                mlflow.log_param(key, val)
            #Log each model run
            model_info = mlflow.transformers.log_model(
                transformers_model=pipeline_object,
                artifact_path="text_generation",
                input_example=input_example,
                signature=signature,
            )
            #Evaluate using the defined evaluation prompts
            results = mlflow.evaluate(
                model=model_info.model_uri,
                model_type='text',
                data=eval_data,
            )

