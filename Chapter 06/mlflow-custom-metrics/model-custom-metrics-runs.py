import os
import transformers
import pandas as pd
import mlflow
from mlflow.models import infer_signature, make_metric
from mlflow.metrics import MetricValue
import numpy as np
import torch
import evaluate
import textstat
from transformers import pipeline

#Configuration
experiment_name = "Model Comparison - Custom Metrics"
task = "text-generation"
model_names = ["gpt2", "MBZUAI/LaMini-Flan-T5-77M"]


#Grab Tracking server information from Environment vars
tracking_server_ip = os.environ['MLFLOW_SERVER_IP']
tracking_server_port = os.environ['MLFLOW_SERVER_PORT']

#Set MLFlow Server and experiment name
mlflow.set_tracking_uri(uri="http://"+tracking_server_ip+":"+tracking_server_port)
mlflow.set_experiment(experiment_name)

# Define the parameters (and their defaults) for optional overrides at inference time.
parameters_sets = [
                    {"max_length": 20, "do_sample": True, "temperature": 0.1},
                    {"max_length": 20, "do_sample": True, "temperature": 0.9},
                    {"max_length": 50, "do_sample": True, "temperature": 0.1},
                    {"max_length": 50, "do_sample": True, "temperature": 0.9}                    
                ]

def evaluate_positive_keywords(predictions, metrics):
    scores = []    
    positive_words = ["good", "excellent", "great", "five", "recommend","amazing","awesome","best"]
    for a_prediction in predictions:
        normalized_prediction = a_prediction.lower()
        words = normalized_prediction.split()
        positive_word_count = 0
        positive_word_count = sum(words.count(word) for word in positive_words)
        scores.append(positive_word_count)    
    return MetricValue(
        scores=list(scores),
        aggregate_results={"mean": np.mean(scores), "sum": np.sum(scores)},
    )

keywordMetric = make_metric(eval_fn=evaluate_positive_keywords, greater_is_better=True, name="keywordMetric")

eval_data = pd.DataFrame(
    {
        "inputs": [
            "Apples taste good and",
            "I give the driver five stars and "
        ]
    }
)

#Loop through models and parameter sets and run evaluations with each configuration
for model_name in model_names:
    for parameters in parameters_sets:
        a_pipeline = transformers.pipeline(
            task=task,
            model=model_name,
            do_sample=parameters.get("do_sample", False),
            max_lenth=parameters.get("max_length", 20),
            temperature=parameters.get("temperature", 0.9)
        )

        # Generate the signature for the model for validation
        signature = mlflow.models.infer_signature(
            model_input="What is 1 x 8?",
            model_output="8",
            params=parameters,
        )

        #Create run name with the current model and current temperature
        run_name = model_name + ' (len:' + str(parameters.get("max_length", "NA")) + ', temp:' + str(parameters.get("temperature","NA")) + ')'

        with mlflow.start_run(run_name = run_name):
            #Log parameters used for each run
            for key, val in parameters.items():
                mlflow.log_param(key, val)
            #Log each model run
            model_info = mlflow.transformers.log_model(
                transformers_model=a_pipeline,
                artifact_path="text_generation",
                signature=signature,
                registered_model_name=model_name,
            )
            #Evaluate using the defined evaluation prompts
            results = mlflow.evaluate(
                model=model_info.model_uri,
                evaluators=["default"],
                model_type="text",
                model_config=parameters,
                data=eval_data,
                extra_metrics=[keywordMetric]
            )
