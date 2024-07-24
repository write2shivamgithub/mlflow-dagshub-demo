from mlflow.tracking import MlflowClient
import mlflow

# Initialize the Mlflow Client 
client = MlflowClient()

# Replace with run_id of the run where the model was logged
run_id = "c9b279a9b25f4ecd948c0833520e172e"

# Replace with the path to be logged model within the run
model_path = "mlflow-artifacts:/0d694a8e6d1a47c58a443f95cede55f7/c9b279a9b25f4ecd948c0833520e172e/artifacts/random forest"

# Construct the model URI
model_uri = f"runs:/{run_id}/{model_path}"

# Register the model in the model registry
model_name = "diabetes-rf"
result = mlflow.register_model(model_uri, model_name)

import time
time.sleep(5)

# Add a description to the registered model version 
client.update_model_version(
    name = model_name,
    version = result.version,
    description = "This is a RandomForest model trained to predict diabetes outcome based on PIMA Indian diabetes"
)

# Add tags to the registered model version 
client.set_model_version_tag(
    name = model_name,
    version = result.version,
    key = "experiment",
    value = "diabetes prediction"
)

client.set_model_version_tag(
    name = model_name,
    version = result.version,
    key = "day",
    value = "wednesday"
)