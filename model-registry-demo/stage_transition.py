from mlflow.tracking import MLflowClient

#Initialize the MLflow Client
client = MLflowClient()

# Define the model name and version 
model_name = "diabetes-rf"
model_version = 3  # Replace with the specific version number you want to transition

# Transition the model to a new stage 
new_stage = "Staging"                      # Possible values: None, Staging, Production, Archived
client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage=new_stage,
    archive_existing_versions=False
)
print(f"Model version {model_version} transitioned to {new_stage}")