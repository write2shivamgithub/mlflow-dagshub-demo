import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='write2shivamgithub', repo_name='mlflow-dagshub-demo', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/write2shivamgithub/mlflow-dagshub-demo.mlflow")

# Load the iris dataset
iris = pd.read_csv("https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv")

X = iris.iloc[:, 0:-1]
y = iris.iloc[:, -1]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameters for the RandomForestClassifier
max_depth = 15
n_estimators = 150

# Apply mlflow

mlflow.autolog()

mlflow.set_experiment('autolog')
with mlflow.start_run(run_name="Shankaraacharya"):
    rf = RandomForestClassifier(max_depth=max_depth)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)



    # Create a confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['setosa', 'versicolor', 'virginica'], yticklabels=['setosa', 'versicolor', 'virginica'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # Save the plot as artifacts
    plt.savefig('confusion_matrix.png')

    # Log code
    mlflow.log_artifact(__file__)

    # Log model
    mlflow.sklearn.log_model(rf, 'random_forest')

    # Log tags
    mlflow.set_tag('author', "mohini")
    mlflow.set_tag('model', 'random_forest')
  
    print('Accuracy:', accuracy)
