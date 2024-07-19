import mlflow
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='write2shivamgithub', repo_name='mlflow-dagshub-demo', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/write2shivamgithub/mlflow-dagshub-demo.mlflow")

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameters for the RandomForestClassifier
max_depth = 5
n_estimators = 100

# Apply mlflow
mlflow.set_experiment('iris_dt')
with mlflow.start_run(run_name="CampusX_exp"):
    rf = RandomForestClassifier(max_depth=max_depth)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    # Create a confusion matrix
    cm =confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels= iris.target_names, yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # Save the plot as artifacts
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')

    #log code
    mlflow.log_artifact(__file__)

    #log model
    mlflow.sklearn.log_model(rf,'random_forest')

    #log tag
    mlflow.set_tag('author',"kanak")
    mlflow.set_tag('model','random_forest')
    
    print('Accuracy:', accuracy)




