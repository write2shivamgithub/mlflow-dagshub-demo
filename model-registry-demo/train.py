from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow

import dagshub
dagshub.init(repo_owner='write2shivamgithub', repo_name='mlflow-dagshub-demo', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/write2shivamgithub/mlflow-dagshub-demo.mlflow")

df = pd.read_csv("https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv")

# Splitting data into features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Splitting into training and testing test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the RandomForestClassifier model
rf = RandomForestClassifier(random_state=42)

# Defining the parameter grid for GridSearchCV
param_grid = {
    "n_estimators": [5, 50, 100, 150, 200],
    "max_depth": [None, 3, 6, 7]
}

# Applying GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

mlflow.set_experiment('Model_Registry')

with mlflow.start_run(description = "Best parameter trained on RF model",run_name= "grid_search") as parent:
    grid_search.fit(X_train, y_train)

    # log all the children
    for i in range(len(grid_search.cv_results_['params'])):

        print(i)
        with mlflow.start_run(nested=True) as child:

            mlflow.log_params(grid_search.cv_results_['params'][i])
            mlflow.log_metric('accuracy',grid_search.cv_results_['mean_test_score'][i])

    # Displaying the best parameter and the best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    #params
    mlflow.log_params(best_params)

    #metrics
    mlflow.log_metric('accuracy', best_score)

    #data
    train_df = X_train.copy()
    train_df['Outcome'] = y_train

    test_df = X_test.copy()
    test_df['Outcome'] = y_test


    train_df = mlflow.data.from_pandas(train_df)
    test_df = mlflow.data.from_pandas(test_df)

    mlflow.log_input(train_df, 'train')
    mlflow.log_input(test_df, 'validation')

    #source code
    mlflow.log_artifact(__file__)

    #signature
    signature = mlflow.models.infer_signature(X_train, grid_search.best_estimator_.predict(X_train))

    #model
    mlflow.sklearn.log_model(grid_search.best_estimator_,'random forest', signature=signature)

    #tag
    mlflow.set_tag('author','shankaraacharya')

    print(best_params)
    print(best_score)


