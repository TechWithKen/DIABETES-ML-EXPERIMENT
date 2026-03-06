import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


diabetes_dataset = pd.read_csv("./diabetes_experiment_dataset.csv")
diabetes_dataset.corr()["Outcome"].sort_values(ascending=False)

diabetes_label = diabetes_dataset["Outcome"]
diabetes_dataset.drop(columns=["Outcome"], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(diabetes_dataset, diabetes_label, test_size=0.2, random_state=42, stratify=diabetes_label)


scale_features = ColumnTransformer(transformers=[
    ("scaling", StandardScaler(), X_train.columns)
])

randomForestModel = RandomForestClassifier(random_state=42)
xgboostModel = XGBClassifier()
kneighboursModel = KNeighborsClassifier()
svcModel = SVC(kernel="rbf")
logisticModel = LogisticRegression(max_iter=5000)

model_pipeline = Pipeline(steps=[
    ("scale", scale_features),
    ("model", logisticModel)
])

my_folds = StratifiedKFold(
    n_splits=5, 
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model_pipeline,
    X_train,
    y_train,
    scoring="recall",
    cv=my_folds,
    n_jobs=-1,
)

print(scores.mean())
print(scores.std())

param_randomforest_grid = [
    {
        'model': [RandomForestClassifier(random_state=42, class_weight="balanced")],
        'model__n_estimators': [100, 200, 500],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__max_features': ['sqrt', 'log2', None]
    },
]

param_xgbclassifier_grid = [
 {
        'model': [XGBClassifier()],
        'model__n_estimators': [100, 200, 500],
        'model__max_depth': [3, 5, 7],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__subsample': [0.7, 0.8, 1],
        'model__colsample_bytree': [0.7, 0.8, 1]
    },
]

param_knearest_grid = [
    {
        'model': [KNeighborsClassifier()],
        'model__n_neighbors': [3, 5, 7, 9],
        'model__weights': ['uniform', 'distance'],
        'model__p': [1, 2]
    },

]

param_svc_grid = [

    {
        'model': [SVC(kernel='rbf')],
        'model__C': [0.1, 1, 10, 100],
        'model__gamma': ['scale', 'auto', 0.01, 0.1, 1]
    },
]

param_decision_grid = [
    {
        'model': [DecisionTreeClassifier(random_state=42)],
        'model__max_depth': [None, 5, 10, 20],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__max_features': ['sqrt', 'log2', None]
    },
]

param_logisticReg_grid = [
    {
        'model': [LogisticRegression(max_iter=5000)],
        'model__C': [0.01, 0.1, 1, 10],
        'model__penalty': ['l2'],
        'model__solver': ['lbfgs', 'saga']
    }

]


grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_randomforest_grid,
    scoring="recall",
    n_jobs=-1, 
    cv=my_folds,
    return_train_score=True
)

grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)