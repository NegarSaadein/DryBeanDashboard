import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score
)

from preprocessing import (
    X_train_sc,
    X_test_sc,
    X_train_sm,
    y_train,
    y_test,
    y_train_sm,
    best_k,
    best_kernel
)

# Define Machine Learning Models

def get_models(best_k, best_kernel):

    return {

        "Logistic Regression": LogisticRegression(
            max_iter=5000,
            C=1.0,
            solver="lbfgs",
            random_state=42
        ),

        f"KNN (k={best_k})": KNeighborsClassifier(
            n_neighbors=best_k,
            weights="distance",
            metric="euclidean",
            n_jobs=-1
        ),

        "Naive Bayes": GaussianNB(),

        f"SVM ({best_kernel})": SVC(
            kernel=best_kernel,
            C=50,
            gamma="scale",
            probability=True,
            cache_size=2000,
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            colsample_bytree=0.8,
            subsample=0.8,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0
        ),

        "LightGBM": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=127,
            colsample_bytree=0.8,
            subsample=0.8,
            verbose=-1,
            random_state=42,
            n_jobs=-1
        ),

        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=3,
                min_samples_split=5
            ),
            n_estimators=300,
            learning_rate=0.1,
            random_state=42,
            algorithm="SAMME"
        )

    }

# Model Training & Evaluation

def run_evaluation(X_tr, y_tr, X_te, y_te, best_k, best_kernel):

    models = get_models(best_k, best_kernel)

    results = []

    for name, model in models.items():

        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_te)

        acc = accuracy_score(y_te, y_pred) * 100

        f1 = f1_score(
            y_te,
            y_pred,
            average="macro"
        ) * 100

        results.append({

            "Model": name,

            "Accuracy": round(acc, 2),

            "F1": round(f1, 2)

        })

    return results

# Train Models

res_no = run_evaluation(

    X_train_sc,
    y_train,

    X_test_sc,
    y_test,

    best_k,
    best_kernel

)

res_sm = run_evaluation(

    X_train_sm,
    y_train_sm,

    X_test_sc,
    y_test,

    best_k,
    best_kernel

)

# Create Final Results Table

df_no = pd.DataFrame(res_no).rename(

    columns={

        "Accuracy": "Acc_NoSMOTE",

        "F1": "F1_NoSMOTE"

    }

)

df_sm = pd.DataFrame(res_sm).rename(

    columns={

        "Accuracy": "Acc_SMOTE",

        "F1": "F1_SMOTE"

    }

)

results_df = df_no.merge(

    df_sm,

    on="Model"

)

results_df["Best_Accuracy"] = results_df[

    [

        "Acc_NoSMOTE",

        "Acc_SMOTE"

    ]

].max(axis=1)

results_df = results_df.sort_values(

    "Best_Accuracy",

    ascending=False

).reset_index(drop=True)

# Best Model

best_row = results_df.loc[
    results_df["Best_Accuracy"].idxmax()
]

