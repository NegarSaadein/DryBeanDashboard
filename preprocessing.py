import numpy as np
import pandas as pd

from scipy import stats

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from imblearn.over_sampling import SMOTE

from data_loader import df

# Check Missing Values

X = df.drop("Class", axis=1)
y = df["Class"]

missing_count = X.isnull().sum().sum()

# Outlier Removal

def remove_outliers_zscore_per_class(df, target_col="Class", threshold=3.0):

    df_clean = pd.DataFrame()

    feature_cols = [c for c in df.columns if c != target_col]

    for class_name in df[target_col].unique():

        class_df = df[df[target_col] == class_name].copy()

        z_scores = np.abs(
            stats.zscore(
                class_df[feature_cols],
                nan_policy="omit"
            )
        )

        mask = (z_scores < threshold).all(axis=1)

        df_clean = pd.concat(
            [df_clean, class_df[mask]]
        )

    return df_clean.reset_index(drop=True)


df_clean = remove_outliers_zscore_per_class(
    df,
    threshold=3.0
)

# Feature Engineering

def engineer_features(X_df):

    X = X_df.copy()

    # Log Transform Features
    for col in [
        "Area",
        "Perimeter",
        "MajorAxisLength",
        "MinorAxisLength",
        "ConvexArea",
        "EquivDiameter"
    ]:
        X[f"{col}_log"] = np.log1p(X_df[col])

    # Interaction Features
    X["SF2_x_MinorAxis"] = X_df["ShapeFactor2"] * X_df["MinorAxisLength"]
    X["SF1_x_EquivD"] = X_df["ShapeFactor1"] * X_df["EquivDiameter"]
    X["SF2_x_SF1"] = X_df["ShapeFactor2"] * X_df["ShapeFactor1"]
    X["Round_x_SF3"] = X_df["Roundness"] * X_df["ShapeFactor3"]
    X["Compact_x_SF4"] = X_df["Compactness"] * X_df["ShapeFactor4"]
    X["ConvexArea_x_SF2"] = X_df["ConvexArea"] * X_df["ShapeFactor2"]
    X["SF3_x_SF4"] = X_df["ShapeFactor3"] * X_df["ShapeFactor4"]
    X["Round_x_Compact"] = X_df["Roundness"] * X_df["Compactness"]
    X["Eccen_x_Aspect"] = X_df["Eccentricity"] * X_df["AspectRatio"]
    X["Extent_x_Solid"] = X_df["Extent"] * X_df["Solidity"]

    # Ratio Features
    X["Major_Minor_ratio"] = (
        X_df["MajorAxisLength"] /
        (X_df["MinorAxisLength"] + 1e-8)
    )

    X["Area_Perim_ratio"] = (
        X_df["Area"] /
        (X_df["Perimeter"] ** 2 + 1e-8)
    )

    X["ConvArea_Area_ratio"] = (
        X_df["ConvexArea"] /
        (X_df["Area"] + 1e-8)
    )

    X["SF1_div_SF2"] = (
        X_df["ShapeFactor1"] /
        (X_df["ShapeFactor2"] + 1e-8)
    )

    return X


X_fe = engineer_features(
    df_clean.drop("Class", axis=1)
)

# Label Encoding

le = LabelEncoder()

y_enc = le.fit_transform(df_clean["Class"])

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X_fe,
    y_enc,
    test_size=0.2,
    stratify=y_enc,
    random_state=42
)

# Feature Scaling

scaler = StandardScaler()

X_train_sc = scaler.fit_transform(X_train)

X_test_sc = scaler.transform(X_test)

# SMOTE

smote = SMOTE(
    random_state=42,
    k_neighbors=5
)

X_train_sm, y_train_sm = smote.fit_resample(
    X_train_sc,
    y_train
)

# Find Best K for KNN

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

k_scores = {}

for k in range(1, 21):

    knn = KNeighborsClassifier(
        n_neighbors=k,
        weights="distance",
        n_jobs=-1
    )

    scores = []

    for tr_i, val_i in cv.split(X_train_sc, y_train):

        knn.fit(
            X_train_sc[tr_i],
            y_train[tr_i]
        )

        scores.append(
            accuracy_score(
                y_train[val_i],
                knn.predict(X_train_sc[val_i])
            )
        )

    k_scores[k] = np.mean(scores)

best_k = max(
    k_scores,
    key=k_scores.get
)

# Find Best Kernel for SVM

kernel_scores = {}

for kernel in [
    "linear",
    "rbf",
    "poly",
    "sigmoid"
]:

    svm = SVC(
        kernel=kernel,
        C=50,
        gamma="scale",
        probability=True,
        random_state=42
    )

    scores = []

    for tr_i, val_i in cv.split(X_train_sc, y_train):

        svm.fit(
            X_train_sc[tr_i],
            y_train[tr_i]
        )

        scores.append(
            accuracy_score(
                y_train[val_i],
                svm.predict(X_train_sc[val_i])
            )
        )

    kernel_scores[kernel] = np.mean(scores)

best_kernel = max(
    kernel_scores,
    key=kernel_scores.get
)