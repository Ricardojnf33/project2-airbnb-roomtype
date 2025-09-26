"""Projeto 2 – Classificação do tipo de acomodação em anúncios do Airbnb NYC.

Este script implementa as etapas especificadas no Projeto 2 da disciplina de
Introdução à Ciência de Dados. A tarefa consiste em utilizar o conjunto de
dados `AB_NYC_2019.csv` (reutilizado do Projeto 1) para treinar diferentes
modelos de classificação. O objetivo é prever o tipo de quarto (`room_type`)
com base em variáveis numéricas e categóricas do anúncio. O script realiza
pré‑processamento, divisão em conjuntos de treinamento/validação/teste,
seleção de hiper‑parâmetros, rastreamento dos experimentos com um stub de
MLFlow e avaliação do desempenho final.

Principais etapas:

1. Carregamento e tratamento dos dados.
2. Definição de variáveis preditoras e alvo (room_type).
3. Divisão em conjuntos de treinamento, validação e teste com estratificação.
4. Construção de pipelines de pré‑processamento (normalização/one‑hot encoding).
5. Seleção de hiper‑parâmetros para quatro algoritmos de classificação
   (Regressão Logística, RandomForestClassifier, SVC e KNeighborsClassifier).
6. Avaliação de cada modelo na validação e seleção do melhor em cada família.
7. Registro dos resultados em arquivos e com o stub de MLFlow.
8. Avaliação final no conjunto de teste, relatório de classificação e
   visualização da matriz de confusão para o melhor modelo global.

Requisitos: pandas, numpy, scikit‑learn, matplotlib, seaborn. O pacote
mlflow não está disponível no ambiente; em vez disso, utilizamos o módulo
``mlflow_stub`` para criar registros simples de experimentos em
``mlruns/``. Caso deseje usar mlflow real, instale-o e substitua
``mlflow_stub`` por ``mlflow``.
"""

import json
import os
import uuid
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Import the mlflow stub for logging
import mlflow_stub as mlflow


def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    """Load the Airbnb dataset and perform preprocessing.

    The function handles missing values, converts dates into numerical
    differences, removes extreme outliers in the `price` column using the IQR
    method and returns a cleaned DataFrame ready for modeling.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Preprocessed pandas DataFrame.
    """
    df = pd.read_csv(csv_path)

    # Convert last_review to datetime and compute days since last review
    df["last_review"] = pd.to_datetime(df["last_review"])
    max_date = df["last_review"].max()
    df["days_since_last_review"] = (max_date - df["last_review"]).dt.days
    df["days_since_last_review"] = df["days_since_last_review"].fillna(0)

    # Fill missing numeric values
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
    df["days_since_last_review"] = df["days_since_last_review"].fillna(0)

    # Remove extreme price outliers using the IQR method
    q1 = df["price"].quantile(0.25)
    q3 = df["price"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df = df[(df["price"] >= lower_bound) & (df["price"] <= upper_bound)].copy()

    # Drop columns that will not be used as predictors
    df = df.drop(
        columns=[
            "id",
            "name",
            "host_id",
            "host_name",
            "last_review",
        ]
    )

    # Drop rows with missing values in essential categorical variables
    df = df.dropna(subset=["neighbourhood_group", "room_type"])

    return df


def get_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    """Construct a ColumnTransformer for preprocessing.

    Numeric features are standardized and categorical features are one‑hot encoded.

    Args:
        numeric_features: List of column names for numeric features.
        categorical_features: List of column names for categorical features.

    Returns:
        A configured ColumnTransformer.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def evaluate_model(model: Pipeline, X_val, y_val) -> float:
    """Compute classification accuracy on the validation set."""
    preds = model.predict(X_val)
    return accuracy_score(y_val, preds)


def run_experiments(df: pd.DataFrame, target: str = "room_type") -> None:
    """Run experiments for several classifiers and log results.

    Args:
        df: Preprocessed DataFrame.
        target: Name of the target column to predict.
    """

    # Optionally downsample the dataset for computational efficiency.
    # The Airbnb dataset contains tens of thousands of rows; exhaustive
    # hyper‑parameter searches on the full dataset can be slow.  Here we
    # subsample up to 15 000 examples while preserving class proportions.
    if len(df) > 15000:
        # Stratified sampling: determine number per class
        samples_per_class = int(15000 / df[target].nunique())
        df_sampled = (
            df.groupby(target, group_keys=False)
            .apply(lambda x: x.sample(min(len(x), samples_per_class), random_state=42))
        )
    else:
        df_sampled = df

    # Separate features and target
    X = df_sampled.drop(columns=[target])
    y = df_sampled[target]

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    # Split into train, validation and test (60% train, 20% val, 20% test)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,  # 0.25 * 0.8 = 0.2 of the full dataset
        random_state=42,
        stratify=y_temp,
    )

    # Prepare preprocessor
    preprocessor = get_preprocessor(numeric_features, categorical_features)

    # Define models and hyperparameter grids (max 3 hyperparameters per algorithm)
    algorithms = {
        # Simplified grids for efficiency. Each grid contains at most two
        # hyper‑parameter values per parameter, ensuring no more than
        # ∼2 combinations per algorithm.
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000, multi_class="auto"),
            "param_grid": {
                # Single combination: baseline logistic regression
                "model__C": [1.0],
                "model__penalty": ["l2"],
                "model__solver": ["lbfgs"],
            },
        },
        "RandomForestClassifier": {
            "model": RandomForestClassifier(random_state=42),
            "param_grid": {
                "model__n_estimators": [100],
                "model__max_depth": [None, 20],
                "model__min_samples_split": [2],
            },
        },
        "SVC": {
            "model": SVC(probability=True),
            "param_grid": {
                "model__C": [1.0],
                "model__kernel": ["rbf", "linear"],
                "model__gamma": ["scale"],
            },
        },
        "KNeighborsClassifier": {
            "model": KNeighborsClassifier(),
            "param_grid": {
                "model__n_neighbors": [5, 10],
                "model__weights": ["uniform"],
                "model__p": [2],
            },
        },
    }

    # Prepare result storage
    os.makedirs("project2/results", exist_ok=True)
    results = []
    mlflow.set_experiment("Projeto2_RoomTypeClassification")

    best_overall = None
    best_overall_score = -np.inf

    # Iterate over algorithms
    for algo_name, config in algorithms.items():
        model_base = config["model"]
        param_grid = config["param_grid"]

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model_base),
            ]
        )

        best_score = -np.inf
        best_params = None
        best_model = None

        # Iterate over parameter combinations (limiting to <= 50 combos for efficiency)
        combos = list(ParameterGrid(param_grid))
        for params in combos:
            # Set parameters
            pipeline.set_params(**params)
            pipeline.fit(X_train, y_train)
            score = evaluate_model(pipeline, X_val, y_val)
            if score > best_score:
                best_score = score
                best_params = params.copy()
                best_model = Pipeline(
                    steps=[("preprocessor", preprocessor), ("model", model_base.__class__(**{
                        k.split("__")[1]: v for k, v in params.items() if k.startswith("model__")
                    }))]
                )
                # Fit best_model on full training data for evaluation
                best_model.fit(X_train, y_train)

        # Evaluate best model on test set
        test_acc = evaluate_model(best_model, X_test, y_test)
        results.append(
            {
                "algorithm": algo_name,
                "best_params": best_params,
                "val_accuracy": round(best_score, 4),
                "test_accuracy": round(test_acc, 4),
            }
        )

        # Check if this is best overall
        if test_acc > best_overall_score:
            best_overall_score = test_acc
            best_overall = {
                "name": algo_name,
                "model": best_model,
                "params": best_params,
                "val_accuracy": best_score,
                "test_accuracy": test_acc,
            }

        # Log results with mlflow stub
        with mlflow.start_run(run_name=algo_name) as run:
            # Log hyperparameters (flatten model__ prefix)
            for key, value in best_params.items():
                ml_key = key.replace("model__", "")
                run.log_param(ml_key, value)
            run.log_metric("val_accuracy", best_score)
            run.log_metric("test_accuracy", test_acc)

    # Save summary results to JSON and CSV
    summary_csv = os.path.join("project2", "results", "model_results.csv")
    summary_json = os.path.join("project2", "results", "model_results.json")
    pd.DataFrame(results).to_csv(summary_csv, index=False)
    with open(summary_json, "w") as fjson:
        json.dump(results, fjson, indent=2)

    # Diagnostics for the best overall model
    best_model = best_overall["model"]
    # Predict on test set
    y_pred_test = best_model.predict(X_test)
    report = classification_report(y_test, y_pred_test)

    report_path = os.path.join("project2", "results", "best_model_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Best Model: {best_overall['name']}\n")
        f.write(f"Parameters: {best_overall['params']}\n\n")
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test, labels=sorted(y.unique()))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()), ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix – {best_overall['name']}")
    fig.tight_layout()

    cm_fig_path = os.path.join("project2", "figures", "confusion_matrix.png")
    fig.savefig(cm_fig_path)
    plt.close(fig)

    # Save overall info as JSON
    overall_info_path = os.path.join("project2", "results", "best_model_overall.json")
    with open(overall_info_path, "w") as f:
        json.dump(
            {
                "best_algorithm": best_overall["name"],
                "parameters": best_overall["params"],
                "val_accuracy": best_overall["val_accuracy"],
                "test_accuracy": best_overall["test_accuracy"],
            },
            f,
            indent=2,
        )


def main() -> None:
    # Ensure result directories exist
    Path("project2/results").mkdir(parents=True, exist_ok=True)
    Path("project2/figures").mkdir(parents=True, exist_ok=True)

    df = load_and_preprocess("AB_NYC_2019.csv")
    run_experiments(df, target="room_type")


if __name__ == "__main__":
    main()