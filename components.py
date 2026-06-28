from shiny import ui
from data_loader import df
import pandas as pd
import matplotlib.pyplot as plt
from shiny import render

def create_header():

    return ui.div(

        ui.h1("Dry Bean Classification Dashboard"),

        ui.p(
            "Comparison of Machine Learning Models for Dry Bean Classification"
        ),

        ui.hr()

    )

def create_dataset():

    # Class Distribution
    class_count = df["Class"].value_counts().reset_index()
    class_count.columns = ["Class", "Count"]

    # Missing Values
    missing = df.isnull().sum().reset_index()
    missing.columns = ["Feature", "Missing Values"]

    return ui.div(

        ui.h2("Dataset Overview"),

        ui.p(
            "This page presents the main characteristics of the Dry Bean Dataset."
        ),

        ui.br(),

        # Statistics

        ui.layout_columns(

            ui.value_box(
                str(df.shape[0]),
                "Samples"
            ),

            ui.value_box(
                str(df.shape[1] - 1),
                "Features"
            ),

            ui.value_box(
                str(df["Class"].nunique()),
                "Classes"
            )

        ),

        ui.br(),

        #First Records

        ui.card(

            ui.card_header("First 10 Records"),

            ui.HTML(
                df.head(10).to_html(
                    index=False,
                    classes="table table-striped table-hover"
                )
            )

        ),

        ui.br(),

        #Feature Names

        ui.card(

            ui.card_header("Feature Names"),

            ui.tags.ul(

                *[
                    ui.tags.li(col)
                    for col in df.columns
                ]

            )

        )

    )

def create_home():

    return ui.div(

        #معرفی پروژه 

        ui.card(

            ui.card_header("Project Overview"),

            ui.p(
                """
                This dashboard presents a complete machine learning pipeline for Dry Bean Classification.
                The project includes data preprocessing, hyperparameter tuning, comparison of nine machine
                learning models,and comprehensive performance evaluation.
                """
            )

        ),

        ui.br(),

        # کارت‌های آماری 

        ui.layout_columns(

            ui.value_box(
                "13,611",
                "Samples"
            ),

            ui.value_box(
                "16",
                "Features"
            ),

            ui.value_box(
                "7",
                "Classes"
            ),

            ui.value_box(
                "9",
                "Models"
            )

        ),

        ui.br(),

        #  دو ستون 

        ui.layout_columns(

            # ستون اول
            ui.card(

                ui.card_header("Dataset Information"),

                ui.tags.ul(

                    ui.tags.li("Dataset : Dry Bean Dataset"),

                    ui.tags.li("Source : UCI Machine Learning Repository"),

                    ui.tags.li("Samples : 13,611"),

                    ui.tags.li("Features : 16"),

                    ui.tags.li("Classes : 7"),

                    ui.tags.li("Classification : Multi-Class")

                )

            ),

            # ستون دوم
            ui.card(

                ui.card_header("Project Objectives"),

                ui.tags.ul(

                    ui.tags.li("Data Preprocessing"),

                    ui.tags.li("Feature Scaling"),

                    ui.tags.li("SMOTE Oversampling"),

                    ui.tags.li("Hyperparameter Tuning"),

                    ui.tags.li("Comparison of 9 Models"),

                    ui.tags.li("Performance Evaluation"),

                    ui.tags.li("Final Prediction")

                )

            )

        ),

        ui.br(),

        #  Workflow 

        ui.card(

            ui.card_header("Machine Learning Workflow"),

            ui.markdown("""

Dataset

↓

Preprocessing

↓

Feature Engineering

↓

StandardScaler

↓

SMOTE

↓

Model Training

↓

Model Evaluation

""")

        )

    )
def create_eda():

    return ui.div(

        ui.h2("Exploratory Data Analysis"),

        ui.p(
            "Visualization and statistical exploration of the Dry Bean Dataset."
        ),

        ui.br(),

        ui.card(

            ui.card_header("Class Distribution"),

            ui.output_plot("class_distribution")

        )

    )
def create_eda():

    return ui.div(

        ui.h2("Exploratory Data Analysis"),

        ui.p(
            "Exploratory Data Analysis of the Dry Bean Dataset"
        ),

        ui.br(),

        ui.card(

            ui.card_header("Class Distribution"),

            ui.output_plot("class_distribution", height="700px")

        ),

        ui.br(),

        ui.card(

            ui.card_header("Correlation Heatmap"),

            ui.output_plot("correlation_heatmap", height="1100px")

        ),

        ui.br(),

        ui.card(

            ui.card_header("Boxplots"),

            ui.output_plot("boxplots", height="1400px")

        ),

        ui.br(),

        ui.card(

            ui.card_header("Pairplot"),

            ui.output_plot("pairplot", height="1200px")

        )

    )

def create_statistics():

    return ui.div(

        ui.h2("Statistical Analysis"),

        ui.p(
            "Statistical analysis of the Dry Bean dataset."
        ),

        ui.br(),

        ui.card(

            ui.card_header("Descriptive Statistics"),

            ui.output_ui("describe_table")

        ),

        ui.br(),

        ui.card(

            ui.card_header("Statistics by Class"),

            ui.output_ui("stats_by_class_table")

        ),

        ui.br(),

        ui.card(

            ui.card_header("Coefficient of Variation (%)"),

            ui.output_ui("cv_table")

        )

    )
def create_preprocessing():

    return ui.div(

        ui.h2("Preprocessing"),

        ui.p("Data preprocessing steps applied before model training."),

        ui.br(),

        ui.layout_column_wrap(

            ui.card(
                ui.card_header("Missing Values"),
                ui.output_ui("missing_values"),
                full_screen=True
            ),

            ui.card(
                ui.card_header("Outlier Removal"),
                ui.output_ui("outlier_info"),
                full_screen=True
            ),

            fill=False
        ),

        ui.br(),

        ui.layout_column_wrap(

            ui.card(
                ui.card_header("Feature Engineering"),
                ui.output_ui("feature_engineering"),
                full_screen=True
            ),

            ui.card(
                ui.card_header("Label Encoding"),
                ui.output_ui("label_encoding"),
                full_screen=True
            ),

            fill=False
        ),

        ui.br(),

        ui.layout_column_wrap(

            ui.card(
                ui.card_header("Train / Test Split"),
                ui.output_ui("train_test_info"),
                full_screen=True
            ),

            ui.card(
                ui.card_header("Feature Scaling"),
                ui.output_ui("scaling_info"),
                full_screen=True
            ),

            fill=False
        ),

        ui.br(),

        ui.layout_column_wrap(

            ui.card(
                ui.card_header("SMOTE"),
                ui.output_ui("smote_info"),
                full_screen=True
            ),

            fill=False
        )

    )

def create_models():

    return ui.div(

        ui.h2("Model Development"),

        ui.p(
            "Training and comparison of machine learning models."
        ),

        ui.br(),

        # Hyperparameter Tuning

        ui.card(

            ui.card_header("Hyperparameter Tuning"),

            ui.output_ui("hyperparameter_info")

        ),

        ui.br(),

        # Results Table

        ui.card(

            ui.card_header("Model Comparison"),

            ui.output_ui("results_table")

        ),

        ui.br(),

        # Best Model

        ui.card(

            ui.card_header("Best Model"),

            ui.output_ui("best_model")

        )

    )

def create_evaluation():
    return ui.div(

        ui.h2(
            "Model Evaluation & Insights",
            style="text-align:center; margin-bottom:30px;"
        ),

        
        # Model Comparison
        
        ui.card(
            ui.card_header("Final Model Comparison"),
            ui.img(
                src="Final_Model_Comparison.png",
                style="""
                width:100%;
                display:block;
                margin:auto;
                """
            ),
            full_screen=True
        ),

        ui.br(),

        
        # Confusion Matrix
        
        ui.card(
            ui.card_header("Confusion Matrix"),
            ui.img(
                src="Confusion_Matrix.png",
                style="""
                width:100%;
                display:block;
                margin:auto;
                """
            ),
            full_screen=True
        ),

        ui.br(),

        
        # ROC Curve
        
        ui.card(
            ui.card_header("ROC Curve"),
            ui.img(
                src="ROC_Curve.png",
                style="""
                width:100%;
                display:block;
                margin:auto;
                """
            ),
            full_screen=True
        ),

        ui.br(),

        
        # Learning Curve
        
        ui.card(
            ui.card_header("Learning Curve"),
            ui.img(
                src="Learning_Curve.png",
                style="""
                width:100%;
                display:block;
                margin:auto;
                """
            ),
            full_screen=True
        ),

        ui.br(),

        
        # Feature Importance
        
        ui.navset_card_tab(

            ui.nav_panel(
                "Best Model (SHAP)",
                ui.img(
                    src="Computing_SHAP_for_best_model.png",
                    style="width:100%; display:block; margin:auto;"
                )
            ),

            ui.nav_panel(
                "Best Model Importance",
                ui.img(
                    src="Feature_Importance_for_best_model.png",
                    style="width:100%; display:block; margin:auto;"
                )
            ),

            ui.nav_panel(
                "Random Forest",
                ui.img(
                    src="Feature_Importance_for_Random_Forest.png",
                    style="width:100%; display:block; margin:auto;"
                )
            ),

            ui.nav_panel(
                "Logistic Regression",
                ui.img(
                    src="Feature_Importance_for_Logistic_Regression.png",
                    style="width:100%; display:block; margin:auto;"
                )
            ),

            title="Feature Importance Analysis"

        )

    )

