from shiny import ui
from shiny import render
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from preprocessing import (
    missing_count,
    df_clean,
    X_fe,
    X_train,
    X_test,
    X_train_sc,
    X_test_sc,
    X_train_sm,
    y_train,
    y_test,
    y_train_sm,
    best_k,
    best_kernel,
    le
)
from models import (
    results_df,
    best_row
)

from data_loader import df


def server(input, output, session):

    # Class Distribution
    @output
    @render.plot
    def class_distribution():

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        class_counts = df["Class"].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))

        # Pie Chart
        axes[0].pie(
            class_counts.values,
            labels=class_counts.index,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90
        )
        axes[0].set_title("Class Distribution")

        # Bar Chart
        axes[1].bar(
            class_counts.index,
            class_counts.values,
            color=colors
        )
        axes[1].set_title("Number of Samples per Class")
        axes[1].tick_params(axis="x", rotation=45)

        for i, v in enumerate(class_counts.values):
            axes[1].text(
                i,
                v + 30,
                str(v),
                ha="center",
                fontsize=9
            )

        plt.tight_layout()

        return fig


    # Correlation Heatmap
    @output
    @render.plot
    def correlation_heatmap():

        feature_cols = [c for c in df.columns if c != "Class"]

        fig, ax = plt.subplots(figsize=(18, 16))

        corr_matrix = df[feature_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(
            corr_matrix,
            annot=False,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            mask=mask,
            linewidths=0.5,
            annot_kws={"size": 8},
            ax=ax
        )

        ax.set_title("Correlation Heatmap")

        plt.tight_layout()

        return fig


    # Boxplot for Each Feature
    @output
    @render.plot
    def boxplots():

        feature_cols = [c for c in df.columns if c != "Class"]

        fig, axes = plt.subplots(4, 4, figsize=(22, 18))
        axes = axes.flatten()

        for i, col in enumerate(feature_cols):

            df.boxplot(
                column=col,
                by="Class",
                ax=axes[i]
            )

            axes[i].set_title(col, fontsize=9)
            axes[i].set_xlabel("")
            axes[i].tick_params(
                axis="x",
                rotation=45,
                labelsize=7
            )

        plt.suptitle(
            "Feature Distribution by Bean Class",
            fontsize=14
        )

        plt.tight_layout()

        return fig


    # Pairplot of Important Features
    @output
    @render.plot
    def pairplot():

        important_features = [
            "ShapeFactor2",
            "ShapeFactor1",
            "MinorAxisLength",
            "Roundness",
            "Eccentricity",
            "Class"
        ]

        available = [
            f for f in important_features
            if f in df.columns
        ]

        sample_df = df[available].sample(500, random_state=42)

        g = sns.pairplot(
            sample_df,
            hue="Class",
            diag_kind="kde",
            plot_kws={
                "alpha": 0.5,
                "s": 20
            }
        )

        return g.fig


    # Descriptive Statistics
    @output
    @render.ui
    def describe_table():

        describe = df.describe().round(2)

        return ui.HTML(
            describe.T.to_html(
                classes="table table-striped table-hover"
            )
        )


    # Statistics by Class
    @output
    @render.ui
    def stats_by_class_table():

        feature_cols = [
            c for c in df.columns
            if c != "Class"
        ]

        stats = (
            df.groupby("Class")[feature_cols]
            .agg(["mean", "std", "median"])
            .round(3)
        )

        return ui.HTML(
            stats.T.to_html(
                classes="table table-striped table-hover"
            )
        )


    # Coefficient of Variation (CV)
    @output
    @render.ui
    def cv_table():

        feature_cols = [
            c for c in df.columns
            if c != "Class"
        ]

        cv = (
            df.groupby("Class")[feature_cols]
            .agg(lambda x: round((x.std() / x.mean()) * 100, 2))
        )

        return ui.HTML(
            cv.T.to_html(
                classes="table table-striped table-hover"
            )
        )
    

    # Missing Values
    @output
    @render.ui
    def missing_values():

        return ui.HTML(f"""
        <table class="table table-striped">
            <tr>
                <th>Total Missing Values</th>
                <td>{missing_count}</td>
            </tr>
        </table>
        """)
    
    # Outlier Removal
    @output
    @render.ui
    def outlier_info():

        original_samples = len(df)
        cleaned_samples = len(df_clean)
        removed_samples = original_samples - cleaned_samples

        return ui.HTML(f"""
        <table class="table table-striped">
            <tr>
                <th>Original Samples</th>
                <td>{original_samples}</td>
            </tr>
            <tr>
                <th>Samples After Outlier Removal</th>
                <td>{cleaned_samples}</td>
            </tr>
            <tr>
                <th>Removed Samples</th>
                <td>{removed_samples}</td>
            </tr>
        </table>
        """)
    
    # Feature Engineering
    @output
    @render.ui
    def feature_engineering():

        original_features = df.shape[1] - 1
        engineered_features = X_fe.shape[1]
        new_features = engineered_features - original_features

        return ui.HTML(f"""
        <table class="table table-striped">
            <tr><th>Original Features</th><td>{original_features}</td></tr>
            <tr><th>Engineered Features</th><td>{engineered_features}</td></tr>
            <tr><th>New Features Added</th><td>{new_features}</td></tr>
        </table>
        """)
    
    # Label Encoding
    @output
    @render.ui
    def label_encoding():

        return ui.HTML("""
        <table class="table table-striped">
            <tr><th>Method</th><td>LabelEncoder</td></tr>
            <tr><th>Target Variable</th><td>Class</td></tr>
            <tr><th>Number of Classes</th><td>7</td></tr>
        </table>
        """)
    
   
    # Train-Test Split
    @output
    @render.ui
    def train_test_info():

        return ui.HTML(f"""
        <table class="table table-striped">
            <tr><th>Training Samples</th><td>{len(X_train)}</td></tr>
            <tr><th>Testing Samples</th><td>{len(X_test)}</td></tr>
            <tr><th>Split Ratio</th><td>80% / 20%</td></tr>
        </table>
        """)

     # Feature Scaling
    @output
    @render.ui
    def scaling_info():

        return ui.HTML(f"""
        <table class="table table-striped">
            <tr><th>Method</th><td>StandardScaler</td></tr>
            <tr><th>Training Shape</th><td>{X_train_sc.shape}</td></tr>
            <tr><th>Testing Shape</th><td>{X_test_sc.shape}</td></tr>
        </table>
        """)

    # SMOTE
    @output
    @render.ui
    def smote_info():

        return ui.HTML(f"""
        <table class="table table-striped">
            <tr><th>Method</th><td>SMOTE</td></tr>
            <tr><th>Training Before</th><td>{len(y_train)}</td></tr>
            <tr><th>Training After</th><td>{len(y_train_sm)}</td></tr>
        </table>
        """)
    
    # Hyperparameter Tuning

    @output
    @render.ui
    def hyperparameter_info():

        return ui.HTML(f"""
        <table class="table table-striped table-hover">

            <tr>
                <th>Best K for KNN</th>
                <td>{best_k}</td>
            </tr>

            <tr>
                <th>Best Kernel for SVM</th>
                <td>{best_kernel}</td>
            </tr>

        </table>
        """)

    #results
    @output
    @render.ui
    def results_table():

        return ui.HTML(
            results_df.round(2).to_html(
                index=False,
                classes="table table-striped table-hover"
            )
        )
    
    #best model
    @output
    @render.ui
    def best_model():

        return ui.HTML(f"""
        <table class="table table-striped">
            <tr>
                <th>Best Model</th>
                <td>{best_row["Model"]}</td>
            </tr>
            <tr>
                <th>Best Accuracy</th>
                <td>{best_row["Best_Accuracy"]:.2f}%</td>
            </tr>
        </table>
        """)


    
