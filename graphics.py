from result_paths import EXAM_DIR
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_model_culture_accuracy(df_final):
    """
    Plots the accuracy of different models across cultures.
    
    Parameters:
    - df_final: DataFrame containing columns 'culture', 'accuracy', and 'model'.
    """
        
    plt.figure(figsize=(8,5))

    sns.barplot(
        data=df_final,
        x="culture",
        y="accuracy",
        hue="model",
        errorbar=None
    )

    plt.ylabel("Accuracy")
    plt.xlabel("Culture")

    plt.title(
        "LLM performance across cultures"
    )

    plt.ylim(0,1)

    plt.tight_layout()

    plt.savefig(
        EXAM_DIR / "model_culture_accuracy.png",
        dpi=300
    )

    # plt.show()


def plot_model_questiontype_accuracy(df_final):
    """
    Plots the accuracy of different models across question types.
    
    Parameters:
    - df_final: DataFrame containing columns 'question_type', 'accuracy', and 'model'.
    """
    plt.figure(figsize=(8,5))


    sns.barplot(
        data=df_final,
        x="question_type",
        y="accuracy",
        hue="model",
        errorbar=None
    )


    plt.ylabel("Accuracy")
    plt.xlabel("Question type")

    plt.ylim(0,1)

    plt.xticks(rotation=30)

    plt.title(
        "LLM performance by question type"
    )


    plt.tight_layout()

    plt.savefig(
        EXAM_DIR / "model_questiontype_accuracy.png",
        dpi=300
    )


    # plt.show()


def plot_model_culture_qtype_accuracy(df_final):

    heatmap_data = df_final.pivot_table(
        values="accuracy",
        index="model",
        columns=[
            "culture",
            "question_type"
        ]
    )


    plt.figure(figsize=(12,4))


    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1, 
        cmap="viridis"
    )


    plt.title(
        "Accuracy across models, cultures and question types"
    )


    plt.tight_layout()


    plt.savefig(
        EXAM_DIR / "heatmap_accuracy.png",
        dpi=300
    )


    # plt.show()


def df_model_culture_dimension(df_detail):
    culture_scope = pd.read_csv(
        EXAM_DIR.parent.parent / "cultural_parameters" / "cultureScope.csv"
    )

    dimension_categories = (
        culture_scope[
            ["Fine-grained Dimension", "Category"]
        ]
        .drop_duplicates()
        .rename(
            columns={
                "Fine-grained Dimension": "dimension",
                "Category": "category"
            }
        )
    )

    df_dimension = (
        df_detail
        .merge(dimension_categories, on="dimension", how="left")
        .groupby(
            [
                "model",
                "culture",
                "category"
            ]
        )
        .agg(
            {
                "correct":"sum",
                "total":"sum"
            }
        )
        .reset_index()
    )

    df_dimension["accuracy"] = (
        df_dimension["correct"] /
        df_dimension["total"]
    )
    return df_dimension

def plot_model_culture_dimension(df_dimension):
    
    plt.figure(figsize=(12,6))


    sns.barplot(
        data=df_dimension,
        x="category",
        y="accuracy",
        hue="model", 
        errorbar=None
    )


    plt.ylabel("Accuracy")
    plt.xlabel("Category")


    plt.title("Performance across cultural categories")


    plt.ylim(0,1)

    plt.xticks(rotation=45)

    plt.tight_layout()


    plt.savefig(
        EXAM_DIR / "model_category.png",
        dpi=300
    )

    # plt.show()