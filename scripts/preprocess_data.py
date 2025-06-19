import argparse
import polars as pl
import re
import os
from sklearn.preprocessing import StandardScaler


TEXT_COLUMNS = [
    "gen",
    "color",
    "transmission",
    "drive",
    "wheel_type",
    "state",
    "model_name",
    "description",
]
NUMERICAL_FEATURES = ["tax", "mileage", "year"]


def clean_text(text: str) -> str:
    text = re.sub(r"\n\n+", "\n", text)
    text = re.sub(r"\t+", " ", text)
    text = re.sub(r" +", " ", text)
    text = text.capitalize()
    return text.strip()


def normalize_numerical_features(df: pl.DataFrame) -> pl.DataFrame:
    scaler = StandardScaler()
    X = df.select(NUMERICAL_FEATURES)
    scaler = StandardScaler()
    scaler.fit(X)
    normalized_features = scaler.transform(X)
    for numerical_feature_idx, numerical_feature_name in enumerate(NUMERICAL_FEATURES):
        df = df.with_columns(
            pl.Series(normalized_features[:, numerical_feature_idx]).alias(
                numerical_feature_name
            ),
        )
    return df


def process_df(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.concat_str(
            TEXT_COLUMNS,
            separator=" ",
        ).alias("text")
    )
    df = df.with_columns(
        pl.col("text")
        .map_elements(lambda text: clean_text(text), return_dtype=pl.String)
        .alias("text")
    )
    df = normalize_numerical_features(df)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset-folder", help="Path to dataset folder", type=str
    )
    args = parser.parse_args()
    dataset_folder = args.dataset_folder
    path_to_dataset = os.path.join(dataset_folder, "auto_ru_cars.parquet")
    df = pl.read_parquet(path_to_dataset)
    df = process_df(df)
    df.write_parquet(path_to_dataset)
    print(f"Write preprocess dataset to folder: {path_to_dataset}")
    return 0


if __name__ == "__main__":
    exit(main())
