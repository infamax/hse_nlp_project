import argparse
import os
import polars as pl
import pathlib
from utils.consts import SEED
from sklearn.model_selection import train_test_split

TRAIN_FOLDER, VAL_FOLDER, TEST_FOLDER = "train", "val", "test"


def _save_dataset(df: pl.DataFrame, path_to_dataset: str):
    folder_name = pathlib.Path(path_to_dataset).parent
    os.makedirs(folder_name, exist_ok=True)
    df.write_parquet(path_to_dataset)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset-folder", help="Path to dataset folder", type=str
    )
    args = parser.parse_args()
    dataset_folder = args.dataset_folder
    path_to_dataset = os.path.join(dataset_folder, "auto_ru_cars.parquet")
    df = pl.read_parquet(path_to_dataset)
    df_train, df_tmp = train_test_split(df, test_size=0.4, random_state=SEED)
    df_val, df_test = train_test_split(df_tmp, test_size=0.5, random_state=SEED)
    path_to_train_dataset = os.path.join(dataset_folder, TRAIN_FOLDER, "train.parquet")
    path_to_val_dataset = os.path.join(dataset_folder, VAL_FOLDER, "val.parquet")
    path_to_test_dataset = os.path.join(dataset_folder, TEST_FOLDER, "test.parquet")
    _save_dataset(df_train, path_to_train_dataset)
    _save_dataset(df_val, path_to_val_dataset)
    _save_dataset(df_test, path_to_test_dataset)
    print(f"""
          Successfully split dataset. 
          Path to train dataset: {path_to_train_dataset}
          Path to val dataset: {path_to_val_dataset}
          Path to test dataset: {path_to_test_dataset}
          """)
    return 0


if __name__ == "__main__":
    exit(main())
