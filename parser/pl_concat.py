import os 
import polars as pl 
import pathlib 


path_to_current_dir = os.getcwd();
parquet_files = os.listdir(path_to_current_dir)
parquet_files = list(filter(lambda parquet_file: pathlib.Path(parquet_file).suffix == ".parquet", parquet_files))
dfs = []

for parquet_file in parquet_files:
    path_to_parquet_file = os.path.join(path_to_current_dir, parquet_file)
    df = pl.read_parquet(path_to_parquet_file)
    if df.shape[0] > 0:
        dfs.append(
            df
        )

res_df = pl.concat(dfs)
res_df = (
    res_df.with_columns(
        pl.col("price").cast(pl.Int64).alias("price"),
        (pl.col("is_available") == "True").alias("is_available"),
        pl.col("year").cast(pl.Int64).alias("year"),
        pl.col("mileage").cast(pl.Int64).alias("mileage",)
    )
)
print(f"res_df shape: {res_df.shape}")
print(f"res_df schema: {res_df.schema}")
res_df.write_parquet("auto_ru_cars_price.parquet")
