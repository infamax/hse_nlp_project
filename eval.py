import argparse
import torch 
import torch.nn as nn
import numpy as np
import polars as pl
import os
import yaml
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from models.nn.dataset import AutoRUDataset
from models.nn.models import TransformerRegression
from transformers import AutoTokenizer


def compute_regression_metrics(model: nn.Module, test_dataset: torch.utils.data.Dataset, batch_size: int, device: str) -> dict[str, int]:
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    true_values = []
    predicted_values = []
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        true_values.extend(labels.cpu().numpy())
        with torch.no_grad():
            predicted_values.extend(model(input_ids.squeeze(), attention_mask.squeeze()).cpu().numpy())
        
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)
    metrics = {
        "mse": mean_squared_error(true_values, predicted_values),
        "r2_score": r2_score(true_values, predicted_values),
        "mae": mean_absolute_error(true_values, predicted_values),
    }

    return metrics
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset-folder",
        help="Folder that contain a test of dataset",
        type=str,
    )
    parser.add_argument("-c", "--config", help="Experiment config path", type=str)
    args = parser.parse_args()
    dataset_folder, config_path = args.dataset_folder, args.config

    path_to_test_dataset = os.path.join(dataset_folder, "test/test.parquet")
    
    if not os.path.exists(path_to_test_dataset):
        print(f"Check exists test dataset: {path_to_test_dataset}")
        return -1
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["model_name"]
    dropout_rate = config["dropout_rate"]
    max_len = config["max_len"]
    text_splitter = config["text_splitter"]
    batch_size = config["batch_size"]
    experiment_name = config["experiment_name"]
    path_to_save_model_best_weights = config["path_to_save_model_best_weights"]

    path_to_best_model_weigths = os.path.join(path_to_save_model_best_weights, f"{experiment_name}_best_model.pth")
    model = TransformerRegression(model_name=model_name, dropout_rate=dropout_rate)
    state_dict = torch.load(path_to_best_model_weigths)
    model.load_state_dict(state_dict)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_dataset = AutoRUDataset(path_to_test_dataset, tokenizer, max_len=max_len, text_splitter=text_splitter)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    metrics = compute_regression_metrics(
        model=model,
        test_dataset=test_dataset,
        batch_size=batch_size,
        device=device,
    )
    metrics = pl.DataFrame(metrics)
    print("metrics")
    print(metrics)
    
    
    return 0

if __name__ == "__main__":
    exit(main())
